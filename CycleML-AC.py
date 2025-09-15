import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import datetime
from typing import Dict, List, Tuple
import copy
from torch.autograd import grad
from data_process import load_fault_diagnosis_data, create_data_loader_generator
from evaluation_module import integrate_evaluation_into_training, train_with_evaluation


class Discriminator(nn.Module):
    def __init__(self, sample_shape, data_length, df):
        super(Discriminator, self).__init__()
        
        # 存储参数
        self.sample_shape = sample_shape
        self.data_length = data_length
        self.df = df  # discriminator filters
        
        # 定义判别器层
        self.d1 = self._d_layer(1, self.df, normalization=False)
        self.d2 = self._d_layer(self.df, self.df*2)
        self.d3 = self._d_layer(self.df*2, self.df*4)
        self.d4 = self._d_layer(self.df*4, self.df*8, normalization=False)
        
        # 计算最终特征图大小
        # 经过4次步长为2的卷积，32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        final_feature_size = 2 * 2 * self.df * 8
        
        # 输出层 - 只输出validity
        self.validity_head = nn.Linear(final_feature_size, 1)
        
    def _d_layer(self, in_channels, out_channels, f_size=5, normalization=True, zero_padding=False):
        """判别器层"""
        layers = []
        
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=f_size, stride=2, padding=2))
        
        # 零填充（如果需要）
        if zero_padding:
            layers.append(nn.ZeroPad2d((0, 1, 0, 1)))
            
        # 批归一化（如果需要）
        if normalization:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.8))
            
        # 激活函数和dropout
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout2d(0.25))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 输入形状：(batch_size, data_length)
        batch_size = x.size(0)
        
        # 重塑为 (batch_size, data_length, 1)
        x = x.unsqueeze(-1)
        
        # 计算需要填充的零的数量
        zeros_needed = 32 * 32 - self.data_length
        
        # 在末尾填充零：(batch_size, data_length, 1) -> (batch_size, 32*32, 1)
        if zeros_needed > 0:
            padding = torch.zeros(batch_size, zeros_needed, 1, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # 重塑为图像格式：(batch_size, 32*32, 1) -> (batch_size, 1, 32, 32)
        x = x.view(batch_size, 1, 32, 32)
        
        # 通过判别器层
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 输出 - 只返回validity
        validity = self.validity_head(x)
        
        return validity
    
class Generator(nn.Module):
    def __init__(self, sample_shape, data_length, gf, num_classes):
        super(Generator, self).__init__()
        
        # 存储参数
        self.sample_shape = sample_shape
        self.data_length = data_length
        self.gf = gf
        self.num_classes = num_classes
        
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, 32 * 32)
        
        # 编码器（下采样）层
        self.conv1 = self._conv2d_block(2, self.gf)
        self.conv2 = self._conv2d_block(self.gf, self.gf * 2)
        self.conv3 = self._conv2d_block(self.gf * 2, self.gf * 4)
        self.conv4 = self._conv2d_block(self.gf * 4, self.gf * 8)
        
        # 解码器（上采样）层 - 修改这里，分开定义上采样和卷积
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = self._deconv2d_layer(self.gf * 8 + self.gf * 4, self.gf * 4)  # +skip
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2 = self._deconv2d_layer(self.gf * 4 + self.gf * 2, self.gf * 2)  # +skip
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv3 = self._deconv2d_layer(self.gf * 2 + self.gf, self.gf)  # +skip
        
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(self.gf, 1, kernel_size=5, stride=1, padding=2)
        
        # 输出层
        self.output_dense = nn.Linear(32 * 32, self.data_length)
        self.output_bn = nn.BatchNorm1d(self.data_length)
    
    def _conv2d_block(self, in_channels, out_channels, f_size=5):
        """编码器卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=f_size, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def _deconv2d_layer(self, in_channels, out_channels, f_size=5, dropout_rate=0):
        """解码器卷积层（不包括上采样）"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=f_size, stride=1, padding=2),
            nn.ReLU(inplace=True)
        ]
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
            
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, sample, label):
        batch_size = sample.size(0)
        
        # 处理标签形状
        if label.dim() > 1:
            label = label.squeeze(-1)
        
        # 1. 处理样本输入
        sample_expanded = sample.unsqueeze(-1)
        zeros_needed = 32 * 32 - self.data_length
        
        if zeros_needed > 0:
            padding = torch.zeros(batch_size, zeros_needed, 1, device=sample.device)
            padded_sample = torch.cat([sample_expanded, padding], dim=1)
        else:
            padded_sample = sample_expanded
        
        reshaped_sample = padded_sample.view(batch_size, 1, 32, 32)
        
        # 2. 处理标签输入
        embedded_label = self.label_embedding(label.long())
        reshaped_label = embedded_label.view(batch_size, 1, 32, 32)
        
        # 3. 连接样本和标签
        concatenated = torch.cat([reshaped_sample, reshaped_label], dim=1)
        
        # 4. 编码器（下采样）
        d1 = self.conv1(concatenated)  # (batch_size, gf, 16, 16)
        d2 = self.conv2(d1)            # (batch_size, gf*2, 8, 8)
        d3 = self.conv3(d2)            # (batch_size, gf*4, 4, 4)
        d4 = self.conv4(d3)            # (batch_size, gf*8, 2, 2)
        
        # 5. 解码器（上采样）带跳跃连接
        u1 = self.up1(d4)              # 上采样到 (batch_size, gf*8, 4, 4)
        u1 = torch.cat([u1, d3], dim=1)  # 跳跃连接
        u1 = self.deconv1(u1)          # (batch_size, gf*4, 4, 4)
        
        u2 = self.up2(u1)              # 上采样到 (batch_size, gf*4, 8, 8)
        u2 = torch.cat([u2, d2], dim=1)  # 跳跃连接
        u2 = self.deconv2(u2)          # (batch_size, gf*2, 8, 8)
        
        u3 = self.up3(u2)              # 上采样到 (batch_size, gf*2, 16, 16)
        u3 = torch.cat([u3, d1], dim=1)  # 跳跃连接
        u3 = self.deconv3(u3)          # (batch_size, gf, 16, 16)
        
        # 6. 最终上采样和卷积
        u4 = self.up4(u3)              # (batch_size, gf, 32, 32)
        u4 = self.final_conv(u4)       # (batch_size, 1, 32, 32)
        
        # 7. 输出处理
        flattened = u4.view(batch_size, -1)
        output = self.output_dense(flattened)
        output = self.output_bn(output)
        
        return output


class AuxiliaryClassifier(nn.Module):
    def __init__(self, input_size=512, num_classes=10):
        super(AuxiliaryClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Conv1: 16 filters, kernel size 16
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv2: 32 filters, kernel size 3
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Conv3: 64 filters, kernel size 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Conv4: 64 filters, kernel size 3
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        
        # Conv5: 64 filters, kernel size 3
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.pool5 = nn.MaxPool1d(kernel_size=2)
        
        # 计算经过所有卷积和池化后的特征尺寸
        # 初始尺寸: 512
        # 经过5次MaxPool1d(kernel_size=2): 512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.flattened_size = 960 #个通道，每个通道16个特征
        
        # 全连接层
        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
        # Dropout层（可选，用于防止过拟合）
        self.dropout = nn.Dropout(0.3)
        
    def _conv_block(self, conv_layer, bn_layer, pool_layer, x):
        """卷积块：Conv + ReLU + BatchNorm + MaxPool1D"""
        x = conv_layer(x)
        x = F.relu(x)
        x = bn_layer(x)
        x = pool_layer(x)
        return x
    
    def forward(self, x):
        # 输入: (batch_size, 512)
        batch_size = x.size(0)
        
        # Reshape为1D卷积格式: (batch_size, 1, 512)
        # 这里将512个输入单元reshape为1个通道，512个时间步
        x = x.view(batch_size, 1, self.input_size)
        
        # Conv1 + ReLU + BN + Max1D
        x = self._conv_block(self.conv1, self.bn1, self.pool1, x)  # (batch_size, 16, 256)
        
        # Conv2 + ReLU + BN + Max1D
        x = self._conv_block(self.conv2, self.bn2, self.pool2, x)  # (batch_size, 32, 128)
        
        # Conv3 + ReLU + BN + Max1D
        x = self._conv_block(self.conv3, self.bn3, self.pool3, x)  # (batch_size, 64, 64)
        
        # Conv4 + ReLU + BN + Max1D
        x = self._conv_block(self.conv4, self.bn4, self.pool4, x)  # (batch_size, 64, 32)
        
        # Conv5 + ReLU + BN + Max1D
        x = self._conv_block(self.conv5, self.bn5, self.pool5, x)  # (batch_size, 64, 16)
        
        # 展平特征
        x = x.view(batch_size, -1)  # (batch_size, 64*16)
        
        # FC1
        x = self.fc1(x)  # (batch_size, 100)
        x = F.relu(x)
        x = self.dropout(x)  # 添加dropout防止过拟合
        
        # FC2
        x = self.fc2(x)  # (batch_size, 1)
        
        return x

class CycleML_AC:
    def __init__(self, data_length=512, num_classes=10):
        self.data_length = data_length
        self.sample_shape = (data_length,)
        self.num_classes = num_classes
        
        # Network parameters
        self.gf = 32  # Generator filters
        self.df = 32  # Discriminator filters
        
        # Loss weights
        self.lambda_adv = 1
        self.lambda_cycle = 10
        self.lambda_ac = 1  # Auxiliary classifier weight
        self.lambda_gp = 10  # Gradient penalty weight
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build networks
        self.d_1 = Discriminator(self.sample_shape, self.data_length, self.df).to(self.device)
        self.d_2 = Discriminator(self.sample_shape, self.data_length, self.df).to(self.device)
        
        self.g_AB = Generator(self.sample_shape, self.data_length, self.gf, self.num_classes).to(self.device)
        self.g_BA = Generator(self.sample_shape, self.data_length, self.gf, self.num_classes).to(self.device)
        
        self.ac = AuxiliaryClassifier(input_size=self.data_length, num_classes=self.num_classes).to(self.device)
        
        # Meta-learning optimizers for auxiliary classifier
        self.ac_optimizer_inner = optim.SGD(self.ac.parameters(), lr=0.01)
        self.ac_optimizer_outer = optim.Adam(self.ac.parameters(), lr=0.001)
        
        # Meta-learning optimizers for discriminators
        self.d1_optimizer_inner = optim.SGD(self.d_1.parameters(), lr=0.01)
        self.d1_optimizer_outer = optim.Adam(self.d_1.parameters(), lr=0.0005)
        
        self.d2_optimizer_inner = optim.SGD(self.d_2.parameters(), lr=0.01)
        self.d2_optimizer_outer = optim.Adam(self.d_2.parameters(), lr=0.0005)
        
        # Meta-learning optimizers for generators
        self.g_optimizer_inner = optim.SGD(
            list(self.g_AB.parameters()) + list(self.g_BA.parameters()), 
            lr=0.01
        )
        self.g_optimizer_outer = optim.Adam(
            list(self.g_AB.parameters()) + list(self.g_BA.parameters()), 
            lr=0.0005
        )
        
    def wasserstein_loss(self, y_pred, y_true):
        """Wasserstein loss for WGAN"""
        return -torch.mean(y_true * y_pred)
    
    def gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_samples)
        
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        validity = discriminator(interpolated)
        
        gradients = grad(outputs=validity, inputs=interpolated,
                        grad_outputs=torch.ones_like(validity),
                        create_graph=True, retain_graph=True)[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty
    
    def meta_train_auxiliary_classifier(self, tasks_data, num_inner_steps=1):
        """Meta-learning training for auxiliary classifier"""
        meta_grads = []
        
        for task_data in tasks_data:
            
            original_params = [p.clone() for p in self.ac.parameters()]

            
            support_x, support_y = task_data['support']
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device).long()
            
            for _ in range(num_inner_steps):
                pred = self.ac(support_x)
                inner_loss = F.cross_entropy(pred, support_y)

                # 计算梯度但不更新
                grads = torch.autograd.grad(inner_loss, self.ac.parameters(), create_graph=True)
                
                # 手动更新参数（创建计算图）
                for p, g in zip(self.ac.parameters(), grads):
                    p.data = p.data - 0.01 * g  # 0.01 是学习率
                
                # inner_optimizer.zero_grad()
                # inner_loss.backward()
                # inner_optimizer.step()
            
            # Outer loop - compute gradients on query set
            query_x, query_y = task_data['query']
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device).long()
            
            pred = self.ac(query_x)
            outer_loss = F.cross_entropy(pred, query_y)
            
            # Compute gradients w.r.t original model parameters
            task_grad = grad(outer_loss, self.ac.parameters(), create_graph=False)
            meta_grads.append(task_grad)

            for p, orig_p in zip(self.ac.parameters(), original_params):
                p.data = orig_p.data
        
        # Average gradients across tasks
        avg_grads = []
        for i in range(len(meta_grads[0])):
            avg_grad = torch.mean(torch.stack([g[i] for g in meta_grads]), dim=0)
            avg_grads.append(avg_grad)
        
        # Update original model
        self.ac_optimizer_outer.zero_grad()
        for param, g in zip(self.ac.parameters(), avg_grads):
            param.grad = g
        self.ac_optimizer_outer.step()
    
    def meta_train_discriminator(self, discriminator, generator, optimizer_inner, optimizer_outer,
                                tasks_data, source_domain='A', num_inner_steps=1):
        """Meta-learning training for discriminator"""
        meta_grads = []
        
        for task_data in tasks_data:
            # Clone discriminator for inner loop
            # d_clone = copy.deepcopy(discriminator)
            # inner_optimizer = optim.SGD(d_clone.parameters(), lr=0.01)

            original_params = [p.clone() for p in discriminator.parameters()]
            
            # Inner loop - adaptation on support set
            if source_domain == 'A':
                source_samples, source_labels = task_data['support_A']
                target_samples, target_labels = task_data['support_B']
            else:
                source_samples, source_labels = task_data['support_B']
                target_samples, target_labels = task_data['support_A']
            
            source_samples = source_samples.to(self.device)
            source_labels = source_labels.to(self.device).long()
            target_samples = target_samples.to(self.device)
            target_labels = target_labels.to(self.device).long()
            
            for _ in range(num_inner_steps):
                # Generate fake samples
                with torch.no_grad():
                    fake_samples = generator(source_samples, target_labels)
                
                # Discriminator predictions
                real_validity = discriminator(target_samples)
                fake_validity = discriminator(fake_samples.detach())
                
                # WGAN loss
                d_loss_real = -torch.mean(real_validity)
                d_loss_fake = torch.mean(fake_validity)
                
                # Gradient penalty
                gp = self.gradient_penalty(discriminator, target_samples, fake_samples.detach())
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gp

                # 计算梯度但保持计算图
                grads = torch.autograd.grad(d_loss, discriminator.parameters(), create_graph=True)
                
                # 手动更新参数
                for p, g in zip(discriminator.parameters(), grads):
                    p.data = p.data - 0.01 * g  # 0.01 是学习率
            
            # Outer loop - compute gradients on query set
            if source_domain == 'A':
                source_samples, source_labels = task_data['query_A']
                target_samples, target_labels = task_data['query_B']
            else:
                source_samples, source_labels = task_data['query_B']
                target_samples, target_labels = task_data['query_A']
            
            source_samples = source_samples.to(self.device)
            source_labels = source_labels.to(self.device).long()
            target_samples = target_samples.to(self.device)
            target_labels = target_labels.to(self.device).long()
            
            # Generate fake samples
            with torch.no_grad():
                fake_samples = generator(source_samples, target_labels)
            
            # Discriminator predictions
            real_validity = discriminator(target_samples)
            fake_validity = discriminator(fake_samples.detach())
            
            # WGAN loss
            d_loss_real = -torch.mean(real_validity)
            d_loss_fake = torch.mean(fake_validity)
            
            # Gradient penalty
            gp = self.gradient_penalty(discriminator, target_samples, fake_samples.detach())
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gp
            
            # Compute gradients w.r.t original model parameters
             # 计算梯度
            task_grad = torch.autograd.grad(d_loss, discriminator.parameters(), retain_graph=True)
            meta_grads.append(task_grad)

            # 恢复原始参数
            for p, orig_p in zip(discriminator.parameters(), original_params):
                p.data = orig_p.data
        
        # Average gradients across tasks
        avg_grads = []
        for i in range(len(meta_grads[0])):
            avg_grad = torch.mean(torch.stack([g[i] for g in meta_grads]), dim=0)
            avg_grads.append(avg_grad)
        
        # Update original model
        optimizer_outer.zero_grad()
        for param, g in zip(discriminator.parameters(), avg_grads):
            param.grad = g
        optimizer_outer.step()
    
    def meta_train_generators(self, tasks_data, num_inner_steps=1):
        """Meta-learning training for generators"""
        meta_grads = []
        
        for task_data in tasks_data:

            # 保存原始参数
            original_params_AB = [p.clone() for p in self.g_AB.parameters()]
            original_params_BA = [p.clone() for p in self.g_BA.parameters()]
            
            
            # Inner loop - adaptation on support set
            samples_A, labels_A = task_data['support_A']
            samples_B, labels_B = task_data['support_B']
            
            samples_A = samples_A.to(self.device)
            labels_A = labels_A.to(self.device).long()
            samples_B = samples_B.to(self.device)
            labels_B = labels_B.to(self.device).long()
            
            for _ in range(num_inner_steps):
                # Generate fake samples
                fake_B = self.g_AB(samples_A, labels_B)
                fake_A = self.g_BA(samples_B, labels_A)
                
                # Cycle consistency
                reconstr_A = self.g_BA(fake_B, labels_A)
                reconstr_B = self.g_AB(fake_A, labels_B)
                
                # Adversarial loss
                adv_loss_B = -torch.mean(self.d_2(fake_B))
                adv_loss_A = -torch.mean(self.d_1(fake_A))
                
                # Cycle consistency loss
                cycle_loss_A = F.l1_loss(reconstr_A, samples_A)
                cycle_loss_B = F.l1_loss(reconstr_B, samples_B)
                
                # Auxiliary classifier loss
                ac_pred_fake_A = self.ac(fake_A)
                ac_pred_fake_B = self.ac(fake_B)
                ac_loss_A = F.cross_entropy(ac_pred_fake_A, labels_A)
                ac_loss_B = F.cross_entropy(ac_pred_fake_B, labels_B)
                
                # Total generator loss
                g_loss = (self.lambda_adv * (adv_loss_A + adv_loss_B) +
                         self.lambda_cycle * (cycle_loss_A + cycle_loss_B) +
                         self.lambda_ac * (ac_loss_A + ac_loss_B))
                
                # 计算梯度并手动更新
                all_params = list(self.g_AB.parameters()) + list(self.g_BA.parameters())
                grads = torch.autograd.grad(g_loss, all_params, create_graph=True)
                
                # 手动更新参数
                for p, g in zip(all_params, grads):
                    p.data = p.data - 0.01 * g
                
                
            
            # Outer loop - compute gradients on query set
            samples_A, labels_A = task_data['query_A']
            samples_B, labels_B = task_data['query_B']
            
            samples_A = samples_A.to(self.device)
            labels_A = labels_A.to(self.device).long()
            samples_B = samples_B.to(self.device)
            labels_B = labels_B.to(self.device).long()
            
            # Generate fake samples
            fake_B = self.g_AB(samples_A, labels_B)
            fake_A = self.g_BA(samples_B, labels_A)
            
            # Cycle consistency
            reconstr_A = self.g_BA(fake_B, labels_A)
            reconstr_B = self.g_AB(fake_A, labels_B)
            
            # Adversarial loss
            adv_loss_B = -torch.mean(self.d_2(fake_B))
            adv_loss_A = -torch.mean(self.d_1(fake_A))
            
            # Cycle consistency loss
            cycle_loss_A = F.l1_loss(reconstr_A, samples_A)
            cycle_loss_B = F.l1_loss(reconstr_B, samples_B)
            
            # Auxiliary classifier loss
            ac_pred_fake_A = self.ac(fake_A)
            ac_pred_fake_B = self.ac(fake_B)
            ac_loss_A = F.cross_entropy(ac_pred_fake_A, labels_A)
            ac_loss_B = F.cross_entropy(ac_pred_fake_B, labels_B)
            
            # Total generator loss
            g_loss = (self.lambda_adv * (adv_loss_A + adv_loss_B) +
                     self.lambda_cycle * (cycle_loss_A + cycle_loss_B) +
                     self.lambda_ac * (ac_loss_A + ac_loss_B))
            
            # Compute gradients w.r.t original model parameters
            all_params = list(self.g_AB.parameters()) + list(self.g_BA.parameters())
            task_grad = torch.autograd.grad(g_loss, all_params, retain_graph=True)
            meta_grads.append(task_grad)

            # 恢复原始参数
            for p, orig_p in zip(self.g_AB.parameters(), original_params_AB):
                p.data = orig_p.data
            for p, orig_p in zip(self.g_BA.parameters(), original_params_BA):
                p.data = orig_p.data
        
        # Average gradients across tasks
        avg_grads = []
        for i in range(len(meta_grads[0])):
            avg_grad = torch.mean(torch.stack([g[i] for g in meta_grads]), dim=0)
            avg_grads.append(avg_grad)
        
        # Update original models
        self.g_optimizer_outer.zero_grad()
        all_params = list(self.g_AB.parameters()) + list(self.g_BA.parameters())
        for param, g in zip(all_params, avg_grads):
            param.grad = g
        self.g_optimizer_outer.step()
    
    def train(self, epochs, tasks_data_loader, num_tasks=9):
        """Main training loop"""
        start_time = datetime.datetime.now()
        
        for epoch in range(epochs):
            # Get batch of tasks
            tasks_data = next(tasks_data_loader)
            
            # Step 1: Meta-train auxiliary classifier
            print(f"Epoch {epoch}: Training auxiliary classifier...")
            self.meta_train_auxiliary_classifier(tasks_data)
            
            # Step 2: Meta-train discriminators
            print(f"Epoch {epoch}: Training discriminators...")
            self.meta_train_discriminator(
                self.d_1, self.g_BA, self.d1_optimizer_inner, self.d1_optimizer_outer,
                tasks_data, source_domain='B'
            )
            self.meta_train_discriminator(
                self.d_2, self.g_AB, self.d2_optimizer_inner, self.d2_optimizer_outer,
                tasks_data, source_domain='A'
            )
            
            # Step 3: Meta-train generators
            print(f"Epoch {epoch}: Training generators...")
            self.meta_train_generators(tasks_data)
            
            elapsed_time = datetime.datetime.now() - start_time
            
            # Evaluate and print losses (simplified version)
            if epoch % 10 == 0:
                with torch.no_grad():
                    # Get a sample task for evaluation
                    eval_task = tasks_data[0]
                    samples_A, labels_A = eval_task['query_A']
                    samples_B, labels_B = eval_task['query_B']
                    
                    samples_A = samples_A.to(self.device)
                    labels_A = labels_A.to(self.device).long()
                    samples_B = samples_B.to(self.device)
                    labels_B = labels_B.to(self.device).long()
                    
                    # Generate fake samples
                    fake_B = self.g_AB(samples_A, labels_B)
                    fake_A = self.g_BA(samples_B, labels_A)
                    
                    # Discriminator predictions
                    d1_real = self.d_1(samples_A)
                    d1_fake = self.d_1(fake_A)
                    d2_real = self.d_2(samples_B)
                    d2_fake = self.d_2(fake_B)
                    
                    # AC predictions
                    ac_pred_A = self.ac(samples_A)
                    ac_pred_B = self.ac(samples_B)
                    ac_acc_A = (ac_pred_A.argmax(1) == labels_A).float().mean()
                    ac_acc_B = (ac_pred_B.argmax(1) == labels_B).float().mean()
                    
                    print(f"[Epoch {epoch}] D1_real: {d1_real.mean():.4f}, "
                          f"D1_fake: {d1_fake.mean():.4f}, "
                          f"D2_real: {d2_real.mean():.4f}, "
                          f"D2_fake: {d2_fake.mean():.4f}, "
                          f"AC_acc_A: {ac_acc_A:.4f}, AC_acc_B: {ac_acc_B:.4f}, "
                          f"Time: {elapsed_time}")
            
            # Save models periodically
            if epoch % 100 == 0:
                self.save_models(epoch)
    
    def save_models(self, epoch):
        """Save all models"""
        torch.save(self.g_AB.state_dict(), f'g_AB_epoch_{epoch}.pth')
        torch.save(self.g_BA.state_dict(), f'g_BA_epoch_{epoch}.pth')
        torch.save(self.d_1.state_dict(), f'd_1_epoch_{epoch}.pth')
        torch.save(self.d_2.state_dict(), f'd_2_epoch_{epoch}.pth')
        torch.save(self.ac.state_dict(), f'ac_epoch_{epoch}.pth')
        print(f"Models saved at epoch {epoch}")
    
    def load_models(self, epoch):
        """Load all models"""
        self.g_AB.load_state_dict(torch.load(f'g_AB_epoch_{epoch}.pth'))
        self.g_BA.load_state_dict(torch.load(f'g_BA_epoch_{epoch}.pth'))
        self.d_1.load_state_dict(torch.load(f'd_1_epoch_{epoch}.pth'))
        self.d_2.load_state_dict(torch.load(f'd_2_epoch_{epoch}.pth'))
        self.ac.load_state_dict(torch.load(f'ac_epoch_{epoch}.pth'))
        print(f"Models loaded from epoch {epoch}")


def prepare_tasks_data(domain_A_data, domain_B_data, num_tasks=9, support_size=2, query_size=3):
    """Prepare tasks data for meta-learning"""
    tasks = []
    
    for task_id in range(num_tasks):
        # For auxiliary classifier training (using domain B data)
        support_idx = np.random.choice(len(domain_B_data[task_id]['X']), support_size, replace=False)
        query_idx = np.random.choice(
            [i for i in range(len(domain_B_data[task_id]['X'])) if i not in support_idx],
            query_size, replace=False
        )
        
        task_data = {
            'support': (
                torch.FloatTensor(domain_B_data[task_id]['X'][support_idx]),
                torch.LongTensor(domain_B_data[task_id]['Y'][support_idx])
            ),
            'query': (
                torch.FloatTensor(domain_B_data[task_id]['X'][query_idx]),
                torch.LongTensor(domain_B_data[task_id]['Y'][query_idx])
            ),
            'support_A': (
                torch.FloatTensor(domain_A_data['X'][:support_size]),
                torch.LongTensor(domain_A_data['Y'][:support_size])
            ),
            'query_A': (
                torch.FloatTensor(domain_A_data['X'][support_size:support_size+query_size]),
                torch.LongTensor(domain_A_data['Y'][support_size:support_size+query_size])
            ),
            'support_B': (
                torch.FloatTensor(domain_B_data[task_id]['X'][support_idx]),
                torch.LongTensor(domain_B_data[task_id]['Y'][support_idx])
            ),
            'query_B': (
                torch.FloatTensor(domain_B_data[task_id]['X'][query_idx]),
                torch.LongTensor(domain_B_data[task_id]['Y'][query_idx])
            )
        }
        tasks.append(task_data)
    
    return tasks


# Example usage
if __name__ == '__main__':
    

    CycleML_AC.train_with_evaluation = train_with_evaluation

    # Initialize the model
    model = CycleML_AC(data_length=512, num_classes=10)

    # 加载数据
    dataset, tasks = load_fault_diagnosis_data(
        data_path='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_2/github/dataset_fft_for_cyclegan_case1_512.npz',
        num_samples_per_class=5,  # 每个故障类使用5个样本
        data_length=512,  # FFT数据长度
        support_size=2,
        query_size=3
    )
    # 创建数据生成器
    tasks_data_loader = create_data_loader_generator(
        dataset, 
        support_size=2, 
        query_size=3
    )

    # 使用带评估的训练
    eval_history = model.train_with_evaluation(
        epochs=3000,
        tasks_data_loader=tasks_data_loader,
        dataset=dataset,  # 传入dataset用于评估
        num_tasks=9,
        eval_frequency=10,  # 每10个epoch评估一次
        save_frequency=100  # 每100个epoch保存模型
    )
    
    # # Prepare dummy data (replace with your actual data loading)
    # # This is just an example structure
    # domain_A_data = {
    #     'X': np.random.randn(1000, 512).astype(np.float32),
    #     'Y': np.zeros(1000, dtype=np.int64)  # Healthy samples (class 0)
    # }
    
    # domain_B_data = {}
    # for i in range(9):  # 10 fault types
    #     domain_B_data[i] = {
    #         'X': np.random.randn(100, 512).astype(np.float32),
    #         'Y': np.full(100, i+1, dtype=np.int64)  # Fault classes 1-9
    #     }
    
    # Training loop
    # epochs = 100
    # model.train(epochs, tasks_data_loader, num_tasks=9)
    # for epoch in range(epochs):
    #     # Prepare tasks for this epoch
    #     tasks_data = prepare_tasks_data(domain_A_data, domain_B_data)
        
    #     # Create a simple data loader (generator)
    #     def tasks_data_loader():
    #         while True:
    #             yield tasks_data
        
    #     # Train for one epoch
    #     model.train(1, tasks_data_loader(), num_tasks=9)
        
    #     print(f"Completed epoch {epoch+1}/{epochs}")



