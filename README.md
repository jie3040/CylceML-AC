# CylceML-AC
The implementation of CylceML-AC for intelligent fault diagnosis with limited data

# Dataset
This example is based on the Case Western Reserve University bearing dataset

# Requirements 
```
python = 3.9.20
torch version: 2.4.0
pandas = 1.3.5
tiktoken version: 0.7.0
scikit-learn = 1.0.2                                                                                                     `
```
The codes was tested with pytorch 2.4.0, CUDA driver 12.7, CUDA toolkit 11.8, Ubuntu 22.04.

# Quick strat
```
python CycleML_AC.py
```
The dataset has been convert to fresuency domain by FFT and each sample contains 512 points.
