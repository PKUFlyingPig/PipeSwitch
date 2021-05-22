# Reproduce experiment figures

## Setup

All experiments are conducted on AWS. We use two EC2 instance types. One is p3.2xlarge, which is configured with 8 vCPUs (Intel Xeon E5-2686 v4), 1 GPU (NVIDIA V100 with 16 GB GPU memory), PCIe 3.0 ×16, and 61 GB memory. The other is g4dn.2xlarge, which is configured with 8 vCPUs (Intel Platinum 8259CL), 1 GPU (NVIDIA T4 with 16 GB GPU memory), PCIe 3.0 ×8, and 32 GB memory. The software environment includes PyTorch-1.3.0, torchvision- 0.4.2, scipy-1.3.2, and CUDA-10.1.

