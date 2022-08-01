

## Set up your system
  1. Install [PyTorch](https://pytorch.org/get-started/locally/) version 1.9.0
  2. Install [RLlib](https://docs.ray.io/en/latest/rllib.html) version 1.11
  3. Install [Google Research Football](https://github.com/google-research/football/) version 2.10.1

```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch

```

## Running the code
  1. Make sure your environment is set up properly by following the installation and verification steps for all packages above.
  2. Then, run `python scripts\run_impala.py`


