# InventoryRL

```bash
conda create -n inv python=3.10
conda activate inv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib numpy pandas scipy
conda install -c conda-forge scikit-learn
pip install pyyaml munch tqdm tensorboard gym[atari] pathos wandb[azure]
```