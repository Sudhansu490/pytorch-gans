# pytorch-gans

Vanilla GAN (MLP Generator + Discriminator) on CelebA with PyTorch. DCGAN coming next in the same repo.

## Notebooks

- `vanilla_gan.ipynb` — vanilla GAN, 64x64 CelebA, BCELoss, Adam(lr=0.0002, betas=(0.5, 0.999)), z_dim=100, batch_size=128
- `dcgan.ipynb` — planned (convolutional Generator/Discriminator)

## Dataset setup (not committed)

Download CelebA aligned faces (`img_align_celeba`, 202599 images) from Kaggle and place it next to the notebooks:

- Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset (`img_align_celeba.zip` — 202,599 cropped/aligned face images)
- Original source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (non-commercial research only)

```
pytorch-gans/
├── vanilla_gan.ipynb
├── dcgan.ipynb
├── pyproject.toml
├── README.md
└── img_align_celeba/   # git-ignored, local only
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

The notebook expects:

```python
root_dir_path = "./img_align_celeba/"
```

## Install

```bash
pip install -e ".[notebook]"
# or: pip install torch torchvision numpy matplotlib pillow jupyter ipykernel
```

## Run

```bash
jupyter notebook vanilla_gan.ipynb
# Run All top-to-bottom: dataset -> dataloader -> models -> train(epochs=5)
```

Training prints `G-loss / D-loss` every 50 batches and shows one 8-image grid per epoch.

## Notes

- CPU training on full CelebA (202k x 5 epochs) is very slow; use CUDA/MPS if available (`device` cell auto-detects).
- `img_align_celeba/`, `.ipynb_checkpoints/`, `.virtual_documents/` are git-ignored — never commit them.
- Clear notebook outputs before committing (`Kernel -> Restart & Clear Output`).
