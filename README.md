<h1 align="center">● MicroFoundation</h1>

### 2D Segmentation
```bash
CUDA_VISIBLE_DEVICES=0 python train_2d.py \
  -net sam2 \
  -exp_name micro \
  -vis 1 \
  -sam_config sam2_hiera_s \
  -image_size 1024 \
  -out_size 1024 \
  -b 2 \
  -val_freq 10 \
  -dataset Micro2D \
  -data_path {data_path} \
  -sam_ckpt pretrain.pt

### 3D Segmentation
```bash
CUDA_VISIBLE_DEVICES=0 python train_3d.py \
  -net sam2 \
  -exp_name micro \
  -sam_config sam2_hiera_s \
  -image_size 1024 \
  -val_freq 10 \
  -prompt bbox \
  -prompt_freq 2 \
  -dataset btcv \
  -data_path {data_path} \
  -sam_ckpt pretrain.pt

### Deblurring / Super-Resolution
CUDA_VISIBLE_DEVICES=0 python train_restore.py \
  -net sam2 \
  -exp_name micro \
  -vis 1 \
  -sam_config sam2_hiera_s \
  -image_size 1024 \
  -out_size 1024 \
  -b 1 \
  -val_freq 10 \
  -dataset SR \
  -data_path {data_path} \
  -SR \
  --config configs/sam/sam_diffsr_df2k4x.yaml \
  --reset \
  --exp_name sam_diffsr_df2k4x \
  --work_dir exp/ \
  -sam_ckpt pretrain.pt

The dataset used for pre-training is available on Hugging Face:
- **Name:** `Micro`  
- **URL:** [https://huggingface.co/datasets/Charlotte188/MicroSegmentation](https://huggingface.co/datasets/Charlotte188/MicroSegmentation)
