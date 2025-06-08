# ELIP: Efficient Language-Image Pre-training with Fewer Vision Tokens

We implement ELIP on the [OpenCLIP](https://github.com/mlfoundations/open_clip) model.

## Installing, Dataset Preparation

Please refer to the [OpenCLIP](https://github.com/mlfoundations/open_clip) repo to install all dependencies and prepare all required datasets.

## Pre-Training

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node <NUM_GPUS> -m open_clip_train.main --save-frequency <SAVE_FREQUENCY> --report-to tensorboard --train-data=<DATA_DIR> --dataset-type webdataset --train-num-samples <NUM_SAMPLES> --batch-size=<BATCH_SIZE> --lr=<LR> --wd=<WD> --epochs=<NUM_EPOCHS> --model <BACKBONE>
```

Here is an example:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node 4 -m open_clip_train.main --save-frequency 4 --report-to tensorboard --train-data='/data/web-data/cc12m/{00000..01242}.tar::/data/web-data/mscoco/{00000..00059}.tar::/data/web-data/sbucaptions/{00000..00099}.tar' --dataset-type webdataset --train-num-samples 10108501 --batch-size=440 --lr=1e-3 --wd=0.1 --epochs=32 --model ViT-S-16
```