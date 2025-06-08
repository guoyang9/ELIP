# ELIP: Efficient Language-Image Pre-training with Fewer Vision Tokens

We implement ELIP on the [METER](https://github.com/zdou0830/METER) model.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

## Pre-training

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_mlm_itm_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_mlm_itm_clip_bert per_gpu_batchsize=32 clip16 text_roberta image_size=288
```

## Fine-tuning on Downstream Tasks

### VQAv2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=576 clip_randaug
```

### Flickr30k IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_f30k_clip_bert get_recall_metric=False per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_f30k_clip_bert get_recall_metric=False per_gpu_batchsize=32 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=384 clip_randaug
```

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES>  task_finetune_nlvr2_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1  task_finetune_nlvr2_clip_bert per_gpu_batchsize=32 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=288 clip_randaug
```

### SNLI-VE

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_snli_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> <IMAGE_AUGMENTATION>
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_snli_clip_bert per_gpu_batchsize=8 load_path=meter_pretrain.ckpt clip16 text_roberta image_size=384 clip_randaug
```

## Evaluation on Downstream Tasks

### VQAv2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=meter_vqa.ckpt clip16 text_roberta image_size=576 test_only=True
```

Then, submit the json file in the `result` directory to [eval.ai](https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get the test-dev and/or test-std scores.


### Flickr30k IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_f30k_clip_bert get_recall_metric=True per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_f30k_clip_bert get_recall_metric=True per_gpu_batchsize=32 load_path=meter_f30k.ckpt clip16 text_roberta image_size=384 test_only=True
```

The returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10.

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES>  task_finetune_nlvr2_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1  task_finetune_nlvr2_clip_bert per_gpu_batchsize=32 load_path=meter_nlvr2.ckpt clip16 text_roberta image_size=288 test_only=True
```

### SNLI-VE

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_snli_clip_bert per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> <IMAGE_ENCODER> <TEXT_ENCODER> image_size=<IMAGE_SIZE> test_only=True
```

Here is an example:
```bash
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_snli_clip_bert per_gpu_batchsize=8 load_path=meter_snli.ckpt clip16 text_roberta image_size=384 test_only=True
```



<!-- ## Citation

```
@inproceedings{dou2022meter,
  title={An Empirical Study of Training End-to-End Vision-and-Language Transformers},
  author={Dou, Zi-Yi and Xu, Yichong and Gan, Zhe and Wang, Jianfeng and Wang, Shuohang and Wang, Lijuan and Zhu, Chenguang and Zhang, Pengchuan and Yuan, Lu and Peng, Nanyun and Liu, Zicheng and Zeng, Michael},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  url={https://arxiv.org/abs/2111.02387},
}
``` -->

