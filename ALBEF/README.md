## ELIP: Efficient Language-Image Pre-training with Fewer Vision Tokens

We build our model based upon the original [ALBEF](https://github.com/salesforce/ALBEF) repo.
We sincerely appreciate Junnan's generous sharing and great contribution!

#### Most of the settings follow ALBEF except for the decreased batch-size due to resource constraints.

### Requirements:
* pytorch 1.8.0
* transformers 4.8.1
* timm 0.4.9

### Download:
We reuse the json file extracted by ALBEF.
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz"> Dataset json files for downstream tasks</a>
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip"> Dataset json files for pre-training</a> (the image paths in each json file need to be changed to your own directory)

### Pre-training on custom datasets:
1. Prepare training json files where each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'image': path_of_image, 'caption': text_of_image}.
2. In configs/Pretrain.yaml, set the paths for the json files.
3. Pre-train the model using 4 A5000 GPUs:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env Pretrain.py
```

All the downstream tutorial can be found at ALBEF, we simply copy them here.
#### Fine-tuning VQA
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env VQA.py
```
#### Fine-tuning Visual Entailment
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env VE.py
```
#### Fine-tuning Retrieval@Flickr30K
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py
```
#### Fine-tuning Retrieval@MSCOCO
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
--config ./configs/Retrieval_coco.yaml \
```

<!-- ### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{ALBEF,
      title={Align before Fuse: Vision and Language Representation Learning with Momentum Distillation},
      author={Junnan Li and Ramprasaath R. Selvaraju and Akhilesh Deepak Gotmare and Shafiq Joty and Caiming Xiong and Steven Hoi},
      year={2021},
      booktitle={NeurIPS},
}</pre> -->
