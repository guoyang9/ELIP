## ELIP: Efficient Language-Image Pre-training with Fewer Vision Tokens

We build our model based upon the original [BLIP](https://github.com/salesforce/BLIP) repo.
We sincerely appreciate the Junnan' generous sharing and great contribution!

#### Most of the settings follow BLIP except for the decreased batch-size due to resource constraints.
We found that due to the batch reduction, some downstrem tasks do not work normally (The model performance drops with more pre-training steps).
In addition, due to the heavy computation of VQA, we keep only two retrieval task and the image captioning task.

All the downstream tutorial can be found at BLIP, we simply copy them here.
#### Fine-tuning Retrieval@Flickr30K
```
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train_retrieval.py
```
#### Fine-tuning Retrieval@MSCOCO
```
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train_retrieval.py \
--config ./configs/Retrieval_coco.yaml \
```
#### Fine-tuning Captioning
```
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train_caption.py \
--config ./configs/Retrieval_coco.yaml \
```

### Pre-train:
1. Prepare training json files where each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'image': path_of_image, 'caption': text_of_image}.
2. In configs/pretrain.yaml, set 'train_file' as the paths for the json files .
3. Pre-train the model using 4 A5000 GPUs:
```
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 pretrain.py
```

### Pre-training datasets download:
We used the 4M version image-caption datasets for pre-training.

<!-- ### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}</pre> -->