# CIIC
Implementation of __Show, Deconfound and Tell: Image Captioning with Causal Inference__ (Updating)

## Requirements (Our Main Enviroment)
+ Python 3.7.4
+ PyTorch 1.5.1
+ TorchVision 0.6.0
+ [coco-caption](https://github.com/tylin/coco-caption)
+ numpy
+ tqdm
+ yacs
+ lmdbdict

## Preparation
### 1. Download Bottom-up features. Prepare the training dataset as in https://github.com/ruotianluo/self-critical.pytorch
### 2. Download our features. https://pan.baidu.com/s/1_8B95prrHS2aLUUddKxc5Q [key]:y2mf

## Training
*Note: our repository is mainly based on [https://github.com/ruotianluo/self-critical.pytorch).

### 1. Training the model
```
# for training
python train.py --id exp --caption_model CIIC --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_att_dir data/cocobu_att --input_att_dir_iod data/IOD --glove_embedding_dict data/Glove_embedding.npy --visual_dict data/vis.npy --lin_dict data/lin.npy --batch_size 10 --N_enc 6
--N_dec 6 --d_model 512 --d_ff 2048 --num_att_heads 8 --dropout 0.1 --learning_rate 0.0003 --learning_rate_decay_start 3 --learning_rate_decay_rate 0.5 --noamopt_warmup 20000 --self_critical_after 30
```
### 2. Evaluating the model
```bash
# for evaluating
python eval.py --model checkpoint_path/model-best.pth --infos_path checkpoint_path/infos-best.pkl
```
## Acknowledgements
This code is implemented based on Ruotian Luo's implementation of image captioning in https://github.com/ruotianluo/self-critical.pytorch.
