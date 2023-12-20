
# [NEUROCOM] Class Similarity Weighted Knowledge Distillation for Few Shot Incremental Learning


## Requirements
- [PyTorch >= version 1.4](https://pytorch.org)


## Datasets
We follow the setting of [FSCIL](https://github.com/xyutao/fscil) to use simillar data index_list for training. 
you need strictly follow the guidelines found in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.
Scripts for experiments on mini-imagenet are as follows, and the full codes will be available upon acceptance:

## Pretrain scripts 
mini-imagenet 
    $ python trainfeya.py --dataset mini-imagenet --exp_dir experiment --epoch 200 --batch_size 128 --init_lr 0.1 --milestones 80 160 --val_start 100 --change_val_interval 160


## Testing scripts    
mini-imagenet

    $ python test.py --dataset mini-imagenet --exp_dir experiment --finetune_needs --ft_iters 100 --ft_lr 0.001 --ft_factor 1.0 --ft_T 16 --w_d 100 --part_frozen --ft_KD_all --ft_teacher fixed --dual --BC_hidden_dim 64 --BC_lr 0.01 --w_BC_binary 50 --logits_EMA --w_l 1 --EMA_FC_lr 0.01

## Acknowledgment
Our project references the codes in the following repos.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [fscil](https://github.com/xyutao/fscil)
- [BiDistFSCIL](https://github.com/LinglanZhao/BiDistFSCIL.git)
- [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)


