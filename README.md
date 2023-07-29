# SparDL: Distributed Deep Learning Training with Efficient Sparse Communication

Top-k sparsification has recently been widely used to reduce the communication volume in distributed deep learning. However, due to the Sparse Gradient Accumulation (SGA) dilemma, the performance of top-k sparsification still has limitations. Recently, a few methods have been put forward to handle the SGA dilemma. Regrettably, even the state-of-the-art method suffers from several drawbacks, e.g., it relies on an inefficient communication algorithm and requires extra transmission steps. Motivated by the limitations of existing methods, we propose a novel efficient sparse communication framework, called SparDL. Specifically, SparDL uses the Spar-Reduce-Scatter algorithm, which is based on an efficient Reduce-Scatter model, to handle the SGA dilemma without additional communication operations. Besides, to further reduce the latency cost and improve the efficiency of SparDL, we propose the Spar-All-Gather algorithm. Moreover, we propose the global residual collection algorithm to ensure fast convergence of model training. Finally, extensive experiments are conducted to validate the superiority of SparDL. 

## Requirements

- Python 3.8.13
- torch 1.11.0+cu113
- torchvision 0.12.0+cu113
- MPI 3.3.2
- mpi4py 3.0.3
- numpy 1.21.5
- ...

## Models and Datasets

We use seven deep learning models, VGG-16, VGG-19, ResNet-50, VGG-11, LSTM-IMDB, LSTM-PTB, BERT, and seven datasets as five different learning tasks.

CIFAR-10 : Download from https://pytorch.org/vision/stable/datasets.html#cifar

CIFAR-100 : Download from https://pytorch.org/vision/stable/datasets.html#cifar

ImageNet : Download from http://www.image-net.org/challenges/LSVRC/2012/2012-downloads

House : Download from https://github.com/emanhamed/Houses-dataset

IMDB : Download from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

PTB : Download from http://www.fit.vutbr.cz/∼imikolov/rnnlm/

Wikipedi : Download from https://github.com/microsoft/AzureML-BERT/blob/master/pretrain/PyTorch/notebooks/BERT_Pretrain.ipynb

## Quick Start

To train and evaluate with SparDL on 2 workers, i.e., node0, node1

```
mpiexec -n 2 -host node0,node1 python main_trainer.py --dnn vgg16 --dataset cifar10 --max-epochs 121 --batch-size 16 --nworkers 2 --data-dir vgg_data --lr 0.1 --compression --density 0.01 --compressor spardl
```

or using shell script

```
sh vgg16_spardl.sh
```


The meaning of the flags:

- `--batch-size`: Batch size.
- `--nworkers`: Number of workers.
- `--compression`: Compress gradients or not.
- `--compressor`: Specify the compressors if 'compression' is open.
- `--density`: Density for sparsification.
- `--dataset`: Specify the dataset for training. options: {imagenet,cifar10,cifar100,mnist,imdb}
- `--dnn`: Specify the neural network for training. options: {resnet50,resnet20,resnet56,resnet110,vgg19,vgg16,alexnet,lstm,lstmimdb}
- `--data-dir`: Specify the data root path.
- `--lr`: Default learning rate.
- `--max-epochs`: Default maximum epochs to train.
