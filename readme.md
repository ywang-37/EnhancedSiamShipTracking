# EnhancedSiamShipTracking



## Introduction

Code for ["An Enhanced SiamMask Network for Coastal Ship Tracking"](https://ieeexplore.ieee.org/abstract/document/9584864). 

## Setup

- Clone the repository

  ```bash
  git clone https://github.com/ywang-37/EnhancedSiamShipTracking.git
  cd EnhancedSiamShipTracking
  export EnhancedSiamShipTracking=$PWD
  export PYTHONPATH=$PWD:$PYTHONPATH
  ```

- Pytorch environment

  ```bash
  conda create -n torch python=3.6
  conda activate torch
  pip install -r requirements.txt
  bash make.sh
  ```

## Train

Pretrain model can download by: 

```bash
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
```

```bash
cd $EnhancedSiamShipTracking/experiments/siammask_base
export PYTHONPATH=$PWD:$PYTHONPATH
```

```bash
python -u ../..//tools/train_siammask.py \
    --config=config.json # configuration
    -b 8 \ # minibatch
    --epochs 20 \ # epoch
    --log logs/log.txt 2>&1 | tee logs/train.log # train logs
```

## Test

```bash
python -u ../../tools/test.py \
    --config config.json \ # configuration
    --resume snapshot/checkpoint_e10.pth \ # which model for testing
    --mask \ # use mask
    --dataset $dataset # which dataset
    2>&1 | tee logs/test.log # test log
```

## Eval

```bash
python ../..//tools/eval.py 
	--dataset $dataset # which dataset
	--num 1 # workers
	--tracker_prefix C # results
	--result_dir ./test/$dataset # result dir
	2>&1 | tee logs/eval_test_$dataset.log
```

## Acknowledgment

Our ship tracker is based on [SiamMask](https://github.com/foolwood/SiamMask). We sincerely thank the authors for providing the project. 

## Bibtex 
If you need to cite, you can use: 

```
@article{yang2021enhanced,
  title={An Enhanced SiamMask Network for Coastal Ship Tracking},
  author={Yang, Xi and Wang, Yan and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
```
