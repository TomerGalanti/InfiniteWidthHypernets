# On Infinite-Width Hypernetworks([arxiv](https://arxiv.org/abs/2003.12193)).

Pytorch Implementation of "On Infinite-Width Hypernetworks" (NeurIPS 2020)

## Prerequisites
- Python 3.6+
- Pytorch 0.4

### Rotations Prediction
Run ```rotation_experiment/main.py```. You can use the following examples to run:
```
python main.py --dataset mnist --lr 0.01 --epochs 100 --var epoch
python main.py --dataset cifar --lr 0.01 --epochs 100 --var epoch
python main.py --dataset cifar --epochs 100 --var lr
```

### Acknowledgements
The file utils/input_data is taken from the open-source code of Tensorflow 1.4.

## Reference
If you found this code useful, please cite the following paper:
```
@inproceedings{galanti2020modularity,
  title={On the Modularity of Hypernetworks},
  author={Tomer Galanti and Lior Wolf},
  booktitle={NeurIPS},
  year={2020}
}
```


