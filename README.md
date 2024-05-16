# Lookbehind-SAM: k steps back, 1 step forward (ICML 2024)

This is the official repository for [Lookbehind-SAM](https://arxiv.org/abs/2307.16704), accepted at ICML 2024. Our code base is a modification and extension of the existing [ASAM](https://github.com/SamsungLabs/ASAM) and [Lookahead](https://github.com/michaelrzhang/lookahead) repositories.

## Abstract
Sharpness-aware minimization (SAM) methods have gained increasing popularity by formulating the problem of minimizing both loss value and loss sharpness as a minimax objective. In this work, we increase the efficiency of the maximization and minimization parts of SAM's objective to achieve a better loss-sharpness trade-off. By taking inspiration from the Lookahead optimizer, which uses multiple descent steps ahead, we propose Lookbehind, which performs multiple ascent steps behind to enhance the maximization step of SAM and find a worst-case perturbation with higher loss. Then, to mitigate the variance in the descent step arising from the gathered gradients across the multiple ascent steps, we employ linear interpolation to refine the minimization step. Lookbehind leads to a myriad of benefits across a variety of tasks. Particularly, we show increased generalization performance, greater robustness against noisy weights, as well as improved learning and less catastrophic forgetting in lifelong learning settings.

## Install
```
pip install -r requirements.txt
```

## Usage
Adaptive $\alpha$:
```
python example_cifar.py --minimizer Lookbehind_SAM --rho 0.05 --k 2 --alpha -1
python example_cifar.py --minimizer Lookbehind_ASAM --rho 0.5 --k 2 --alpha -1
```
To use a static $\alpha$, set $\alpha$ to a value larger than 0 and smaller than 1 (e.g. ```--alpha 0.5```). In the paper, we used $\alpha \in \\{0.2, 0.5, 0.8\\}$. 

To use CIFAR-100, set ```--dataset CIFAR100``` and, optionally, ```--model resnet50```.

## Citation
```
@inproceedings{
lookbehindsam,
title={Lookbehind-{SAM}: k steps back, 1 step forward},
author={Mordido, Gon{\c{c}}alo and Malviya, Pranshu and Baratin, Aristide and Chandar, Sarath},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=vCN5lwcWWE}
}
```
