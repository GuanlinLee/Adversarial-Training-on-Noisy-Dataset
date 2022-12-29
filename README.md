# Adversarial-Training-on-Noisy-Dataset

We simply implement several training strategies combined with adversarial training methods to explore the effectiveness of previous methods on adversarial robustness.
## Requirements
> 1. Pytorch (>=1.7.0)
> 2. torchattacks
> 3. numpy
> 4. tqdm
> 5. matplotlib
> 6. scipy

## How To Use
```
python train.py --arch resnet --dataset cifar10 \\
--nr 0.2 --noise_type [sym, asy] \\
--method [pgd, trades, pgd_te, trades_te, pgd_sat, trades_sat, pencil, labelcorr, elr, selfie, plc] \\
--save save_name --exp experiment_name 
```
## Implementations
> 1. [PGD](https://github.com/MadryLab/cifar10_challenge)
> 2. [TRADES](https://github.com/yaodongyu/TRADES)
> 3. [TE](https://github.com/dongyp13/memorization-AT)
> 4. [SAT](https://github.com/LayneH/self-adaptive-training)
> 5. [PENCIL](https://github.com/yikun2019/PENCIL)
> 6. [LABELCORRECTION](https://github.com/PaulAlbert31/LabelNoiseCorrection)
> 7. [ELR](https://github.com/shengliu66/ELR)
> 8. [SELFIE](https://github.com/kaist-dmlab/SELFIE)
> 9. [PLC](https://github.com/pxiangwu/PLC)
