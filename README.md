# adversarial-playground
This repo reproduces several popular adversarial attacks and defenses, applying such methods to different datasets and applications. The objectives of this project are threefold:

- Give a glimpse at the current state of the art when it comes to adversarial machine learning
- See if there are any underlying trends among datasets/types of machine learning models that make them more vulnerable to adversarial examples
- Make it easier to reproduce popular adversarial attacks and defenses, applying them to a wider set of application fields

## Install

This repo requires the following dependencies:

- [PyTorch 1.4](https://pytorch.org/)
- [Numpy](https://docs.scipy.org/doc/numpy-1.15.0/user/index.html)
- [Sklearn](https://scikit-learn.org/stable/index.html)
- [ART](https://github.com/IBM/adversarial-robustness-toolbox)
- [TensorboardX](https://github.com/lanpa/tensorboardX)

Additionally, I evaluate on the following datasets:

- [Chest X-Rays](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [GTSRB](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- [Kaggle Stock Market](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)

## Attacks and Defenses

I test the following attacks and defenses in this repo:

Attacks

- [FGSM](https://arxiv.org/abs/1412.6572)
- [PGD](https://arxiv.org/abs/1706.06083)
- [Query-Efficient Black Box Attacks](https://arxiv.org/pdf/1804.08598.pdf)
- [HopSkipJump](https://arxiv.org/abs/1904.02144v4)
- [DeepFool](https://arxiv.org/abs/1511.04599)

Defenses

- Adversarial training
- [PixelDefend](https://arxiv.org/abs/1710.10766)
- [I-Defender](https://papers.nips.cc/paper/8016-robust-detection-of-adversarial-attacks-by-modeling-the-intrinsic-properties-of-deep-neural-networks.pdf)
- [Thermometer Encoding](https://openreview.net/pdf?id=S18Su--CW)
- [Total Variance Minimization](https://arxiv.org/abs/1711.00117)
