# Code for On Intrinsic Dataset Properties for Adversarial Machine Learning [[arXiv](https://arxiv.org/abs/2005.09170)]

In this work, we explore the effect of two intrinsic dataset properties, input size and image contrast, on adversarial robustness, testing on five popular image classification datasets — MNIST, Fashion-MNIST, CIFAR10/CIFAR100, and ImageNet. We find that input size and image contrast play key roles in attack and defense success. Our discoveries highlight that dataset design and data preprocessing steps are important to boost the adversarial robustness of DNNs.

```
@misc{pan2020intrinsic,
    title={On Intrinsic Dataset Properties for Adversarial Machine Learning},
    author={Jeffrey Z. Pan and Nicholas Zufelt},
    year={2020},
    eprint={2005.09170},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Install

This code requires the following dependencies

- [PyTorch 1.2](https://pytorch.org/)
- Torchvision 0.4.0
- [Numpy](https://docs.scipy.org/doc/numpy-1.15.0/user/index.html)
- [Sklearn](https://scikit-learn.org/stable/index.html)
- [ART](https://github.com/IBM/adversarial-robustness-toolbox)
- [Foolbox](https://foolbox.jonasrauber.de/)

Additionally, we evaluate on the following datasets:

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR-10/CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](http://image-net.org/)

All of the above datasets can be installed through their respective `torchvision.datasets` classes by setting `download=True` in `lib/datasets/data_utils.py`, with the exception of ImageNet, which must be manually downloaded.

## Attacks and Defenses

I test the following attacks and defenses in this repo:

Attacks

- [FGSM](https://arxiv.org/abs/1412.6572)
- [BIM](https://arxiv.org/abs/1607.02533)
- [PGD](https://arxiv.org/abs/1706.06083)
- [DeepFool](https://arxiv.org/abs/1511.04599)

Defenses

- [Total Variance Minimization](https://arxiv.org/abs/1711.00117)
- [JPEG Compression](https://arxiv.org/abs/1711.00117)

## Input Size Results 

![images](./figs/mnist_input_size.png)
*The effects of input image size on adversarial robustness for the MNIST dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

![images](./figs/fmnist_input_size.png)
*The effects of input image size on adversarial robustness for the Fashion-MNIST dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

![images](./figs/cifar10_input_size.png)
*The effects of input image size on adversarial robustness for the CIFAR-10 dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

![images](./figs/cifar100_input_size.png)
*The effects of input image size on adversarial robustness for the CIFAR-100 dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

![images](./figs/imagenet_input_size.png)
*The effects of input image size on adversarial robustness for the ImageNet dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

## Image Contrast Results 

![images](./figs/mnist_contrast.png)
*The effects of image contrast on adversarial robustness for the MNIST dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

![images](./figs/fmnist_contrast.png)
*The effects of image contrast on adversarial robustness for the Fashion-MNIST dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

![images](./figs/cifar10_contrast.png)
*The effects of image contrast on adversarial robustness for the CIFAR-10 dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

![images](./figs/cifar100_contrast.png)
*The effects of image contrast on adversarial robustness for the CIFAR-100 dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*

![images](./figs/imagenet_contrast.png)
*The effects of image contrast on adversarial robustness for the ImageNet dataset. Each row represents a given defense (no defense, TVM, and JPEG) while each column represents a given attack (FGSM, BIM, PGD, DeepFool). Four different epsilon values are used in every attack-defense combination: 2/255, 4/255, 8/255, and 16/255.*
