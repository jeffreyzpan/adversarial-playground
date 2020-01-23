# adversarial-playground
This repo reproduces several popular adversarial attacks and defenses, applying such methods to different datasets and applications. The objectives of this project are threefold:

- Give a glimpse at the current state of the art when it comes to adversarial machine learning
- See if there are any underlying trends among datasets/types of machine learning models that make them more vulnerable to adversarial examples
- Make it easier to reproduce popular adversarial attacks and defenses, applying them to a wider set of application fieldss

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
