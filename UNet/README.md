# UNet: semantic segmentation with PyTorch

Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Multiple Surface Segmentation of the Retinal Layer in OCT Images.

This model was trained from scratch with 2600 images (no data augmentation) and scored a cross-entropy of 7e10-3.

### Remarks

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively.

## Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard. For example:

`tensorboard --logdir=runs`

## Notes on memory

The model has be trained from scratch on NVIDIA DGX-2.
Predicting images of 1918*1280 takes 1.5GB of memory.
Training takes much approximately 3GB, so if you are a few MB shy of memory, consider turning off all graphical displays.
This assumes you use bilinear up-sampling, and not transposed convolution in the model.

---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
