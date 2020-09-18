from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import PIL
from torchvision.transforms import transforms

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.mapping = {
            29: 1,
            76: 2,
            150: 3
        }

        self.ids = [file.split(' uint8')[0] for file in listdir(imgs_dir)]
        logging.info(f'Creating dataset with {len(self.ids)} examples\nfirst is {self.ids[0]}')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask=False):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH), PIL.Image.NEAREST)
        img_nd = np.array(pil_img)
        img_nd = np.expand_dims(img_nd, axis=2)
        img_trans = img_nd.transpose((2, 0, 1))
        if not is_mask:
            img_trans = img_trans / 255
        return img_trans

    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + ' black.tif')
        img_file = glob(self.imgs_dir + idx + ' uint8.tif')
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = Image.open(mask_file[0]).convert('L')
        img = Image.open(img_file[0])
        
        my_min_w = 485
        my_min_h = 496
        if img.size[0] < my_min_w or mask.size[0] < my_min_w or img.size[1] < my_min_h or mask.size[1] < my_min_h:
            raise Exception('check sizes')
            
        img, mask = img.crop((0, 0, my_min_w, my_min_h)), mask.crop((0, 0, my_min_w, my_min_h))
        
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(img, self.scale), self.preprocess(mask, self.scale, is_mask=True)
        mask = self.mask_to_class(mask)
        
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
