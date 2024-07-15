import os
import random
import h5py
import numpy as np
import torch
import json
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa 
import cv2 as cv
import torch.nn.functional as F

def mask_to_onehot(mask):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

def augment_seg(img_aug, img, seg ):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic() 
    image_aug = aug_det.augment_image( img )

    segmap = ia.SegmentationMapsOnImage( seg, shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug , segmap_aug

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class SegPC2021_dataset(Dataset):
    def __init__(self,base_dir,split,img_size, norm_x_transform=None, norm_y_transform=None,shuffle=True,test_mode=False, one_hot=True) -> None:
        super().__init__()
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.data_dir = os.path.join(base_dir,split)
        self.split = split
        self.sample_list = os.listdir(self.data_dir)
        self.one_hot = one_hot
        
        if test_mode:
            self.sample_list = self.sample_list[:int(len(self.sample_list)*0.1)]

        if shuffle:
            random.shuffle(self.sample_list)

        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0,2),[
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"), # type: ignore
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5), # type: ignore
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train_npz":
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image,_ = augment_seg(self.img_aug, image, label)
            x, y, _ = image.shape
            
            if x != self.img_size or y != self.img_size:
                image = cv.resize(image,(self.img_size,self.img_size),interpolation=cv.INTER_CUBIC) # type: ignore
                label = cv.resize(label,(self.img_size,self.img_size),interpolation=cv.INTER_NEAREST) # type: ignore
        else:
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label'] # type: ignore
            x, y, _ = image.shape
            
            if x != self.img_size or y != self.img_size:
                image = cv.resize(image,(self.img_size,self.img_size),interpolation=cv.INTER_CUBIC) # type: ignore
                label = cv.resize(label,(self.img_size,self.img_size),interpolation=cv.INTER_NEAREST)
        
        # print(label.shape,np.unique(label))
        label = np.where(label == 2, 1, 0)
        if self.norm_x_transform is not None:
            image = self.norm_x_transform(image.copy())
        if self.norm_y_transform is not None:
            label = self.norm_y_transform(label.copy())
        if self.one_hot:
            label = F.one_hot(torch.squeeze(label).to(torch.int64))
            label = torch.moveaxis(label, -1, 0).to(torch.float)
        
        sample = {'image': image, 'label': label,'name':slice_name.strip('.pnz')}
        return sample

class ISIC2018_dataset(Dataset):
    def __init__(self,base_dir,split,img_size, norm_x_transform=None, norm_y_transform=None,shuffle=True,test_mode=False) -> None:
        super().__init__()
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.data_dir = os.path.join(base_dir,split)
        self.split = split
        self.sample_list = os.listdir(self.data_dir)
        
        if test_mode:
            self.sample_list = self.sample_list[:int(len(self.sample_list)*0.1)]

        if shuffle:
            random.shuffle(self.sample_list)

        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0,4),[
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"), # type: ignore
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5), # type: ignore
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train_npz":
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image,label = augment_seg(self.img_aug, image, label)
            x, y, _ = image.shape
            
            if x != self.img_size or y != self.img_size:
                image = cv.resize(image,(self.img_size,self.img_size),interpolation=cv.INTER_CUBIC) # type: ignore
                label = cv.resize(label,(self.img_size,self.img_size),interpolation=cv.INTER_NEAREST) # type: ignore
        else:
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label'] # type: ignore
            x, y, _ = image.shape
            
            if x != self.img_size or y != self.img_size:
                image = cv.resize(image,(self.img_size,self.img_size),interpolation=cv.INTER_CUBIC) # type: ignore
                label = cv.resize(label,(self.img_size,self.img_size),interpolation=cv.INTER_NEAREST)
        
        sample = {'image': image, 'label': label,'name':slice_name.strip('.pnz')}
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())
        return sample


class SegPC2021_dataset_full(Dataset):
    def __init__(self,base_dir,split,img_size, norm_x_transform=None, norm_y_transform=None,shuffle=True,test_mode=False) -> None:
        super().__init__()
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.data_dir = os.path.join(base_dir,split)
        self.split = split
        self.sample_list = os.listdir(self.data_dir)
        
        if test_mode:
            self.sample_list = self.sample_list[:int(len(self.sample_list)*0.1)]

        if shuffle:
            random.shuffle(self.sample_list)

        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0,2),[
            # iaa.Flipud(0.5, name="Flipud"),
            # iaa.Fliplr(0.5, name="Fliplr"), # type: ignore
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5), # type: ignore
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            # iaa.Affine(rotate=(-10, 10)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image,label = augment_seg(self.img_aug, image, label)
            x, y, _ = image.shape
            
            if x != self.img_size or y != self.img_size:
                image = cv.resize(image,(self.img_size,self.img_size),interpolation=cv.INTER_CUBIC) # type: ignore
                label = cv.resize(label,(self.img_size,self.img_size),interpolation=cv.INTER_NEAREST) # type: ignore
        else:
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label'] # type: ignore
            x, y, _ = image.shape
            
            if x != self.img_size or y != self.img_size:
                image = cv.resize(image,(self.img_size,self.img_size),interpolation=cv.INTER_CUBIC) # type: ignore
                label = cv.resize(label,(self.img_size,self.img_size),interpolation=cv.INTER_NEAREST)
        
        sample = {'image': image, 'label': label,'name':slice_name.strip('.pnz')}
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, norm_x_transform=None, norm_y_transform=None,shuffle=True, test_mode=False):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        
        if test_mode:
            self.sample_list = self.sample_list[:int(len(self.sample_list)*0.1)]

        if shuffle:
            random.shuffle(self.sample_list)

        self.data_dir = base_dir
        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0,4),[
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"), # type: ignore
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5), # type: ignore
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = np.float32(data['image']), np.float32(data['label'])
            image,label = augment_seg(self.img_aug, image, label)
            x, y = image.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)  # why not 3?
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:] # type: ignore
        
        sample = {'image': image, 'label': label}
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class PDGM_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, norm_x_transform=None, norm_y_transform=None,shuffle=True, test_mode=False):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        
        if test_mode:
            self.sample_list = self.sample_list[:int(len(self.sample_list)*0.1)]

        if shuffle:
            random.shuffle(self.sample_list)

        self.data_dir = base_dir
        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0,1),[
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"), # type: ignore
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5), # type: ignore
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image,label = augment_seg(self.img_aug, image, label)

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'], data['label'] # type: ignore
        
        sample = {'image': np.float32(image), 'label': np.float32(label)}
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    


