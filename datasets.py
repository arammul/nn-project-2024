import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np



class GTA5Dataset(Dataset):
    def __init__(self, 
                 dataset_name,
                 image_dir, 
                 label_dir, 
                 image_files, 
                 num_classes, 
                 preprocessing_fn=None, 
                 resize_dims=None, 
                 crop_dims=None
        ):
        self.dataset_name = dataset_name
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = image_files
        self.num_classes = num_classes
        self.preprocessing_fn = preprocessing_fn
        self.resize = transforms.Resize(resize_dims) if resize_dims is not None else None
        self.crop_dims = crop_dims
        
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file)

        image = transforms.functional.to_tensor(Image.open(image_path)).type(torch.float32)
        label = torch.Tensor(np.array(Image.open(label_path))).type(torch.long).unsqueeze(0)
        
        # Apply transformations
        if self.resize:
            image = self.resize(image)
            label = self.resize(label)
        if self.crop_dims:
            params = transforms.RandomCrop.get_params(image, self.crop_dims)
            image = transforms.functional.crop(image, *params)
            label = transforms.functional.crop(label, *params)

        # Convert label from (W, H) to (C, W, H) - each class would have its own mask
        mask = F.one_hot(label, self.num_classes).squeeze(0).permute(2, 0, 1)

        # Applying standartization using the statistics of the pretrained model
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image.permute(1,2,0)).permute(2,0,1)

        return {    
            "dataset_name": self.dataset_name,
            "image": image.type(torch.float32),
            "mask": mask.type(torch.float32),
        }

class VistasDataset(Dataset):
    def __init__(self, 
                 dataset_name,
                 image_dir, 
                 label_dir, 
                 image_files, 
                 num_classes, 
                 preprocessing_fn=None, 
                 downscale_to_height=None, 
                 crop_dims=None
        ):
        self.dataset_name = dataset_name
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = image_files
        self.num_classes = num_classes
        self.preprocessing_fn = preprocessing_fn
        self.resize = transforms.Resize(downscale_to_height) if downscale_to_height is not None else None
        self.crop_dims = crop_dims
        
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file.replace(".jpg", ".png"))

        image = transforms.functional.to_tensor(Image.open(image_path)).type(torch.float32)
        label = torch.Tensor(np.array(Image.open(label_path))).type(torch.long).unsqueeze(0)
        
        # Apply transformations
        if self.resize:
            image = self.resize(image)
            label = self.resize(label)
        if self.crop_dims:
            params = transforms.RandomCrop.get_params(image, self.crop_dims)
            image = transforms.functional.crop(image, *params)
            label = transforms.functional.crop(label, *params)

        # Convert label from (W, H) to (C, W, H) - each class would have its own mask
        mask = F.one_hot(label, self.num_classes).squeeze(0).permute(2, 0, 1)

        # Applying standartization using the statistics of the pretrained model
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image.permute(1,2,0)).permute(2,0,1)

        return {    
            "dataset_name": self.dataset_name,
            "image": image.type(torch.float32),
            "mask": mask.type(torch.float32),
        }


class CityscapesDataset(Dataset):
    def __init__(self, 
                 dataset_name,
                 image_dir, 
                 label_dir, 
                 image_files, 
                 preprocessing_fn=None, 
                 downscale_to_height=None, 
                 crop_dims=None
        ):
        self.dataset_name = dataset_name
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = image_files
        self.preprocessing_fn = preprocessing_fn
        self.resize = transforms.Resize(downscale_to_height) if downscale_to_height is not None else None
        self.crop_dims = crop_dims
        self.color_to_class = {  (128, 64, 128): 0,
                                 (244, 35, 232): 1,
                                 (70, 70, 70): 2,
                                 (102, 102, 156): 3,
                                 (190, 153, 153): 4,
                                 (153, 153, 153): 5,
                                 (250, 170, 30): 6,
                                 (220, 220, 0): 7,
                                 (107, 142, 35): 8,
                                 (152, 251, 152): 9,
                                 (70, 130, 180): 10,
                                 (220, 20, 60): 11,
                                 (255, 0, 0): 12,
                                 (0, 0, 142): 13,
                                 (0, 0, 70): 14,
                                 (0, 60, 100): 15,
                                 (0, 80, 100): 16,
                                 (0, 0, 230): 17,
                                 (119, 11, 32): 18 }   
        self.num_classes = len(self.color_to_class)
    
    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def get_image_files(directory):
        png_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.png'):
                    full_path = os.path.join(root, file)
                    rel_filename = '/'.join(full_path.split('/')[-2:])
                    png_files.append(rel_filename)
        return png_files


    def convert_label_to_mask(self, image):
        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for color, class_index in self.color_to_class.items():
            color_mask = np.all(image == color, axis=-1)
            mask[color_mask] = class_index
        return mask

    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file.replace('_leftImg8bit', '_gtFine_color'))

        image = transforms.functional.to_tensor(Image.open(image_path)).type(torch.float32)
        label = np.array(Image.open(label_path))[:,:,:3]
        label = torch.Tensor(self.convert_label_to_mask(label)).type(torch.long).unsqueeze(0)
        
        # Apply transformations
        if self.resize:
            image = self.resize(image)
            label = self.resize(label)
        if self.crop_dims:
            params = transforms.RandomCrop.get_params(image, self.crop_dims)
            image = transforms.functional.crop(image, *params)
            label = transforms.functional.crop(label, *params)

        # Convert label from (W, H) to (C, W, H) - each class would have its own mask
        mask = F.one_hot(label, self.num_classes).squeeze(0).permute(2, 0, 1)

        # Applying standartization using the statistics of the pretrained model
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image.permute(1,2,0)).permute(2,0,1)

        return {    
            "dataset_name": self.dataset_name,
            "image": image.type(torch.float32),
            "mask": mask.type(torch.float32),
        }

class ADE20KDataset(Dataset):
    def __init__(self, 
                 dataset_name,
                 image_dir, 
                 label_dir, 
                 image_files, 
                 preprocessing_fn=None, 
                 downscale_to_height=None, 
                 crop_dims=None
        ):
        self.dataset_name = dataset_name
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = image_files
        self.preprocessing_fn = preprocessing_fn
        self.resize = transforms.Resize(downscale_to_height) if downscale_to_height is not None else None
        self.crop_dims = crop_dims
        self.color_to_class = {'165': 1,
                             '3055': 2,
                             '350': 3,
                             '1831': 4,
                             '774': 5,
                             '783': 5,
                             '2684': 6,
                             '687': 7,
                             '471': 8,
                             '401': 9,
                             '1735': 10,
                             '2473': 11,
                             '2329': 12,
                             '1564': 13,
                             '57': 14,
                             '2272': 15,
                             '907': 16,
                             '724': 17,
                             '2985': 18,
                             '533': 18,
                             '1395': 19,
                             '155': 20,
                             '2053': 21,
                             '689': 22,
                             '266': 23,
                             '581': 24,
                             '2380': 25,
                             '491': 26,
                             '627': 27,
                             '2388': 28,
                             '943': 29,
                             '2096': 30,
                             '2530': 31,
                             '420': 32,
                             '1948': 33,
                             '1869': 34,
                             '2251': 35,
                             '239': 36,
                             '571': 37,
                             '2793': 38,
                             '978': 39,
                             '236': 40,
                             '181': 41,
                             '629': 42,
                             '2598': 43,
                             '1744': 44,
                             '1374': 45,
                             '591': 46,
                             '2679': 47,
                             '223': 48,
                             '47': 49,
                             '327': 50,
                             '2821': 51,
                             '1451': 52,
                             '2880': 53,
                             '480': 54,
                             '77': 55,
                             '2616': 56,
                             '246': 57,
                             '247': 57,
                             '2733': 58,
                             '14': 59,
                             '38': 60,
                             '1936': 61,
                             '120': 62,
                             '1702': 63,
                             '249': 64,
                             '2928': 65,
                             '2337': 66,
                             '1023': 67,
                             '2989': 68,
                             '1930': 69,
                             '2586': 70,
                             '131': 71,
                             '146': 72,
                             '95': 73,
                             '1563': 74,
                             '1708': 75,
                             '103': 76,
                             '1002': 77,
                             '2569': 78,
                             '2833': 79,
                             '1551': 80,
                             '1981': 81,
                             '29': 82,
                             '187': 83,
                             '747': 84,
                             '2254': 85,
                             '2262': 86,
                             '1260': 87,
                             '2243': 88,
                             '2932': 89,
                             '2836': 90,
                             '2850': 91,
                             '64': 92,
                             '894': 93,
                             '1919': 94,
                             '1583': 95,
                             '318': 96,
                             '2046': 97,
                             '1098': 98,
                             '530': 99,
                             '954': 100}
        self.num_classes = len(np.unique(list(self.color_to_class.values())))

    @staticmethod
    def get_image_files(directory):
        jpg_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.jpg'):
                    full_path = os.path.join(root, file)
                    rel_filename = '/'.join(full_path.split('/')[-3:])
                    jpg_files.append(rel_filename)
        return jpg_files
    
    def __len__(self):
        return len(self.image_files)


    def convert_label_to_mask(self, seg):
        R = seg[:,:,0]
        G = seg[:,:,1]
        mask = (R/10).astype(np.int32)*256+(G.astype(np.int32))
    
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        for class_code, id_code in self.color_to_class.items():
            new_mask[mask == int(class_code)] = id_code
        return new_mask

    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file.replace('.jpg', '_seg.png'))

        image = transforms.functional.to_tensor(Image.open(image_path)).type(torch.float32)
        label = np.array(Image.open(label_path))[:,:,:3]
        label = torch.Tensor(self.convert_label_to_mask(label)).type(torch.long).unsqueeze(0)
        
        # Apply transformations
        if self.resize:
            image = self.resize(image)
            label = self.resize(label)
        if self.crop_dims:
            params = transforms.RandomCrop.get_params(image, self.crop_dims)
            image = transforms.functional.crop(image, *params)
            label = transforms.functional.crop(label, *params)

        # Convert label from (W, H) to (C, W, H) - each class would have its own mask
        mask = F.one_hot(label, self.num_classes).squeeze(0).permute(2, 0, 1)

        # Applying standartization using the statistics of the pretrained model
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image.permute(1,2,0)).permute(2,0,1)

        return {    
            "dataset_name": self.dataset_name,
            "image": image.type(torch.float32),
            "mask": mask.type(torch.float32),
        }


class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_lengths_cumsum = []
        for i, dataset in enumerate(datasets):
            length = len(dataset)
            if i == 0:
                self.dataset_lengths_cumsum.append(length)
            else:
                self.dataset_lengths_cumsum.append(self.dataset_lengths_cumsum[-1] + length)

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, idx):
        for i, dataset_length in enumerate(self.dataset_lengths_cumsum):
            if idx < dataset_length:
                if i == 0:
                    return self.datasets[i][idx]
                else:
                    return self.datasets[i][idx - self.dataset_lengths_cumsum[i - 1]]

