import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from src import config
from src.utils import Converter

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.img_lst = os.listdir(data_dir)
        self.transform = transform
        self.vocab = Converter.getVocab()
        self.num_characters = self.vocab.num_class
    
    def __len__(self,):
        return len(self.img_lst)
    
    def __getitem__(self, index):
        img_idx = self.img_lst[index]
        label = img_idx.split('.')[0]
        encoded_label = self.vocab.encode(label)

        image = Image.open(os.path.join(self.data_dir, img_idx))
        image = np.array(image)

        if self.transform:
            image = self.transform(image)
        
        return image, torch.IntTensor(encoded_label)
    

class CaptchaDataModule(LightningDataModule):
    def __init__(self, data_dir:str, batch_size:int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def setup(self, stage: str):
        self.train_data = CaptchaDataset(os.path.join(self.data_dir, 'train'), transform=self.trans)
        self.val_data = CaptchaDataset(os.path.join(self.data_dir, 'val'), transform=self.trans)
        self.test_data = CaptchaDataset(os.path.join(self.data_dir, 'test'), transform=self.trans)
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size,shuffle=True, pin_memory=True, num_workers=config.NUM_WORKERS)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, pin_memory=True, num_workers=config.NUM_WORKERS)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, pin_memory=True, num_workers=config.NUM_WORKERS)