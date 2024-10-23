import torch
import pytorch_lightning as L
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gdown
import zipfile
from pathlib import Path
from typing import List
from PIL import Image, UnidentifiedImageError
import os
from torchvision.datasets.folder import default_loader

class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = self._load_samples()
        self.imgs = self.samples

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _load_samples(self):
        samples = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self._is_valid_image(path):
                        item = path, class_index
                        samples.append(item)
        return samples

    def _is_valid_image(self, filepath):
        try:
            with Image.open(filepath) as img:
                img.verify()
            return True
        except (IOError, SyntaxError, UnidentifiedImageError):
            print(f"Corrupted image found: {filepath}")
            return False

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_workers: int = 2,
        batch_size: int = 32,
        splits: List[float] = [0.8, 0.1, 0.1],
        pin_memory: bool = True,
        image_size: int = 224,
        normalization_mean: List[float] = [0.485, 0.456, 0.406],
        normalization_std: List[float] = [0.229, 0.224, 0.225],
        dataset_file_id: str = "1bDvcKFwPMWjmZwppu58xM2KFuGnGUmg-",
        dataset_subfolder: str = "PetImages",
        output_file: str = "cat_dog_dataset.zip",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.splits = splits
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.dataset_file_id = dataset_file_id
        self.dataset_subfolder = dataset_subfolder
        self.output_file = output_file
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std)
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage == 'test' or stage is None:
            dataset_path = Path(f"{self.data_dir}/{self.dataset_subfolder}")
            if not dataset_path.exists():
                self._download_and_extract_dataset()
            
            full_dataset = FilteredImageFolder(root=str(dataset_path), transform=self.transform)
            n_data = len(full_dataset)
            n_train = int(self.splits[0] * n_data)
            n_val = int(self.splits[1] * n_data)
            n_test = n_data - n_train - n_val
            
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                full_dataset, [n_train, n_val, n_test]
            )
            
            print(f"Total valid data: {n_data}")
            print(f"Train data: {len(self.train_dataset)}")
            print(f"Validation data: {len(self.val_dataset)}")
            print(f"Test data: {len(self.test_dataset)}")

    def _download_and_extract_dataset(self):
        gdown.download(f"https://drive.google.com/uc?id={self.dataset_file_id}", self.output_file, quiet=False)
        dataset_path = Path(f"{self.data_dir}/{self.dataset_subfolder}")
        with zipfile.ZipFile(self.output_file, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        # Remove the zip file after extraction
        Path(self.output_file).unlink()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
