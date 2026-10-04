
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DishDataset(Dataset):
    def __init__(self, df, data_dir, tokenizer, transform, max_len=64):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row['dish_id']

        img_path = os.path.join(self.data_dir, "images", str(dish_id), "rgb.png")
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        text = row['ingredients_text']
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        target = row['total_calories']

        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'target': target,
            'dish_id': dish_id
        }


def get_transforms(image_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


def build_dataloaders(df, data_dir, tokenizer, cfg):
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    train_ds = DishDataset(
        train_df, data_dir, tokenizer,
        get_transforms(cfg.IMAGE_SIZE, is_train=True),
        max_len=cfg.MAX_LEN
    )
    test_ds = DishDataset(
        test_df, data_dir, tokenizer,
        get_transforms(cfg.IMAGE_SIZE, is_train=False),
        max_len=cfg.MAX_LEN
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )

    return train_loader, test_loader
