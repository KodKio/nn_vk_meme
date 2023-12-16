import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPProcessor


class HatefulMemesDataset(Dataset):
    def __init__(self, root_folder, image_folder, split='train', image_size=224):
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.split = split

        self.image_size = image_size
        if split == 'train':
            self.info_file = os.path.join(root_folder, 'train.jsonl')
        if split == 'dev':
            self.info_file = os.path.join(root_folder, 'dev.jsonl')
        if split == 'test':
            self.info_file = os.path.join(root_folder, 'test.jsonl')
        print(self.info_file)
        self.df = pd.read_json(self.info_file, lines=True).reset_index(drop=True)
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        image_fn = row['img'].split('/')[1]
        item['image'] = Image.open(f"{self.image_folder}/{image_fn}").convert('RGB').resize(
            (self.image_size, self.image_size))
        item['text'] = row['text']
        item['label'] = row['label']

        return item


class CustomCollator(object):

    def __init__(self, args):
        self.args = args
        self.image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
        self.text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")[
            'pixel_values']
        text_output = self.text_processor([item['text'] for item in batch], padding=True, return_tensors="pt",
                                          truncation=True)
        labels = torch.LongTensor([item['label'] for item in batch])

        batch_new = {}
        batch_new['pixel_values'] = pixel_values,
        batch_new['input_ids'] = text_output['input_ids']
        batch_new['attention_mask'] = text_output['attention_mask']
        batch_new['labels'] = labels

        return batch_new


def load_dataset(args, split):
    image_folder = '/kaggle/input/facebook-hateful-meme-dataset/data/img'

    dataset = HatefulMemesDataset(root_folder='/kaggle/input/facebook-hateful-meme-dataset/data',
                                  image_folder=image_folder, split=split,
                                  image_size=args.image_size)

    return dataset
