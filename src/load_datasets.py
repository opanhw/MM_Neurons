#!/usr/local/bin/python3

import os
import json
import random
import requests
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from io import BytesIO
from torch.utils.data import Dataset


class SBU(Dataset):
    def __init__(self, dataframe, folder_dir, transform=None):
        self.folder_dir = folder_dir
        self.transform = transform
        self.file_names = dataframe.index
        self.captions = dataframe.captions.values.tolist()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image = Image.open(self.folder_dir + '/images/' + self.file_names[index])
        caption = self.captions[index]
        sample = {'image': image, 'caption': caption}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'caption': caption}
        return sample


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_sbu(file_path, train_num=1000, valid_num=200, test_num=200, rebuild=False, random_state=2023):
    if os.path.exists(file_path + "/train.pkl") and os.path.exists(file_path + "/valid.pkl") and os.path.exists(
            file_path + "/test.pkl") and not rebuild:
        print("Loading from prepared...")
        train_df = pd.read_pickle(file_path + "/train.pkl")
        valid_df = pd.read_pickle(file_path + "/valid.pkl")
        test_df = pd.read_pickle(file_path + "/test.pkl")

        return train_df, valid_df, test_df

    random.seed(random_state)
    with open(file_path + "/sbu-captions-all.json", 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        sample = data['image_urls'][:2 * (train_num + valid_num + test_num)]
        image_urls, captions = [], []
        for s in range(len(sample)):
            image_urls.append(data['image_urls'][s])
            captions.append(data['captions'][s])

    if os.path.exists(file_path + '/images') and not rebuild:
        pass
    else:
        if not os.path.exists(file_path + '/images'):
            os.makedirs(file_path + '/images')

        cnt = 0

        for image_url in tqdm(image_urls):
            try:
                image = load_image(image_url)
                image.save(file_path + f"/images/{image_url.split('/')[-1]}")
                cnt += 1
            except:
                pass

            if cnt >= train_num + valid_num + test_num:
                break

    new_captions = []
    new_images = []
    images = [x.split('/')[-1] for x in glob(file_path + '/images/*')]

    for i, image_url in enumerate(image_urls):
        if image_url.split('/')[-1] in images:
            new_captions.append(captions[i])
            new_images.append(image_url.split('/')[-1])

    np.random.seed(random_state)
    shuffle = np.random.permutation(len(new_captions))

    print('Preparing train captions...')

    train_dict = {}
    train_file_names = []
    for i in tqdm(shuffle[:train_num]):
        file_name = new_images[i]
        caption = new_captions[i]
        train_dict[file_name] = caption
        train_file_names.append(file_name)
    train_df = pd.DataFrame({'values': train_dict.values()})
    train_df.index = train_file_names
    train_df.columns = ['captions']

    train_df.to_pickle(file_path + '/train.pkl')

    print('Successfully preparing train captions!')

    print('Preparing valid captions...')

    valid_dict = {}
    valid_file_names = []
    for i in tqdm(shuffle[train_num:train_num + valid_num]):
        file_name = new_images[i]
        caption = new_captions[i]
        valid_dict[file_name] = caption
        valid_file_names.append(file_name)
    valid_df = pd.DataFrame({'values': valid_dict.values()})
    valid_df.index = valid_file_names
    valid_df.columns = ['captions']

    valid_df.to_pickle(file_path + '/valid.pkl')

    print('Successfully preparing valid captions!')

    print('Preparing test captions...')

    test_dict = {}
    test_file_names = []
    for i in tqdm(shuffle[train_num + valid_num:train_num + valid_num + test_num]):
        file_name = new_images[i]
        caption = new_captions[i]
        test_dict[file_name] = caption
        test_file_names.append(file_name)
    test_df = pd.DataFrame({'values': test_dict.values()})
    test_df.index = test_file_names
    test_df.columns = ['captions']

    test_df.to_pickle(file_path + '/test.pkl')
    print('Successfully preparing test captions!')

    return train_df, valid_df, test_df



