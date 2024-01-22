import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir 
        self.ann_path = args.ann_path
        self.split_path = args.split_path

        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        
        cases = self.clean_data(pd.read_csv(self.split_path).loc[:, self.split].dropna())

        
        self.examples = []
        root = self.ann_path

        count=0

        for dir in os.listdir(root):
            if not dir in cases.keys(): # check whther contained in the split
                continue
            else:
                img_name = cases[dir]
                
            image_path = os.path.join(self.image_dir,img_name)

            if not os.path.exists(image_path+'.pt'):
                continue
                
            file_name = os.path.join(root, dir, 'annotation')

            anno = json.loads(open(file_name, 'r').read())
            report_ids = tokenizer(anno)
            if len(report_ids) < self.max_seq_length:
                padding = [0] * (self.max_seq_length-len(report_ids))  
                report_ids.extend(padding)
            #report_ids = tokenizer(anno)[:self.max_seq_length]
            self.examples.append({'id':dir, 'image_path': image_path+'.pt','report': anno, 'split': self.split,'ids':report_ids, 'mask': [1]*len(report_ids)})

        
        print(f'The size of {self.split} dataset: {len(self.examples)}')


    def __len__(self):
        return len(self.examples)



class TcgaImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        
        image = torch.load(image_path)
        image = image[:self.max_fea_length]
        report_ids = example['ids']
        report_masks = example['mask']

        seq_length = len(report_ids)


        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


    def clean_data(self,data):
        cases = {}
        for idx in range(len(data)):
            case_name = data[idx]

            case_id = '-'.join(case_name.split('-')[:3])
            cases[case_id] = case_name
        return cases 
    
    def filter_df(self,df, filter_dict):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            # assert 'label' not in filter_dict.keys()
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def df_prep(self,data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data