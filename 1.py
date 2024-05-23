import os
import random

from paddlenlp.datasets import MapDataset


# for train and dev sets
def load_ds(datafiles, split_train=False, dev_size=0):
    '''
    intput:
        datafiles -- str or list[str] -- the path of train or dev sets
        split_train -- Boolean -- split from train or not
        dev_size -- int -- split how much data from train

    output:
        MapDataset
    '''

    datas = []

    def read(ds_file):
        with open(ds_file, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                data = line[:-1].split('\t')
                if len(data) == 2:
                    yield ({'text': data[1], 'label': int(data[0])})
                elif len(data) == 3:
                    yield ({'text': data[2], 'label': int(data[1])})

    def write_tsv(tsv, datas):
        with open(tsv, mode='w', encoding='UTF-8') as f:
            for line in datas:
                f.write(line)

    # 从train切出一部分给dev
    def spilt_train4dev(train_ds, dev_size):
        with open(train_ds, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(f):
                datas.append(line)
        datas_tmp = datas[1:]  # title line should not shuffle
        random.shuffle(datas_tmp)
        if 1 - os.path.exists(os.path.dirname(train_ds) + '/tem'):
            os.mkdir(os.path.dirname(train_ds) + '/tem')
        # remember the title line
        write_tsv(os.path.dirname(train_ds) + '/tem/train.tsv', datas[0:1] + datas_tmp[:-dev_size])
        write_tsv(os.path.dirname(train_ds) + '/tem/dev.tsv', datas[0:1] + datas_tmp[-dev_size:])

    if split_train:
        if 1 - isinstance(datafiles, str):
            print("If you want to split the train, make sure that \'datafiles\' is a train set path str.")
            return None
        if dev_size == 0:
            print("Please set size of dev set, as dev_size=...")
            return None
        spilt_train4dev(datafiles, dev_size)
        datafiles = [os.path.dirname(datafiles) + '/tem/train.tsv', os.path.dirname(datafiles) + '/tem/dev.tsv']

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def load_test(datafile):
    '''
    intput:
        datafile -- str -- the path of test set

    output:
        MapDataset
    '''

    def read(test_file):
        with open(test_file, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                data = line[:-1].split('\t')
                yield {'text': data[1], 'label': '', 'qid': data[0]}

    return MapDataset(list(read(datafile)))


train_ds, dev_ds = load_ds(datafiles=['./data/ChnSentiCorp/train.tsv', './data/ChnSentiCorp/dev.tsv'])
print(train_ds[0])
print(dev_ds[0])
print(type(train_ds[0]))

test_ds = load_test(datafile='./data/ChnSentiCorp/test.tsv')
print(test_ds[0])

train_ds, dev_ds = load_ds(datafiles='./data/NLPCC14-SC/train.tsv', split_train=True, dev_size=1000)
print(train_ds[0])
print(dev_ds[0])
test_ds = load_test(datafile='./data/NLPCC14-SC/test.tsv')
print(test_ds[0])

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class SentimentDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    data.append({'text': parts[1], 'label': int(parts[0])})
                elif len(parts) == 3:
                    data.append({'text': parts[2], 'label': int(parts[1])})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]['text']
        label = self.data[index]['label']
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


# 初始化BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  # 最大序列长度

# 创建数据集实例
train_dataset = SentimentDataset(file_path='./data/ChnSentiCorp/train.tsv', tokenizer=tokenizer, max_len=max_len)
dev_dataset = SentimentDataset(file_path='./data/ChnSentiCorp/dev.tsv', tokenizer=tokenizer, max_len=max_len)
test_dataset = SentimentDataset(file_path='./data/ChnSentiCorp/test.tsv', tokenizer=tokenizer, max_len=max_len)
