import pandas as pd
import random
from torch.utils.data import Dataset
import torch


def load_dataset(csv_path: str = "data/PubMed/pm.csv",
                 train_ratio=0.8,
                 initial_train_size=100,
                 test_size=300,
                 ):
    full_data = pd.read_csv(csv_path)
    sample_count = len(full_data)
    train_sample_count = int(sample_count*train_ratio)
    indices = list(range(sample_count))
    random.shuffle(indices[:train_sample_count])

    train_indices = indices[:train_sample_count]
    test_indices = indices[train_sample_count:]

    train_set = PubMedDataset(full_data, train_indices, initial_train_size)
    test_set = PubMedDataset(full_data, test_indices, test_size)

    return train_set, test_set


class PubMedDataset(Dataset):
    # Title,abstractText,meshMajor,pmid,meshid,meshroot,A,B,C,D,E,F,G,H,I,J,L,M,N,Z
    def __init__(self, dataframe: pd.DataFrame, indices: list, len: int) -> None:
        super().__init__()
        self.indices = indices
        self.len = len
        self.full_data = dataframe

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        row = self.full_data.iloc[self.indices[index]]
        pmid, title, abstract = row[["pmid", "Title", "abstractText"]]
        multi_hot = row.iloc[-14:].tolist()
        title=str(title)
        text = title+abstract
        return [pmid, text, torch.Tensor(multi_hot)]

