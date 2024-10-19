import pandas as pd
import random
from torch.utils.data import Dataset
import torch


def load_dataset(csv_path: str = "data/PubMed/pm.csv", train_ratio=0.8):
    full_data = pd.read_csv(csv_path)
    sample_count = len(full_data)
    train_sample_count = int(sample_count*train_ratio)
    indices = list(range(sample_count))
    random.shuffle(indices)

    train_df = full_data.loc[indices[:train_sample_count]]
    test_df = full_data.loc[indices[train_sample_count:]]

    train_set = PubMedDataset(train_df.copy().reindex())
    test_set = PubMedDataset(test_df.copy().reindex())
    return train_set, test_set


class PubMedDataset(Dataset):
    # Title,abstractText,meshMajor,pmid,meshid,meshroot,A,B,C,D,E,F,G,H,I,J,L,M,N,Z
    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__()
        self.pmid=dataframe["pmid"].to_list()
        self.title=dataframe["Title"].to_list()
        self.abstract=dataframe["abstractText"].to_list()
        self.multi_hot=dataframe.iloc[:,-14:].values

    def __len__(self):
        return len(self.pmid)

    def __getitem__(self, index):
        text=self.title[index]+" | "+self.abstract[index]
        return [self.pmid[index],text,torch.Tensor(self.multi_hot[index])]
