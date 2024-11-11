import pandas as pd
import random
from torch.utils.data import Dataset
from models.text_encoder import BertEncoder
from torch.utils.data import DataLoader
from models.classifier import MLPClassifier
from models.auto_encoder import AutoEncoder
import torch
import torch.nn.functional as F
import sys


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
        title = str(title)
        text = title+abstract
        return [pmid, text, torch.Tensor(multi_hot)]

    def get_unlabeled_dataset(self):
        unlabeled_dataset = PubMedDataset(
            self.full_data,
            self.indices[self.len:],
            len(self.indices)-self.len
        )
        return unlabeled_dataset


def fake_active_sample_selection(model: dict, dataset: PubMedDataset, select=100, gamma=0.5, batch_size=64):
    return PubMedDataset(
        dataset.full_data,
        dataset.indices,
        dataset.len+select
    )


def active_sample_selection(model: dict, dataset: PubMedDataset, select=100, gamma=0.5, batch_size=64):
    # correspond to part 3.2 (active learning sampling strategy) of original paper
    text_encoder: BertEncoder = model["text_encoder"]
    auto_encoder: AutoEncoder = model["auto_encoder"]
    backbone: MLPClassifier = model["backbone"]
    text_encoder = text_encoder.eval()
    auto_encoder = auto_encoder.eval()
    backbone = backbone.eval()
    device = text_encoder.device
    unlabeled_dataset = dataset.get_unlabeled_dataset()
    labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=False)
    C = auto_encoder.n_classes

    # 1. Compute mean similarity with cluster prototypes and mean cardinality
    sum_cardinality = 0
    sum_similarity = torch.zeros(C, device=device)  # [C]
    sample_count = torch.zeros(C, device=device)  # [C]
    prototypes = F.normalize(
        auto_encoder.mean[:-1], dim=-1).unsqueeze(0)  # [1 x C x sub_dim]
    with torch.no_grad():
        for x in labeled_loader:
            pmid, text, multi_hot = x
            multi_hot = multi_hot.to(device)  # [B x C]
            sum_cardinality += multi_hot.sum().item()
            txt_emb = text_encoder(text)  # [B x in_dim]
            sub_rep = auto_encoder.get_sub_rep(txt_emb)  # [B x C x sub_dim]
            similarity = (sub_rep*prototypes).sum(dim=-1)  # [B x C]
            similarity *= multi_hot  # [B x C]
            sum_similarity += similarity.sum(dim=0)  # [C]
            sample_count += multi_hot.sum(dim=0)  # [C]
    mean_similarity = sum_similarity/sample_count
    mean_cardinality = sum_cardinality/len(dataset)

    print(f"mean_cardinality: {mean_cardinality}")
    sum_sample = sample_count.sum().item()
    print(f"mean_similarity:")
    print(mean_similarity.cpu().tolist())
    print("sample_ratio:")
    print((sample_count/sum_sample).cpu().tolist())

    # 2. Compute informativeness for unlabeled pool
    individual_informativeness = []
    _all_pred_cardinality = []
    with torch.no_grad():
        for x in unlabeled_loader:
            pmid, text, multi_hot = x
            multi_hot = multi_hot.to(device)  # [B x C]
            txt_emb = text_encoder(text)  # [B x in_dim]
            sub_rep, recon = auto_encoder(txt_emb)  # [B x C x sub_dim]

            similarity = (sub_rep*prototypes).sum(dim=-1)  # [B x C]
            positive_mask = (
                similarity > mean_similarity.unsqueeze(0)).float()  # [B x C]
            cardinality = positive_mask.sum(dim=-1)  # [B]
            slci = (cardinality-mean_cardinality).abs()

            logits = backbone(txt_emb)  # [B x C]
            pred_mask = (logits.sigmoid() > 0.5).float()  # [B x C]
            _pred_cardinality = pred_mask.sum(dim=-1)  # [B], just for record
            pfd = ((1-similarity)*pred_mask).sum(dim=-1)  # [B]

            informativeness = (slci**gamma)*(pfd**(1-gamma))
            individual_informativeness.append(informativeness)
            _all_pred_cardinality.append(_pred_cardinality)
    _all_pred_cardinality = torch.cat(_all_pred_cardinality, dim=-1)
    print(f"mean_pred_cardinality: {_all_pred_cardinality.mean():.4f}")
    individual_informativeness = torch.cat(individual_informativeness, dim=-1)

    # 3. Select samples
    old_indices = dataset.indices
    old_len = len(dataset)
    unlabeled_len = len(unlabeled_dataset)
    value, index = individual_informativeness.topk(select)
    index = index.cpu().tolist()

    selected_sample_index = [old_indices[x+old_len] for x in index]
    unselected_sample_index = [old_indices[x+old_len]
                               for x in range(unlabeled_len) if x not in index]
    new_indices = dataset.indices[:old_len]
    new_indices += selected_sample_index
    new_indices += unselected_sample_index

    sys.stdout.flush()

    return PubMedDataset(
        dataset.full_data,
        new_indices,
        old_len+select
    )
