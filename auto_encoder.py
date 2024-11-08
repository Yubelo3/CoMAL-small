import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import SupConLoss


class AutoEncoder(nn.Module):
    def __init__(self,
                 num_classes=14,
                 in_dim=768,
                 hidden_dim=64,
                 sub_dim=64,
                 ) -> None:
        super().__init__()
        self.h_dim = hidden_dim
        self.sub_dim=sub_dim
        self.n_classes = num_classes
        # cluster sample count
        self.register_buffer("count", torch.zeros(
            num_classes+1, requires_grad=False), persistent=True)
        # cluster mean
        self.register_buffer("mean", torch.zeros(
            num_classes+1, sub_dim, requires_grad=False), persistent=True)

        # Encoder side
        # [B x in_dim]
        self.fc0 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*num_classes),
            nn.LeakyReLU()
        )
        # [B x C*h_dim] -> [B x C x h_dim]
        self.fc1 = nn.Linear(hidden_dim, sub_dim)
        # [B x C x sub_dim]

        # Decoder side
        self.fc2 = nn.Linear(sub_dim*num_classes, in_dim)

        # Loss
        self.contrastive_loss = SupConLoss()
        self.reconstruction_loss = nn.MSELoss()

    def get_sub_rep(self, x:torch.Tensor):
        # x: [B x in_dim]
        B = x.shape[0]
        y: torch.Tensor = self.fc0(x).view(B, self.n_classes, self.h_dim)
        sub_rep = self.fc1(y)  # [B x C x sub_dim]
        sub_rep = F.normalize(sub_rep, dim=-1)
        return sub_rep

    def forward(self, x: torch.Tensor):
        # x: [B x in_dim], text embedding
        B = x.shape[0]
        sub_rep = self.get_sub_rep(x)
        recon = self.fc2(sub_rep.view(B, -1))

        return sub_rep, recon

    def train_forward(self, x: torch.Tensor, multi_hot: torch.Tensor):
        # x: [B x in_dim], text embedding
        # multi_hot: [B x C], text multi label vector
        B = x.shape[0]
        sub_rep=self.get_sub_rep(x)
        multi_hot = multi_hot.view(B, self.n_classes, 1)
        agg_feat = (sub_rep*multi_hot).sum(dim=0)  # [C x sub_dim]
        agg_neg_feat = (sub_rep*(1-multi_hot)
                        ).sum(dim=[0, 1]).unsqueeze(0)  # [1 x sub_dim]
        agg_all_feat = torch.cat(
            [agg_feat, agg_neg_feat], dim=-2)  # [C+1 x sub_dim]

        new_postive_count = multi_hot.sum(dim=0).view(self.n_classes)  # [C]
        new_negtive_count = B*self.n_classes-new_postive_count.sum()  # scalar

        new_count = self.count.clone()
        new_count[:self.n_classes] += new_postive_count
        new_count[self.n_classes] += new_negtive_count

        self.mean = (self.count/new_count).unsqueeze(-1)*self.mean
        self.mean += agg_all_feat/new_count.unsqueeze(-1)
        self.count = new_count

        recon = self.fc2(sub_rep.view(B, -1))

        return sub_rep, recon

    def forward_and_get_losses(self, x: torch.Tensor, multi_hot: torch.Tensor):
        sub_rep, recon = self.train_forward(x, multi_hot)
        B, C, sub_dim = sub_rep.shape

        reconstruction_loss = self.reconstruction_loss(recon, x)  # L_rec

        flatten_label = multi_hot.flatten()  # [B*C]
        contrastive_mask = (flatten_label.view(-1, 1) ==
                            flatten_label.view(1, -1)).float()  # [B*C x B*C]
        neg_mask = (flatten_label == 0)
        contrastive_mask[neg_mask, :] = 0
        contrastive_mask[:, neg_mask] = 0
        contrastive_loss = self.contrastive_loss(
            # L_mscl
            sub_rep.view(B*C, sub_dim), contrastive_mask, batch_size=B*C)

        return sub_rep, recon, reconstruction_loss, contrastive_loss
