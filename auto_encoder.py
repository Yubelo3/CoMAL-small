import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self,
                 num_classes=14,
                 in_dim=768,
                 hidden_dim=64,
                 sub_dim=64,
                 ) -> None:
        super().__init__()
        self.h_dim = hidden_dim
        self.n_classes = num_classes
        self.cluster_sample_count = self.register_buffer(
            torch.zeros(num_classes+1, requires_grad=False))
        self.cluster_center = self.register_buffer(
            torch.zeros(num_classes+1, sub_dim, requires_grad=False))

        # Encoder side
        # [B x in_dim]
        self.fc0 = nn.Sequential([
            nn.Linear(in_dim, hidden_dim*num_classes),
            nn.LeakyReLU()
        ])
        # [B x C*h_dim] -> [B x C x h_dim]
        self.fc1 = nn.Linear(hidden_dim, sub_dim)
        # [B x C x sub_dim]

        # Decoder side
        self.fc2 = nn.Linear(sub_dim*num_classes, in_dim)

    def forward(self, x: torch.Tensor, multi_hot: torch.Tensor):
        B = x.shape[0]
        y: torch.Tensor = self.fc0(x).view(0, self.n_classes, self.h_dim)
        sub_rep = self.fc1(y)  # [B x C x sub_dim]
        sub_rep = F.normalize(sub_rep, dim=-1)
        multi_hot = multi_hot.view(B, self.n_classes, 1)
        agg_feat = (sub_rep*multi_hot).sum(dim=0)  # [C x sub_dim]
        agg_neg_feat = (sub_rep*(1-multi_hot)).sum(dim=[0, 1])  # [sub_dim]

        new_postive_count = multi_hot.sum(dim=0).view(self.n_classes)  # [C]
        new_negtive_count = B*self.n_classes-new_postive_count.sum()  # scalar

        self.cluster_center = self.cluster_center*self.cluster_sample_count
        self.cluster_center[:self.n_classes] += agg_feat
        self.cluster_center[-1] += agg_neg_feat
        self.cluster_sample_count[:self.n_classes] += new_postive_count
        self.cluster_sample_count[-1] += new_negtive_count
        self.cluster_center /= self.cluster_sample_count

        recon = self.fc2(sub_rep.view(B, -1))

        return sub_rep, recon
