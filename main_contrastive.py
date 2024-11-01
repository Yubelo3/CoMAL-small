from dataset import load_dataset
from text_encoder import BertEncoder
from torch.utils.data import DataLoader
from classifier import MLPClassifier
from auto_encoder import AutoEncoder
from tqdm import tqdm
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from metric import topk_precision
from logger import TBWriter

device = "cuda"

train_set, test_set = load_dataset(initial_train_size=1000)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

model = {
    "text_encoder": BertEncoder(device),
    "auto_encoder": AutoEncoder().to(device),
    "classifer": MLPClassifier().to(device)
}
for v in model.values():
    v = torch.compile(v)
optimizer = {
    "text_encoder": torch.optim.Adam(model["text_encoder"].parameters(), lr=1e-4),
    "auto_encoder": torch.optim.Adam(model["auto_encoder"].parameters(), lr=0.004, betas=(0.9, 0.999)),
    "classifier": torch.optim.Adam(model["classifer"].parameters(), lr=0.004, betas=(0.9, 0.999))
}
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160, 240])

writer = TBWriter("contrastive")


def train_loop():
    batches = 0
    sum_recon_loss, sum_contra_loss, sum_cls_loss = 0.0, 0.0, 0.0
    for v in model.values():
        v.train()
    for x in train_loader:
        for k, v in optimizer.items():
            v.zero_grad()

        pmid, text, multi_hot = x
        multi_hot = multi_hot.to(device)
        txt_emb = model["text_encoder"](text)
        sub_rep, recon, recon_loss, contra_loss = model["auto_encoder"].forward_and_get_losses(
            txt_emb, multi_hot)
        contra_loss *= 0.2
        cls_loss = model["classifer"].get_loss(recon, multi_hot)
        loss = recon_loss+0.2*contra_loss+cls_loss

        sum_recon_loss += recon_loss.item()
        sum_contra_loss += contra_loss.item()
        sum_cls_loss += cls_loss.item()
        batches += 1
        loss.backward()
        for k, v in optimizer.items():
            v.step()
    return sum_recon_loss/batches, sum_contra_loss/batches, sum_cls_loss/batches


def test():
    for v in model.values():
        v.eval()
    test_loader = DataLoader(test_set, batch_size=32,
                             shuffle=False, drop_last=True)
    batches = 0
    sum_p1, sum_p3 = 0.0, 0.0
    with torch.no_grad():
        for x in test_loader:
            pmid, text, multi_hot = x
            multi_hot = multi_hot.to(device)
            txt_emb = model["text_encoder"](text)  # [B x in_dim]
            sub_rep, recon = model["auto_encoder"](txt_emb)
            logits = model["classifer"](recon)  # [B x C]
            sum_p1 += topk_precision(logits, multi_hot, k=1)
            sum_p3 += topk_precision(logits, multi_hot, k=3)
            batches += 1
    return sum_p1/batches, sum_p3/batches


bar = tqdm(range(100))
for epoch in bar:
    recon_loss, contra_loss, cls_loss = train_loop()
    loss = recon_loss+contra_loss+cls_loss
    writer.add_scalar("loss", loss, epoch)
    writer.add_scalar("recon_loss", recon_loss, epoch)
    writer.add_scalar("contra_loss", contra_loss, epoch)
    writer.add_scalar("cls_loss", cls_loss, epoch)
    p1, p3 = test()
    writer.add_scalar("p1", p1, epoch)
    writer.add_scalar("p3", p3, epoch)
    if (epoch+1)%20==0:
        writer.save_ckpt(model, epoch)
    bar.set_description(f"loss: {loss:.4f}, p@1:{p1:.4f}, p@3: {p3:.4f}")
