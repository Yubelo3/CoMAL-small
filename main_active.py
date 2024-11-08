from logger import TBWriter
from metric import topk_precision
from tqdm import tqdm
from auto_encoder import AutoEncoder
from classifier import MLPClassifier
from torch.utils.data import DataLoader
from text_encoder import BertEncoder
from dataset import load_dataset, active_sample_selection, fake_active_sample_selection
import random
import torch
import numpy
random.seed(2024)
torch.manual_seed(2024)
numpy.random.seed(2024)

device = "cuda"
batch_size = 64
fake_active = False


train_set, test_set = load_dataset(initial_train_size=200, test_size=1000)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


def reinitialize():
    model = {
        "text_encoder": BertEncoder(device),
        "auto_encoder": AutoEncoder().to(device),
        "classifier": MLPClassifier().to(device),
        "backbone": MLPClassifier().to(device)
    }
    for v in model.values():
        v = torch.compile(v)
    optimizer = {
        "text_encoder": torch.optim.Adam(model["text_encoder"].parameters(), lr=2e-5),
        "auto_encoder": torch.optim.Adam(model["auto_encoder"].parameters(), lr=0.005, betas=(0.9, 0.999)),
        "classifier": torch.optim.Adam(model["classifier"].parameters(), lr=0.005, betas=(0.9, 0.999)),
        "backbone": torch.optim.Adam(model["backbone"].parameters(), lr=0.005, betas=(0.9, 0.999)),
    }
    return model, optimizer


model, optimizer = reinitialize()
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160, 240])

writer = TBWriter("fake_active") if fake_active else TBWriter("active")


def train_loop_first_stage():
    batches = 0
    sum_recon_loss, sum_contra_loss, sum_cls_loss, sum_backbone_loss = 0.0, 0.0, 0.0, 0.0
    for v in model.values():
        v.train()

    while True:
        for x in train_loader:
            for k, v in optimizer.items():
                v.zero_grad()
            pmid, text, multi_hot = x
            multi_hot = multi_hot.to(device)
            txt_emb = model["text_encoder"](text)
            backbone_loss = model["backbone"].get_loss(txt_emb, multi_hot)
            sub_rep, recon, recon_loss, contra_loss = model["auto_encoder"].forward_and_get_losses(
                txt_emb, multi_hot)
            cls_loss = model["classifier"].get_loss(recon, multi_hot)
            loss = backbone_loss+recon_loss+contra_loss+cls_loss

            sum_recon_loss += recon_loss.item()
            sum_contra_loss += contra_loss.item()
            sum_cls_loss += cls_loss.item()
            sum_backbone_loss += backbone_loss.item()
            batches += 1
            loss.backward()
            for k, v in optimizer.items():
                v.step()
            if batches >= 16:
                break
        if batches >= 16:
            break
    return sum_recon_loss/batches, sum_contra_loss/batches, sum_cls_loss/batches, sum_backbone_loss/batches


def train_loop_second_stage():
    batches = 0
    sum_recon_loss, sum_contra_loss, sum_cls_loss = 0.0, 0.0, 0.0
    for k in ["auto_encoder", "classifier"]:
        model[k].train()
    for k in ["text_encoder", "backbone"]:
        model[k].eval()

    while True:
        for x in train_loader:
            for k, v in optimizer.items():
                v.zero_grad()
            pmid, text, multi_hot = x
            multi_hot = multi_hot.to(device)
            with torch.no_grad():
                txt_emb = model["text_encoder"](text)
            sub_rep, recon, recon_loss, contra_loss = model["auto_encoder"].forward_and_get_losses(
                txt_emb, multi_hot)
            # contra_loss *= 0.2
            cls_loss = model["classifier"].get_loss(recon, multi_hot)
            loss = recon_loss+contra_loss+cls_loss

            sum_recon_loss += recon_loss.item()
            sum_contra_loss += contra_loss.item()
            sum_cls_loss += cls_loss.item()
            batches += 1
            loss.backward()
            for k in ["auto_encoder", "classifier"]:
                optimizer[k].step()
            if batches >= 16:
                break
        if batches >= 16:
            break
    return sum_recon_loss/batches, sum_contra_loss/batches, sum_cls_loss/batches


def train_loop_end_stage():
    batches = 0
    sum_backbone_loss = 0.0
    for k in ["text_encoder", "backbone"]:
        model[k].train()

    while True:
        for x in train_loader:
            for k, v in optimizer.items():
                v.zero_grad()
            pmid, text, multi_hot = x
            multi_hot = multi_hot.to(device)
            txt_emb = model["text_encoder"](text)
            backbone_loss = model["backbone"].get_loss(txt_emb, multi_hot)
            sum_backbone_loss += backbone_loss.item()
            batches += 1
            loss = backbone_loss
            loss.backward()
            for k in ["text_encoder", "backbone"]:
                optimizer[k].step()
            if batches >= 16:
                break
        if batches >= 16:
            break
    return sum_backbone_loss/batches


def test(first_stage=True):
    for v in model.values():
        v.eval()
    test_loader = DataLoader(test_set, batch_size=64,
                             shuffle=False, drop_last=True)
    batches = 0
    sum_p1, sum_p3, sum_p5 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x in test_loader:
            pmid, text, multi_hot = x
            multi_hot = multi_hot.to(device)
            txt_emb = model["text_encoder"](text)  # [B x in_dim]
            # if not first_stage:
            #     sub_rep, recon = model["auto_encoder"](txt_emb)
            #     logits = model["classifier"](recon)  # [B x C]
            # else:
            logits = model["backbone"](txt_emb)
            sum_p1 += topk_precision(logits, multi_hot, k=1)
            sum_p3 += topk_precision(logits, multi_hot, k=3)
            sum_p5 += topk_precision(logits, multi_hot, k=5)
            batches += 1
    return sum_p1/batches, sum_p3/batches, sum_p5/batches


cycle_epoch = 250
cycles = 10
cur_cycle = 0

bar = tqdm(range(cycle_epoch*cycles))
for epoch in bar:
    first_stage = (epoch+1) % cycle_epoch < 100
    if first_stage:
        recon_loss, contra_loss, cls_loss, backbone_loss = train_loop_first_stage()
        loss = recon_loss+contra_loss+cls_loss+backbone_loss
    else:
        recon_loss, contra_loss, cls_loss = train_loop_second_stage()
        loss = recon_loss+contra_loss+cls_loss
    writer.add_scalar("loss", loss, epoch)
    writer.add_scalar("recon_loss", recon_loss, epoch)
    writer.add_scalar("contra_loss", contra_loss, epoch)
    writer.add_scalar("cls_loss", cls_loss, epoch)
    if first_stage:
        writer.add_scalar("backbone_loss", backbone_loss, epoch)
    p1, p3, p5 = test(first_stage)
    writer.add_scalar("p1", p1, epoch)
    writer.add_scalar("p3", p3, epoch)
    writer.add_scalar("p5", p5, epoch)
    if (epoch+1) % 500 == 0:
        writer.save_ckpt(model, epoch)
    bar.set_description(
        f"cycle{cur_cycle+1}, loss: {loss:.4f}, p@1:{p1:.4f}, p@3: {p3:.4f}, p@5: {p5:.4f}")

    if (epoch+1) % cycle_epoch == 0 and cur_cycle != cycles-1:
        cur_cycle += 1
        # train_set = active_sample_selection(model, train_set, 100)
        if fake_active:
            train_set = fake_active_sample_selection(model, train_set, 200)
        else:
            train_set = active_sample_selection(model, train_set, 200)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        model, optimizer = reinitialize()

model, optimizer = reinitialize()
bar = tqdm(range(cycle_epoch*cycles, cycle_epoch*cycles+200))
for rest_epoch in bar:
    backbone_loss = train_loop_end_stage()
    writer.add_scalar("backbone_loss", backbone_loss, epoch)
    p1, p3, p5 = test(first_stage=True)
    writer.add_scalar("p1", p1, epoch)
    writer.add_scalar("p3", p3, epoch)
    writer.add_scalar("p5", p5, epoch)
    if (epoch+1) % 200 == 0:
        writer.save_ckpt(model, epoch)
    bar.set_description(
        f"loss: {loss:.4f}, p@1:{p1:.4f}, p@3: {p3:.4f}, p@5: {p5:.4f}")
