from dataset import load_dataset
from text_encoder import BertEncoder
from torch.utils.data import DataLoader
from classifier import MLPClassifier
from tqdm import tqdm
import torch

device="cuda"

train_set,test_set=load_dataset()
train_loader=DataLoader(train_set,batch_size=4)

text_encoder=BertEncoder(device)
classifier=MLPClassifier().to(device)

optimizer=torch.optim.Adam(classifier.parameters(),lr=0.001)

bar=tqdm(range(10))
for epoch in bar:
    sum_loss=0.0
    for x in train_loader:
        optimizer.zero_grad()
        pmid,text,multi_hot=x
        multi_hot=multi_hot.to(device)
        txt_emb=text_encoder(text)
        loss=classifier.get_loss(txt_emb,multi_hot)
        print(loss.item())
        sum_loss+=loss.item()
        loss.backward()
        optimizer.step()
    bar.set_description(f"loss: {sum_loss/len(bar):.4f}")
