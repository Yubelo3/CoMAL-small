from dataset import load_dataset
from text_encoder import BertEncoder
from torch.utils.data import DataLoader

device="cuda"

train_set,test_set=load_dataset()
train_loader=DataLoader(train_set,batch_size=2)

text_encoder=BertEncoder(device)

for x in train_loader:
    pmid,text,multi_hot=x
    multi_hot=multi_hot.to(device)
    print(text_encoder(text))
    break

