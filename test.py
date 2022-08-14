from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from model import Mymodel
from dgllife.utils import Meter
from train import Mydataset

Count = 1564
smiles = pd.read_csv('../Data/smiles.csv')
bigraphs = []
node_featurizer = CanonicalAtomFeaturizer()
edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
for smile in smiles['smiles']:
    bigraphs.append(
        smiles_to_bigraph(
            smile,
            add_self_loop=True,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            canonical_atom_order=False,
        ))
bigraphs = bigraphs[Count:]

labels = []
for i in smiles['Class']:
    labels.append(i)
labels = labels[Count:]

fp = pd.read_csv('../Data/feature/CDKExt.csv')
fp = fp.iloc[Count:,:]
fp = np.array(fp).tolist()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Data=Mydataset(bigraphs, labels, fp)
collate = Data.get_collate_fn()
test_loader = DataLoader(dataset = Data, batch_size = 32, shuffle =False, collate_fn=collate)

Ind = torch.tensor(fp).shape[-1]
graph = bigraphs[0]
n_feats = graph.ndata["h"].shape[1]
model = Mymodel(n_feats = n_feats, fp = Ind)
model.load_state_dict(torch.load('checkpoint.pt'))
model.to(device)

model.eval()
TP, TN, FP, FN = 0, 0, 0, 0
eval_meter = Meter()
for i, (features, labels, fp) in enumerate(test_loader):
    features = features.to(device)
    fp = fp.to(device)
    labels_1 = torch.eye(2).index_select(dim=0, index=labels).to(device)
    logits = model(features, fp)

    for idx, i in enumerate(logits):
        if torch.argmax(i) == torch.argmax(labels_1[idx]) and torch.argmax(i) == 1:
            TP += 1
        if torch.argmax(i) == torch.argmax(labels_1[idx]) and torch.argmax(i) == 0:
            TN += 1
        if torch.argmax(i) != torch.argmax(labels_1[idx]) and torch.argmax(i) == 1:
            FP += 1
        if torch.argmax(i) != torch.argmax(labels_1[idx]) and torch.argmax(i) == 0:
            FN += 1

    y_true = labels.view(-1, 1)
    y_score = logits[:, 1].view(-1, 1)
    eval_meter.update(y_pred=y_score, y_true=y_true)

acc = 100*(TP+TN) / (TP+TN+FP+FN)
se = TP / (TP + FN)
sp = TN / (TN + FP)
auc = np.mean(eval_meter.compute_metric('roc_auc_score'))

print('Acc: {:.3f}%，AUC: {:.3f}, SE: {:.3f}, SP: {:.3f}\n'
      .format(acc, auc, se, sp))

