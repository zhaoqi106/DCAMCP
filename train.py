from torch.utils.data import DataLoader, Dataset
import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from model import *
import dgl
from dgllife.utils import Meter
import numpy as np
import random
from pytorchtools import EarlyStopping


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
bigraphs = bigraphs[:Count]

fp = pd.read_csv('../Data/feature/CDKExt.csv')
fp = fp.iloc[:Count,:]
fp = np.array(fp).tolist()

labels = []
for i in smiles['Class']:
    labels.append(i)
labels = labels[:Count]

def setup_seed(seed):
    random.seed((seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class Mydataset(Dataset):
    def __init__(self, train_features, train_labels, fp):
        self.x_data = train_features
        self.y_data = train_labels
        self.fp = fp
        self.len = len(train_labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.fp[index]

    def __len__(self):
        return self.len

    def get_collate_fn(self):
        def _collate(data):
            graphs, labels, fp = map(list, zip(*data))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.tensor(labels), torch.tensor(fp)
        return _collate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_kfold_data(k, i, X, y, z):
    fold_size = len(X) // k
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid, z_valid = X[val_start:val_end], y[val_start:val_end], z[val_start:val_end]
        X_train = X[0:val_start] + X[val_end:]
        y_train = y[0:val_start] + y[val_end:]
        z_train = z[0:val_start] + z[val_end:]
    else:
        X_valid, y_valid, z_valid = X[val_start:], y[val_start:], z[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]
        z_train = z[0:val_start]

    return X_train, y_train, z_train, X_valid, y_valid, z_valid

def traink(model,X_train, y_train, z_train, X_val, y_val, z_val, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):

    data = Mydataset(X_train, y_train, z_train)
    collate = data.get_collate_fn()
    train_loader = DataLoader(data, BATCH_SIZE,collate_fn=collate, shuffle=True)
    val_loader = DataLoader(Mydataset(X_val, y_val, z_val), BATCH_SIZE,collate_fn=collate, shuffle=True)
    model=model.to(device)

    criterion = CapsuleLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor= 0.5)
    early_stopping = EarlyStopping(patience = 50, verbose=True)

    losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    SE = []
    SP = []
    AUC = []

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        correct = 0
        for i, (features, labels, fp) in enumerate(train_loader):

            features = features.to(device)
            fp = fp.to(device)
            labels = torch.eye(2).index_select(dim=0, index=labels).to(device)

            optimizer.zero_grad()
            logits = model(features, fp)
            loss = criterion(logits,labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
            correct += torch.sum(
                torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()
        accuracy = 100. * correct / len(X_train)
        print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'
              .format(epoch + 1, loss.item(), correct, len(X_train), accuracy))
        train_acc.append(accuracy)

        model.eval()
        val_loss = 0
        correct = 0
        eval_meter = Meter()
        TP, TN, FP, FN = 0, 0, 0, 0
        with torch.no_grad():
            for i, (features,labels,fp) in enumerate(val_loader):

                features = features.to(device)
                fp = fp.to(device)
                labels_1 = torch.eye(2).index_select(dim=0, index=labels).to(device)

                optimizer.zero_grad()
                logits= model(features, fp)
                loss = criterion(logits, labels_1).item()
                val_loss += loss * len(labels_1)

                for idx, i in enumerate(logits):
                    if torch.argmax(i) == torch.argmax(labels_1[idx]) and torch.argmax(i) == 1:
                        TP += 1
                    if torch.argmax(i) == torch.argmax(labels_1[idx]) and torch.argmax(i) == 0:
                        TN += 1
                    if torch.argmax(i) != torch.argmax(labels_1[idx]) and torch.argmax(i) == 1:
                        FP += 1
                    if torch.argmax(i) != torch.argmax(labels_1[idx]) and torch.argmax(i) == 0:
                        FN += 1
                            
                correct += torch.sum(
                    torch.argmax(logits, dim=1) == torch.argmax(labels_1, dim=1)).item()

                y_true = labels.view(-1,1)
                y_score = logits[:,1].view(-1,1)
                eval_meter.update(y_pred=y_score,y_true=y_true)
        sp = TN / (TN + FP)
        se = TP / (TP + FN)
        auc = np.mean(eval_meter.compute_metric('roc_auc_score'))


        val_losses.append(val_loss / len(X_val))
        SE.append(se)
        SP.append(sp)
        AUC.append(auc)

        accuracy = 100. * correct / len(X_val)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n AUC: {:.3f}'
              ', SE: {:.3f}, SP: {:.3f}\n'.format(
            val_loss, correct, len(X_val), accuracy, auc, se, sp))

        val_acc.append(accuracy)
        scheduler.step(auc)

        early_stopping(auc, model)

        if early_stopping.early_stop:
            print("Early stopping")

            break


    return losses, val_losses, train_acc, val_acc, AUC, SE, SP

def k_fold(k, X_train, y_train, z_train, num_epochs=3, learning_rate=0.0001, batch_size=16):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    SE_sum, SP_sum, AUC_sum= 0,0,0

    for i in range(k):
        print('*' * 25, '第', i + 1, '折', '*' * 25)
        data = get_kfold_data(k, i, X_train, y_train, z_train)
        fps = torch.tensor(z_train)
        Ind = fps.shape[-1]
        graph = X_train[0]
        n_feats = graph.ndata["h"].shape[1]
        model = Mymodel(n_feats = n_feats, fp = Ind)
        train_loss, val_loss, train_acc, val_acc, AUC, SE, SP= traink(model, *data, batch_size, learning_rate, num_epochs)
        index = AUC.index(max(AUC))

        print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss[index], train_acc[index]))
        print('valid loss:{:.5f}, valid_acc:{:.3f}%\n'.format(val_loss[index], val_acc[index]))

        train_loss_sum += train_loss[index]
        valid_loss_sum += val_loss[index]
        train_acc_sum += train_acc[index]
        valid_acc_sum += val_acc[index]

        SE_sum += SE[index]
        SP_sum += SP[index]
        AUC_sum += AUC[index]



    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))
    print('average valid AUC:{:.3f}, average valid SE:{:.3f}, average valid SP:{:.3f}'
          .format(AUC_sum / k,SE_sum / k,SP_sum / k))

    return

if __name__ == '__main__':
    setup_seed(2)
    k_fold(5,bigraphs,labels,fp,num_epochs=100,learning_rate=0.001 ,batch_size=256)


