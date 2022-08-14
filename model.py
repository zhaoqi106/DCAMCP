import torch
from dgllife.model import GAT
from dgl.nn.pytorch import MaxPooling
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def squash2(x,dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    u = (1-1/torch.exp(squared_norm.sqrt()))*x / (squared_norm+1e-8).sqrt()
    return u

class PrimaryCaps2(nn.Module):

    def __init__(self, out_channels):
        super(PrimaryCaps2, self).__init__()
        self.out_channels = out_channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = squash2(x.contiguous().view(batch_size, -1, self.out_channels), dim=-1)
        return x

class DigitCaps2(nn.Module):

    def __init__(self, in_dim, in_caps, num_caps, dim_caps, D):

        super(DigitCaps2, self).__init__()
        self.D = D
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.device = device
        self.attention_coef = 1/torch.sqrt(torch.tensor([self.D])).to(self.device)
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)
        self.B = nn.Parameter(0.01 * torch.randn(num_caps, 1, in_caps),requires_grad=True)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(-1)
        a = self.attention_coef*torch.matmul(u_hat,u_hat.transpose(2,3)).to(self.device)
        c = a.sum(dim=-2,keepdim=True).softmax(dim=1)
        s = torch.matmul((c + self.B), u_hat).squeeze(-2)
        v = squash2(s)
        return v

class CapsuleLoss(nn.Module):

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda

    def forward(self, labels, logits):
        left = (self.upper - logits).relu() ** 2
        right = (logits - self.lower).relu() ** 2
        margin_loss = torch.sum((labels * left + self.lmda * (1 - labels) * right),dim=-1)
        margin_loss = torch.mean(margin_loss)

        return margin_loss

class GAT1(nn.Module):

    def __init__(self, n_feats, fp):
        super(GAT1,self).__init__()
        self.n_feats = n_feats
        self.fp = fp
        self.gnn_layers = GAT(
            in_feats=self.n_feats,
            hidden_feats=[50,50],
            num_heads=[4,4],
            feat_drops=[0.2,0.2],
            attn_drops=[0.2,0.2],
            alphas=[0.2,0.2],
            residuals=[True,True],
            agg_modes=["flatten", "mean"],
            activations=[nn.LeakyReLU(), None]
        )
        self.pool = MaxPooling()
        self.linear = nn.Sequential(nn.Linear(50 + self.fp, 128),
                                    nn.Dropout(0.2),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128),
                                    nn.Linear(128,128),
                                    nn.ReLU())

    def forward(self, input_d, fps):
        graph = input_d
        out = graph.ndata["h"]
        out = self.gnn_layers(graph, out)
        out = self.pool(graph,out)
        out = torch.cat([out, fps], dim=-1)
        out = self.linear(out)
        return out

class Mymodel(nn.Module):

    def __init__(self, n_feats, fp):
        super(Mymodel, self).__init__()
        self.GAT = GAT1(n_feats, fp)
        self.pri = PrimaryCaps2(out_channels=8)
        self.dig = DigitCaps2(in_dim=8,
                             in_caps=16,
                             num_caps=2,
                             dim_caps=2,
                             D = 128)

    def forward(self, x, fp):
        out = self.GAT(x,fp)
        out = self.pri(out)
        out = self.dig(out)

        logits = (out ** 2).sum(dim=-1)
        logits = (logits + 1e-8).sqrt()

        return logits


