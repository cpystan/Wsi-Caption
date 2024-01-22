import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 1024
        self.D = 512
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x):
        N, C, L = x.shape
        o = []

        H = x.reshape(-1,L)# NxL

        A = self.attention(H)  # NxK
        #A = torch.transpose(A, 1, 0)  # KxN
        A = A.reshape(N,C,self.K)
        A = F.softmax(A, dim=1)  # softmax over N
        A = A.permute(0,2,1)

        M = torch.bmm(A, x)  # KxL

        return M

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x





class MS2MIL(nn.Module):
    def __init__(self, n_classes, feat_size=1024, scales=5):
        super(MS2MIL, self).__init__()
        self.feat_size = feat_size
        self.L = 512
        self.scales = scales
        self.feature_extractor = nn.Sequential(
            nn.Linear(feat_size, self.L),
            nn.ReLU(),
        )
        #self.scale_token = nn.Parameter(torch.randn(1, 1, 3000,1024))
        self.n_classes = n_classes
        #self.layer0 = nn.TransformerEncoderLayer(d_model=feat_size,nhead=8,batch_first=True)
        self.layer0 = Attention()

        #self.token_fc = nn.Linear(feat_size*self.scales,feat_size)
        self.token_fc = nn.Linear(feat_size,feat_size)
        self._fc2 = nn.Linear(feat_size, self.n_classes)




    def forward(self, **kwargs):
        h = kwargs['data'].float().cuda()# [B, C, n, 1024]

        h = h.permute(0,2,1,3)


        B, C, N, L = h.shape
        h = self.feature_extractor(h.reshape(-1,N,L))
        h = h.reshape(B, C, N, self.L)
        #---->Translayer x0
        h = h.permute(0,2,1,3).reshape(-1,C,self.L)
        hh = self.layer0(h[:,[1,2],...].reshape(1,-1,self.feat_size))




        hh = self.token_fc((torch.flatten(hh)).unsqueeze(0))
        # ---->predict
        logits = self._fc2(hh)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict


if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data=data)
    print(results_dict)
