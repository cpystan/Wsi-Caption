import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPooling(nn.Module):
    def __init__(self, n_classes, feat_size=512):
        super(MaxPooling, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024,feat_size), nn.ReLU())
        self.n_classes = n_classes
        self._fc0 = nn.Linear(feat_size, 1)
        self._fc2 = nn.Linear(feat_size, self.n_classes)

    def forward(self, **kwargs):

        h = kwargs['data'].float().cuda()[:,:,1,:] #[B, n, 1024]
        # pdb.set_trace()
        h = self._fc1(h) #[B, n, 512]
        h = torch.max(h, dim=1)[0]
        # pdb.set_trace()
        # logits0 = self._fc0(h)
        # max_pos = torch.argmax(logits0)
        # h = h[:,max_pos]
        # pdb.set_trace()
        # pick max


        logits = self._fc2(h) #[B, n_classes]
        # pdb.set_trace()
        Y_prob = torch.softmax(logits,dim=-1)
        # max_pos = torch.argmax(Y_prob[:,:,1])
        # Y_prob = Y_prob[:,max_pos,]
        Y_hat = torch.argmax(Y_prob, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 384)).cuda()
    model = MaxPooling(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
