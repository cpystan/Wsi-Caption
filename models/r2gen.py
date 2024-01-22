import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, encoder_decoder=None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.prompt = nn.Parameter(torch.randn(1, 1, args.d_vf))
        self.fc = nn.Sequential(nn.LayerNorm(args.d_model),nn.Linear(args.d_model,args.d_model),nn.Linear(args.d_model,args.n_classes))
        if not encoder_decoder:
            print('use encoder_decoder: default')
            self.encoder_decoder = EncoderDecoder(args, tokenizer)
            
        if args.dataset_name:
            self.forward = self.forward_brca
        else:
            raise ValueError('no forward function')


    def cal_parameters(self):
        # 定义总参数量、可训练参数量及非可训练参数量变量
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        # 遍历model.parameters()返回的全局参数列表
        for param in self.parameters():

            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_brca(self, images, targets=None, mode='train'):

        att_feats = images  # shape 1*N*384
        att_feats = torch.cat([self.prompt,att_feats],dim=1)
        fc_feats = torch.sum(att_feats,dim=1) #shape 1*384

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        elif mode == 'encode':
            output = self.encoder_decoder(fc_feats, att_feats, mode='encode')

            logits = self.fc(output[0,0,:]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
        else:
            raise ValueError
        return output

