import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .layers import SimpleAttLayer, Attn_head

class GAT(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim,
                 heads_1=8, heads_2=1, att_dropout=0.6, input_dropout=0.6):
        
        super(GAT, self).__init__()

        self.att_dropout = att_dropout
        self.input_dropout = input_dropout

        self.conv1 = GATConv(in_channels=input_dim,
                             out_channels=hidden_dim // heads_1,
                             heads=heads_1,
                             concat=True,
                             dropout=att_dropout)
        self.conv2 = GATConv(in_channels=hidden_dim,
                             out_channels=output_dim,
                             heads=heads_2,
                             concat=False,
                             dropout=att_dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class HeteGAT_multi(nn.Module):
    def __init__(self, inputs_list, nb_classes, nb_nodes, attn_drop, ffd_drop,
                 bias_mat_list, hid_units, n_heads, activation=nn.ELU(), residual=False):
        super(HeteGAT_multi, self).__init__()
        self.inputs_list = inputs_list
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.bias_mat_list = bias_mat_list
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual 
        self.mp_att_size = 128
        self.layers = self._make_attn_head()
        self.simpleAttLayer = SimpleAttLayer(64,self.mp_att_size,time_major=False,return_alphas=True)
        self.fc = nn.Linear(64,self.nb_classes)
        
    def _make_attn_head(self):
        layers = []
        for inputs,biases in zip(self.inputs_list,self.bias_mat_list):
            layers.append(Attn_head(in_channel=inputs.shape[1],out_sz=self.hid_units[0],bias_mat=biases,in_drop=self.ffd_drop,coef_drop=self.attn_drop,activation=self.activation,residual=self.residual))
        #print("当前有{}个注意力头".format(len(layers)))
        return nn.Sequential(*list(m for m in layers))
        
    def forward(self,x):
        embed_list = []
        for i,(inputs, biases) in enumerate(zip(x,self.bias_mat_list)):

            attns = []
            jhy_embeds = []
            for _ in range(self.n_heads[0]):
                attns.append(self.layers[i](inputs))
            h_1 = torch.cat(attns,dim=1)
            #print("h_1.shape:",h_1.shape)
            #print("torch.squeeze(h_1).shape",torch.squeeze(h_1).shape)
            #print("torch.squeeze(h_1).reshape(h_1.shap[-1],1,-1.shape)",torch.squeeze(h_1).reshape(h_1.shape[-1],1,-1).shape)
            embed_list.append(torch.squeeze(h_1).reshape(h_1.shape[-1],1,-1))
        multi_embed = torch.cat(embed_list,dim=1)
        #print("multi_embed.shape:",multi_embed.shape)
        final_embed,att_val = self.simpleAttLayer(multi_embed)
        out = [] 
        for i in range(self.n_heads[-1]):
           out.append(self.fc(final_embed))
        #print("out[0].shape:",out[0].shape)
        return out[0]