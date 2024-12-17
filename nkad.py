import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from pygod.detector import Detector
# from .gat import GAT
from torch_geometric.nn import GCN, SAGEConv, PNAConv, GIN
# from basic_nn import GCN
from gat import GAT

from pygod.utils import validate_device
# from ..metrics import eval_roc_auc
from pygod.metric import *
from modules import *
from torch_scatter import scatter
from pygod.nn.nn import MLP_generator, FNN_GAD_NR
from pygod.nn.functional import KL_neighbor_loss, W2_neighbor_loss
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import random

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)



def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, negative_slope=0.2, concat_out=True, aggr='sum') -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation="relu",
            feat_drop=dropout,
            attn_drop=0.1,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
        # mod = GAT(
        #     in_channels=in_dim,
        #     hidden_channels=num_hidden,
        #     out_channels=out_dim,
        #     num_layers=num_layers,
        #     act="relu",
        #     dropout=dropout
        # )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation="relu",
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod= GCN(
                in_channels=int(in_dim),
                hidden_channels=int(num_hidden),
                num_layers=num_layers,
                out_channels=int(out_dim),
                dropout=dropout,
                act=activation,
                aggr=aggr)
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod

class NKAD(Detector):
    def __init__(self,
                 hid_dim=64,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=None,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False,
                 max_batch_size=20000,
                 aggr='add',
                 loss_name='rec',
                 loss_weight=0.1,
                 T=100,
                 attr_encoder_name='gcn',
                 attr_decoder_name='mlp',
                 struct_decoder_name='mlp',
                 node_encoder_num_layers=2,
                 attr_decoder_num_layers=1,
                 struct_decoder_num_layers=1,
                 highpass_layer_num=2,
                 neighbor_rec_loss_coe=1,
                 high_coe=0.5,
                 center_rec_coef=0.2
                 ):
        super(NKAD, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers=6
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh

        # other param
        self.verbose = verbose
        self.model = None

        self.max_batch_size=max_batch_size

        self.aggr=aggr
        self.loss_name=loss_name
        self.loss_weight=loss_weight
        self.T=T

        self.attr_encoder_name = attr_encoder_name

        self.attr_decoder_name = attr_decoder_name
        self.struct_decoder_name = struct_decoder_name

        self.node_encoder_num_layers = node_encoder_num_layers

        # self.decoder_num_layers=decoder_num_layers
        self.attr_decoder_num_layers = attr_decoder_num_layers
        self.struct_decoder_num_layers = struct_decoder_num_layers

        self.dropout=dropout
        self.highpass_layer_num = highpass_layer_num
        self.neighbor_rec_loss_coe = neighbor_rec_loss_coe
        self.high_coe = high_coe
        self.center_rec_coef = center_rec_coef
    
    def fit(self, G, y_true=None):
        
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]


        if self.alpha is None:
            self.alpha = torch.std(G.s).detach() / \
                         (torch.std(G.x).detach() + torch.std(G.s).detach())

        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]


        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model = NKAD_Base(in_dim=G.x.shape[1],
                                   hid_dim=self.hid_dim,
                                   dropout=self.dropout,
                                   act=self.act,
                                   aggr=self.aggr,
                                   attr_encoder_name=self.attr_encoder_name,
                                   attr_decoder_name=self.attr_decoder_name,
                                   struct_decoder_name=self.struct_decoder_name,
                                   node_encoder_num_layers=self.node_encoder_num_layers,
                                   attr_decoder_num_layers=self.attr_decoder_num_layers,
                                   struct_decoder_num_layers=self.struct_decoder_num_layers,
                                   highpass_layer_num=self.highpass_layer_num,
                                   high_coe=self.high_coe,
                                   center_rec_coef=self.center_rec_coef).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        decision_score = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            for sampled_data in loader:
                x, s, edge_index = self.process_graph(sampled_data)
                x_, s_, neighbor_rec_loss= self.model(x, edge_index)

                rank_score, score = self.loss_func(x, x_, s, s_)

                if neighbor_rec_loss != None:
                    decision_score = rank_score.detach().cpu() #+ self.neighbor_rec_loss_coe * neighbor_rec_loss.detach().cpu()
                    loss = torch.mean(score) + self.neighbor_rec_loss_coe * torch.mean(neighbor_rec_loss)
                else:
                    decision_score = rank_score.detach().cpu()
                    loss = torch.mean(score)

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.verbose = False
            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, epoch_loss / G.x.shape[0]), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_score)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_score_ = decision_score
        self._process_decision_score()
        return self
        
    def decision_function(self, G):
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G, [self.num_neigh] * self.num_layers, batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = torch.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, s, edge_index = self.process_graph(sampled_data)
            x_, s_, neighbor_rec_loss = self.model(x, edge_index)
            rank_score, score = self.loss_func(x[:batch_size],
                                   x_[:batch_size],
                                   s[:batch_size, node_idx],
                                   s_[:batch_size])

            outlier_scores[node_idx[:batch_size]] = rank_score.detach().cpu() #+ self.neighbor_rec_loss_coe * neighbor_rec_loss.detach().cpu()
        return outlier_scores

    def decision_struct_function(self, G):
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G, [self.num_neigh] * self.num_layers, batch_size=self.batch_size)

        self.model.eval()
        outlier_edge_scores = np.zeros((G.x.shape[0],G.x.shape[0]))
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx
            x, s, edge_index = self.process_graph(sampled_data)
            x_, s_, _ = self.model(x, edge_index)
            s_score = self.s_loss_func(s[:batch_size, node_idx],s_[:batch_size])

            s_score[G.s==0]=0
            outlier_edge_scores[:batch_size, node_idx] = s_score.detach().cpu()
        return outlier_edge_scores

    def process_graph(self, G):
        s = G.s.to(self.device)
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)
        return x, s, edge_index

    def s_loss_func(self,s,s_):
        diff_structure=torch.abs(s-s_)
        return diff_structure

    def single_sce_loss(self,x,y,alpha=3):        
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    def sce_loss(self,x,x_,s,s_,alpha=3):
        diff_attribute = self.single_sce_loss(x,x_,alpha)
        diff_structure = self.single_sce_loss(s,s_, alpha)

        score = self.alpha * diff_attribute + (1 - self.alpha) * diff_structure
        return score

    def loss_func(self, x, x_, s, s_):

        if self.loss_name=='rec':
            score=self.rec_loss(x,x_,s,s_)
            return score, score

        elif self.loss_name=='add_log_t_entropy':
            score=self.rec_loss(x,x_,s,s_)
            entropy_loss=self.log_t_entropy_loss(x,x_,s,s_,score)

            rank_score=score + self.loss_weight * entropy_loss
            return rank_score, rank_score
        else:
            assert(False,'wrong loss func')

    def entropy_loss(self, x, x_, s, s_, score):
        diag_s=torch.eye(s.size()[0]).to(s.device)+s
        all_score=score.repeat(score.size()[0],1)

        all_score=torch.where(diag_s>0.1,all_score,0)+1e-6

        all_score=all_score/torch.sum(all_score,1)
        all_log_score=-torch.log(all_score)
        all_log_score=torch.sum(all_log_score,1)

        return all_log_score

    def log_t_entropy_loss(self, x, x_, s, s_, score):

        diag_s=torch.eye(s.size()[0]).to(s.device)+s
        all_score=score.repeat(score.size()[0],1).float()

        all_score=torch.where(diag_s.float()>0.1,all_score,torch.tensor(0.0, dtype=torch.float).to(s.device))+1e-6
        log_all_score=torch.log(all_score)/self.T

        all_score=F.softmax(log_all_score,dim =1)

        all_log_score=-torch.log(all_score)*all_score
        all_log_score=torch.sum(all_log_score,1)
        
        return all_log_score

    def rec_loss(self, x, x_, s, s_):
        
        diff_attribute = torch.pow(x_ - x, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        diff_structure = torch.pow(s_ - s, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))
        
        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors
        return score


class NKAD_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 dropout,
                 act,
                 aggr,
                 attr_encoder_name='gcn',
                 attr_decoder_name='mlp',
                 struct_decoder_name='mlp',
                 node_encoder_num_layers=2,
                 attr_decoder_num_layers=1,
                 struct_decoder_num_layers=1,
                 highpass_layer_num=2,
                 high_coe=0.5,
                 center_rec_coef=0.2
                 ):
        super(NKAD_Base, self).__init__()

        self.attr_encoder_name = attr_encoder_name
        self.attr_decoder_name = attr_decoder_name
        self.struct_decoder_name = struct_decoder_name
        self.aggr = aggr
        self.hid_dim = hid_dim
        self.linear = nn.Linear(in_dim, hid_dim)
        self.attr_encoder = setup_module(
            m_type=attr_encoder_name,
            enc_dec="encoding",
            in_dim=hid_dim,
            num_hidden=hid_dim,
            out_dim=hid_dim,
            num_layers= node_encoder_num_layers,
            nhead=1,
            nhead_out=1,
            concat_out=False,
            activation=act,
            dropout=dropout,
            negative_slope=0.2,
            residual=False,
            norm=None,
            aggr=aggr,
        )

        self.attr_decoder = setup_module(
                m_type=attr_decoder_name,
                enc_dec="decoding",
                in_dim=hid_dim,
                num_hidden=hid_dim,
                out_dim=in_dim,
                num_layers=attr_decoder_num_layers,
                nhead=1,
                nhead_out=1,
                concat_out=False,
                activation=act,
                dropout=dropout,
                negative_slope=0.2,
                residual=False,
                norm=None,
                aggr=aggr,
            )

        self.struct_decoder = setup_module(
                m_type=struct_decoder_name,
                enc_dec="decoding",
                in_dim=hid_dim,
                num_hidden=hid_dim,
                out_dim=in_dim,
                num_layers=struct_decoder_num_layers,
                nhead=1,
                nhead_out=1,
                concat_out=False,
                activation=act,
                dropout=dropout,
                negative_slope=0.2,
                residual=False,
                norm=None,
                aggr=aggr,
            )
        
        self.highpass_layer_num = highpass_layer_num
        self.high_coe = high_coe
        self.center_rec_coef = center_rec_coef
        self.k_neighbors = 10
        self.high_pass_encoder = HighPassGCN(in_channels=in_dim,
                                   hidden_channels=hid_dim,
                                   num_layers=highpass_layer_num,
                                   out_channels=hid_dim,
                                   dropout=dropout,
                                   add_self_loops=False,
                                   act=act)
                     
        self.mlp_mean = nn.Linear(hid_dim, hid_dim)
        self.mlp_sigma = nn.Linear(hid_dim, hid_dim)
        self.mlp_gen = MLP_generator(hid_dim, hid_dim)
        self.feature_decoder = FNN_GAD_NR(hid_dim, hid_dim, hid_dim, 2)
        self.reconstructor_from_neighbor = GATAggregator(hid_dim, hid_dim, hid_dim)

    def forward(self, x, edge_index):
        # attribute encode
        h0 = self.linear(x)
        # low-pass
        h_low = self.attr_encoder(h0, edge_index).to(torch.float32)
        # high-pass
        h_high = self.high_pass_encoder(x, edge_index)
        
        embed = (1 - self.high_coe) * h_low + self.high_coe * h_high

        h0_rec = self.feature_decoder(embed)
        h0_diff = (h0_rec - h0)  # [N, d]
        h0_rec_loss = torch.sum(h0_diff**2, dim=-1) # Scalar
        generated_mean, generated_cov, neighbor_rec_loss = self.full_batch_neigh_recon(h_low, h_high, h0, edge_index)
        neighbor_rec_loss = neighbor_rec_loss + h0_rec_loss

        mean_agg, cov_agg = self.reconstructor_from_neighbor(generated_mean, generated_cov, edge_index)
        cov_agg = cov_agg / (torch.norm(cov_agg, dim=(-1, -2), keepdim=True) + 1e-6)
        h_reconstructed = torch.einsum('nd,nbd->nb', mean_agg, cov_agg)
        embed = (1 - self.center_rec_coef) * embed + self.center_rec_coef * h_reconstructed
            
        # attribute decode
        if self.attr_decoder_name=='mlp' or self.attr_decoder_name=='linear':
            x_ = self.attr_decoder(embed)
        else:
            x_ = self.attr_decoder(embed, edge_index)
        # structure decode
        if self.struct_decoder_name=='mlp' or self.struct_decoder_name=='linear':
            h_ = self.struct_decoder(embed)
        else:
            h_ = self.struct_decoder(embed, edge_index)
        s_ = h_ @ h_.T

        return x_, s_, neighbor_rec_loss
    
    def compute_neighbors_stats(self, x, edge_index):
        src, dst = edge_index
        x_neighbors = x[src]
        mean_neighbors = scatter(x_neighbors, dst, dim=0, reduce="mean", dim_size=x.size(0))
        mean_neighbors = torch.nan_to_num(mean_neighbors, nan=0.0, posinf=0.0, neginf=0.0)
        mean_neighbors_expanded = mean_neighbors[dst]
        variance_neighbors = scatter(
            (x_neighbors - mean_neighbors_expanded) ** 2,
            dst,
            dim=0,
            reduce="mean",
            dim_size=x.size(0),
        )
        variance_neighbors = torch.nan_to_num(variance_neighbors, nan=0.0, posinf=0.0, neginf=0.0)
        std_neighbors = (variance_neighbors + 1e-6).sqrt()

        return mean_neighbors, std_neighbors
    
    def gather_neighbors_features(self, h0, edge_index):
        N, d = h0.shape
        src, dst = edge_index 
        neighbors_indices = [[] for _ in range(N)]
        for s, t in zip(src.tolist(), dst.tolist()):
            neighbors_indices[t].append(s)
        max_neighbors = max(len(neighbors) for neighbors in neighbors_indices)
        padded_neighbors = torch.full((N, max_neighbors), -1, dtype=torch.long, device=h0.device)
        for i, neighbors in enumerate(neighbors_indices):
            padded_neighbors[i, :len(neighbors)] = torch.tensor(neighbors, device=h0.device)
        rand_indices = torch.randint(0, max_neighbors, (N, self.k_neighbors), device=h0.device)
        sampled_indices = torch.gather(padded_neighbors, 1, rand_indices)
        sampled_indices[sampled_indices == -1] = 0 
        neighbors_features = h0[sampled_indices] 
        mask_no_neighbors = (padded_neighbors[:, 0] == -1)
        neighbors_features[mask_no_neighbors] = 0
        return neighbors_features

    def full_batch_neigh_recon(self, h1, h1_high, h0, edge_index):
        mean_neigh, std_neigh = self.compute_neighbors_stats(h0, edge_index)
        target_mean = mean_neigh
        target_std = std_neigh

        neigh_features = self.gather_neighbors_features(h0, edge_index)
        neigh_centered = neigh_features - mean_neigh.unsqueeze(1)

        target_cov = torch.einsum('nji,njk->nik', neigh_centered, neigh_centered)
        target_cov = target_cov / (self.k_neighbors - 1) 

        reg_eye = torch.eye(h0.shape[1], device=h0.device).unsqueeze(0).repeat(h0.shape[0], 1, 1)
        target_cov = target_cov + 1e-3 * reg_eye

        generated_mean = self.mlp_mean(h1)
        generated_std = self.mlp_sigma(h1_high)
        generated_std = torch.clamp(generated_std, min=1e-6, max=10.0)

        generated_neigh_centered = neigh_features - generated_mean.unsqueeze(1)
        generated_cov = torch.einsum('nji,njk->nik', generated_neigh_centered, generated_neigh_centered)
        generated_cov = generated_cov / (self.k_neighbors - 1) 
        generated_cov = generated_cov + 1e-3 * reg_eye

        mean_diff = (generated_mean - target_mean)
        mean_term = torch.sum(mean_diff**2, dim=-1)
        std_diff = (generated_std - target_std)
        std_term = torch.sum(std_diff**2, dim=-1)

        self.KL = False
        
        if self.KL:
            k = h1.shape[1]  # Dimensionality of node features
            trace_term = torch.einsum('nij,nji->n', torch.linalg.pinv(generated_cov), target_cov) 
            mean_diff = (generated_mean - target_mean).unsqueeze(-1) 
            mahalanobis_term = torch.einsum('nij,njk,nki->n', mean_diff.transpose(1, 2), torch.linalg.pinv(generated_cov), mean_diff)
            
            def logdet_safe(cov_matrix):
                eigvals = torch.linalg.eigvalsh(cov_matrix)
                eigvals = torch.clamp(eigvals, min=1e-6)  
                logdet = torch.sum(torch.log(eigvals), dim=-1) 
                return logdet

            logdet_gen = logdet_safe(generated_cov)  
            logdet_tar = logdet_safe(target_cov)
            det_term = logdet_gen - logdet_tar

            kl_loss = 0.5 * (trace_term + mahalanobis_term.squeeze() - k + det_term)
            kl_loss = kl_loss.mean()  # Scalar loss for the entire batch
            neighbor_loss = mean_term.mean() + std_term.mean() + kl_loss
        else:
            cov_diff = generated_cov - target_cov  # [N, d, d]
            cov_term = torch.norm(cov_diff, p='fro', dim=(1, 2)) 

            neighbor_loss = mean_term.mean() + std_term.mean() + cov_term.mean()

        return target_mean, generated_cov, neighbor_loss
