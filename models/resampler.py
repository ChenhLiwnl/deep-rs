from pytorch3d.ops.knn import knn_points
import torch
from torch.nn import Module
import pytorch3d.ops
from .common import *

from .encoders import get_encoder
from .vecfields import get_vecfield

class PointSetResampler(Module):

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        #print(config)
        self.encoder = get_encoder(config.encoder)
        self.vecfield = get_vecfield(config.vecfield, self.encoder.out_channels)
        
    def forward(self, p_query, p_ctx):
        """
        Args:
            p_query:  Query point set, (B, N_query, 3).
            p_ctx:    Context point set, (B, N_ctx, 3).
        Returns:
            (B, N_query, 3)
        """
        h_ctx = self.encoder(p_ctx)  # (B, N_query, H), Features of each point in `p_ctx`.
        vec = self.vecfield(p_query, p_ctx, h_ctx)  # (B, N_query, 3)
        return vec

    def get_loss_vec(self, p_query, p_ctx, vec_gt):
        """
        Computes loss according to ground truth vectors.
        Args:
            vec_gt:  Ground truth vectors, (B, N_query, 3).
        """
        vec_pred = self(p_query, p_ctx)  # (B, N_query, 3)
        loss = ((vec_pred - vec_gt) ** 2.0).sum(dim=-1).mean() 
        return loss 

    def get_loss_pc(self, p_query, p_ctx, p_gt, avg_knn):
        """
        Computes loss according to ground truth point clouds.
        Args:
            p_gt:     Ground truth point clouds, (B, N_gt, 3).
            avg_knn:  For each point in `p_query`, use how many nearest points in `p_gt`
                        to estimate the ground truth vector.
        """
        _, _, gt_nbs = pytorch3d.ops.knn_points(
            p_query,
            p_gt,
            K=avg_knn,
            return_nn=True,
        )   # (B, p_query, K, 3)

        vec_gt = (gt_nbs - p_query.unsqueeze(-2)).mean(-2)  # (B, N_query, 3)
        return self.get_loss_vec(p_query, p_ctx, vec_gt)
    def get_cd_loss(self, ref , gen):
        P = batch_pairwise_dist(ref, gen)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins)
        return loss_1 + loss_2
    @torch.no_grad()
    def resample(self, p_init, p_ctx, step_size=0.2, step_decay=0.95, num_steps=40):
        traj =  [p_init.clone().cpu()]
        h_ctx = self.encoder(p_ctx) # (B, N_ctx, 3)

        p_current = p_init
        for step in range(num_steps):
            vec_pred = self.vecfield(p_current, p_ctx, h_ctx)   # (B, N_query, 3)
                        # vec_pred: f(x_i) -> v_i + dL(x)/dx
            s = step_size * (step_decay ** step)
            p_next = p_current + s *  vec_pred 
            if step % 5 == 4 :
                traj.append(p_next.clone().cpu())
            p_current = p_next
        return p_current, traj
    def get_repulsion_loss(self, p_cur):
        h = 7e-3
        dist2, _, _ = pytorch3d.ops.knn_points(
            p1=p_cur,
            p2=p_cur,
            K=7,
            return_nn=True
        )   # (B, N_query,K ) , dist2 is a squared number
        dist2 = dist2[: , :, 1:] # dist2[: ,:, 0] = 0
        dist = torch.sqrt(dist2)
        #print(dist.mean())
        weight = torch.exp(-dist2 / h ** 2)
        loss = torch.mean((- dist) * weight) 
        #print('Loss %.6f' % (
        #     0.05*loss.item(),
        #))
        return 0.05*loss + 1e-4
    '''
    def repulsion(self, p_cur) :
        _, _, knn_points = pytorch3d.ops.knn_points(
            p1=p_cur,
            p2=p_cur,
            K=6,
            return_nn=True
        )   # (B, N_query,K , 3)
        p_rel = knn_points - p_cur.unsqueeze(-2)  # (B, N_query, K, 3)
        p_rel = p_rel[:,:,1:,:]
        p_normed = torch.norm(p_rel, p=2, dim = -1,keepdim=True)
        p_normed = p_rel / p_normed
        dist = torch.sqrt((p_rel ** 2 ).sum(dim = -1)) # (B,N_query,K)
        #p_normed = (-dist * torch.exp ( -1 * dist ** 2 / 6e-3)).unsqueeze(-1) * p_normed
        p_normed =  -(5e-4/dist ** 2).unsqueeze(-1) * p_normed
        target = torch.sum(p_normed,dim = -2 , keepdim=False) # target of the repulsion power
        return target
    '''
    def glr (self, p_cur):
        _, _, knn_points = pytorch3d.ops.knn_points(
            p1=p_cur,
            p2=p_cur,
            K=6,
            return_nn=True
        )   # (B, N_query,K , 3)
        p_rel = p_cur.unsqueeze(-2) - knn_points # (B, N_query, K, 3)
        p_rel = p_rel[:,:,1:,:]
        glr_grad = 2 * p_rel
        dist = (p_rel ** 2 ).sum(dim = -1) # (B,N_query,K)
        target =  torch.exp ( -1 * dist ** 2 / 1e-9).unsqueeze(-1) * glr_grad
        target = torch.sum(target,dim = -2 , keepdim=False)
        return target
    def gtv (self, p_cur) :
        _, _, knn_points = pytorch3d.ops.knn_points(
            p1=p_cur,
            p2=p_cur,
            K=6,
            return_nn=True
        )   # (B, N_query,K , 3)
        p_rel = (p_cur.unsqueeze(-2) - knn_points)
        p_rel = p_rel[:,:,1:,:]
        gtv_grad = p_rel / (torch.abs(p_rel)+1e-7)
        dist = (p_rel ** 2 ).sum(dim = -1) # (B,N_query,K)
        target =  torch.exp ( -1 * dist ** 2 / 5e-9).unsqueeze(-1) * gtv_grad
        target = torch.sum(target,dim = -2 , keepdim=False)
        return target