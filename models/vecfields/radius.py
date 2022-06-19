import torch
from torch.nn import Module, Linear, Conv2d, BatchNorm2d, Conv1d, BatchNorm1d, Sequential, ReLU
import pytorch3d.ops
import numpy as np
from ..common import *


class RadiusVectorField(Module):

    def __init__(
        self,
        radius,
        num_points , 
        ctx_point_feature_dim,
        style = "normal" ,
        max_points = 60,
        raise_xyz_channels=64,
        point_dim=3,
        hidden_dims_pointwise=[128, 128, 256],
        hidden_dims_global=[128, 64]
    ):
        super().__init__()
        self.radius = radius
        self.style = style
        self.num_points = num_points
        self.max_points = max_points
        self.raise_xyz = Linear(3, raise_xyz_channels)

        dims = [raise_xyz_channels+ctx_point_feature_dim] + hidden_dims_pointwise
        conv_layers = []
        for i in range(len(dims)-1):
            if self.style == "normal":
                conv_layers += [
                Conv2d(dims[i], dims[i+1], kernel_size=(1, 1)),
                BatchNorm2d(dims[i+1]),
            ]
            elif self.style == "residual":
                conv_layers += [
                ResnetBlockConv2d(dims[i], dims[i+1]),
                BatchNorm2d(dims[i+1]),
            ]
            if i < len(dims)-2:
                conv_layers += [
                    ReLU(),
                ]
        self.pointwise_convmlp = Sequential(*conv_layers)

        dims = [hidden_dims_pointwise[-1]] + hidden_dims_global + [point_dim]
        conv_layers = []
        for i in range(len(dims)-1):
            if self.style == "normal":
                conv_layers += [
                Conv1d(dims[i], dims[i+1], kernel_size=1),
            ]
            elif self.style == "residual":
                conv_layers += [
                    ResnetBlockConv1d(dims[i] , dims[i+1]) , 
                ]
            if i < len(dims)-2:
                conv_layers += [
                    BatchNorm1d(dims[i+1]),
                    ReLU(),
                ]
        self.global_convmlp = Sequential(*conv_layers)


    def forward(self, p_query, p_context, h_context):
        """
        Args:
            p_query:   Query point set, (B, N_query, 3).
            p_context: Context point set, (B, N_ctx, 3).
            h_context: Point-wise features of the context point set, (B, N_ctx, H_ctx).
        Returns:
            (B, N_query, 3)
        """
        dist, knn_idx, _ = pytorch3d.ops.knn_points(
            p1=p_query,
            p2=p_context,
            K=self.max_points,
            return_nn=True,
            return_sorted=True
        )   # (B,N_query , K ) , (B, N_query, K), (B, N_query, K, 3) , K should be a large number
        # dist : [B, N_query, K] records the distance between the query points and its K nearest neighbours
        B , N_query , _ = p_query.shape
        _ , N , _ = p_context.shape
        #radius_first_point = knn_idx[ : , : , 0].view(B , N_query , 1).repeat([1, 1, self.num_points])
        knn_idx[dist > self.radius ** 2] = N + 1  # set index of points out of range as N + 1
        radius_graph_idx = knn_idx[ : , : , :self.num_points] # knn_idx is sorted in ascending order before

        #  actually copy the value of the first point in radius_graph  for subsequent replacement.
        mask = (radius_graph_idx == N + 1) # The number of points within their( these points' ) radius is less than num_points
        radius_graph_idx = torch.where(mask, torch.zeros_like(radius_graph_idx),radius_graph_idx)
        radius_graph_points =  pytorch3d.ops.knn_gather( x = p_context , idx = radius_graph_idx) # ( B, N_query, num_points, 3 )
        dist = dist [ : , : , :self.num_points] 
        dist = torch.where(mask , torch.zeros_like(dist) , dist)
        # Relative coordinates and their embeddings
        p_rel = radius_graph_points - p_query.unsqueeze(-2)  # (B, N_query, num_points, 3)
        h_rel = self.raise_xyz(p_rel) # (B, N_query, num_points , H_rel)

        # Grouped features of neighbor points
        h_group = pytorch3d.ops.knn_gather(
            x=h_context,
            idx=radius_graph_idx,
        )   # (B, N_query, num_points, H_ctx)
        # Combine
        h_combined = torch.cat([h_rel, h_group], dim=-1)  # (B, N_query, num_points, H_rel+H_ctx)

        # Featurize
        h_combined = h_combined.permute(0, 3, 1, 2).contiguous()  # (B, H_rel+H_ctx, N_query, num_points)
        y = self.pointwise_convmlp(h_combined)        # (B, H_out, N_query, num_points)
        y = y.permute(0, 2, 3, 1).contiguous() # (B, N_query, num_points, H_out)
        dist = dist.unsqueeze(-1) #(B, N_query,  num_points, 1)
        c = 0.5*(torch.cos(dist * np.pi / self.radius) + 1.0)
        c = c * (dist <= self.radius) * (dist > 0.0) # (B, N_query,  num_points, 1)
        y = torch.mul(y , c).permute(0, 3, 1, 2).contiguous()
        y = y.sum(-1) # (B, H_out, N_query)
        #y, _ = torch.max(y, dim=3)  # (B, H_out, N_query)
        # Vectorize
        y = self.global_convmlp(y)  # (B, 3, N_query)
        y = y.permute(0, 2, 1).contiguous()  # (B, N_query, 3)
        return y
