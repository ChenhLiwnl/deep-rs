import torch
from torch.nn import Module, Linear, Conv2d, BatchNorm2d, Conv1d, BatchNorm1d, Sequential, ReLU
import pytorch3d.ops
import numpy as np


class KNearestVectorField(Module):

    def __init__(
        self, 
        knn, 
        ctx_point_feature_dim, 
        raise_xyz_channels=64, 
        point_dim=3, 
        radius = 0.1,
        hidden_dims_pointwise=[128, 128, 256],
        hidden_dims_global=[128, 64]
    ):
        super().__init__()
        self.knn = knn
        self.radius = radius
        self.raise_xyz = Linear(3, raise_xyz_channels)
        print(self.radius)
        dims = [raise_xyz_channels+ctx_point_feature_dim] + hidden_dims_pointwise
        conv_layers = []
        for i in range(len(dims)-1):
            conv_layers += [
                Conv2d(dims[i], dims[i+1], kernel_size=(1, 1)),
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
            conv_layers += [
                Conv1d(dims[i], dims[i+1], kernel_size=1),
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
        b , N_query , _ = p_query.shape
        dist, knn_idx, knn_points = pytorch3d.ops.knn_points(
            p1=p_query,
            p2=p_context,
            K=self.knn,
            return_nn=True
        )   # (B, N_query,K), (B, N_query, K), (B, N_query, K, 3)
        # Relative coordinates and their embeddings
        p_rel = knn_points - p_query.unsqueeze(-2)  # (B, N_query, K, 3)

        h_rel = self.raise_xyz(p_rel) # (B, N_query, K, H_rel)

        # Grouped features of neighbor points
        h_group = pytorch3d.ops.knn_gather(
            x=h_context,
            idx=knn_idx,
        )   # (B, N_query, K, H_ctx)
        
        # Combine
        h_combined = torch.cat([h_rel, h_group], dim=-1)  # (B, N_query, K, H_rel+H_ctx)

        # Featurize
        h_combined = h_combined.permute(0, 3, 1, 2).contiguous()  # (B, H_rel+H_ctx, N_query, K)
        y = self.pointwise_convmlp(h_combined)        # (B, H_out, N_query, K)
        y = y.permute(0, 2, 3, 1).contiguous() # (B, N_query, K, H_out)
        dist = torch.sqrt(dist)
        dist = dist.unsqueeze(-1) #(B, N_query, K, 1)
        c = 0.5*(torch.cos(dist * np.pi / self.radius) + 1.0)
        c = c * (dist <= self.radius) * (dist > 0.0) # (B, N_query, K, 1)
        #c = 1
        y = torch.mul(y , c).permute(0, 3, 1, 2).contiguous()
        y = y.sum(-1) # (B, H_out, N_query)
        #y, _ = torch.max(y, dim=3)  # (B, H_out, N_query)

        # Vectorize
        y = self.global_convmlp(y)  # (B, 3, N_query)
        y = y.permute(0, 2, 1).contiguous()  # (B, N_query, 3)
        return y
