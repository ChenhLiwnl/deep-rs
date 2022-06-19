import math
import torch
import numpy as np
import pytorch3d.ops
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, KDTree
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import itertools
from models.common import farthest_point_sampling
from .transforms import NormalizeUnitSphere
def split_tensor_to_segments(x, segsize):
    num_segs = math.ceil(x.size(0) / segsize)
    segs = []
    for i in range(num_segs):
        segs.append(x[i*segsize : (i+1)*segsize])
    return segs

def patch_based_denoise(model, pcl_noisy, step_size=0.15, num_steps=50, patch_size=4000, seed_k=5, denoise_knn=4, step_decay=0.98, get_traj=False):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    #seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    #_, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    #patches = patches[0]
    patches = pcl_noisy
    with torch.no_grad():
        #model.eval()
        patches_denoised, traj = model.resample(
        p_init=patches, 
        p_ctx=patches,
        step_size=step_size, 
        step_decay=step_decay, 
        num_steps=num_steps
    )
    '''
    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
    pcl_denoised = pcl_denoised[0]
    fps_idx = fps_idx[0]
'''
    if get_traj:
        return patches_denoised[0], traj
    else:
        return patches_denoised[0]
def patch_based_upsample(model, pcl_low, pcl_noisy, step_size=0.15, num_steps=50, patch_size=512, seed_k=3, denoise_knn=4, step_decay=0.98, get_traj=False):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()
    M , d = pcl_low.size()
    rate = int(N/M)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    pcl_low = pcl_low.unsqueeze(0)
    #seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    #_, _, patches_noisy = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy,K=int(patch_size) ,return_nn=True)
    #patches_noisy = patches_noisy[0]
    #_, _, patches_low = pytorch3d.ops.knn_points(seed_pnts, pcl_low, K=patch_size, return_nn=True)
    #patches_low = patches_low[0]
    patches_noisy = pcl_noisy
    patches_low = pcl_low
    with torch.no_grad():
        model.eval()
        patches_denoised, traj = model.resample(
        p_init=patches_noisy, 
        p_ctx=patches_low,
        step_size=step_size, 
        step_decay=step_decay, 
        num_steps=num_steps
    )
    
    if get_traj : 
        return patches_denoised[0] , traj
    else : 
        return patches_denoised[0]
    return patches_denoised[0]
    '''
    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
    pcl_denoised = pcl_denoised[0]
    fps_idx = fps_idx[0]

    if get_traj:
        for i in range(len(traj)):
            traj[i] = traj[i].view(-1, d)[fps_idx, :]
        return pcl_denoised, traj
    else:
        return pcl_denoised
    '''
    
def patch_based_upsample_big(model, pcl_low, pcl_noisy, step_size=0.15, num_steps=50, patch_size=512, seed_k=3, denoise_knn=4, step_decay=0.99, get_traj=False):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()
    M , d = pcl_low.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    pcl_low = pcl_low.unsqueeze(0)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches_noisy = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=int(patch_size), return_nn=True)
    patches_noisy = patches_noisy[0]
    _, _, patches_low = pytorch3d.ops.knn_points(seed_pnts, pcl_low, K=patch_size, return_nn=True)
    patches_low = patches_low[0]
    patches_low = split_tensor_to_segments( patches_low, 5)
    patches_noisy = split_tensor_to_segments(patches_noisy ,5 )
    n = len(patches_low)
    patches_denoised = []
    for i in range(n) :
        patch_denoised, traj = model.resample(
        p_init=patches_noisy[i], 
        p_ctx=patches_low[i],
        step_size=step_size, 
        step_decay=step_decay, 
        num_steps=num_steps
    )   
        patches_denoised.append(patch_denoised)
    patches_denoised = torch.cat(patches_denoised , dim=0)
    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
    pcl_denoised = pcl_denoised[0]
    fps_idx = fps_idx[0]

    if get_traj:
        for i in range(len(traj)):
            traj[i] = traj[i].view(-1, d)[fps_idx, :]
        return pcl_denoised, traj
    else:
        return pcl_denoised


def patch_based_denoise_big(model, pcl_noisy, step_size=0.15, num_steps=50, patch_size=10000, seed_k=3, denoise_knn=4, step_decay=0.95, get_traj=False):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    print("here!")
    patches = patches[0]    # (N, K, 3)
    patches = split_tensor_to_segments( patches, 5)
    n = len(patches)
    patches_denoised = []
    for i in range(n) :
        patch_denoised, traj = model.resample(
        p_init=patches[i], 
        p_ctx=patches[i],
        step_size=step_size, 
        step_decay=step_decay, 
        num_steps=num_steps
    )
        patches_denoised.append(patch_denoised)
    patches_denoised = torch.cat(patches_denoised , dim=0)
    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
    pcl_denoised = pcl_denoised[0]
    fps_idx = fps_idx[0]

    if get_traj:
        for i in range(len(traj)):
            traj[i] = traj[i].view(-1, d)[fps_idx, :]
        return pcl_denoised, traj
    else:
        return pcl_denoised

def denoise_large_pointcloud(model, pcl, cluster_size, seed=0):
    device = pcl.device
    pcl = pcl.cpu().numpy()

    print('Running KMeans to construct clusters...')
    n_clusters = math.ceil(pcl.shape[0] / cluster_size)
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(pcl)

    pcl_parts = []
    for i in tqdm(range(n_clusters), desc='Denoise Cluster'):
        pts_idx = kmeans.labels_ == i

        pcl_part_noisy = torch.FloatTensor(pcl[pts_idx]).to(device)
        pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)
        pcl_part_denoised = patch_based_denoise(
            model,
            pcl_part_noisy,
            seed_k=3
        )
        pcl_part_denoised = pcl_part_denoised * scale + center
        pcl_parts.append(pcl_part_denoised)

    return torch.cat(pcl_parts, dim=0)
