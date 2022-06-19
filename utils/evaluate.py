import os
import torch
import pytorch3d
import pytorch3d.loss
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import point_cloud_utils as pcu
from tqdm.auto import tqdm

from .misc import BlackHole
import glob
import h5py
def load_h5():
    DATA_DIR = "./"
    all_data = []
    all_label = []
    h5_name = os.path.join(DATA_DIR, 'PUGAN_poisson_256_poisson_1024.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_xyz(xyz_dir):
    all_pcls = {}
    for fn in tqdm(os.listdir(xyz_dir), desc='Loading'):
        if fn[-3:] != 'xyz':
            continue
        name = fn[:-4]
        path = os.path.join(xyz_dir, fn)
        all_pcls[name] = torch.FloatTensor(np.loadtxt(path, dtype=np.float32))
    return all_pcls

def load_off(off_dir):
    all_meshes = {}
    for fn in tqdm(os.listdir(off_dir), desc='Loading'):
        if fn[-3:] != 'off':
            continue
        name = fn[:-4]
        path = os.path.join(off_dir, fn)
        verts, faces, _ = pcu.read_off(path)
        verts = torch.FloatTensor(verts)
        faces = torch.LongTensor(faces)
        all_meshes[name] = {'verts': verts, 'faces': faces}
    return all_meshes


class Evaluator(object):

    def __init__(self, output_pcl_dir, dataset_root, dataset, summary_dir, experiment_name, device='cuda', res_gts='8192_poisson', logger=BlackHole()):
        super().__init__()
        self.output_pcl_dir = output_pcl_dir
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.summary_dir = summary_dir
        self.experiment_name = experiment_name
        self.gts_pcl_dir = os.path.join(dataset_root, dataset, 'pointclouds', 'test', res_gts)
        self.gts_mesh_dir = os.path.join(dataset_root, dataset, 'meshes', 'test')
        self.res_gts = res_gts
        self.device = device
        self.logger = logger
        self.load_data()

    def load_data(self):
        self.pcls_up = load_xyz(self.output_pcl_dir)
        self.pcls_high = load_xyz(self.gts_pcl_dir)
        self.meshes = load_off(self.gts_mesh_dir)
        self.pcls_name = list(self.pcls_up.keys())

    def run(self):
        pcls_up, pcls_high, pcls_name = self.pcls_up, self.pcls_high, self.pcls_name
        results = {}
        for name in tqdm(pcls_name, desc='Evaluate'):
            pcl_up = pcls_up[name][:,:3].unsqueeze(0).to(self.device)
            if name not in pcls_high:
                self.logger.warning('Shape `%s` not found, ignored.' % name)
                continue
            pcl_high = pcls_high[name].unsqueeze(0).to(self.device)
            verts = self.meshes[name]['verts'].to(self.device)
            faces = self.meshes[name]['faces'].to(self.device)

            cd = pytorch3d.loss.chamfer_distance(pcl_up, pcl_high)[0].item()
            cd_sph = chamfer_distance_unit_sphere(pcl_up, pcl_high)[0].item()
            hd_sph = hausdorff_distance_unit_sphere(pcl_up, pcl_high)[0].item()

            # p2f = point_to_mesh_distance_single_unit_sphere(
            #     pcl=pcl_up[0],
            #     verts=verts,
            #     faces=faces
            # ).sqrt().mean().item()
            if 'blensor' in self.experiment_name:
                rotmat = torch.FloatTensor(Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()).to(pcl_up[0])
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_up[0].matmul(rotmat.t()),
                    verts=verts,
                    faces=faces
                ).item()
            else:
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_up[0],
                    verts=verts,
                    faces=faces
                ).item()

            results[name] = {
                'cd': cd,
                'cd_sph': cd_sph,
                'p2f': p2f,
                'hd_sph': hd_sph,
            }

        results = pd.DataFrame(results).transpose()
        res_mean = results.mean(axis=0)
        self.logger.info("\n" + repr(results))
        self.logger.info("\nMean\n" + '\n'.join([
            '%s\t%.12f' % (k, v) for k, v in res_mean.items()
        ]))

        update_summary(
            os.path.join(self.summary_dir, 'Summary_%s.csv' % self.dataset),
            model=self.experiment_name,
            metrics={
                # 'cd(mean)': res_mean['cd'],
                'cd_sph(mean)': res_mean['cd_sph'],
                'p2f(mean)': res_mean['p2f'],
                'hd_sph(mean)': res_mean['hd_sph'],
            }
        )


def update_summary(path, model, metrics):
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, sep="\s*,\s*", engine='python')
    else:
        df = pd.DataFrame()
    for metric, value in metrics.items():
        setting = metric
        if setting not in df.columns:
            df[setting] = np.nan
        df.loc[model, setting] = value
    df.to_csv(path, float_format='%.12f')
    return df
def hausdorff_distance_unit_sphere(gen, ref):
    """
    Args:
        gen:    (B, N, 3)
        ref:    (B, N, 3)
    Returns:
        (B, )
    """
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)

    dists_ab, _, _ = pytorch3d.ops.knn_points(ref, gen, K=1)
    dists_ab = dists_ab[:,:,0].max(dim=1, keepdim=True)[0]  # (B, 1)
    # print(dists_ab)

    dists_ba, _, _ = pytorch3d.ops.knn_points(gen, ref, K=1)
    dists_ba = dists_ba[:,:,0].max(dim=1, keepdim=True)[0]  # (B, 1)
    # print(dists_ba)
    
    dists_hausdorff = torch.max(torch.cat([dists_ab, dists_ba], dim=1), dim=1)[0]

    return dists_hausdorff
def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale

def normalize_pcl(pc, center, scale):
    return (pc - center) / scale
'''
def pointwise_p2m_distance_normalized(pcl, verts, faces):
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    # Normalize mesh
    #print(' before :verts %.6f pcl %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]
    #print('after :verts %.6f pcl %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))
    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()
    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face
'''
def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Args:
        pcl:    (N, 3).
        verts:  (M, 3).
        faces:  LongTensor, (T, 3).
    Returns:
        Squared pointwise distances, (N, ).
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'

    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls)