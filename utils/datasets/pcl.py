import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import point_cloud_utils as pcu
import random

class PointCloudDataset(Dataset):
    def __init__(self, root, dataset , split, resolution, from_saved = False  , noise_level = None , input_root = None , need_mesh = False , transform=None):
        super().__init__()
        if from_saved == False : 
            self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution) #during training, set from_saved = False
        else : 
            self.pcl_dir = os.path.join(input_root , dataset+"_"+resolution+"_"+str(noise_level)) #during evaluating, set from_saved=True
        print(self.pcl_dir)
        self.resolution = resolution
        self.transform = transform
        self.need_mesh = need_mesh
        if self.need_mesh == False:
            self.pointclouds = []
            self.pointcloud_names = []
            for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
                if fn[-3:] != 'xyz':
                    continue
                pcl_path = os.path.join(self.pcl_dir, fn)
                if not os.path.exists(pcl_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
                self.pointclouds.append(pcl)
                self.pointcloud_names.append(fn[:-4])
        if self.need_mesh == True:
            self.mesh_dir = os.path.join(root, dataset, 'meshes', 'test')
            self.meshes = {}
            self.meshes_names = []
            self.pointclouds = {}
            self.pointcloud_names = []
            for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
                if fn[-3:] != 'xyz':
                    continue
                pcl_path = os.path.join(self.pcl_dir, fn)
                if not os.path.exists(pcl_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
                name = fn[:-4]
                self.pointclouds[name] = pcl
                self.pointcloud_names.append(name)
            for fn in tqdm(os.listdir(self.mesh_dir), desc='Loading'):
                if fn[-3:] != 'off':
                    continue
                mesh_path = os.path.join(self.mesh_dir , fn)
                if not os.path.exists(mesh_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                verts, faces, = pcu.load_mesh_vf(mesh_path)                
                verts = torch.FloatTensor(verts)
                faces = torch.LongTensor(faces)
                name = fn[:-4]
                self.meshes[name] = {'verts': verts, 'faces': faces}
                self.meshes_names.append(name)
    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        if self.need_mesh == False : 
            data = {
            'pcl_clean': self.pointclouds[idx].clone(), 
            'name': self.pointcloud_names[idx]
            }
            if self.transform is not None:
                data = self.transform(data)
            return data
        if self.need_mesh == True : 
            name = self.pointcloud_names[idx]
            data = {
                'pcl_clean': self.pointclouds[name].clone(), 
                'name': name,
            }
            if self.transform is not None:
                data = self.transform(data)
            data['meshes'] = self.meshes[name]
            return data

class UpsampleDataset(Dataset):

    def __init__(self, root, dataset , split, resolution, rate , noise_min , noise_max , from_saved = False , input_root = None , need_mesh = False , transform=None):
        super().__init__()
        self.resolution = resolution
        self.rate = rate
        self.transform = transform
        self.need_mesh = need_mesh
        self.noise_min = noise_min
        self.noise_max = noise_max
        if from_saved == False : 
            self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        else : 
            self.pcl_dir = os.path.join(input_root , dataset+"_"+resolution+"_"+"0.03")
        self.gt_resolution = str(int( self.resolution[:self.resolution.index('_')]) * self.rate )
        #self.gt_resolution = "32768"
        self.gt_dir = os.path.join(root, dataset ,'pointclouds' , split , self.gt_resolution+"_poisson" )
        print(self.pcl_dir)
        print(self.gt_dir)
        if self.need_mesh == False:
            self.pointclouds = []
            self.gt_pointclouds = []
            self.pointcloud_names = []
            for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
                if fn[-3:] != 'xyz':
                    continue
                pcl_path = os.path.join(self.pcl_dir, fn)
                gt_path = os.path.join(self.gt_dir , fn)
                if not os.path.exists(pcl_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
                gt = torch.FloatTensor(np.loadtxt(gt_path, dtype=np.float32))
                self.pointclouds.append(pcl)
                self.gt_pointclouds.append(gt)
                self.pointcloud_names.append(fn[:-4])
        if self.need_mesh == True:
            self.mesh_dir = os.path.join(root, dataset, 'meshes', 'test')
            self.meshes = {}
            self.meshes_names = []
            self.pointclouds = []
            self.gt_pointclouds = []
            self.pointcloud_names = []
            for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
                if fn[-3:] != 'xyz':
                    continue
                pcl_path = os.path.join(self.pcl_dir, fn)
                gt_path = os.path.join(self.gt_dir , fn)
                if not os.path.exists(pcl_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
                gt = torch.FloatTensor(np.loadtxt(gt_path, dtype=np.float32))
                self.pointclouds.append(pcl)
                self.gt_pointclouds.append(gt)
                self.pointcloud_names.append(fn[:-4])
            for fn in tqdm(os.listdir(self.mesh_dir), desc='Loading'):
                if fn[-3:] != 'off':
                    continue
                mesh_path = os.path.join(self.mesh_dir , fn)
                if not os.path.exists(mesh_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                verts, faces, = pcu.load_mesh_vf(mesh_path)
                verts = torch.FloatTensor(verts)
                faces = torch.LongTensor(faces)
                name = fn[:-4]
                self.meshes[name] = {'verts': verts, 'faces': faces}
                self.meshes_names.append(name)
    def __len__(self):
        return len(self.pointclouds)
    def normalize(self,pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2    # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale
    def __getitem__(self, idx):
        if self.need_mesh == False : 
            pcl_noisy = []
            #gt , center , scale = self.normalize(self.gt_pointclouds[idx])
            gt = self.gt_pointclouds[idx].clone()
            #print(center,scale)
            original = self.pointclouds[idx]
            #original, _ , _ = self.normalize(self.pointclouds[idx],center , scale)
            for i in range(self.rate-1) :
                noise_std = random.uniform(self.noise_min, self.noise_max)
                data = original + torch.randn_like(original) * noise_std
                #data = original + (2*torch.rand_like(original)-1) * noise_std
                pcl_noisy.append(data)
            pcl_noisy.append(original)
            pcl_noisy = torch.cat(pcl_noisy,dim=0)
            #pcl_noisy , _ , _ = self.normalize(pcl_noisy , center , scale)
            data = {
            'gt': gt,
            'ups' : pcl_noisy,
            'original' : original,
            'name': self.pointcloud_names[idx]
            }
            if self.transform is not None:
                data = self.transform(data)
            return data
        if self.need_mesh == True : 
            pcl_noisy = []
            #gt , center , scale = self.normalize(self.gt_pointclouds[idx])
            gt = self.gt_pointclouds[idx].clone()
            #print(center,scale)
            original = self.pointclouds[idx]
            #original, _ , _ = self.normalize(self.pointclouds[idx],center , scale)
            for i in range(self.rate-1) :
                noise_std = random.uniform(self.noise_min, self.noise_max)
                data = original + torch.randn_like(original) * noise_std
                #data = original + (2*torch.rand_like(original)-1) * noise_std
                pcl_noisy.append(data)
            pcl_noisy.append(original)
            pcl_noisy = torch.cat(pcl_noisy,dim=0)
            #pcl_noisy , _ , _ = self.normalize(pcl_noisy , center , scale)
            name = self.pointcloud_names[idx]
            data = {
            'gt': gt,
            'ups' : pcl_noisy,
            'original' : original,
            'name': name
            }
            if self.transform is not None:
                data = self.transform(data)
            data['meshes'] = self.meshes[name]
            return data
 