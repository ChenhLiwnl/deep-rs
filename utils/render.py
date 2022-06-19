import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import point_cloud_utils as pcu
from matplotlib import cm

from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from models.common import *
from .mitsuba import *


def get_icosahedron(size):
    X = .525731112119133606
    Z = .850650808352039932
    verts = torch.FloatTensor([
        [-X, 0.0, Z], [X, 0.0, Z], [-X, 0.0, -Z], [X, 0.0, -Z],
        [0.0, Z, X], [0.0, Z, -X], [0.0, -Z, X], [0.0, -Z, -X],
        [Z, X, 0.0], [-Z, X, 0.0], [Z, -X, 0.0], [-Z, -X, 0.0]
    ]) * size
    faces = torch.LongTensor([
        [1, 4, 0], [4, 9, 0], [4, 5, 9], [8, 5, 4], [1, 8, 4],
        [1, 10, 8], [10, 3, 8], [8, 3, 5], [3, 2, 5], [3, 7, 2],
        [3, 10, 7], [10, 6, 7], [6, 11, 7], [6, 0, 11], [6, 1, 0],
        [10, 1, 6], [11, 0, 9], [2, 11, 9], [5, 2, 9], [11, 2, 7],
    ])
    return verts, faces


class NoisyPointCloudXMLMaker(object):

    def __init__(self, point_size=0.01, max_noise=0.03):
        super().__init__()
        self.point_size = point_size
        self.max_noise = max_noise

    def get_color(self, noise, showing_noisy=False):
        """
        Args:
            noise: (N, 1)
        """
        max_noise = self.max_noise

        N = noise.shape[0]
        noise_level = np.clip(noise / max_noise, 0, 1)  # (N, 1)
        
        base_color = np.repeat(np.array([[0, 0, 1]], dtype=np.float), N, axis=0)    # Blue, (N, 3)
        noise_color = np.repeat(np.array([[1, 1, 0]], dtype=np.float), N, axis=0)   # Yellow
        if showing_noisy:
            noise_level = (np.power(10, noise_level)-1) / (np.power(10, 1)-1)
        else:
            noise_level = (np.power(25, noise_level)-1) / (np.power(25, 1)-1)

        inv_mix = (1-noise_level)*(1-base_color) + noise_level*(1-noise_color)
        return 1 - inv_mix     # (N, 3)

    def render(self, pcl, verts, faces, rotation=None, lookat=(2.0, 30, -45), distance='p2m'):
        assert distance in ('p2m', 'nn')

        device = pcl.device
        # Normalize mesh
        verts, center, scale = normalize_sphere(verts.unsqueeze(0))
        verts = verts[0]
        # Normalize pcl
        pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
        pcl = pcl[0]
        if distance == 'p2m':
            # Compute point-to-surface distance
            p2m = pointwise_p2m_distance_normalized(pcl.to(device), verts.to(device), faces.to(device)).sqrt().unsqueeze(-1)
        elif distance == 'nn':
            p2m, _, _ = pytorch3d.ops.knn_points(pcl.unsqueeze(0), verts.unsqueeze(0), K=1)
            p2m = p2m[0,:,:].mean(dim=-1,keepdim=True)
            p2m = p2m 
        print(p2m.max() , p2m.mean() , p2m.min())



        # Rotate point cloud
        if rotation is not None:
            pcl = torch.matmul(pcl, rotation.t())
        # pcl[:, 1] = -pcl[:, 1]

        xml = make_xml(pcl.cpu().numpy(), self.get_color(p2m.cpu().numpy()), radius=self.point_size , max_points=None)
        return xml


class SimpleMeshRenderer(object):

    def __init__(self, image_size=1024):
        super().__init__()
        self.image_size = image_size

    def render(self, verts, faces, lookat=(2.0, 30, -45), color=(1, 1, 1)):
        device = verts.device
        # Normalize mesh
        verts, _, _ = normalize_sphere(verts.unsqueeze(0))
        verts = verts[0]

        textures = TexturesVertex([
            torch.FloatTensor(color).to(device).view(1, 3).repeat(verts.size(0), 1)
        ])
        meshes = Meshes([verts], [faces], textures)

        # Render
        R, T = look_at_view_transform(*lookat)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.,
            faces_per_pixel=1,
        )

        lights = PointLights(
            device=device,
            ambient_color=torch.ones(1,3) * 0.5,
            diffuse_color=torch.ones(1,3) * 0.4,
            specular_color=torch.ones(1,3) * 0.1,
            location=cameras.get_camera_center()
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )

        images = renderer(meshes)




        return images
