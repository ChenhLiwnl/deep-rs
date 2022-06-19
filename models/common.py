import math
import torch
from torch import nn
import pytorch3d.loss
import pytorch3d.structures
import pytorch3d.ops
from pytorch3d.loss.point_mesh_distance import point_face_distance
from torch_cluster import fps
from torch.nn import Module, ModuleList, Identity, ReLU, Parameter, Sequential,  Conv2d, BatchNorm2d, Conv1d, BatchNorm1d

def batch_pairwise_dist( x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


class EdgeConv(Module):

    def __init__(self, in_channels, num_layers, layer_out_dim, knn=16, aggr='max', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.knn = knn
        assert num_layers > 2
        self.num_layers = num_layers
        self.layer_out_dim = layer_out_dim
        
        # Densely Connected Layers
        self.layer_first = FullyConnected(3*in_channels, layer_out_dim, bias=True, activation=activation)
        self.layer_last = FullyConnected(in_channels + (num_layers - 1) * layer_out_dim, layer_out_dim, bias=True, activation=None)
        self.layers = ModuleList()
        for i in range(1, num_layers-1):
            self.layers.append(FullyConnected(in_channels + i * layer_out_dim, layer_out_dim, bias=True, activation=activation))

        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_layers * self.layer_out_dim

    def get_edge_feature(self, x, knn_idx):
        """
        :param  x:          (B, N, d)
        :param  knn_idx:    (B, N, K)
        :return (B, N, K, 2*d)
        """
        knn_feat = knn_group(x, knn_idx)   # B * N * K * d
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)
        return edge_feat

    def forward(self, x, pos):
        """
        :param  x:  (B, N, d)
        :return (B, N, d+L*c)
        """
        knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)

        # First Layer
        edge_feat = self.get_edge_feature(x, knn_idx)
        y = torch.cat([
            self.layer_first(edge_feat),              # (B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, self.knn, 1) # (B, N, K, d)
        ], dim=-1)  # (B, N, K, d+c)

        # Intermediate Layers
        for layer in self.layers:
            y = torch.cat([
                layer(y),           # (B, N, K, c)
                y,                  # (B, N, K, c+d)
            ], dim=-1)  # (B, N, K, d+c+...)
        
        # Last Layer
        y = torch.cat([
            self.layer_last(y), # (B, N, K, c)
            y                   # (B, N, K, d+(L-1)*c)
        ], dim=-1)  # (B, N, K, d+L*c)

        # Pooling
        y = self.aggr(y, dim=-2)
        
        return y

def gen_grid(rate):
    '''
    in : int
    out : (rate , 2)
    '''

    sqrted = int(math.sqrt(rate))+1
    for i in range(1,sqrted+1).__reversed__():
        if (rate%i) == 0:
            num_x = i
            num_y = rate//i
            break
    grid_x = torch.linspace(-0.2, 0.2, num_x)
    grid_y = torch.linspace(-0.2, 0.2, num_y)

    x, y = torch.meshgrid(grid_x, grid_y)
    grid = torch.reshape(torch.stack([x,y],dim=-1) ,[-1,2])
    return grid

class GCN_feature_extractor(torch.nn.Module):

    def __init__(self ,conv_growth_rate , knn , block_num) :
        super().__init__()

        self.layer_out_dim = conv_growth_rate 
        self.knn = knn
        self.block_num = block_num

        dims = [conv_growth_rate , conv_growth_rate * 5, conv_growth_rate * 10]
        print(dims)
        self.layers = []
        self.edgeconvs = []
        for i in range(block_num):
            if i == 0 :
                self.layers +=  [nn.Conv1d(in_channels= 3 , out_channels = self.layer_out_dim, kernel_size = 1 )]
                self.edgeconvs += [EdgeConv(in_channels= self.layer_out_dim  , num_layers= 3 , layer_out_dim= self.layer_out_dim , knn = self.knn)]
            else : 
                self.layers +=  [nn.Conv1d(in_channels= dims[i] , out_channels = self.layer_out_dim * 2 , kernel_size = 1 )]
                self.edgeconvs += [EdgeConv(in_channels= self.layer_out_dim * 2 , num_layers= 3 , layer_out_dim= self.layer_out_dim , knn = self.knn)]
        self.layers = Sequential(*self.layers)
        self.edgeconvs = Sequential(*self.edgeconvs)
    def forward(self, points) :
        '''
        points : ( B , N , 3)
        out_feature : ( B , N , D)
        '''
        out_feature = points.permute(0,2,1).contiguous()
        for i in range(self.block_num):
            cur_feat = self.layers[i](out_feature).permute(0,2,1).contiguous() # (B, N ,D1)
            if i == 0 :
                out_feature = cur_feat.permute(0,2,1).contiguous()
            cur_feat = self.edgeconvs[i](cur_feat , points) # (B ,N ,D2)
            out_feature = torch.cat([out_feature.permute(0,2,1).contiguous(),cur_feat],dim=-1) 
            out_feature = out_feature.permute(0,2,1).contiguous()

        return out_feature.permute(0,2,1).contiguous()


class duplicate_up(torch.nn.Module):

    def __init__(self , rate) :
        super().__init__()
        
        self.rate = rate 
        dims = [360+2 , 256, 128]
        conv_layers = []

        for i in range(len(dims)-1):
            conv_layers += [
                Conv1d(dims[i], dims[i+1], kernel_size=1),
                BatchNorm1d(dims[i+1]),
            ]
            if i < len(dims)-2:
                conv_layers += [
                    ReLU(),
                ]
        self.layers = Sequential(*conv_layers)

    def forward(self, coarse_feature):

        B, N, d = coarse_feature.shape
        feat = coarse_feature.repeat(1, self.rate, 1) # ( B , N*rate , d)
        grid = gen_grid(self.rate).unsqueeze(0).to("cuda") # (1, rate , 2)
        grid = grid.repeat( B , 1 , N) # ( B ,  rate , 2*N)
        grid = grid.reshape(B, N*self.rate,  2) #(B , N*rate , 2)
        feat = torch.cat([feat,grid] ,dim = -1).permute(0,2,1).contiguous() # ( B , d+2 , N*R)

        feat = self.layers(feat)
        return feat


class regressor (torch.nn.Module):

    def __init__(self) :
        super().__init__()

        dims = [128,256,64,3]
        conv_layers = []

        for i in range(len(dims)-1):
            conv_layers += [
                Conv1d(dims[i], dims[i+1], kernel_size=1, stride=1),
            ]
            if i < len(dims)-2:
                conv_layers += [
                    ReLU(),
                ]
        self.layers = Sequential(*conv_layers)

    def forward(self , coarse_feature):
        B, d, N = coarse_feature.shape
        #coarse_feature = coarse_feature.permute(0,2,1).contiguous()
        coarse = self.layers(coarse_feature)

        coarse = coarse.permute(0,2,1).contiguous()
        return coarse

class coarsesGenerator(torch.nn.Module):

    def __init__(self, rate , block_num = 3 , knn = 24 , conv_growth_rate = 24 ):
        super().__init__()
        self.rate = rate
        self.block_num = block_num 
        self.conv_growth_rate = conv_growth_rate
        self.knn = knn

        self.GCN = GCN_feature_extractor( self.conv_growth_rate , self.knn, self.block_num)
        self.duplicate_up = duplicate_up(self.rate)
        self.regressor = regressor()

    def forward(self, points) :
        coarse_feature = self.GCN(points)
        coarse_feature = self.duplicate_up(coarse_feature)
        coarse = self.regressor(coarse_feature)
        return coarse

class FullyConnected(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))

class FeatureExtraction(Module):

    def __init__(self, in_channels=3, dynamic_graph=True, conv_channels=24, num_convs=3, conv_num_layers=3, conv_layer_out_dim=12, conv_knn=16, conv_aggr='max', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.dynamic_graph = dynamic_graph
        self.num_convs = num_convs

        # Edge Convolution Units
        self.transforms = ModuleList()
        self.convs = ModuleList()
        for i in range(num_convs):
            if i == 0:
                trans = FullyConnected(in_channels, conv_channels, bias=True, activation=None)
            else:
                trans = FullyConnected(in_channels, conv_channels, bias=True, activation=activation)
            conv = EdgeConv(conv_channels, num_layers=conv_num_layers, layer_out_dim=conv_layer_out_dim, knn=conv_knn, aggr=conv_aggr, activation=activation)
            self.transforms.append(trans)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def dynamic_graph_forward(self, x):
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, x)
        return x

    def static_graph_forward(self, pos):
        x = pos
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, pos)
        return x 

    def forward(self, x):
        if self.dynamic_graph:
            return self.dynamic_graph_forward(x)
        else:
            return self.static_graph_forward(x)

class FCLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))


class Aggregator(torch.nn.Module):

    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret


class ResnetBlockConv1d(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out ,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes

        self.size_in = size_in
        self.size_out = size_out
        # Submodules
        if norm_method == 'batch_norm':
            norm = nn.BatchNorm1d
        elif norm_method == 'sync_batch_norm':
            norm = nn.SyncBatchNorm
        else:
             raise Exception("Invalid norm method: %s" % norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_out, 1)
        self.actvn = nn.ReLU()
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1)

    def forward(self, x):
        dx = self.fc_0(x)
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx

        return out

class ResnetBlockConv2d(nn.Module):
    """ 2D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
    """

    def __init__(self, size_in, size_out,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes

        self.size_in = size_in
        self.size_out = size_out
        # Submodules
        if norm_method == 'batch_norm':
            norm = nn.BatchNorm1d
        elif norm_method == 'sync_batch_norm':
            norm = nn.SyncBatchNorm
        else:
             raise Exception("Invalid norm method: %s" % norm_method)

        self.fc_0 = nn.Conv2d(size_in, size_out, (1,1))
        self.actvn = nn.ReLU()
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, (1,1))

    def forward(self, x):
        dx = self.fc_0(x)
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx

        return out


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


def normalize_std(pc, std=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    center = pc.mean(dim=-2, keepdim=True)   # (B, 1, 3)
    pc = pc - center
    scale = pc.view(pc.size(0), -1).std(dim=-1).view(pc.size(0), 1, 1) / std
    pc = pc / scale
    return pc, center, scale


def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def denormalize_pcl(pc, center, scale):
    return pc * scale + center


def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean'):
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale) #ups注释掉这两句之后会有略微的上升，因为已经noramlize过了
    return pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)


def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    indices = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]
        sampled.append(pcls[i:i+1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    return sampled, indices


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

    # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls)


def pointwise_p2m_distance_normalized(pcl, verts, faces):
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


def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(x, y, K=k+offset)
    return knn_idx[:, :, offset:]


def knn_group(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)

