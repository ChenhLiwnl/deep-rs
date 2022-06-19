from .knn import KNearestGCNNPointEncoder
from .mccnn import MCCNNEncoder


def get_encoder(cfg):
    if cfg.type == 'knn':
        return KNearestGCNNPointEncoder(
            dynamic_graph=cfg.dynamic_graph,
            conv_channels=cfg.conv_channels,
            num_convs=cfg.num_convs,
            conv_num_fc_layers=cfg.conv_num_fc_layers,
            conv_growth_rate=cfg.conv_growth_rate,
            conv_knn=cfg.conv_knn,
        )
    elif cfg.type == 'mccnn':
        return MCCNNEncoder(
            radius = cfg.radius , 
            num_points = cfg.num_points,
            kde_bandwidth = cfg.kde_bandwidth , 
            point_dim= cfg.point_dim,
            first_block_dim= cfg.first_block_dim,
            #hidden_dims_pointwise= cfg.hidden_dims_pointwise,
        )
    else:
        raise NotImplementedError('Encoder `%s` is not implemented.' % cfg.type)
