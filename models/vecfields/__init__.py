from .knn import KNearestVectorField
from .radius import RadiusVectorField


def get_vecfield(cfg, ctx_point_feature_dim):
    if cfg.type == 'knn':
        return KNearestVectorField(
            knn=cfg.knn,
            radius = cfg.radius, 
            ctx_point_feature_dim=ctx_point_feature_dim,
        )
    elif cfg.type == 'radius' : 
        return RadiusVectorField(
            radius = cfg.radius , 
            ctx_point_feature_dim = ctx_point_feature_dim , 
            num_points = cfg.num_points ,
            style = cfg.style
        )
    else:
        raise NotImplementedError('Vecfield `%s` is not implemented.' % cfg.type)
