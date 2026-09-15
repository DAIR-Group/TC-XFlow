"""
env_net/ — package chứa Env_net encoder, tách ra từ file gốc
env_net_transformer_gphsplit.py (466 dòng, 3 trách nhiệm khác nhau)
thành 3 module theo đúng ranh giới trách nhiệm:

    feature_encoding.py  — one-hot encoding thuần Python/numpy, không
                            phụ thuộc torch.nn (bearing/distance/velocity/
                            intensity encoding, build_env_features_one_step)
    tensor_utils.py       — chuyển feature dict/batch sang torch.Tensor
                            (feat_to_tensor, build_env_vector)
    env_net_model.py       — class Env_net (nn.Module thật, Transformer
                            encoder cho environmental stream)

File này re-export toàn bộ public API để `from Model.Encoder.env_net
import X` hoạt động giống hệt import cũ từ file gốc gộp chung.
"""

from Model.Encoder.env_net.feature_encoding import (
    ENV_FEATURE_DIMS,
    ENV_DIM_TOTAL,
    ENV_1D_DIM,
    ENV_3D_DIM,
    SCS_BBOX,
    SCS_CENTER,
    SCS_DIAGONAL_KM,
    BOUNDARY_THRESHOLDS,
    DELTA_VEL_BINS,
    bearing_to_scs_center_onehot,
    dist_to_scs_boundary_onehot,
    delta_velocity_onehot,
    intensity_class_onehot,
    build_env_features_one_step,
)
from Model.Encoder.env_net.tensor_utils import feat_to_tensor, build_env_vector
from Model.Encoder.env_net.env_net_model import Env_net

__all__ = [
    "ENV_FEATURE_DIMS",
    "ENV_DIM_TOTAL",
    "ENV_1D_DIM",
    "ENV_3D_DIM",
    "SCS_BBOX",
    "SCS_CENTER",
    "SCS_DIAGONAL_KM",
    "BOUNDARY_THRESHOLDS",
    "DELTA_VEL_BINS",
    "bearing_to_scs_center_onehot",
    "dist_to_scs_boundary_onehot",
    "delta_velocity_onehot",
    "intensity_class_onehot",
    "build_env_features_one_step",
    "feat_to_tensor",
    "build_env_vector",
    "Env_net",
]
