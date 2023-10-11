import torch
import torch.nn.functional as F
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS,listNet,rankNet

def FocusedNet(y_pred, y_true, eps=DEFAULT_EPS, coe=0.5):
  lk_ = listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE)
  rk_ = rankNet(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False)

  return (1-coe)*lk_ + coe*rk_