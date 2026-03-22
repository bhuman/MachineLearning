import os

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch

import numpy as np
from .t1_stand_up import T1_Stand_Up

from utils.utils import apply_randomization


class K1_Stand_Up(T1_Stand_Up):

    def __init__(self, cfg):
        super().__init__(cfg)
