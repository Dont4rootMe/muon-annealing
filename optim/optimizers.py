#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
import torch
from safetensors.torch import load_file, save_file

from .src.shampoo import Shampoo
from .src.muon import MuonWithAuxAdam
from .src.soap import SOAP

from lerobot.common.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from lerobot.common.datasets.utils import flatten_dict, unflatten_dict, write_json
from lerobot.common.utils.io_utils import deserialize_json_into_object


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "adam"

    @abc.abstractmethod
    def build(self, params) -> torch.optim.Optimizer:
        raise NotImplementedError


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.Adam(params, **kwargs)


@OptimizerConfig.register_subclass("adamw")
@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    def build(self, params) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.AdamW(params, **kwargs)


@OptimizerConfig.register_subclass("sgd")
@dataclass
class SGDConfig(OptimizerConfig):
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.SGD(params, **kwargs)


@OptimizerConfig.register_subclass("soap")
@dataclass
class SOAPConfig(OptimizerConfig):
    """
    SOAP optimizer configuration.
    
    SOAP (Shampoo Optimized for Adam Preconditioning) is a second-order optimizer that improves
    and stabilizes Shampoo using Adam. It maintains preconditioning matrices to improve convergence
    while being more stable than vanilla Shampoo.
    
    Reference: "SOAP: Improving and Stabilizing Shampoo using Adam" (https://arxiv.org/abs/2409.11321)
    """
    lr: float = 3e-3
    betas: tuple[float, float] = (0.95, 0.95)
    shampoo_beta: float = -1  # If >= 0, use this beta for preconditioner instead of betas[1]
    eps: float = 1e-8
    weight_decay: float = 0.01
    precondition_frequency: int = 10  # How often to update the preconditioner
    max_precond_dim: int = 10000  # Maximum dimension of the preconditioner
    merge_dims: bool = False  # Whether to merge dimensions of the preconditioner
    precondition_1d: bool = False  # Whether to precondition 1D gradients
    normalize_grads: bool = False  # Whether to normalize gradients per layer
    data_format: str = "channels_first"  # Data format for convolutional layers
    correct_bias: bool = True  # Whether to use bias correction in Adam
    grad_clip_norm: float = 10.0

    def build(self, params) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return SOAP(params, **kwargs)


@OptimizerConfig.register_subclass("shampoo")
@dataclass
class ShampooConfig(OptimizerConfig):
    """
    Shampoo optimizer configuration.
    
    Shampoo is a second-order optimizer that maintains and uses preconditioning matrices
    to improve convergence. It's particularly effective for training neural networks with
    better conditioning than first-order methods like Adam.
    
    Reference: "Shampoo: Preconditioned Stochastic Tensor Optimization" (Gupta et al., 2018)
    """
    lr: float = 2e-5
    momentum: float = 0.0
    weight_decay: float = 0.0
    epsilon: float = 1e-4  # Small value for numerical stability
    update_freq: int = 1   # Frequency of updating the preconditioning matrices
    grad_clip_norm: float = 10.0
    

    def build(self, params) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return Shampoo(params, **kwargs)


@OptimizerConfig.register_subclass("muon")
@dataclass
class MuonConfig(OptimizerConfig):
    """
    Muon optimizer configuration with automatic parameter group separation.
    
    Muon (MomentUm Orthogonalized by Newton-schulz) is designed for training large neural networks.
    It uses orthogonalization for 2D parameters (weights) and falls back to AdamW for other parameters.
    
    The optimizer automatically separates parameters into two groups:
    1. Parameters matching muon_include_patterns (and not muon_exclude_patterns) -> use Muon
    2. All other parameters -> use AdamW
    
    Reference: https://kellerjordan.github.io/posts/muon/
    """
    # AdamW defaults (used as base for both Muon and AdamW parameters)
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0
    
    # Muon-specific parameters
    muon_lr: float | None = None  # If None, uses lr
    muon_momentum: float = 0.95
    muon_weight_decay: float | None = None  # If None, uses weight_decay
    
    # Parameter selection patterns
    muon_include_patterns: tuple[str, ...] = ()  # Parameters matching these patterns are candidates for Muon
    muon_exclude_patterns: tuple[str, ...] = ()  # Parameters matching these patterns are excluded from Muon

    def build(self, named_params) -> torch.optim.Optimizer:
        """
        Build the Muon optimizer with automatic parameter group separation.
        
        Args:
            named_params: List of (name, parameter) tuples from model.named_parameters()
        """
        # Separate parameters based on patterns
        muon_params = []
        adamw_params = []
        
        for name, param in named_params:
            # Check if parameter should use Muon
            use_muon = False
            
            # First check if it matches any include pattern
            if self.muon_include_patterns:
                for pattern in self.muon_include_patterns:
                    if pattern in name:
                        use_muon = True
                        break
            
            # Then check if it should be excluded
            if use_muon and self.muon_exclude_patterns:
                for pattern in self.muon_exclude_patterns:
                    if pattern in name:
                        use_muon = False
                        break
            
            # Also require that Muon parameters are 2D (weights)
            if use_muon and param.ndim >= 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        # Create parameter groups
        param_groups = []
        
        # Muon group for 2D parameters matching patterns
        if muon_params:
            muon_group = {
                "params": muon_params,
                "use_muon": True,
                "lr": self.muon_lr if self.muon_lr is not None else self.lr,
                "momentum": self.muon_momentum,
                "weight_decay": self.muon_weight_decay if self.muon_weight_decay is not None else self.weight_decay,
            }
            param_groups.append(muon_group)
        
        # AdamW group for all other parameters
        if adamw_params:
            adamw_group = {
                "params": adamw_params,
                "use_muon": False,
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }
            param_groups.append(adamw_group)
        
        return MuonWithAuxAdam(param_groups)


def save_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> None:
    state = optimizer.state_dict()
    param_groups = state.pop("param_groups")
    flat_state = flatten_dict(state)
    flat_state = {k: (torch.tensor(v) if not isinstance(v, torch.Tensor) else v) for k, v in flat_state.items()}
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def load_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> torch.optim.Optimizer:
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)
    state = unflatten_dict(flat_state)
    loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}

    if "param_groups" in current_state_dict:
        param_groups = deserialize_json_into_object(
            save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
        )
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer
