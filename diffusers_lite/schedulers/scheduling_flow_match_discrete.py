# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        sigma_max=1.0,
        reverse: bool = True,
        solver: str = "euler",
    ):
        # sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        sigmas = torch.linspace(sigma_max,0,num_train_timesteps+1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

        self.supported_solver = ["euler"]
        if solver not in self.supported_solver:
            raise ValueError(
                f"Solver {solver} not supported. Supported solvers: {self.supported_solver}"
            )

        self.sigma_max = sigma_max

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        dtype: torch.Tensor = torch.float32,
    ):
        self.num_inference_steps = num_inference_steps
        
        sigmas = torch.linspace(self.sigma_max, 0, num_inference_steps + 1)
        sigmas = (self.config.shift * sigmas) / (1 + (self.config.shift - 1) * sigmas)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(
            dtype=dtype, device=device
        )

        # Reset step index
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma_ = self.sigmas[self.step_index + 1]
        sigma = self.sigmas[self.step_index]
        dt = sigma_ - sigma

        if self.config.solver == "euler":
            prev_sample = sample + model_output.to(torch.float32) * dt
        else:
            raise ValueError(
                f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}"
            )

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps

    def get_train_timestep_and_sigma(
        self,
        weighting_scheme: str = "logit_normal",   # logit_norm, uniform
        batch_size: int = 1,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        generator: Optional[torch.Generator] = None,
        n_dim: int = 4,
    ):
        if weighting_scheme == "logit_normal":
            # NOTE: sigma from 1 to 0.
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator)
            u = torch.nn.functional.sigmoid(u)
        else:
            u = torch.rand(size=(batch_size,), generator=generator)

        indices = (u * self.config.num_train_timesteps).long()
        timestep = self.timesteps[indices].to(device=device)
        sigma = self.sigmas[indices].to(device=device, dtype=torch.float32)

        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return timestep, sigma
    
    def get_train_timestep(
        self,
        weighting_scheme: str = "logit_normal",   # logit_norm, uniform
        batch_size: int = 1,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        generator: Optional[torch.Generator] = None,
    ):
        if weighting_scheme == "logit_normal":
            # NOTE: sigma from 1 to 0.
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator)
            u = torch.nn.functional.sigmoid(u)
        else:
            u = torch.rand(size=(batch_size,), generator=generator)

        indices = (u * self.config.num_train_timesteps).long()
        timestep = self.timesteps[indices].to(device=device)
        return timestep

    def get_train_sigma(
        self,
        timestep: Union[float, torch.FloatTensor],
        n_dim: int = 4,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], dtype=dtype)

        sigmas = self.sigmas.to(device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timestep = timestep.to(device)

        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timestep]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        sigma: Union[float, torch.FloatTensor],
    ) -> torch.FloatTensor:
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def get_train_target(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
    ):
        target = noise - original_samples
        return target
        
    def get_train_loss_weighting(
        self, 
        sigma: torch.FloatTensor,
    ):
        weighting = torch.ones_like(sigma)
        return weighting

    def get_x0(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        sigma_t: torch.FloatTensor,
    ):
        sigma_0 = torch.zeros_like(sigma_t)
        dt = sigma_0 - sigma_t
        prev_sample = sample + model_output.to(torch.float32) * dt
        return prev_sample