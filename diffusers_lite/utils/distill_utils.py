import numpy as np
import torch
import torch.nn as nn
from .diffusion_utils import list2batch


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))

def get_phase_endpoint(index, num_teacher_timesteps=32, multiphase=8):
    interval = num_teacher_timesteps // multiphase
    max_endpoint = num_teacher_timesteps - interval
    
    if index >= max_endpoint:
        return max_endpoint

    else:
        quotient = index // interval
        return quotient * interval

class EulerSolver:
    def __init__(self, sigmas, timesteps=1000, euler_timesteps=50):
        # sigmas: 0.0 -> 1.0, length = 1001
        self.num_timesteps = timesteps

        step_ratio = timesteps / euler_timesteps
        euler_timesteps = np.round(np.arange(timesteps, 0, -step_ratio)).astype(np.int64) - 1   # 999,...,0
        self.euler_timesteps = euler_timesteps[::-1].copy() + 1 # 1,...,1000

        self.sigmas = sigmas[self.euler_timesteps]  # 0.001,...,1.0
        self.sigmas_prev = np.asarray(
            [sigmas[0]] + sigmas[self.euler_timesteps[:-1]].tolist()    # 0.000,...,0.999
        )
        self.sigmas_all = sigmas.copy()

        self.euler_timesteps = torch.from_numpy(self.euler_timesteps).long()
        self.sigmas = torch.from_numpy(self.sigmas)
        self.sigmas_prev = torch.from_numpy(self.sigmas_prev)
        self.sigmas_all = torch.from_numpy(self.sigmas_all)
        

    def to(self, device):
        self.euler_timesteps = self.euler_timesteps.to(device)
        self.sigmas = self.sigmas.to(device)
        self.sigmas_prev = self.sigmas_prev.to(device)
        self.sigmas_all = self.sigmas_all.to(device)
        return self

    def euler_step(self, sample, model_pred, timestep_index):
        sigma = extract_into_tensor(self.sigmas, timestep_index, model_pred.shape)
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index, model_pred.shape)
        x_prev = sample + (sigma_prev - sigma) * model_pred
        return x_prev
    
    def euler_step_to_target(self, sample, model_pred, timestep_index, target_timestep_index):
        sigma = extract_into_tensor(self.sigmas, timestep_index, model_pred.shape)
        sigma_target = extract_into_tensor(self.sigmas_prev, target_timestep_index, model_pred.shape)

        x_target = sample + (sigma_target - sigma) * model_pred
        return x_target


class DiscriminatorHead(nn.Module):
    def __init__(self, in_channels=1280, reduced_channels=512):
        super(DiscriminatorHead, self).__init__()
        
        # Reduce channels using 1x1 convolution
        self.reduce_ch_conv = nn.Conv3d(in_channels, reduced_channels, kernel_size=(1, 1, 1))
        
        # Main convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(reduced_channels, reduced_channels * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(reduced_channels * 2, reduced_channels * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(reduced_channels * 4, reduced_channels * 8, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
            nn.LeakyReLU(0.2)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(reduced_channels * 8, 1)

    def forward(self, feature):
        # Reduce channels
        reduced_feature = self.reduce_ch_conv(feature)
        
        # Apply main convolutional layers
        x = self.conv_layers(reduced_feature)
        
        # Global pooling
        x = self.global_pool(x)

        # Fully connected layer
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        
        return out
    

class Discriminator(nn.Module):

    def __init__(
        self,
        num_h_per_head=1,
        selected_layers=[20,30,40],
        adapter_channel_dims=[1280],
    ):
        super().__init__()
        if isinstance(adapter_channel_dims, int):
            adapter_channel_dims = [adapter_channel_dims]

        adapter_channel_dims = adapter_channel_dims * len(selected_layers)
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)
        self.heads = nn.ModuleList([
            nn.ModuleList([DiscriminatorHead(adapter_channel) for _ in range(self.num_h_per_head)])
            for adapter_channel in adapter_channel_dims
        ])

    def forward(self, features):
        outputs = []
        assert len(features) == len(self.heads)
        for i in range(0, len(features)):
            for h in self.heads[i]:
                if isinstance(features[i], list):
                    input_features = list2batch(features[i])
                else:
                    input_features = features[i]
                out = h(input_features)
                outputs.append(out)
        return outputs