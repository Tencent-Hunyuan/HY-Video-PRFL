import torch
import torch.distributed as dist
import math


class ModulePlugin:
    def __init__(self, module, module_id, global_state=None):
        self.module = module
        self.module_id = module_id
        self.global_state = global_state
        self.enable = True
        self.implement_forward()

    @property
    def is_log_node(self):
        return self.global_state.get('dist_controller').rank == 0 and self.module_id[1] == 0

    @property
    def t(self):
        return self.global_state.get('timestep')
    
    @property
    def p(self):
        return self.t / 1000

    def implement_forward(self):
        module = self.module
        if not hasattr(module, "old_forward"):
            module.old_forward = module.forward
        self.new_forward = self.get_new_forward()
        def forward(*args, **kwargs):
            self.update_config() # update config
            return self.new_forward(*args, **kwargs) if self.enable else self.old_forward(*args, **kwargs)
        module.forward = forward

    def set_enable(self, enable=True):
        self.enable = enable
        
    def get_new_forward(self):
        raise NotImplementedError
    
    def update_config(self, config:dict=None):
        if config is None:
            config = self.global_state.get('plugin_configs', {}).get(self.module_id[0], {})
        for key, value in config.items():
            setattr(self, key, value)


class GroupNormPlugin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)

    def get_new_forward(self):
        module = self.module
    
        def new_forward(x):
            shape = x.shape
            N, C, G = shape[0], shape[1], module.num_groups
            assert C % G == 0

            x = x.reshape(N, G, -1)
            
            mean = x.mean(-1, keepdim=True).to(torch.float32)
            dist.all_reduce(mean)
           
            mean = mean / dist.get_world_size()
            
            var = ((x - mean.to(x.dtype)) ** 2).mean(-1, keepdim=True).to(torch.float32) 
            
            dist.all_reduce(var)
            var = var / dist.get_world_size()

            x = (x - mean.to(x.dtype)) / (var.to(x.dtype) + module.eps).sqrt()
            x = x.view(shape)

            new_shape = [1 for _ in shape]
            new_shape[1] = -1

            return x * module.weight.view(new_shape) + module.bias.view(new_shape)

        return new_forward
    
    
class Conv3DSafeNewPligin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)
        
        self.kernel_size = getattr(module, 'kernel_size', (1, 1, 1))
        
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)
      
        kernel_width = self.kernel_size[2]  
        d = kernel_width - 1  
        self.padding_left = d // 2  
        self.padding_right = d - self.padding_left  
        self.padding_flag = self.padding_left if d > 0 else 0  
        
        self.rank = dist.get_rank()
        self.adj_groups = self.global_state.get('dist_controller').adj_groups
        

    def pad_context(self, h):
        if self.padding_flag == 0:
            return h
            
      
        share_to_left = h[:, :, :, :self.padding_left].contiguous()
        share_to_right = h[:, :, :, -self.padding_right:].contiguous()
        
        if self.rank % 2:
            # 1. the rank is odd, pad the left first 
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
            # 2. then pad the right
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
        else:
            # 1. the rank is even, pad the right first
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
            # 2. then pad the left
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
        # torch.cuda.synchronize()
        
        h_with_context = torch.cat([left_context, h, right_context], dim=3)
        return h_with_context

    def get_new_forward(self):
        module = self.module
        def new_forward(hidden_states, cache_x=None, *args, **kwargs):
            if self.padding_flag == 0:
                # print(f"padding=0, return old_forward")
                return module.old_forward(hidden_states, cache_x, *args, **kwargs)

            
            hidden_states = self.pad_context(hidden_states)
            if cache_x is not None:
                cache_x = self.pad_context(cache_x)
         
            result = module.old_forward(hidden_states, cache_x, *args, **kwargs)
            result = result[:,:,:,self.padding_left:-self.padding_right if self.padding_right > 0 else None]
      
            return result
        
        return new_forward
    
class Conv2DSafeNewPligin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)
        
        self.kernel_size = getattr(module, 'kernel_size', (1, 1))
        self.stride = getattr(module, 'stride', (1, 1))
        
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
            
        kernel_height = self.kernel_size[0]  # 卷积核的高度维度
        d = kernel_height - 1  # 总padding量
        self.padding_left = d // 2  # 上侧padding
        self.padding_right = d - self.padding_left  # 下侧padding
        self.padding = self.padding_left if d > 0 else 0  
        self.rank = dist.get_rank()
        self.adj_groups = self.global_state.get('dist_controller').adj_groups
    
    def pad_context(self, h):
        if self.padding == 0:
            return h
            
        share_to_left = h[:, :, :self.padding_left].contiguous()
        share_to_right = h[:, :, -self.padding_right:].contiguous()
        if self.rank % 2:
            # 1. the rank is odd, pad the left first 
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
            # 2. then pad the right
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
        else:
            # 1. the rank is even, pad the right first
            if self.rank != dist.get_world_size() - 1:
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
            # 2. then pad the left
            if self.rank:
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
        # torch.cuda.synchronize()
        
        h_with_context = torch.cat([left_context, h, right_context], dim=2)
        return h_with_context

    def get_new_forward(self):
        module = self.module
        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            if self.padding == 0:
                return module.old_forward(hidden_states)               
            
            hidden_states = self.pad_context(hidden_states)           
            hidden_states = module.old_forward(hidden_states)[:,:,self.padding_left:-self.padding_right if self.padding_right > 0 else None]
            return hidden_states

        return new_forward

class Conv2DSafeNewPliginStride2(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)
        
        self.kernel_size = getattr(module, 'kernel_size', (1, 1))
        self.stride = getattr(module, 'stride', (1, 1))
        
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
            
        kernel_height = self.kernel_size[0]  
        d = kernel_height - 1  
        self.padding_left = d // 2  
        self.padding_right = d - self.padding_left 
        self.padding = self.padding_left if d > 0 else 0 
        self.rank = dist.get_rank()
        self.adj_groups = self.global_state.get('dist_controller').adj_groups
    
    def pad_context(self, h):
        if self.padding == 0:
            return h
            
        share_to_left = h[:, :, :self.padding_left].contiguous()

        if self.rank < dist.get_world_size() - 1:
            right_context = torch.zeros_like(share_to_left)
            
            dist.recv(right_context, src=self.rank+1)
        if self.rank >0:
            dist.send(share_to_left, dst=self.rank-1)
        # torch.cuda.synchronize()
        if self.rank < dist.get_world_size() - 1:
            h_with_context = torch.cat([h, right_context], dim=2)
        else:
            h_with_context = h
        return h_with_context

    def get_new_forward(self):
        module = self.module
        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            if self.padding == 0:
                return module.old_forward(hidden_states)               
            
            hidden_states = hidden_states[:, :, :-1, :]
            hidden_states = self.pad_context(hidden_states) 
            hidden_states = torch.nn.functional.pad(hidden_states,(0,0,0,1))          
            hidden_states = module.old_forward(hidden_states)#[:,:,self.padding_left:-self.padding_right if self.padding_right > 0 else None]
            return hidden_states

        return new_forward

class WanAttentionPlugin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        super().__init__(module, module_id, global_state)
    
    def get_new_forward(self):
        module = self.module
        rank = self.rank
        world_size = self.world_size
        
        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            gathered_tensors = [torch.zeros_like(hidden_states) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, hidden_states)
            
            combined_tensor = torch.cat(gathered_tensors, dim=3)
            
            forward_output = module.old_forward(combined_tensor)
            
            chunk_sizes = [t.size(3) for t in gathered_tensors]
            
            start_idx = sum(chunk_sizes[:rank])
            end_idx = start_idx + chunk_sizes[rank]
            
            local_output = forward_output[:, :, :, start_idx:end_idx].contiguous()
            
            return local_output
            
        return new_forward
        