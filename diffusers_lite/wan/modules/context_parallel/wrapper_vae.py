from .tools import GlobalState, DistController

from .plugins import torch, ModulePlugin, GroupNormPlugin, Conv3DSafeNewPligin, Conv2DSafeNewPligin, WanAttentionPlugin, Conv2DSafeNewPliginStride2
#from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanAttentionBlock
from ...modules.vae import CausalConv3d, AttentionBlock


class DistWrapper(object):
    def __init__(self, pipe, dist_controller: DistController, config) -> None:
        super().__init__()
        self.pipe = pipe
        self.dist_controller = dist_controller
        self.config = config
        self.global_state = GlobalState({
            "dist_controller": dist_controller
        })
        self.plugin_mount()

        plugin_configs={
            "attn":{
                "padding": 24,
                "top_k": 24,
                "top_k_chunk_size": 24,
                "attn_scale": 1.,
                "token_num_scale": True,
                "dynamic_scale": True,
            },
            "conv_3d": {
                "padding": 1,
            },
            "conv_layer": {},
        }
        self.global_state.set("plugin_configs", plugin_configs)

        # torch.compile
        #self.pipe.model.encoder = torch.compile(self.pipe.model.encoder)
        #self.pipe.model.decoder = torch.compile(self.pipe.model.decoder)


    def plugin_mount(self):
        self.plugins = {}
        self.group_norm_plugin_mount()
        self.conv_3d_plugin_mount()
        self.conv_2d_plugin_stride2_mount()  ##only for wan vae encoder
        self.conv_2d_plugin_mount()
        self.wanattention_plugin_mount()

    def wanattention_plugin_mount(self):
        self.plugins['wanattention'] = {}
        wanattention_s = []
        for module in self.pipe.model.encoder.named_modules():
            #print("encoder named_modules: ", module[1].__class__.__name__)
            #if self.dist_controller.is_master and module[1].__class__.__name__ == 'AttentionBlock':
            #    print("Encoder attn: ", module[0])
            if ('middle.' in module[0] and module[1].__class__.__name__ == 'AttentionBlock'):
                wanattention_s.append(module[1])
        for module in self.pipe.model.decoder.named_modules():
            #print("decoder named_modules: ", module[1].__class__.__name__)
            #if self.dist_controller.is_master and module[1].__class__.__name__ == 'AttentionBlock':
            #    print("Decoder attn: ", module[0])
            if ('middle.' in module[0] and module[1].__class__.__name__ == 'AttentionBlock'):
                wanattention_s.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(wanattention_s)} wanattention_s')
        for i, wanattention in enumerate(wanattention_s):
            plugin_id = 'wanattention', i
            self.plugins['wanattention'][plugin_id] = WanAttentionPlugin(wanattention, plugin_id, self.global_state)

    def group_norm_plugin_mount(self):
        self.plugins['group_norm'] = {}
        group_norms = []
        for module in self.pipe.model.decoder.named_modules():
            if ('norm_layer' in module[0]) and module[1].__class__.__name__ == 'GroupNorm':
                group_norms.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(group_norms)} group norms')
        for i, group_norm in enumerate(group_norms):
            plugin_id = 'group_norm', i
            self.plugins['group_norm'][plugin_id] = GroupNormPlugin(group_norm, plugin_id, self.global_state)

    def conv_3d_plugin_mount(self):
        self.plugins['conv_3d'] = {}
        conv3d_s = []
        for module in self.pipe.model.encoder.named_modules():
            #if isinstance(module[1], CausalConv3d):
            #    print("Encoder conv3d: ", module[0], module[1].kernel_size[1])
            if (isinstance(module[1], CausalConv3d) and module[1].kernel_size[1] > 1):                  
                # print(f"Found conv3d: {module[1]}")
                conv3d_s.append(module[1])
        for module in self.pipe.model.decoder.named_modules():
            #if isinstance(module[1], CausalConv3d):
            #    print("Decoder conv3d: ", module[0], module[1].kernel_size[1])
            if (isinstance(module[1], CausalConv3d) and module[1].kernel_size[1] > 1):                  
                # print(f"Found conv3d: {module[1]}")
                conv3d_s.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(conv3d_s)} conv3d_s')
        for i, conv in enumerate(conv3d_s):
            plugin_id = 'conv_3d', i
            self.plugins['conv_3d'][plugin_id] = Conv3DSafeNewPligin(conv, plugin_id, self.global_state)

    def conv_2d_plugin_stride2_mount(self):
        self.plugins['conv_2d_stride2'] = {}
        conv2d_stride2_s = []
        for module in self.pipe.model.encoder.named_modules():
            if ('.resample' in module[0] and module[1].__class__.__name__ == 'Conv2d'):                
                conv2d_stride2_s.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(conv2d_stride2_s)} conv2d_stride2_s')
        for i, conv in enumerate(conv2d_stride2_s):
            plugin_id = 'conv_2d_stride2', i
            self.plugins['conv_2d_stride2'][plugin_id] = Conv2DSafeNewPliginStride2(conv, plugin_id, self.global_state)
    
    def conv_2d_plugin_mount(self):
        self.plugins['conv_2d'] = {}
        conv2d_s = []
        for module in self.pipe.model.decoder.named_modules():
            if ('.resample' in module[0] and module[1].__class__.__name__ == 'Conv2d'):                
                conv2d_s.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(conv2d_s)} conv2d_s')
        for i, conv in enumerate(conv2d_s):
            plugin_id = 'conv_2d', i
            self.plugins['conv_2d'][plugin_id] = Conv2DSafeNewPligin(conv, plugin_id, self.global_state)

    def inference(
        self,
        local_pose_image,
        local_latents,
        #prompts="A beagle wearning diving goggles  swimming in the ocean while the camera is moving, coral reefs in the background",
        config={},
        pipe_configs={
            "steps": 50,
            "guidance_scale": 12,
            "fps": 60,
            "num_frames": 24 * 1,
            "height": 320,
            "width": 512,
            "export_fps": 12,
            "base_path": "./work/output",
            "file_name": None
        },
        plugin_configs={
            "attn":{
                "padding": 24,
                "top_k": 24,
                "top_k_chunk_size": 24,
                "attn_scale": 1.,
                "token_num_scale": True,
                "dynamic_scale": True,
            },
            "conv_3d": {
                "padding": 1,
            }, 
            "conv_layer": {},
        },
        additional_info={},
    ):
        self.plugin_mount()
        # print("self.config seed: ", self.config["seed"])

        self.global_state.set("plugin_configs", plugin_configs)
        self.pipe = self.pipe.to(device='cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            local_pose_image = local_pose_image.to(device='cuda', dtype=torch.bfloat16)
            local_latents = local_latents.to(device='cuda', dtype=torch.bfloat16)
            
            tmp_latents = self.pipe.encode(local_pose_image).latent_dist.mode()
        
            latents = self.pipe.decode(local_latents, return_dict=False)[0]
            return latents
        
