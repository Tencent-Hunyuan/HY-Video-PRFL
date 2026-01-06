import argparse
import random
import sys

from .wan.configs import WAN_CONFIGS
from .wan.utils.utils import str2bool


def args_init():
    parser = argparse.ArgumentParser(description="diffusers lite script")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--task", type=str, default="i2v", choices=["i2v", "t2v", "flf2v"])

    ### Inference ###
    # model
    parser.add_argument("--base-dir", type=str, default="")
    parser.add_argument("--transformer_path", nargs="?", const="", default="")

    # lora
    parser.add_argument("--lora-path", nargs="?", const="", default="")
    parser.add_argument("--lora-alpha", type=float, default=1.0)

    # dataset
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--resolution", nargs="+", type=int, default=[512])
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--batch-size", type=int, default=1)

    # inference
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--step", type=int, default=50)

    # save
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--save-dir", type=str, default="")

    ### Preprocess dataset ###
    parser.add_argument("--json-paths", nargs="+", type=str, default=[""])
    parser.add_argument("--video-dir", type=str, default="")
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--extract-fps", type=int, default=24)
    # model
    parser.add_argument("--model-type", type=str, default="wanx", choices=["wanx", "ltx"])
    parser.add_argument("--vae-path", type=str, default="")
    parser.add_argument("--image-processor-path", type=str, default="")
    parser.add_argument("--image-encoder-path", type=str, default="")
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--text-encoder-path", type=str, default="")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--vlm-type", type=str, default="qwenvl2", choices=["qwenvl2", "qwenvl2.5", "nocap"])
    parser.add_argument("--vlm-path", nargs="?", const="", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # prompt
    parser.add_argument("--caption-template", type=str, default="")
    parser.add_argument("--instruct-sentence", type=str, default="")
    parser.add_argument("--negative-prompt", nargs="?", const="", default="")
    parser.add_argument("--null-caption-length", type=int, default=0)
    # save
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--save-path", type=str, default="")

    args = parser.parse_args()
    return args


def args_wan_init():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        # choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        help="The folder to save the generated image or video to.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=6.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=None,
        help="The threshold for caching diffusion model steps.")

    # NOTE: add by diffusers-lite to fill in blank args
    parser.add_argument("--ddp_mode", type=bool, default=False)
    # dataset
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--resolution", nargs="+", type=int, default=[512])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--negative_prompt", nargs="?", const="", default="")
    # transformer
    parser.add_argument("--transformer_path", nargs="?", const="", default="")
    # lora
    parser.add_argument("--lora_path", nargs="?", const="", default="")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--distill_lora_path", nargs="?", const="", default="")
    parser.add_argument("--distill_lora_alpha", type=float, default=1.0)

    args = parser.parse_args()

    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task:#  and args.size in ["832*480", "480*832"]
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)

    return args
