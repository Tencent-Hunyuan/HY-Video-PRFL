import os
import math
import random
from collections import Counter
from typing import List, Optional

import imageio
import torch
import torchvision
import numpy as np
from einops import rearrange
from torch.utils.data import Sampler
from typing import Union, Optional, Iterator, List, Callable
import warnings
import logging

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler



def split_list(input_list, rank=0, num_process=8):

    n = len(input_list)
    base = n // num_process
    remainder = n % num_process

    if rank < remainder:
        start = rank * (base + 1)
        end = start + (base + 1)
    else:
        start = remainder * (base + 1) + (rank - remainder) * base
        end = start + base

    local_input_list = input_list[start:end]

    return local_input_list


def align_floor_to(value, alignment):
    return int(math.floor(value / alignment) * alignment)


def align_ceil_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


def crop_tensor(
    latents,
    image_latents=None,
    crop_width_ratio=1.0,
    crop_height_ratio=1.0,
    crop_type="center",
    crop_time_ratio=1.0,
):
    b, c, t, h, w = latents.shape
    crop_h, crop_w = int(h * crop_height_ratio), int(w * crop_width_ratio)
    crop_t = int(t * crop_time_ratio)

    if crop_type == "center":
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
    elif crop_type == "random":
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

    crop_h = align_floor_to(crop_h, alignment=2)
    crop_w = align_floor_to(crop_w, alignment=2)
    crop_t = align_floor_to(crop_t, alignment=1)

    if image_latents is not None:
        return (
            latents[:, :, :crop_t, top : top + crop_h, left : left + crop_w],
            image_latents[:, :, :crop_t, top : top + crop_h, left : left + crop_w],
        )
    else:
        return latents[:, :, :crop_t, top : top + crop_h, left : left + crop_w], image_latents
    
def crop_tensor_dpo(
    latents,
    latents_lose,
    image_latents=None,
    crop_width_ratio=1.0,
    crop_height_ratio=1.0,
    crop_type="center",
    crop_time_ratio=1.0,
):
    b, c, t, h, w = latents.shape
    crop_h, crop_w = int(h * crop_height_ratio), int(w * crop_width_ratio)
    crop_t = int(t * crop_time_ratio)

    if crop_type == "center":
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
    elif crop_type == "random":
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

    crop_h = align_floor_to(crop_h, alignment=2)
    crop_w = align_floor_to(crop_w, alignment=2)
    crop_t = align_floor_to(crop_t, alignment=1)

    if image_latents is not None:
        return (
            latents[:, :, :crop_t, top : top + crop_h, left : left + crop_w],
            image_latents[:, :, :crop_t, top : top + crop_h, left : left + crop_w],
            latents_lose[:, :, :crop_t, top : top + crop_h, left : left + crop_w]
        )
    else:
        return (latents[:, :, :crop_t, top : top + crop_h, left : left + crop_w],
        image_latents,
        latents_lose[:, :, :crop_t, top : top + crop_h, left : left + crop_w])


def megabatch_frame_alignment(megabatches, lengths):
    aligned_magabatches = []
    for _, megabatch in enumerate(megabatches):
        assert len(megabatch) != 0
        len_each_megabatch = [lengths[i] for i in megabatch]
        idx_length_dict = dict([*zip(megabatch, len_each_megabatch)])
        count_dict = Counter(len_each_megabatch)

        # mixed frame length, align megabatch inside
        if len(count_dict) != 1:
            sorted_by_value = sorted(count_dict.items(), key=lambda item: item[1])
            pick_length = sorted_by_value[-1][0]  # the highest frequency
            candidate_batch = [
                idx for idx, length in idx_length_dict.items() if length == pick_length
            ]
            random_select_batch = [
                random.choice(candidate_batch)
                for i in range(len(idx_length_dict) - len(candidate_batch))
            ]
            aligned_magabatch = candidate_batch + random_select_batch
            aligned_magabatches.append(aligned_magabatch)
        # already aligned megabatches
        else:
            aligned_magabatches.append(megabatch)

    return aligned_magabatches


def split_to_even_chunks(indices, lengths, num_chunks, batch_size):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        chunks = [indices[i::num_chunks] for i in range(num_chunks)]
    else:
        num_indices_per_chunk = len(indices) // num_chunks

        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0 for _ in range(num_chunks)]
        for index in indices:
            shortest_chunk = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest_chunk].append(index)
            chunks_lengths[shortest_chunk] += lengths[index]
            if len(chunks[shortest_chunk]) == num_indices_per_chunk:
                chunks_lengths[shortest_chunk] = float("inf")
    # return chunks

    pad_chunks = []
    for idx, chunk in enumerate(chunks):
        if batch_size != len(chunk):
            assert batch_size > len(chunk)
            if len(chunk) != 0:
                chunk = chunk + [
                    random.choice(chunk) for _ in range(batch_size - len(chunk))
                ]
            else:
                chunk = random.choice(pad_chunks)
                print(chunks[idx], "->", chunk)
        pad_chunks.append(chunk)
    return pad_chunks


def group_frame_fun(indices, lengths):
    # sort by num_frames
    indices.sort(key=lambda i: lengths[i], reverse=True)
    return indices


def get_length_grouped_indices(
    lengths,
    batch_size,
    world_size,
    generator=None,
    group_frame=False,
    group_resolution=False,
    seed=42,
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is None:
        generator = torch.Generator().manual_seed(
            seed
        )  # every rank will generate a fixed order but random index

    indices = torch.randperm(len(lengths), generator=generator).tolist()

    # sort dataset according to frame
    indices = group_frame_fun(indices, lengths)

    # chunk dataset to megabatches
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)
    ]

    # make sure the length in each magabatch is align with each other
    megabatches = megabatch_frame_alignment(megabatches, lengths)

    # aplit aligned megabatch into batches
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size, batch_size)
        for megabatch in megabatches
    ]

    # random megabatches to do video-image mix training
    indices = torch.randperm(len(megabatches), generator=generator).tolist()
    shuffled_megabatches = [megabatches[i] for i in indices]

    # expand indices and return
    return [
        i for megabatch in shuffled_megabatches for batch in megabatch for i in batch
    ]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        rank: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        group_frame=False,
        group_resolution=False,
        generator=None,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.lengths = lengths
        self.group_frame = group_frame
        self.group_resolution = group_resolution
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(
            self.lengths,
            self.batch_size,
            self.world_size,
            group_frame=self.group_frame,
            group_resolution=self.group_resolution,
            generator=self.generator,
        )

        def distributed_sampler(lst, rank, batch_size, world_size):
            result = []
            index = rank * batch_size
            while index < len(lst):
                result.extend(lst[index : index + batch_size])
                index += batch_size * world_size
            return result

        indices = distributed_sampler(
            indices, self.rank, self.batch_size, self.world_size
        )
        return iter(indices)


def save_videos_grid(videos, path, rescale=False, n_rows=1, fps=24):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


class BlockDistributedSampler(TorchDistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False,
                 batch_size=-1, start_index=0, align=1):
        """
        Args:
            dataset: Dataset used for sampling.
            num_replicas: Number of processes participating in distributed training.
            rank: Rank of the current process within num_replicas.
            shuffle: If True, the sampler will shuffle the indices.
            seed: Random seed.
            drop_last: If True, the sampler will drop the last batch if its size would be less than batch_size.
            batch_size: Size of mini-batch. If callable, it should accept a tuple of (w, h) as input and return an integer
                value as the batch size. It is useful for mix-scale(e.g., 256, 512, 1024) training.
            start_index: Start index for the sampler.
            align: Align the indices to the multiple of align for each dp.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if batch_size != -1:
            align = batch_size
            warnings.warn("batch_size is deprecated, please use `align` instead.")
        if align <= 0:
            raise ValueError(f"align should be a positive integer, but got {align}.")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.align = align
        self._start_index = start_index
        self.recompute_sizes()

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        if self._start_index != value:
            self._start_index = value
            self.recompute_sizes()

    def recompute_sizes(self):
        self.num_samples = len(self.dataset) // self.align * self.align // self.num_replicas \
                           - self._start_index
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        raw_num_samples = len(indices) // self.align * self.align // self.num_replicas
        raw_total_size = raw_num_samples * self.num_replicas
        indices = indices[:raw_total_size]

        # subsample with start_index
        indices = indices[self.rank * raw_num_samples + self.start_index:(self.rank + 1) * raw_num_samples]
        assert len(indices) + self.start_index == raw_num_samples, \
            f"{len(indices) + self.start_index} vs {raw_num_samples}"

        # print(f"Iterator of BlockDistributedSampler created.")
        # This is a sequential sampler. The shuffle operation is done by the dataset itself.
        return iter(indices)


class DistributedSampler(TorchDistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False,
                 start_index=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self._start_index = start_index
        self.recompute_sizes()
        self.shuffle = shuffle
        self.seed = seed

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        self._start_index = value
        self.recompute_sizes()

    def recompute_sizes(self):
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and (len(self.dataset) - self._start_index) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                ((len(self.dataset) - self._start_index) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil((len(self.dataset) - self._start_index) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            indices = indices[self._start_index:]
        else:
            indices = list(range(self._start_index, len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample with start_index
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        print(f"Iterator of DistributedSamplerWithStartIndex created.")
        return iter(indices)


# For backward compatibility
DistributedSamplerWithStartIndex = DistributedSampler


def cumsum(sequence):
    r, s = [], 0
    for e in sequence:
        l = len(e)
        r.append(l + s)
        s += l
    return r

def get_infinite_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch
        dataloader.sampler.set_epoch(dataloader.sampler.epoch + 1)
        print(f"epoch: {dataloader.sampler.epoch}, rank: {dataloader.sampler.rank}")
        

class VideoImageBatchIterator:
    def __init__(self,
                 video_dataloader, 
                 image_dataloader = None, 
                 sp_size = 1,
                ):
        assert video_dataloader is not None or image_dataloader is not None
        self.sp_size = sp_size
        self.video_dataloader = video_dataloader
        self.image_dataloader = image_dataloader
        self.video_iterator = iter(self.video_dataloader) if video_dataloader is not None else None
        self.image_iterator = iter(self.image_dataloader) if image_dataloader is not None else None
     
    def get_image_batch(self):
        try:
            if self.sp_size > 1:
                while True:
                    batch = next(self.image_iterator)
                    shape = batch[0].shape
                    if shape[-1]/16 * shape[-2]/16 % self.sp_size == 0:
                        break
                    else:
                        logging.warning(f"skipping one sample due to the shape {shape} and SP {self.sp_size} mismatching")
            else:
                batch = next(self.image_iterator)
            return batch
        except StopIteration:
            logging.info(f"Image dataset start new epoch")
            self.image_iterator = iter(self.image_dataloader)
            raise StopIteration
        
    
    def get_video_batch(self):
        try:
            if self.sp_size > 1:
                while True:
                    batch = next(self.video_iterator)
                    shape = batch[0].shape # [B, C, T, H, W]
                    if (shape[-1]/2 * shape[-2]/2 * shape[-3] % self.sp_size == 0):
                        break
                    else:
                        logging.warning(f"skipping one sample due to the shape {shape} and SP {self.sp_size} mismatching")
            else:
                batch = next(self.video_iterator)
                
            return batch
        except StopIteration:
            logging.info(f"Video dataset start new epoch")
            self.video_iterator = iter(self.video_dataloader)
            return next(self.video_iterator)
            

    def __iter__(self):
        return self

    def __next__(self):
        if self.video_iterator is None:
            return self.get_image_batch()
        if self.image_iterator is None:
            return self.get_video_batch()