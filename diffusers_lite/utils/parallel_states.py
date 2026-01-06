
import torch
import torch.distributed as dist
import os
import time
import random
import functools
from typing import List, Optional, Tuple, Union

class COMM_INFO:

    def __init__(self):
        self.group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_group = 0
        self.group_id = 0
        
        # the group info for teacher-student parallel
        self.ts_group_size = 1
        self.ts_group = None  # for fsdp data parallel communication
        self.ts_group_id = 0
        
        # the group for teacher-student unit union
        self.ts_unit_size = 1
        self.ts_unit_group = None
        self.rank_within_ts_unit_group = 0
        self.ts_unit_group_id = 0


nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False
_TEACHER_STUDENT_PARALLEL_STATE = False

def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state


def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE


def initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % sequence_parallel_size == 0
    ), "world_size must be divisible by sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size
    )
    nccl_info.sp_size = sequence_parallel_size #序列并行size
    nccl_info.global_rank = rank #全局rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            nccl_info.group = group
            nccl_info.rank_within_group = rank - i * sequence_parallel_size #rank在序列并行group中的rank
            nccl_info.group_id = i #sequence parallel group id

def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE


def set_teacher_student_parallel_state(state):
    global _TEACHER_STUDENT_PARALLEL_STATE
    _TEACHER_STUDENT_PARALLEL_STATE = state


def get_teacher_student_parallel_state():
    return _TEACHER_STUDENT_PARALLEL_STATE


            
def initialize_teacher_student_parallel_state(sequence_parallel_size):
    global _TEACHER_STUDENT_PARALLEL_STATE
    """Initialize the teacher-student parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % (2 * sequence_parallel_size) == 0
    ), "world_size must be divisible by 2 * sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size
    )
    _TEACHER_STUDENT_PARALLEL_STATE = True
    nccl_info.global_rank = rank
    # init ts_unit_group and assign info
    # teacher and student must have the same sp size temporally! 
    # In the unit, front is student, back is teacher
    nccl_info.ts_unit_size = sequence_parallel_size * 2
    num_teacher_student_union_groups = world_size // sequence_parallel_size // 2    
    for j in range(num_teacher_student_union_groups):
        ts_unit_ranks = range(j * sequence_parallel_size * 2, (j+1) * sequence_parallel_size * 2)
        ts_unit_group = dist.new_group(ts_unit_ranks)
        if rank in ts_unit_ranks:
            nccl_info.ts_unit_group = ts_unit_group
            nccl_info.ts_unit_group_id = j
            nccl_info.rank_within_ts_unit_group = rank - j * sequence_parallel_size * 2
    
    # init ts_goup and assign info    
    nccl_info.ts_group_size = world_size // 2
    for i in range(2):
        ranks = []
        for j in range(num_teacher_student_union_groups):
            ranks += range((j*2+i) * sequence_parallel_size, (j*2+i+1) * sequence_parallel_size)
                        
        ts_group = dist.new_group(ranks)
        if rank in ranks:
            nccl_info.ts_group = ts_group
            nccl_info.ts_group_id = i
    
def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()

def is_teacher_group():
    if _TEACHER_STUDENT_PARALLEL_STATE:
        return nccl_info.group_id % 2 == 1
    else:
        return True

def is_student_group():
    if _TEACHER_STUDENT_PARALLEL_STATE:
        return nccl_info.group_id % 2 == 0
    else:
        return True
