import os
import sys
import time
import random
import logging

import yaml
import easydict
import jinja2

from torch import distributed as dist

from torchdrug.utils import comm


def meshgrid(dict):
    if len(dict) == 0:
        yield {}
        return

    key = next(iter(dict))
    values = dict[key]
    sub_dict = dict.copy()
    sub_dict.pop(key)

    if not isinstance(values, list):
        values = [values]
    for value in values:
        for result in meshgrid(sub_dict):
            result[key] = value
            yield result


def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
    cfg = yaml.load(raw_text, Loader=yaml.CLoader)
    cfg = easydict.EasyDict(cfg)

    return cfg


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    output_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                              cfg.task["class"], cfg.dataset["class"],
                              cfg.task.model["class"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(output_dir)
        os.makedirs(output_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            output_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(output_dir)
    return output_dir


def create_working_directory_mtl(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    output_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                              "MultitaskLearning", "_".join(dataset["class"] for dataset in cfg.datasets),
                              cfg.model["class"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(output_dir)
        os.makedirs(output_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            output_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(output_dir)
    return output_dir
