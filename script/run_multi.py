import os
import sys
import math
import pprint
import shutil
import logging
import argparse
import numpy as np

import torch

import torchdrug
from torchdrug import core, datasets, tasks, models, layers
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from peer import protbert, util, flip
from peer.engine import MultiTaskEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="config/multi_task/CNN/beta_contact_CNN.yaml")
    parser.add_argument("--seed", help="random seed", type=int, default=0)

    return parser.parse_known_args()[0]


def set_seed(seed):
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def build_solver(cfg, logger):
    # build dataset and model for each task
    model = core.Configurable.load_config_dict(cfg.model)

    train_sets, valid_sets, test_sets = [], [], []
    tasks = []
    for dataset_config, task_config in zip(cfg.datasets, cfg.tasks):
        if "center" in dataset_config:
            is_center = dataset_config.pop("center")
        else:
            is_center = False
        if "test_split" in dataset_config:
            test_split = dataset_config.pop("test_split")
            _dataset = core.Configurable.load_config_dict(dataset_config)
            train_set, valid_set, test_set = _dataset.split(['train', 'valid', test_split])
        else:
            _dataset = core.Configurable.load_config_dict(dataset_config)
            train_set, valid_set, test_set = _dataset.split()
        if comm.get_rank() == 0:
            logger.warning(_dataset)
            logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))
        if is_center:
            train_sets = [train_set] + train_sets
            valid_sets = [valid_set] + valid_sets
            test_sets = [test_set] + test_sets
        else:
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        if task_config["class"] in ["PropertyPrediction", "InteractionPrediction"]:
            task_config.task = _dataset.tasks
        task_config.model = model
        task = core.Configurable.load_config_dict(task_config)
        if is_center:
            tasks = [task] + tasks
        else:
            tasks.append(task)

    # build solver
    cfg.optimizer.params = model.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if not "scheduler" in cfg:
        scheduler = None
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)

    solver = MultiTaskEngine(tasks, train_sets, valid_sets, test_sets, optimizer, scheduler, **cfg.engine)
    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {'params': model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
        ]
        for model in solver.models:
            cfg.optimizer.params.append({
                'params': model.mlp.parameters(), 'lr': cfg.optimizer.lr
            })
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint, load_optimizer=False)

    return solver


def train_and_validate(cfg, solver):
    step = math.ceil(cfg.train.num_epoch / 10)
    best_score = float("-inf")
    best_epoch = -1

    if not cfg.train.num_epoch > 0:
        return solver, best_epoch

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        for model in solver.models:
            model.split = "train"
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        if "test_batch_size" in cfg:
            solver.batch_size = cfg.test_batch_size
        for model in solver.models:
            model.split = "valid"
        metric = solver.evaluate("valid")
        solver.batch_size = cfg.engine.batch_size

        score = []
        for k, v in metric.items():
            if k.startswith(cfg.eval_metric):
                if "root mean squared error" in cfg.eval_metric:
                    score.append(-v)
                else:
                    score.append(v)
        score = sum(score) / len(score)
        if score > best_score:
            best_score = score
            best_epoch = solver.epoch

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver, best_epoch


def test(cfg, solver):
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
    for model in solver.models:
        model.split = "valid"
    solver.evaluate("valid")
    for model in solver.models:
        model.split = "test"
    solver.evaluate("test")

    return


if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config)

    set_seed(args.seed)
    output_dir = util.create_working_directory_mtl(cfg)
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % output_dir)
        shutil.copyfile(args.config, os.path.basename(args.config))
    os.chdir(output_dir)

    solver = build_solver(cfg, logger)
    solver, best_epoch = train_and_validate(cfg, solver)
    if comm.get_rank() == 0:
        logger.warning("Best epoch on valid: %d" % best_epoch)
    test(cfg, solver)
