from collections import defaultdict
import os, sys
import logging

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data

from torchdrug import data, core, utils
from torchdrug.core import Registry as R
from torchdrug.utils import comm


module = sys.modules[__name__]
logger = logging.getLogger(__name__)


class ModelsWrapper(nn.Module):
    """
    Wrapper of multiple task models.

    Parameters:
        models (list of nn.Module): multiple task models.
        names (list of str): the names of all task models.
    """

    def __init__(self, models, names):
        super(ModelsWrapper, self).__init__()
        self.models = nn.ModuleList(models)
        self.names = names

    def forward(self, batches):
        all_loss = []
        all_metric = defaultdict(float)
        for id, batch in enumerate(batches):
            loss, metric = self.models[id](batch)
            for k, v in metric.items():
                name = self.names[id] + " " + k
                if id == 0:
                    name = "Center - " + name
                all_metric[name] = v
            all_loss.append(loss)
        all_loss = torch.stack(all_loss)
        return all_loss, all_metric

    def __getitem__(self, id):
        return self.models[id]


@R.register("core.MultiTaskEngine")
class MultiTaskEngine(core.Configurable):
    """
    General class that handles everything about training and test of a Multi-Task Learning (MTL) task.

    We consider the MTL with a single center task and multiple auxiliary tasks,
    where training is performed on all tasks, and test is only performed on the center task.

    Parameters:
        tasks (list of nn.Module): all tasks in the order of [center_task, auxiliary_task1, auxiliary_task2, ...].
        train_sets (list of data.Dataset): training sets corresponding to all tasks.
        valid_sets (list of data.Dataset): validation sets corresponding to all tasks.
        test_sets (list of data.Dataset): test sets corresponding to all tasks.
        optimizer (optim.Optimizer): optimizer.
        scheduler (lr_scheduler._LRScheduler, optional): scheduler.
        gpus (list of int, optional): GPU ids. By default, CPUs will be used.
            For multi-node multi-process case, repeat the GPU ids for each node.
        batch_size (int, optional): batch size of a single CPU / GPU
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        num_worker (int, optional): number of CPU workers per GPU.
        log_interval (int, optional): log every n gradient updates.
    """

    def __init__(self, tasks, train_sets, valid_sets, test_sets, optimizer, scheduler=None, gpus=None, batch_size=1,
                 gradient_interval=1, num_worker=0, log_interval=100):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        logger = core.LoggingLogger()
        self.meter = core.Meter(log_interval=log_interval, silent=self.rank > 0, logger=logger)

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if self.rank == 0:
            module.logger.warning("Preprocess training set")
        # handle dynamic parameters in optimizer
        for task, train_set, valid_set, test_set in zip(tasks, train_sets, valid_sets, test_sets):
            old_params = list(task.parameters())
            result = task.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params):]})

        tasks = ModelsWrapper(tasks, names=[type(set.dataset).__name__ for set in train_sets])
        if self.world_size > 1:
            tasks = nn.SyncBatchNorm.convert_sync_batchnorm(tasks)
        if self.device.type == "cuda":
            tasks = tasks.cuda(self.device)

        self.models = tasks
        self.train_sets = train_sets
        self.valid_sets = valid_sets
        self.test_sets = test_sets
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, num_epoch=1, batch_per_epoch=None, tradeoff=1.0):
        """
        Train the model.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            num_epoch (int, optional): number of epochs.
            batch_per_epoch (int, optional): number of batches per epoch.
            tradeoff (float, optional): the tradeoff weight of auxiliary tasks.
        """
        samplers = [
            torch_data.DistributedSampler(train_set, self.world_size, self.rank)
                for train_set in self.train_sets
        ]
        models = self.models
        if self.world_size > 1:
            if self.device.type == "cuda":
                models = nn.parallel.DistributedDataParallel(models, device_ids=[self.device],
                                                             find_unused_parameters=True)
            else:
                models = nn.parallel.DistributedDataParallel(models, device_ids=[self.device],
                                                             find_unused_parameters=True)
        models.train()

        for epoch in self.meter(num_epoch):
            for sampler in samplers:
                sampler.set_epoch(epoch)
            dataloaders = [
                iter(data.DataLoader(train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker))
                    for train_set, sampler in zip(self.train_sets, samplers)
            ]
            batch_per_epoch_ = batch_per_epoch or len(dataloaders[0])

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch_ - start_id, self.gradient_interval)

            for batch_id in range(batch_per_epoch_):
                batches = []
                for task_id, dataloader in enumerate(dataloaders):
                    if batch_id % len(dataloader) == 0 and batch_id != 0:
                        dataloader = iter(data.DataLoader(self.train_sets[task_id], self.batch_size,
                                                          sampler=samplers[task_id], num_workers=self.num_worker))
                        dataloaders[task_id] = dataloader
                    batch = next(dataloader)
                    if self.device.type == "cuda":
                        batch = utils.cuda(batch, device=self.device)
                    batches.append(batch)

                loss, metric = models(batches)
                loss = loss / gradient_interval
                weight = [1.0 if i == 0 else tradeoff for i in range(len(dataloaders))]
                all_loss = (loss * torch.tensor(weight, device=self.device)).sum()
                if not all_loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                
                all_loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch_ - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): write the evaluation results to log or not.

        Returns:
            dict: metrics
        """
        if comm.get_rank() == 0:
            logger.warning("Evaluate on %s" % split)
        test_sets = getattr(self, "%s_sets" % split)
        samplers = [
            torch_data.DistributedSampler(test_set, self.world_size, self.rank)
                for test_set in test_sets
        ]
        dataloaders = [
            data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
                for test_set, sampler in zip(test_sets, samplers)
        ]
        models = self.models

        models.eval()
        all_metric = defaultdict(float)
        for task_id, (dataloader, model) in enumerate(zip(dataloaders, models)):
            preds = []
            targets = []
            for batch in dataloader:
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                pred, target = model.predict_and_target(batch)
                preds.append(pred)
                targets.append(target)

            pred = utils.cat(preds)
            target = utils.cat(targets)
            if self.world_size > 1:
                pred = comm.cat(pred)
                target = comm.cat(target)
            metric = model.evaluate(pred, target)
            for k, v in metric.items():
                name = type(dataloader.dataset.dataset).__name__ + ' ' + k
                if task_id == 0:
                    name = "Center - " + name
                all_metric[name] = v
        if log:
            self.meter.log(all_metric)

        return all_metric

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file.
            load_optimizer (bool, optional): load optimizer state or not.
        """
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        for i, model in enumerate(self.models):
            model.load_state_dict(state["model_%d" % i])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file.
        """
        if comm.get_rank() == 0:
            logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "optimizer": self.optimizer.state_dict()
            }
            for i, model in enumerate(self.models):
                state["model_%d" % i] = model.state_dict()
            torch.save(state, checkpoint)

        comm.synchronize()

    @classmethod
    def load_config_dict(cls, config):
        """
        Construct an instance from the configuration dict.

        Parameters:
            config (dict): the dictionary storing configurations.
        """
        if getattr(cls, "_registry_key", cls.__name__) != config["class"]:
            raise ValueError("Expect config class to be `%s`, but found `%s`" % (cls.__name__, config["class"]))

        optimizer_config = config.pop("optimizer")
        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = core.Configurable.load_config_dict(v)
            if k != "class":
                new_config[k] = v
        optimizer_config["params"] = new_config["task"].parameters()
        new_config["optimizer"] = core.Configurable.load_config_dict(optimizer_config)

        return cls(**new_config)

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id
