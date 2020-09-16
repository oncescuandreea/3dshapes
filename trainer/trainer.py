from collections import Counter

import numpy as np
import torch

import torchvision
from base import BaseTrainer
from torchvision.utils import make_grid
from utils import MetricTracker, add_margin, histogram_distribution, inf_loop

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 font_type, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.font_type = font_type
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.no_tasks = len(_FACTORS_IN_ORDER)
        list_metrics = []
        for m in self.metric_ftns:
            for i in range(0, self.no_tasks):
                metric_task = f"{m.__name__}_{_FACTORS_IN_ORDER[i]}"
                list_metrics.append(metric_task)
        list_losses = []
        for i in range(0, self.no_tasks):
            list_losses.append(f"loss_{_FACTORS_IN_ORDER[i]}")
        # import pdb; pdb.set_trace()
        self.train_metrics = MetricTracker('loss', 'loss_floor_hue', 'loss_wall_hue', 'loss_object_hue',
                                           'loss_scale', 'loss_shape', 'loss_orientation', 'accuracy_floor_hue',
                                           'accuracy_wall_hue', 'accuracy_object_hue', 'accuracy_scale',
                                           'accuracy_shape', 'accuracy_orientation', 'accuracy',
                                            writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'loss_floor_hue', 'loss_wall_hue', 'loss_object_hue',
                                           'loss_scale', 'loss_shape', 'loss_orientation', 'accuracy_floor_hue',
                                           'accuracy_wall_hue', 'accuracy_object_hue', 'accuracy_scale',
                                           'accuracy_shape', 'accuracy_orientation', 'accuracy',
                                           writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        if epoch == 1:
            list_of_counters = []
            for i in range(0, 6):
                list_of_counters.append(Counter())
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            no_tasks = len(target[0])
            loss = 0
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for i in range(0, no_tasks):
                output_task = output[i]
                target_task = target[:, i]
                if epoch == 1:
                    list_of_counters[i] += Counter(target_task.tolist())
                new_org = add_margin(img_list=data[0:4, :, :],
                                    labels=target_task,
                                    predictions=output_task,
                                    margins=5,
                                    idx2label=self.data_loader.idx2label[i],
                                    font=self.font_type,
                                    )
                self.writer.add_image(f"Image_train_marg_{_FACTORS_IN_ORDER[i]}_{epoch}",
                                        torchvision.utils.make_grid(new_org),
                                        epoch)
                    
                loss_task = self.criterion(output_task, target_task)
                loss += loss_task
                loss_title = f"loss_{_FACTORS_IN_ORDER[i]}"
                self.train_metrics.update(loss_title,
                                          loss_task.item())
                for met in self.metric_ftns:
                    metric_title = f"{met.__name__}_{_FACTORS_IN_ORDER[i]}"
                    self.train_metrics.update(metric_title, 
                                              met(output_task, target_task))

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target, no_tasks))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        #add histograms for data distribution
        if epoch == 1:
            histogram_distribution(list_of_counters, 'train')

        # self.writer.add_image(f'task_{j}_train', f'task_{j}_train.jpg')
        # self.writer.add_histogram(f'task_{j}_train', counter[:, 1])
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        if epoch == 1:
            list_of_counters = []
            for i in range(0, 6):
                list_of_counters.append(Counter())
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                no_tasks = len(target[0])
                loss = 0
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                for i in range(0, no_tasks):
                    output_task = output[i]
                    target_task = target[:, i]
                    if epoch == 1:
                        list_of_counters[i] += Counter(target_task.tolist())
                    new_org = add_margin(img_list=data[0:4, :, :],
                                            labels=target_task,
                                            predictions=output_task,
                                            margins=5,
                                            idx2label=self.data_loader.idx2label[i],
                                            font=self.font_type,
                                        )
                    self.writer.add_image(f"Image_val_marg_{_FACTORS_IN_ORDER[i]}_{epoch}",
                                            torchvision.utils.make_grid(new_org),
                                            epoch)
                    loss_task = self.criterion(output_task, target_task)
                    loss += loss_task
                    loss_title = f"loss_{_FACTORS_IN_ORDER[i]}"
                    self.train_metrics.update(loss_title,
                                              loss_task.item())
                    for met in self.metric_ftns:
                        metric_title = f"{met.__name__}_{_FACTORS_IN_ORDER[i]}"
                        self.train_metrics.update(metric_title,
                                                  met(output_task, target_task))
                
                self.valid_metrics.update('loss', loss.item())
                # for met in self.metric_ftns:
                #     self.valid_metrics.update(met.__name__, met(output, target, no_tasks))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram for data distribution for validation
        if epoch == 1:
            histogram_distribution(list_of_counters, 'validation')
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
