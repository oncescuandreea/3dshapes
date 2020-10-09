from collections import Counter

import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
from base import BaseTrainerRetrieval, BaseTrainer
from model.metric import accuracy_retrieval
from utils import MetricTracker, add_margin, histogram_distribution, inf_loop

from text_encoder import hue_dict, scale_dict, shape_dict, orientation_dict, FILE_NAME

dicts = [hue_dict, hue_dict, hue_dict, scale_dict, shape_dict, orientation_dict]


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
        self.train_metrics = MetricTracker('loss', 'loss_floor_hue', 'loss_wall_hue',
                                           'loss_object_hue', 'loss_scale', 'loss_shape',
                                           'loss_orientation', 'accuracy_floor_hue',
                                           'accuracy_wall_hue', 'accuracy_object_hue',
                                           'accuracy_scale', 'accuracy_shape',
                                           'accuracy_orientation', 'accuracy',
                                           writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'loss_floor_hue', 'loss_wall_hue',
                                           'loss_object_hue', 'loss_scale', 'loss_shape',
                                           'loss_orientation', 'accuracy_floor_hue',
                                           'accuracy_wall_hue', 'accuracy_object_hue',
                                           'accuracy_scale', 'accuracy_shape',
                                           'accuracy_orientation', 'accuracy',
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
                new_org = add_margin(img_list=data[0:8, :, :],
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
                    new_org = add_margin(img_list=data[0:8, :, :],
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
                    self.valid_metrics.update(loss_title,
                                              loss_task.item())
                    for met in self.metric_ftns:
                        metric_title = f"{met.__name__}_{_FACTORS_IN_ORDER[i]}"
                        self.valid_metrics.update(metric_title,
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

class TrainerRetrieval(BaseTrainerRetrieval):
    """
    Trainer class for retrieval
    """
    def __init__(self, model, model_text, criterion, metric_ftns, optimizer, config,
                 data_loader, font_type,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
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

        self.train_metrics = MetricTracker('loss', 'accuracy_retrieval', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'accuracy_retrieval', writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_text.train()
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target_ret, _) in enumerate(self.data_loader):
            data, target_ret = data.to(self.device), target_ret.to(self.device)
            self.optimizer.zero_grad()
            text_output = self.model_text(target_ret.float())
            output = self.model(data)
            loss = self.criterion(output, text_output, 30)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            try:
                self.train_metrics.update('loss', loss.item())
            except AttributeError:
                print("Not enough data")
            self.train_metrics.update('accuracy_retrieval', accuracy_retrieval(output, text_output))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss ret: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
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
        self.model_text.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target_ret, _) in enumerate(self.valid_data_loader):
                data, target_ret = data.to(self.device), target_ret.to(self.device)

                text_output = self.model_text(target_ret.float())
                output = self.model(data)
                loss = self.criterion(output, text_output, 10)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                try:
                    self.valid_metrics.update('loss', loss.item())
                except AttributeError:
                    print("Not enough data")
                self.valid_metrics.update('accuracy_retrieval',
                                          accuracy_retrieval(output, text_output))

                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.model_text.named_parameters():
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


class TrainerRetrievalAux(BaseTrainerRetrieval):
    """
    Trainer class for retrieval with classification as extra info
    """
    def __init__(self, model, model_text, criterion, criterion_ret,
                 metric_ftns, metric_ftns_ret, optimizer, config,
                 data_loader, font_type,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, model_text, criterion, criterion_ret, metric_ftns, metric_ftns_ret, optimizer, config)
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
        self.train_metrics = MetricTracker('loss_classification', 'accuracy_retrieval',
                                           'loss_floor_hue', 'loss_wall_hue', 'loss_object_hue',
                                           'loss_retrieval', 'loss_tot', 'loss_scale', 'loss_shape',
                                           'loss_orientation', 'accuracy_floor_hue',
                                           'accuracy_wall_hue', 'accuracy_object_hue',
                                           'accuracy_scale', 'accuracy_shape',
                                           'accuracy_orientation', 'acc_ret_MR',
                                           'acc_ret_R1', 'acc_ret_R5', 'acc_ret_R10',
                                           writer=self.writer)
        self.valid_metrics = MetricTracker('loss_classification', 'accuracy_retrieval',
                                           'loss_floor_hue', 'loss_wall_hue', 'loss_object_hue',
                                           'loss_retrieval', 'loss_tot', 'loss_scale', 'loss_shape',
                                           'loss_orientation', 'accuracy_floor_hue',
                                           'accuracy_wall_hue', 'accuracy_object_hue',
                                           'accuracy_scale', 'accuracy_shape',
                                           'accuracy_orientation', 'acc_ret_MR',
                                           'acc_ret_R1', 'acc_ret_R5', 'acc_ret_R10',
                                           writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_text.train()
        self.model.train()
        self.train_metrics.reset()
        if epoch == 1:
            list_of_counters = []
            for i in range(0, 6):
                list_of_counters.append(Counter())
        for batch_idx, (data, target_ret, target_init) in enumerate(self.data_loader):
            # import pdb; pdb.set_trace()
            data, target_ret = data.to(self.device), target_ret.to(self.device)
            target_init = target_init.to(self.device)
            self.optimizer.zero_grad()
            text_output = self.model_text(target_ret.float())
            output_ret, output_init = self.model(data)
            loss_ret = self.criterion_ret(output_ret, text_output)
            no_tasks = len(target_init[0])
            loss_classification = 0

            for i in range(0, no_tasks):
                output_task = output_init[i]
                target_task = target_init[:, i]
                if epoch == 1:
                    list_of_counters[i] += Counter(target_task.tolist())
                self.tensorboard_labeled_image(data, target_task,
                                               output_task, i, epoch, 'train')
                loss_task = self.criterion(output_task, target_task)
                loss_classification += loss_task
                loss_title = f"loss_{_FACTORS_IN_ORDER[i]}"
                self.train_metrics.update(loss_title,
                                          loss_task.item())
                for met in self.metric_ftns:
                    metric_title = f"{met.__name__}_{_FACTORS_IN_ORDER[i]}"
                    self.train_metrics.update(metric_title,
                                              met(output_task, target_task))


            loss_tot = loss_ret + loss_classification
            loss_tot.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            try:
                self.train_metrics.update('loss_retrieval', loss_ret.item())
            except AttributeError:
                print("Not enough data")
            self.train_metrics_update_retrieval(output_ret, text_output)
            self.train_metrics.update('loss_classification', loss_classification.item())
            self.train_metrics.update('loss_tot', loss_tot.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss tot: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_tot.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss ret: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_ret.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss classification: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_classification.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        #add histograms for data distribution
        if epoch == 1:
            histogram_distribution(list_of_counters, 'train')
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
        self.model_text.eval()
        self.valid_metrics.reset()
        if epoch == 1:
            list_of_counters = []
            for i in range(0, 6):
                list_of_counters.append(Counter())
        with torch.no_grad():
            for batch_idx, (data, target_ret, target_init) in enumerate(self.valid_data_loader):
                data, target_ret = data.to(self.device), target_ret.to(self.device)
                target_init = target_init.to(self.device)
                text_output = self.model_text(target_ret.float())
                output_ret, output_init = self.model(data)
                no_tasks = len(target_init[0])
                loss_ret = self.criterion_ret(output_ret, text_output)
                loss_classification = 0
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                for i in range(0, no_tasks):
                    output_task = output_init[i]
                    target_task = target_init[:, i]
                    if epoch == 1:
                        list_of_counters[i] += Counter(target_task.tolist())
                    self.tensorboard_labeled_image(data, target_task,
                                                   output_task, i, epoch, 'val')
                    loss_task = self.criterion(output_task, target_task)
                    loss_classification += loss_task
                    loss_title = f"loss_{_FACTORS_IN_ORDER[i]}"
                    self.valid_metrics.update(loss_title,
                                              loss_task.item())
                    for met in self.metric_ftns:
                        metric_title = f"{met.__name__}_{_FACTORS_IN_ORDER[i]}"
                        self.valid_metrics.update(metric_title,
                                                  met(output_task, target_task))
                self.valid_metrics.update('loss_classification', loss_classification.item())
                loss_tot = loss_ret + loss_classification
                self.valid_metrics.update('loss_tot', loss_tot.item())
                # for met in self.metric_ftns:
                #     self.valid_metrics.update(met.__name__, met(output, target, no_tasks))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))


                try:
                    self.valid_metrics.update('loss_retrieval', loss_ret.item())
                except AttributeError:
                    print("Not enough data")
                self.valid_metrics_update_retrieval(output_ret, text_output)

                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.model_text.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def tensorboard_labeled_image(self, data, target_task, output_task, i, epoch, split):
        new_org = add_margin(img_list=data[0:8, :, :],
                             labels=target_task,
                             predictions=output_task,
                             margins=5,
                             idx2label=self.data_loader.idx2label_init[i],
                             font=self.font_type,
                            )
        self.writer.add_image(f"Image_{split}_marg_{_FACTORS_IN_ORDER[i]}_{epoch}",
                                torchvision.utils.make_grid(new_org),
                                epoch)
    
    def train_metrics_update_retrieval(self, output_ret, text_output):
        for met in self.metric_ftns_ret:
            if met.__name__ == 'compute_metric':
                accuracies = met(output_ret, text_output)
                for val in ['R1', 'R5', 'R10', 'MR']:
                    metric_title = f"acc_ret_{val}"
                    self.train_metrics.update(metric_title,
                                              accuracies[val])
            else:
                metric_title = f"{met.__name__}"
                try:
                    self.train_metrics.update(metric_title,
                                            met(output_ret, text_output))
                except NotImplementedError:
                    import pdb; pdb.set_trace()
    
    def valid_metrics_update_retrieval(self, output_ret, text_output):
        for met in self.metric_ftns_ret:
            if met.__name__ == 'compute_metric':
                accuracies = met(output_ret, text_output)
                for val in ['R1', 'R5', 'R10', 'MR']:
                    metric_title = f"acc_ret_{val}"
                    self.valid_metrics.update(metric_title,
                                              accuracies[val])
            else:
                metric_title = f"{met.__name__}"
                self.valid_metrics.update(metric_title,
                                        met(output_ret, text_output))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class TrainerRetrievalComplete(TrainerRetrievalAux):
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_text.train()
        self.model.train()
        self.train_metrics.reset()
        if epoch == 1:
            list_of_counters = []
            for i in range(0, 6):
                list_of_counters.append(Counter())
        for batch_idx, (data, target_ret, target_init) in enumerate(self.data_loader):
            data = data.to(self.device) #images as retrieved from input document
            target_ret = target_ret.to(self.device) # labels formed from word indexes
            target_init = target_init.to(self.device) # labels as classes
            self.optimizer.zero_grad()

            output_ret, output_init = self.model(data) #model returns image encoding (layer before last) and classification layer
            loss_classification = 0.0
            no_tasks = len(target_init[0])
            predicted_idx_from_words = []
            for i in range(0, no_tasks):
                output_task = output_init[i]
                target_task = target_init[:, i]
                if epoch == 1:
                    list_of_counters[i] += Counter(target_task.tolist())
                self.tensorboard_labeled_image(data, target_task,
                                               output_task, i, epoch, 'train')
                loss_task = self.criterion(output_task, target_task)
                loss_classification += loss_task
                loss_title = f"loss_{_FACTORS_IN_ORDER[i]}"
                self.train_metrics.update(loss_title,
                                          loss_task.item())
                for met in self.metric_ftns:
                    metric_title = f"{met.__name__}_{_FACTORS_IN_ORDER[i]}"
                    self.train_metrics.update(metric_title,
                                              met(output_task, target_task))
                # find class with highest probability for current task
                max_loc = torch.argmax(output_task, dim=1)
                # get label from initial data file corresponding to class
                target_act_task = [self.data_loader.idx2label_init[i][a.item()] for a in max_loc]
                # find the words associated with the initial label
                word_labels = [dicts[i][a] for a in list(target_act_task)]
                # index word to corpus index
                word_to_id_predicted = [self.data_loader.label2idx_ret[0][word] for word in word_labels]
                predicted_idx_from_words.append(word_to_id_predicted)
            
            predicted_idx_from_words = torch.FloatTensor(predicted_idx_from_words).to(self.device).T
            text_output = self.model_text(predicted_idx_from_words)
            
            loss_ret = self.criterion_ret(output_ret, text_output)
            
            loss_ret.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            try:
                self.train_metrics.update('loss_retrieval', loss_ret.item())
            except AttributeError:
                print("Not enough data")
            self.train_metrics_update_retrieval(output_ret, text_output)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss ret: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_ret.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss classification: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_classification.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        #add histograms for data distribution
        if epoch == 1:
            histogram_distribution(list_of_counters, 'train')
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
        print("Starting validation")
        self.model.eval()
        self.model_text.eval()
        self.valid_metrics.reset()
        if epoch == 1:
            list_of_counters = []
            for i in range(0, 6):
                list_of_counters.append(Counter())
        # print("reached 647")
        with torch.no_grad():
            for batch_idx, (data, target_ret, target_init) in enumerate(self.valid_data_loader):
                # print("reached 650")
                data, target_ret = data.to(self.device), target_ret.to(self.device)
                target_init = target_init.to(self.device)
                output_ret, output_init = self.model(data)
                no_tasks = len(target_init[0])
                loss_classification = 0.0
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # print("reached 658")
                predicted_idx_from_words = []
                for i in range(0, no_tasks):
                    output_task = output_init[i]
                    target_task = target_init[:, i]
                    if epoch == 1:
                        list_of_counters[i] += Counter(target_task.tolist())
                    self.tensorboard_labeled_image(data, target_task,
                                                   output_task, i, epoch, 'val')
                    loss_task = self.criterion(output_task, target_task)
                    loss_classification += loss_task
                    loss_title = f"loss_{_FACTORS_IN_ORDER[i]}"
                    self.valid_metrics.update(loss_title,
                                              loss_task.item())
                    for met in self.metric_ftns:
                        metric_title = f"{met.__name__}_{_FACTORS_IN_ORDER[i]}"
                        self.valid_metrics.update(metric_title,
                                                  met(output_task, target_task))
                    # find class with highest probability for current task
                    max_loc = torch.argmax(output_task, dim=1)
                    # get label from initial data file corresponding to class
                    target_act_task = [self.data_loader.idx2label_init[i][a.item()] for a in max_loc]
                    # find the words associated with the initial label
                    word_labels = [dicts[i][a] for a in list(target_act_task)]
                    # index word to corpus index
                    word_to_id_predicted = [self.data_loader.label2idx_ret[0][word] for word in word_labels]
                    predicted_idx_from_words.append(word_to_id_predicted)
                predicted_idx_from_words = torch.FloatTensor(predicted_idx_from_words).to(self.device).T
                
                text_output = self.model_text(predicted_idx_from_words)
                loss_ret = self.criterion_ret(output_ret, text_output)

                self.valid_metrics.update('loss_classification', loss_classification.item())
                
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                try:
                    self.valid_metrics.update('loss_retrieval', loss_ret.item())
                except AttributeError:
                    print("Not enough data")
                self.valid_metrics_update_retrieval(output_ret, text_output)

                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.model_text.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
