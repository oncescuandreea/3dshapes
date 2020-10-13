import argparse

import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.text_model as module_arch_text
from parse_config import ConfigParser
from text_encoder import (FILE_NAME, hue_dict, orientation_dict, scale_dict,
                          shape_dict)

dicts = [hue_dict, hue_dict, hue_dict, scale_dict, shape_dict, orientation_dict]

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    # import pdb; pdb.set_trace()
    # build model architecture, then print to console
    model_text = config.init_obj('arch_text', module_arch_text)
    logger.info(model_text)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    loss_fn_ret = getattr(module_loss, config['retrieval_loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    metric_ret = getattr(module_metric, 'accuracy_retrieval')

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    state_dict_text = checkpoint['state_dict_text']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
        model_text = torch.nn.DataParallel(model_text)
    model.load_state_dict(state_dict)
    model_text.load_state_dict(state_dict_text)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    model_text = model_text.to(device)
    model_text.eval()

    no_tasks = len(_FACTORS_IN_ORDER)
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns), no_tasks)
    total_losses = torch.zeros(no_tasks)
    supervised = False
    if type(model).__name__ == 'ShapeModelRetrieval':
        with torch.no_grad():
            for i, (data, target_ret, _) in enumerate(tqdm(data_loader)):
                
                data, target_ret = data.to(device), target_ret.to(device)
                output = model(data)
                batch_size = data.shape[0]
                text_output = model_text(target_ret.float())

                # computing loss, metrics on test set
                total_loss += loss_fn_ret(output, text_output, 30)
                # for j, metric in enumerate(metric_fns):
                #     total_metrics[j] += metric(output, target) * batch_size

        n_samples = len(data_loader.sampler)
        log = {'loss_retrieval': total_loss / n_samples}
        log = {'accuracy_retrieval': metric_ret(output, text_output)}

    else:
        if type(model).__name__ == 'ShapeModelRetrievalAux' and supervised is True:
            embedding = model_text.embedding
            loss_retrieval = 0.0
            loss_classification = 0.0
            with torch.no_grad():
                for i, (data, target_ret, target_init) in enumerate(tqdm(data_loader)):
                    data, target_ret = data.to(device), target_ret.to(device)
                    target_init = target_init.to(device)
                    output_ret, output_init, output_softmax = model(data)
                    batch_size = data.shape[0]
                    text_output = model_text(target_ret.float())

                    # computing loss, metrics on test set
                    loss_retrieval += loss_fn_ret(output_ret, text_output, 30)
                    loss = 0.0
                    for j in range(0, no_tasks):
                        output_task = output_init[j]
                        target_task = target_init[:, j]
                        loss_task = loss_fn(output_task, target_task) * batch_size
                        loss += loss_task
                        total_losses[j] += loss_task
                        for k, metric in enumerate(metric_fns):
                            total_metrics[k][j] += metric(output_task, target_task) * batch_size

                    #
                    # save sample images, or do something with output here
                    #

                    # computing loss, metrics on test set
                    loss_classification += loss.item()
                    total_loss += loss.item() + loss_fn_ret(output_ret, text_output, 30).item()
                    # for j, metric in enumerate(metric_fns):
                    #     total_metrics[j] += metric(output, target) * batch_size

            n_samples = len(data_loader.sampler)
            log = {'total_loss': total_loss / n_samples}
            log = {'loss_retrieval': loss_retrieval / n_samples}
            log = {'loss_classification': total_loss / n_samples}
            log = {'accuracy_retrieval': metric_ret(output_ret, text_output)}
            for i, met in enumerate(metric_fns):
                for j in range(0, no_tasks):
                    log.update({f"{met.__name__}_{_FACTORS_IN_ORDER[j]}": \
                        total_metrics[i][j].item() / n_samples})
        elif supervised is False:
            embedding = model_text.embedding
            # embedding = nn.Embedding.from_pretrained(model_text.embedding)
            loss_retrieval = 0.0
            loss_classification = 0.0
            with torch.no_grad():
                for i, (data, target_ret, target_init) in enumerate(tqdm(data_loader)):
                    data, target_ret = data.to(device), target_ret.to(device)
                    target_init = target_init.to(device)
                    output_ret, output_init, output_softmax = model(data)
                    batch_size = data.shape[0]
                    predicted_idx_from_words = torch.zeros([len(data), 150])
                    # computing loss, metrics on test set
                    loss = 0.0
                    for j in range(0, no_tasks):
                        output_task = output_init[j]
                        target_task = target_init[:, j]
                        loss_task = loss_fn(output_task, target_task) * batch_size
                        loss += loss_task
                        total_losses[j] += loss_task
                        for k, metric in enumerate(metric_fns):
                            total_metrics[k][j] += metric(output_task, target_task) * batch_size
                        # import pdb; pdb.set_trace()
                        # get number of classes available for current task
                        number_outputs_task = len(dicts[j].keys())
                        list_classes_task = list(range(0, number_outputs_task))
                        # get label from initial data file corresponding to class
                        list_labels_task = [data_loader.idx2label_init[j][a] for a in list_classes_task]
                        # find the words associated with the initial label
                        word_labels = [dicts[j][a] for a in list(list_labels_task)]
                        # index word to corpus index
                        word_to_id_predicted = [data_loader.label2idx_ret[0][word] for word in word_labels]
                        # word ids to embeddings
                        task_words_embeddings = embedding(torch.tensor(word_to_id_predicted).cuda())
                        predicted_mean = torch.matmul(output_softmax[j], task_words_embeddings)
                        if j == 0:
                            predicted_idx_from_words = predicted_mean
                        else:
                            predicted_idx_from_words = torch.cat((predicted_idx_from_words, predicted_mean), dim=1)

                    text_output = model_text(predicted_idx_from_words)
                    loss_retrieval += loss_fn_ret(output_ret, text_output)
                    #
                    # save sample images, or do something with output here
                    #

                    # computing loss, metrics on test set
                    loss_classification += loss.item()
                    total_loss += loss.item() + loss_fn_ret(output_ret, text_output).item()
                    # for j, metric in enumerate(metric_fns):
                    #     total_metrics[j] += metric(output, target) * batch_size

            n_samples = len(data_loader.sampler)
            log = {'total_loss': total_loss / n_samples}
            log = {'loss_retrieval': loss_retrieval / n_samples}
            log = {'loss_classification': total_loss / n_samples}
            log = {'accuracy_retrieval': metric_ret(output_ret, text_output)}
            for i, met in enumerate(metric_fns):
                for j in range(0, no_tasks):
                    log.update({f"{met.__name__}_{_FACTORS_IN_ORDER[j]}": \
                        total_metrics[i][j].item() / n_samples})
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
