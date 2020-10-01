import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.text_model as module_arch_text
from parse_config import ConfigParser

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

    # build model architecture, then print to console
    model_text = config.init_obj('arch_text', module_arch_text)
    logger.info(model_text)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    no_tasks = len(_FACTORS_IN_ORDER)
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns), no_tasks)
    total_losses = torch.zeros(no_tasks)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_size = data.shape[0]
            loss = 0
            for j in range(0, no_tasks):
                output_task = output[j]
                target_task = target[:, j]
                loss_task = loss_fn(output_task, target_task) * batch_size
                loss += loss_task
                total_losses[j] += loss_task
                
                for k, metric in enumerate(metric_fns):
                    total_metrics[k][j] += metric(output_task, target_task) * batch_size

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            
            total_loss += loss.item()
            # for j, metric in enumerate(metric_fns):
            #     total_metrics[j] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    for i, met in enumerate(metric_fns):
        for j in range(0, no_tasks):
            log.update({
                            f"{met.__name__}_{_FACTORS_IN_ORDER[j]}": total_metrics[i][j].item() / n_samples 
                        })
                        
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
