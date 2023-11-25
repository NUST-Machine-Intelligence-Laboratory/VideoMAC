import math
import sys
from typing import Iterable

import torch
from tools import utils

def train_one_epoch(model_online: torch.nn.Module,
                    model_target: torch.nn.Module,
                    target_without_ddp: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    momentum_schedule,
                    log_writer=None,
                    args=None):
    model_online.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq

    optimizer.zero_grad()

    for data_iter_step, (samples1, samples2) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
           utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if not isinstance(samples1, list):
            samples1 = samples1.to(device, non_blocking=True)

        if not isinstance(samples2, list):
            samples2 = samples2.to(device, non_blocking=True)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                online_loss, online_pred, online_mask, ids_shuffle, ids_restore = \
                    model_online(samples1, ids_shuffle=None, ids_restore=None)
                with torch.no_grad():
                    model_target.eval()
                    target_loss, target_pred = model_target(samples2, \
                                                            ids_shuffle=ids_shuffle, ids_restore=ids_restore)
                rec_cons_loss = utils.cons_loss(online_pred, online_mask, target_pred)
                loss = online_loss + target_loss + args.gamma * rec_cons_loss
        else:
            online_loss, online_pred, online_mask, ids_shuffle, ids_restore = \
                model_online(samples1, ids_shuffle=None, ids_restore=None)
            with torch.no_grad():
                model_target.eval()
                target_loss, target_pred = model_target(samples2, \
                                                        ids_shuffle=ids_shuffle, ids_restore=ids_restore)
            rec_cons_loss = utils.cons_loss(online_pred, online_mask, target_pred)
            loss = online_loss + target_loss + args.gamma * rec_cons_loss

        loss_value = loss.item()
        online_loss_value = online_loss.item()
        target_loss_value = target_loss.item()
        rec_cons_loss_value = rec_cons_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
    
        loss /= update_freq
        loss_scaler(loss, optimizer, parameters=model_online.parameters(),
                    update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache() # clear the GPU cache at a regular interval for training ME network
        
        # EMA update for the teacher
        with torch.no_grad():
            ms = momentum_schedule[data_iter_step]  # momentum parameter
            if args.distributed:
                for param_q, param_k in zip(
                    model_online.module.parameters(), target_without_ddp.parameters()
                ):
                    param_k.data.mul_(ms).add_((1 - ms) * param_q.detach().data)
            else:
                for param_q, param_k in zip(
                    model_online.parameters(), target_without_ddp.parameters()
                ):
                    param_k.data.mul_(ms).add_((1 - ms) * param_q.detach().data)

        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(online_loss=online_loss_value)
        metric_logger.update(target_loss=target_loss_value)
        metric_logger.update(rec_cons_loss=rec_cons_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = utils.all_reduce_mean(loss_value)
        online_loss_value_reduce = utils.all_reduce_mean(online_loss_value)
        target_loss_value_reduce = utils.all_reduce_mean(target_loss_value)
        rec_cons_loss_value_reduce = utils.all_reduce_mean(rec_cons_loss_value)
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(train_loss=loss_value_reduce, head="loss", step=epoch_1000x)
            log_writer.update(train_loss=online_loss_value_reduce, head="online_loss", step=epoch_1000x)
            log_writer.update(train_loss=target_loss_value_reduce, head="target_loss", step=epoch_1000x)
            log_writer.update(train_loss=rec_cons_loss_value_reduce, head="rec_cons_loss", step=epoch_1000x)
            log_writer.update(lr=lr, head="opt", step=epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}