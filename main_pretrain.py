import argparse
import datetime
import setproctitle
import numpy as np
import time
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from models.dataloader_mac import YT18Dataset, K400Dataset

import timm.optim.optim_factory as optim_factory

import models.video_mac as video_mac
from engine_pretrain import train_one_epoch


from tools import utils
from tools.utils import NativeScalerWithGradNormCount as NativeScaler
from tools.utils import str2bool

def get_args_parser():
    parser = argparse.ArgumentParser('Video-MAC pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation step')
    parser.add_argument('--use_amp', type=str2bool, default=False)
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    
    # Model parameters
    parser.add_argument('--model', default='convnets', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--patch_size', default=32, type=int,
                        help='Patch size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--save_ckpt_num', default=10, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # loss balance weight gamma
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="loss balance weight gamma"
    )
    # For target encoder
    parser.add_argument(
        "--momentum_target",
        default=0.996,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1
        during training with cosine schedule.
        We recommend setting a higher value with small batches:
        for example use 0.9995 with batch size of 256.""",
    )
    return parser

def main(args):
    # set name
    setproctitle.setproctitle("Video-MAC")
    utils.init_distributed_mode(args)

    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    
    # video dataset
    if 'yt18' in args.data_path or 'ytb18' in args.data_path:
        dataset_train = YT18Dataset(args.input_size, os.path.join(args.data_path, 'train'))
    elif 'k400' in args.data_path:
        dataset_train = K400Dataset(args.input_size, os.path.join(args.data_path, 'train'))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model_online = video_mac.__dict__[args.model](
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth,
        decoder_embed_dim=args.decoder_embed_dim,
        norm_pix_loss=args.norm_pix_loss,
        patch_size=args.patch_size,
        compute_loss=True)
    model_target = video_mac.__dict__[args.model](
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth,
        decoder_embed_dim=args.decoder_embed_dim,
        norm_pix_loss=args.norm_pix_loss,
        patch_size=args.patch_size,
        compute_loss=False)
    model_online.to(device)
    model_target.to(device)

    online_without_ddp = model_online
    target_without_ddp = model_target
    n_parameters = sum(p.numel() for p in model_online.parameters() if p.requires_grad)

    print("Online Model = %s" % str(online_without_ddp))
    print("Target Model = %s" % str(target_without_ddp))
    # print('number of params:', n_parameters)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
        
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model_online = torch.nn.parallel.DistributedDataParallel(model_online, device_ids=[args.gpu], find_unused_parameters=False)
        online_without_ddp = model_online.module
        target_without_ddp.load_state_dict(model_online.module.state_dict())
    else:
        target_without_ddp.load_state_dict(model_online.state_dict())
    for param in model_target.parameters():
        param.requires_grad = False
    print(f"Online and Target are built: they are both {args.model} network.")

    param_groups = optim_factory.param_groups_weight_decay(online_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # momentum parameter is increased to 1. during training with a cosine
    # schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_target, 1, args.epochs, len(data_loader_train)
    )

    utils.auto_load_model_distill(
        args=args, model_online=model_online, online_without_ddp=online_without_ddp,
        model_target=model_target, target_without_ddp=target_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model_online, model_target, target_without_ddp,
            data_loader_train, optimizer, device, epoch, 
            loss_scaler, momentum_schedule, log_writer=log_writer,
            args=args
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model_distill(
                    args=args,
                    model_online=model_online, 
                    online_without_ddp=online_without_ddp,
                    model_target=model_target,
                    target_without_ddp=target_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)