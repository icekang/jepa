# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn import LayerNorm
import src.models.vision_transformer as vit

import torchio as tio

from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    get_logger,)

from app.vjepa.utils import (
    init_video_model,
    CosineWDSchedule, 
    WarmupCosineSchedule
)
from evals.video_segmentation.utils import (
    KFoldNNUNetSegmentationDataModule
)
from src.models.attentive_pooler import (
    AttentiveSegmentator
)
import yaml
import segmentation_models_pytorch.losses as smp_losses
from torchmetrics import Dice

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def main(args_eval, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    load_weight = args_pretrain.get('load_weight', True)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    num_classes = args_data.get('num_classes')
    batch_size = args_data.get('batch_size')

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')
    freeze_encoder = args_opt.get('freeze_encoder')

    # -- DECODER
    args_dec = args_eval.get('decoder')
    decoder_depth = args_dec.get('depth')

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'video_segmentation/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    test_prediction_path = os.path.join(folder, f'{tag}-predictions')
    train_visualization_folder = os.path.join(folder, 'visualization')

    if not os.path.exists(train_visualization_folder):
        os.makedirs(train_visualization_folder, exist_ok=True)
    if not os.path.exists(test_prediction_path):
        os.makedirs(test_prediction_path, exist_ok=True)



    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%.5f', 'train-dice'),
        ('%.5f', 'val-dice'),
    )

    # -- init model
    # Initialize model

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,
        load_weight=load_weight)
    if freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    decoder = AttentiveSegmentator(
        img_size=encoder.input_size,
        patch_size=encoder.patch_size,
        num_frames=encoder.num_frames,
        tubelet_size=encoder.tubelet_size,
        encoder_embed_dim=encoder.embed_dim,
        decoder_embed_dim=768,
        depth=decoder_depth,
        num_heads=12, 
        mlp_ratio=4.0, 
        qkv_bias=True, 
        qk_scale=None, 
        drop_rate=0.0, 
        attn_drop_rate=0.0,
        norm_layer=LayerNorm,
        init_std=0.02, 
        num_classes=num_classes, 
    ).to(device)

    # -- init data-loaders/samplers
    data_module = KFoldNNUNetSegmentationDataModule(args_eval)
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        decoder=decoder,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
        freeze_encoder=freeze_encoder)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    decoder = DistributedDataParallel(decoder, static_graph=True)

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        encoder, decoder, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=latest_path,
            encoder=encoder,
            decoder=decoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path=latest_path):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        train_dice = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            decoder=decoder,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            freeze_encoder=freeze_encoder,
            vis_prefix=f'{train_visualization_folder}/train_{epoch}')

        val_dice = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            decoder=decoder,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            freeze_encoder=freeze_encoder,
            vis_prefix=f'{train_visualization_folder}/val_{epoch}')

        logger.info('[Epoch %5d] train: %.3f%% test: %.3f%%' % (epoch + 1, train_dice, val_dice))
        if rank == 0:
            csv_logger.log(epoch + 1, train_dice, val_dice)
        save_checkpoint(epoch + 1)
    
    # -- TESTING LOOP
    data_module.setup('test')
    test_loaders, test_grid_samplers = data_module.test_dataloader()
    dice_scores = []
    for test_loader, test_grid_sampler, index in zip(test_loaders, test_grid_samplers, range(len(test_loaders))):
        prediction_aggregator = tio.inference.GridAggregator(test_grid_sampler)
        label_aggregator = tio.inference.GridAggregator(test_grid_sampler)
        dice_score = run_test_whole_volumne(
            device=device,
            encoder=encoder,
            decoder=decoder,
            data_loader=test_loader,
            prediction_aggregator=prediction_aggregator,
            label_aggregator=label_aggregator,
            use_bfloat16=use_bfloat16,
            save_prediction_prefix=str(os.path.join(test_prediction_path, f'index_{index}'))
            )
        print(dice_score.item())
        dice_scores.append(dice_score)
    dice_score = dice_score.mean()
    logger.info(f'[0] Test: {dice_score}')
    

def run_one_epoch(
    device,
    training: bool,
    encoder: DistributedDataParallel,
    decoder: AttentiveSegmentator,
    scaler: torch.cuda.amp.GradScaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16: bool,
    freeze_encoder: bool,
    vis_prefix=''):
    if freeze_encoder:
        encoder.train(mode=False)
    else:
        encoder.train(mode=training)
    decoder.train(mode=training)
    criterion_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75]).to(device=device))
    criterion_dice = smp_losses.DiceLoss('multiclass', from_logits=True, smooth=1e-5, ignore_index=0)
    criterion = lambda y_pred, y_gt: 0.5 * criterion_ce(y_pred, y_gt) + 0.5 * criterion_dice(y_pred, y_gt.reshape(y_gt.shape[0], -1))
    dice_meter = Dice(num_classes=decoder.module.num_classes, ignore_index=0).to(device=device)
    has_vised = False
    for itr, data in enumerate(data_loader):

        if training:
            scheduler.step()
            wd_scheduler.step()
        
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            B, C, T, H, W = data['image'].shape
            N_T = T // encoder.module.tubelet_size
            N_H = H // encoder.module.patch_size
            N_W = W // encoder.module.patch_size
            N_CLASS = decoder.module.num_classes

            # Load data and put on GPU
            x = data['image'].to(device)
            y = data['label'].long().to(device)

            # Forward and prediction
            if not training:
                with torch.no_grad():
                    outputs = encoder(x)
                    outputs = decoder(outputs)
            else:
                if freeze_encoder:
                    with torch.no_grad():
                        outputs = encoder(x)
                else:
                    outputs = encoder(x)
                outputs = decoder(outputs)

            # Compute loss
            '''
            outputs shape = B * N_token * (T * P * P * C)
            output.reshape(B, T, H, W)
            '''
            y_pred = outputs.reshape(B, N_T, N_H, N_W, -1)
            y_pred = y_pred.reshape(B, N_T, N_H, N_W, encoder.module.tubelet_size, encoder.module.patch_size, encoder.module.patch_size, N_CLASS)
            # (B, N_T, N_H, N_W, ts, ps, ps, c)
            # (0,   1,   2,   3,  4,  5,  6, 7)
            # to
            # (B,   c, N_T,  ts, N_H,ps,N_W,ps)
            # (0,   7,   1,   4,   2, 5,  3, 6)
            y_pred = y_pred.permute(0, 7, 1, 4, 2, 5, 3, 6)
            # (B, c, T, H, W)
            y_pred = y_pred.reshape(B, N_CLASS, N_T * encoder.module.tubelet_size, N_H * encoder.module.patch_size, N_W * encoder.module.patch_size)

            loss = criterion(y_pred, y)

            with torch.no_grad():
                seg = torch.argmax(y_pred, dim=1).detach().cpu()
                print('seg min, max', seg.min(), seg.max(), seg.sum())
                # print('Positive prediction', seg.sum())
                # print('label min, max', y.detach().min(), y.detach().max())
                # print('label positive prediction', y.detach().sum())
                # print("Pred type", y_pred.dtype, "label type", y.dtype)

            with torch.no_grad():
                dice_meter.update(F.softmax(y_pred, dim=1), y)

        if vis_prefix and not has_vised:
            import matplotlib.pyplot as plt
            # print("data['image'].shape", data['image'].shape)
            # print("data['label'][0]", data['label'].shape)
            # print("y_pred.detach().cpu().shape", y_pred.detach().cpu().shape)
            # print("torch.argmax(y_pred[0].detach().cpu(), dim=0, keepdim=True)", torch.argmax(y_pred[0].detach().cpu(), dim=0, keepdim=True).shape)

            visualize = tio.Subject(
                image1 = tio.ScalarImage(tensor=data['image'][0]),
                gt1 = tio.LabelMap(tensor=data['label'][0].unsqueeze(0)),
                pred1 = tio.LabelMap(tensor=torch.argmax(y_pred[0].detach().cpu(), dim=0, keepdim=True)),
                image2 = tio.ScalarImage(tensor=data['image'][1]),
                gt2 = tio.LabelMap(tensor=data['label'][1].unsqueeze(0)),
                pred2 = tio.LabelMap(tensor=torch.argmax(y_pred[1].detach().cpu(), dim=0,  keepdim=True)),
            )
            visualize.plot()
            plt.savefig(f'{vis_prefix}_vis.png')
            plt.close('all')
            plt.clf()
            plt.cla()
            has_vised = True

        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % 20 == 0:
            logger.info('[Iteration %5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, dice_meter.compute().item(), loss,
                           torch.cuda.max_memory_allocated() / 1024.**2))
    return dice_meter.compute()

def run_test_whole_volumne(
    device,
    encoder: DistributedDataParallel,
    decoder: AttentiveSegmentator,
    data_loader: torch.utils.data.DataLoader,
    prediction_aggregator: tio.inference.GridAggregator,
    label_aggregator: tio.inference.GridAggregator,
    use_bfloat16: bool,
    save_prediction_prefix:str = ''):
    '''
    This function runs evaluation on ONE volumne of torchio gridsampler
    Mentally, I am disabled right now, but Imma keep going...
    Jesus christ
    '''
    encoder.train(mode=False)
    decoder.train(mode=False)
    # Dice meter should be in CPU because torchio aggregator moves all the prediction there
    dice_meter = Dice(num_classes=decoder.module.num_classes, ignore_index=0).to(device=device)

    # Now the fun time :)
    for itr, data in enumerate(data_loader):
        logger.info(f'Testing iteration {itr} / {len(data_loader)}')
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            B, C, T, H, W = data['image'].shape
            N_T = T // encoder.module.tubelet_size
            N_H = H // encoder.module.patch_size
            N_W = W // encoder.module.patch_size
            N_CLASS = decoder.module.num_classes

            # Load data and put on GPU
            x = data['image'].to(device)
            locations = data['location']
            y = data['label'].long() # Don't need to put into GPU

            with torch.no_grad():
                # I don't know why amp would not convert this to float16, but yeah...
                if use_bfloat16:
                    x = x.to(torch.float16)
                outputs = encoder(x)
                outputs = decoder(outputs)

                y_pred = outputs.reshape(B, N_T, N_H, N_W, -1)
                y_pred = y_pred.reshape(B, N_T, N_H, N_W, encoder.module.tubelet_size, encoder.module.patch_size, encoder.module.patch_size, N_CLASS)
                # (B, N_T, N_H, N_W, ts, ps, ps, c)
                # (0,   1,   2,   3,  4,  5,  6, 7)
                # to
                # (B,   c, N_T,  ts, N_H,ps,N_W,ps)
                # (0,   7,   1,   4,   2, 5,  3, 6)
                y_pred = y_pred.permute(0, 7, 1, 4, 2, 5, 3, 6)
                # (B, c, T, H, W)
                y_pred = y_pred.reshape(B, N_CLASS, N_T * encoder.module.tubelet_size, N_H * encoder.module.patch_size, N_W * encoder.module.patch_size)

                y_pred = y_pred.argmax(dim=1, keepdim=True)
                y_pred = y_pred.permute(0, 1, 3, 4, 2)
                y = y.permute(0, 2, 3, 1).unsqueeze(dim=1)
                prediction_aggregator.add_batch(y_pred, locations=locations)
                label_aggregator.add_batch(y, locations=locations)

    predictions = prediction_aggregator.get_output_tensor()
    labels = label_aggregator.get_output_tensor()
    if save_prediction_prefix:
        prediction_path = f'{save_prediction_prefix}_prediction.pt'
        torch.save(predictions, prediction_path)
        label_path = f'{save_prediction_prefix}_label.pt'
        torch.save(labels, label_path)
        logger.info(f'Saved prediction and labels in {prediction_path} and {label_path}')
    predictions = torch.load(f'{save_prediction_prefix}_prediction.pt')
    labels = torch.load(f'{save_prediction_prefix}_label.pt')
    dice_meter.update(preds=predictions, target=labels)
    return dice_meter.compute()


def load_checkpoint(
    r_path,
    encoder,
    decoder,
    opt,
    scaler
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        # -- loading decoder
        pretrained_dict = checkpoint['decoder']
        msg = decoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, decoder, opt, scaler, epoch

def init_opt(
    encoder,
    decoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    freeze_encoder,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
):
    param_groups = [
        {
            'params': (p for n, p in decoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in decoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
        },
    ]
    if not freeze_encoder:
        encoder_param_groups = [
            {
                'params': (p for n, p in encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0,
            }, 
        ]
        param_groups += encoder_param_groups

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler

def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder',
    load_weight=True,
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    if load_weight:
        encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    else:
        logger.info('Not loading pre-trained weight')
    return encoder

def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder