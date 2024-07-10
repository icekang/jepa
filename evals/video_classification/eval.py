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

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.datasets.data_manager import (
    init_data,
)
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

from evals.video_classification.utils import (
    KFoldNNUNetTabularDataModule
)

from torchmetrics import Accuracy, Precision, Recall, F1Score

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


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
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    num_classes = args_data.get('num_classes')
    batch_size = args_eval.get('batch_size')

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')
    freeze_encoder = args_opt.get('freeze_encoder', True)

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'video_classification_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    best_path = os.path.join(folder, f'{tag}-best.pth.tar')

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%.5f', 'train_acc'),
                               ('%.5f', 'train_precision'),
                               ('%.5f', 'train_recall'),
                               ('%.5f', 'train_f1'),
                               ('%.5f', 'val_acc'),
                               ('%.5f', 'val_precision'),
                               ('%.5f', 'val_recall'),
                               ('%.5f', 'val_f1'),
                               )

    # Initialize model

    # -- pretrained encoder
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
        use_sdpa=use_sdpa)

    if freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)

    # -- init data-loaders/samplers
    data_module = KFoldNNUNetTabularDataModule(args_eval)
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    best_val_f1 = 0.0
    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        classifier=classifier,
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
    classifier = DistributedDataParallel(classifier, static_graph=True)

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        encoder, classifier, optimizer, scaler, start_epoch, best_val_f1 = load_checkpoint(
            device=device,
            r_path=latest_path,
            encoder=encoder,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path=latest_path):
        if rank != 0:
            return
        save_dict = {
            'classifier': classifier.state_dict(),
            'encoder': encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'best_val_f1': best_val_f1
        }
        if rank == 0:
            torch.save(save_dict, path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        train_acc, train_precision, train_recall, train_f1 = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            freeze_encoder=freeze_encoder,
            config=args_eval)

        val_acc, val_precision, val_recall, val_f1 = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            freeze_encoder=freeze_encoder,
            config=args_eval)

        logger.info('[%5d] train: %.3f%% test: %.3f%%' % (epoch + 1, train_acc, val_acc))
        logger.info(f'Accuracy: {train_acc:.5f}, Precision: {train_precision:.5f}, Recall: {train_recall:.5f}, F1: {train_f1:.5f}')
        logger.info(f'Val Accuracy: {val_acc:.5f}, Precision: {val_precision:.5f}, Recall: {val_recall:.5f}, F1: {val_f1:.5f}')
        if rank == 0:
            csv_logger.log(epoch + 1, 
                           train_acc,
                           train_precision,
                           train_recall,
                           train_f1,
                           val_acc,
                           val_precision,
                           val_recall,
                           val_f1)
        if val_f1 > best_val_f1:
            logger.info(f'New best F1: {val_f1:.5f} improved from {best_val_f1}, saving to {best_path}')
            best_val_f1 = val_f1
            save_checkpoint(epoch + 1, best_path)
        else:
            logger.info(f'F1: {val_f1:.5f} did not improve from {best_val_f1}')
        save_checkpoint(epoch + 1)
    
    # TEST LOOP
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    encoder, classifier, _, _, _, best_val_f1 = load_checkpoint(
        device=device,
        r_path=best_path,
        encoder=encoder,
        classifier=classifier,
        opt=optimizer,
        scaler=scaler)
    logger.info(f'Loaded best model with F1: {best_val_f1:.5f}')
    test_acc, test_precision, test_recall, test_f1 = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=test_loader,
            use_bfloat16=use_bfloat16,
            freeze_encoder=freeze_encoder,
            config=args_eval)
    logger.info(f'Test Accuracy: {test_acc:.5f}, Precision: {test_precision:.5f}, Recall: {test_recall:.5f}, F1: {test_f1:.5f}')


def run_one_epoch(
    device,
    training: bool,
    encoder: DistributedDataParallel,
    classifier: AttentiveClassifier,
    scaler: torch.cuda.amp.GradScaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16: bool,
    freeze_encoder: bool,
    config: dict
):
    if freeze_encoder:
        encoder.train(mode=False)
    else:
        encoder.train(mode=training)
    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()
    accurary = Accuracy(task='binary').to(device=device)
    precision = Precision(task='binary').to(device=device)
    recall = Recall(task='binary').to(device=device)
    f1 = F1Score(task='binary').to(device=device)

    training_zeros = 0
    training_ones = 0
    training_pred_zeros = 0
    training_pred_ones = 0
    for itr, batch in enumerate(data_loader):
        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            x = batch['image']
            x = x.to(device)
            y = torch.concat([batch[metric] for metric in config['data']['output_metrics']], dim=1)
            y = y.long().squeeze(dim=1)
            y = y.to(device)

            if not training:
                with torch.no_grad():
                    y_hat = encoder(x)
                    y_hat = classifier(y_hat)
            else:
                if freeze_encoder:
                    with torch.no_grad():
                        y_hat = encoder(x)
                else:
                    y_hat = encoder(x)
                y_hat = classifier(y_hat)

            loss = criterion(y_hat, y)
        
            with torch.no_grad():
                y_hat_index = torch.argmax(y_hat, dim=1)
                accurary.update(y_hat_index, y)
                precision.update(y_hat_index, y)
                recall.update(y_hat_index, y)
                f1.update(y_hat_index, y)
                training_zeros += torch.sum(y == 0).item()
                training_ones += torch.sum(y == 1).item()
                training_pred_zeros += torch.sum(y_hat_index == 0).item()
                training_pred_ones += torch.sum(y_hat_index == 1).item()
        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if not freeze_encoder:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if not freeze_encoder:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % 20 == 0:
            logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, accurary.compute(), loss,
                           torch.cuda.max_memory_allocated() / 1024.**2))
    print(f'[Label/Prediction] {"Training" if training else "Validation"} zeros: {training_zeros} / {training_pred_zeros}, Training ones: {training_ones} / {training_pred_ones}')
    return accurary.compute(), precision.compute(), recall.compute(), f1.compute()


def load_checkpoint(
    device,
    r_path,
    encoder,
    classifier,
    opt,
    scaler
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # reverse compatability
        best_val_f1 = 0.0
        if 'best_val_f1' in checkpoint:
            best_val_f1 = checkpoint['best_val_f1']
            logger.info(f'loaded best validation dice from epoch {epoch} with f1: {best_val_f1}')

        # -- loading encoder
        if 'encoder' in checkpoint:
            msg = encoder.load_state_dict(checkpoint['encoder'])
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        else:
            logger.info(f'No encoder found in checkpoint {r_path}')

        # -- loading encoder
        pretrained_dict = checkpoint['classifier']
        msg = classifier.load_state_dict(pretrained_dict)
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

    return encoder, classifier, opt, scaler, epoch, best_val_f1


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
    checkpoint_key='target_encoder'
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
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder


def init_opt(
    encoder,
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    freeze_encoder=True
):
    param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    if not freeze_encoder:
        logger.info('Unfreezing encoder...')
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
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
