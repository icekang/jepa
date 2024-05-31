import torchio as tio
import torch
from app.vjepa.utils import (
    init_video_model,
)
from src.models.attentive_pooler import AttentiveSegmentator
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import KFoldNNUNetSegmentationDataModule
import yaml
import lightning as pl
import segmentation_models_pytorch.losses as smp_losses

pl.seed_everything(42)

with open('configs/evals/kfold_segmentation.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

dm = KFoldNNUNetSegmentationDataModule(config)
dm.setup('fit')

data = next(iter(dm.train_dataloader()))

with open('configs/pretrain/vitl16.yaml', 'r') as y_file:
    args = yaml.load(y_file, Loader=yaml.FullLoader)

# -- set device
if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

# -- META
cfgs_meta = args.get('meta')
use_sdpa = cfgs_meta.get('use_sdpa', False)

# -- MODEL
cfgs_model = args.get('model')
model_name = cfgs_model.get('model_name')
pred_depth = cfgs_model.get('pred_depth')
pred_embed_dim = cfgs_model.get('pred_embed_dim')
uniform_power = cfgs_model.get('uniform_power', True)
use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

# -- MASK
cfgs_mask = args.get('mask')

# -- DATA
cfgs_data = args.get('data')
dataset_type = cfgs_data.get('dataset_type', 'videodataset')
mask_type = cfgs_data.get('mask_type', 'multiblock3d')
dataset_paths = cfgs_data.get('datasets', [])
datasets_weights = cfgs_data.get('datasets_weights', None)
if datasets_weights is not None:
    assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
batch_size = cfgs_data.get('batch_size')
batch_size = 1
num_clips = cfgs_data.get('num_clips')
num_frames = cfgs_data.get('num_frames')
tubelet_size = cfgs_data.get('tubelet_size')
sampling_rate = cfgs_data.get('sampling_rate')
duration = cfgs_data.get('clip_duration', None)
crop_size = cfgs_data.get('crop_size', 224)
patch_size = cfgs_data.get('patch_size')
pin_mem = cfgs_data.get('pin_mem', False)
num_workers = cfgs_data.get('num_workers', 1)
filter_short_videos = cfgs_data.get('filter_short_videos', False)
decode_one_clip = cfgs_data.get('decode_one_clip', True)
log_resource_util_data = cfgs_data.get('log_resource_utilization', False)
attend_across_segments = False
world_size = 1
rank = 0

encoder, _ = init_video_model(
    uniform_power=uniform_power,
    use_mask_tokens=use_mask_tokens,
    num_mask_tokens=len(cfgs_mask),
    zero_init_mask_tokens=zero_init_mask_tokens,
    device=device,
    patch_size=patch_size,
    num_frames=num_frames,
    tubelet_size=tubelet_size,
    model_name=model_name,
    crop_size=crop_size,
    pred_depth=pred_depth,
    pred_embed_dim=pred_embed_dim,
    use_sdpa=use_sdpa,
)

decoder = AttentiveSegmentator(
    img_size=encoder.backbone.input_size,
    patch_size=encoder.backbone.patch_size,
    num_frames=encoder.backbone.num_frames,
    tubelet_size=encoder.backbone.tubelet_size,
    encoder_embed_dim=encoder.backbone.embed_dim,
    decoder_embed_dim=768,
    depth=1, 
    num_heads=12, 
    mlp_ratio=4.0, 
    qkv_bias=True, 
    qk_scale=None, 
    drop_rate=0.0, 
    attn_drop_rate=0.0,
    norm_layer=nn.LayerNorm,
    init_std=0.02, 
    num_classes=2, 
).cuda()
x = data['image'].to(device)
y = data['label'].long().to(device)

B, C, T, H, W = data['image'].shape
N_T = T // encoder.backbone.tubelet_size
N_H = H // encoder.backbone.patch_size
N_W = W // encoder.backbone.patch_size
N_CLASS = decoder.num_classes

# Load the weight
checkpoint = torch.load('vitl16.pth.tar', map_location='cpu')
new_encoder_state_dict = {}
pretrained_dict = checkpoint['target_encoder']
pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
# pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
encoder.load_state_dict(pretrained_dict)

checkpoint = torch.load('decoder.pth', map_location='cpu')
decoder.load_state_dict(checkpoint)

optimizer = torch.optim.Adam(list(decoder.parameters()), lr=1e-4)
encoder.eval()
decoder.train()

dice_loss = smp_losses.DiceLoss('multiclass', from_logits=True)
for i in range(1001, 2001):
    optimizer.zero_grad()
    print(data['image'].shape, data['label'].shape)

    with torch.no_grad():
        encoded = encoder(x)
    decoded = decoder(encoded)
    reshaped_decoded = decoded.reshape(B, N_CLASS, N_T * tubelet_size, N_H * patch_size, N_W * patch_size)
    # print(y_one_hot.shape, reshaped_decoded.shape)
    loss = 0.25 * F.cross_entropy(reshaped_decoded, y) + 0.75 * dice_loss(reshaped_decoded, y.reshape(B, -1))
    loss.backward()
    optimizer.step()
    print(i, loss.item())
    if i % 100 == 0:
        torch.save(decoder.state_dict(), f'decoder.pth')
        B, C, T, H, W = reshaped_decoded.shape

        visualize = reshaped_decoded.detach().cpu().permute(0, 1, 3, 4, 2)
        visualize = torch.argmax(visualize, dim=1).unsqueeze(1)
        print('non-zero', torch.count_nonzero(visualize), 'zero', torch.count_nonzero(visualize == 0))
        gt_visualize = y.cpu().unsqueeze(1).permute(0, 1, 3, 4, 2)
        subject = tio.Subject(
            **{f'gt_{batch_index}': tio.data.LabelMap(tensor=gt_visualize[batch_index]) for batch_index in range(B)},
            **{f'label_{batch_index}': tio.data.LabelMap(tensor=visualize[batch_index]) for batch_index in range(B)},
        )
        # save the subject.plot
        subject.plot(figsize=(100, 10))
        fig = plt.gcf()
        fig.savefig(f'subject_iter_{i:03d}.png')
        #clear the figure
        plt.close('all')
