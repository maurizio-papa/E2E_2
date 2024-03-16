from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision
import torchvision.transforms._transforms_video as transforms_video
from timm.data.loader import MultiEpochsDataLoader
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

from backbone.AVION.avion.data.clip_dataset import get_downstream_dataset
from backbone.AVION.avion.data.tokenizer import tokenize
from backbone.AVION.avion.data.transforms import Permute

import backbone.AVION.avion.models.model_clip as model_clip
from backbone.AVION.avion.models.utils import inflate_positional_embeds
from backbone.AVION.avion.optim.schedulers import cosine_scheduler
import backbone.AVION.avion.utils.distributed as dist_utils


def load_backbone(ckpt_path):

    ckpt = torch.load(ckpt_path, map_location='cpu')
    old_args = ckpt['args']

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print(f'creating model: {old_args.model}')

    model = getattr(model_clip, old_args.model)(
        pretrained=old_args.load_visual_pretrained,
        pretrained2d=old_args.load_visual_pretrained is not None,
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        timesformer_gated_xattn=False,
        timesformer_freeze_space=False,
        num_frames= 16,
        drop_path_rate= 0.1,
    )
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
        model.state_dict(), state_dict,
        num_frames= 16,
        load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(BASE_MODEL, ckpt['epoch']))


    model = model_clip.VideoClassifier(
            model.visual,
            dropout= old_args.dropout_rate,
            num_classes= old_args.num_classes
        )
    return model


def main():
    load_backbone('C:/Users/maurizio.papa/Downloads/avion_finetune_cls_lavila_vitb_best.pt')


if __name__ == '__main__':
    main()