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
        freeze_temperature=True,
        use_grad_checkpointing= False,
        context_length=old_args.context_length,
        vocab_size=old_args.vocab_size,
        patch_dropout= 0,
        num_frames= 16,
        drop_path_rate= 0.1,
        use_fast_conv1= False,
        use_flash_attn= False,
        use_quick_gelu=True,
        project_embed_dim=old_args.project_embed_dim,
        pretrain_zoo=old_args.pretrain_zoo,
        pretrain_path=old_args.pretrain_path,
    )
    model.logit_scale.requires_grad = False
    print('=> inflating PE in models due to different frame numbers')
    state_dict = inflate_positional_embeds(
        model.state_dict(), state_dict,
        num_frames= 16,
        load_temporal_fix='bilinear',
    )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))


    model = model_clip.VideoClassifier(
            model.visual,
            dropout= old_args.dropout_rate,
            num_classes= old_args.num_classes
        )
    return model


def main():
    load_backbone('')


if __name__ == '__main__':
    main()