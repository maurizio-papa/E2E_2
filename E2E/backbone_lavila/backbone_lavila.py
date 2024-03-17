from collections import OrderedDict
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from sklearn.metrics import confusion_matrix
import wandb

from .download_videos_and_convert_to_tensor.untar_img_and_convert_to_h5 import load_images_from_hdf5


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_TENSOR_DIR = 'tensor'
FEATURE_DIR = 'features_lavila'
BASE_MODEL =  'clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth'
FINETUNED_MODEL = 'clip_openai_timesformer_base.ft_ek100_cls.ep_0100.md5sum_4e3575.pth'

from .lavila.lavila.data import datasets
from .lavila.lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from .lavila.lavila.models import models
from .lavila.lavila.models.tokenizer import (MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer)
from .lavila.lavila.models.utils import inflate_positional_embeds
from .lavila.lavila.utils import distributed as dist_utils
from .lavila.lavila.utils.evaluation import accuracy, get_mean_accuracy
from .lavila.lavila.utils.meter import AverageMeter, ProgressMeter
from .lavila.lavila.utils.preprocess import generate_label_map
from .lavila.lavila.utils.random import random_seed
from .lavila.lavila.utils.scheduler import cosine_scheduler
from .lavila.lavila.utils.evaluation_ek100cls import get_marginal_indexes, marginalize
from .lavila.lavila.models.utils import inflate_positional_embeds
from .lavila.lavila.models import models

def load_backbone(BASE_MODEL):
    """
    loads pre-trained and then fine-tuned model,
    removes the last layer and return the fine-tuned model
    """
    ckpt = torch.load(BASE_MODEL, map_location='cpu')

    old_args = ckpt['args']

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print(f'creating model: {old_args.model}')

    model = getattr(models, old_args.model)(
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

    model = models.VideoClassifierMultiHead(
            model.visual,
            dropout= 0.0,
            num_classes_list = [97, 300, 3806]
        )
    return model 