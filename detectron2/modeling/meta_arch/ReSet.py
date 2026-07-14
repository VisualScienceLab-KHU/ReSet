import math
import random
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast


from detectron2.config import configurable, get_cfg
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
import warnings
from ..matcher import Matcher
from .build import META_ARCH_REGISTRY


from detectron2.layers.roi_align import ROIAlign
from torchvision.ops.boxes import box_area, box_iou

from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.box_regression import Box2BoxTransform
from fvcore.nn import smooth_l1_loss

from lib.regionprop import augment_rois, region_coord_2_abs_coord, abs_coord_2_region_coord, SpatialIntegral
from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT


def elementwise_box_iou(boxes1, boxes2) -> Tuple[torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The input boxes should be in (x0, y0, x1, y1) format

    Args:
        boxes1: (torch.Tensor[N, 4]): first set of boxes
        boxes2: (torch.Tensor[M, 4]): second set of boxes

    Returns:
        torch.Tensor: a NxM pairwise matrix containing the pairwise Generalized IoU
        for every element in boxes1 and boxes2.
    """

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = elementwise_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


def interpolate(seq, T, mode='linear', force=False):
    if (seq.shape[-1] < T) or force:
        return F.interpolate(seq, T, mode=mode) 
    else:
        return seq[:, :, -T:]


class GatedMultiScaleConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.conv_dilated = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 5, out_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_dim // 2, 5, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv_fusion = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.conv7x7(x)
        x5 = self.conv_dilated(x)

        multi_scale = torch.cat([x1, x2, x3, x4, x5], dim=1)
        weights = self.scale_attention(multi_scale).view(-1, 5, 1, 1, 1)

        fused = (
            weights[:, 0] * x1 +
            weights[:, 1] * x2 +
            weights[:, 2] * x3 +
            weights[:, 3] * x4 +
            weights[:, 4] * x5
        )
        return self.conv_fusion(fused)


class DifferenceOperatorConv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.pdc(x, self.weight, self.bias, 1, self.padding, self.dilation, self.groups)


def build_difference_conv_operator(operator_type, theta=0.875):
    if operator_type == "standard":
        return F.conv2d
    if operator_type != "center_difference":
        raise ValueError(f"unknown operator type: {operator_type}")

    assert 0 < theta <= 1.0, "theta should be within (0, 1]"

    def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in [1, 2], "dilation for cd_conv should be in 1 or 2"
        assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for cd_conv should be 3x3"
        assert padding == dilation, "padding for cd_conv set wrong"

        weights_c = weights.sum(dim=[2, 3], keepdim=True) * theta
        yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y - yc

    return func


class ResidualDifferenceConvBlock(nn.Module):
    def __init__(self, operator_type, inplane, ouplane, theta=0.875):
        super().__init__()
        conv_func = build_difference_conv_operator(operator_type, theta)
        self.conv1 = DifferenceOperatorConv2d(
            conv_func, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x
        return y


class AdaptiveDifferenceFusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, theta=0.875):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.pdc_cv = ResidualDifferenceConvBlock(
            operator_type="standard", inplane=out_dim, ouplane=out_dim, theta=theta
        )
        self.pdc_cd = ResidualDifferenceConvBlock(
            operator_type="center_difference", inplane=out_dim, ouplane=out_dim, theta=theta
        )
        self.attention_fc = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, 2, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        diff_cv = self.pdc_cv(x)
        diff_cd = self.pdc_cd(x)
        diff_stack = torch.cat([diff_cv, diff_cd], dim=1)
        attention_weights = self.attention_fc(diff_stack)
        attention_weights = F.softmax(attention_weights, dim=1)
        diff_cv_weighted = diff_cv * attention_weights[:, 0:1, :, :]
        diff_cd_weighted = diff_cd * attention_weights[:, 1:2, :, :]
        return diff_cv_weighted + diff_cd_weighted


class PropagateNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.5,
                mask_temperature=0.1
    ):
        super().__init__()
        start_mask_dim = 0
        self.mask_temperature = mask_temperature

        self.main_layers = nn.ModuleList()
        self.mask_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            channels = input_dim if i == 0 else hidden_dim
            self.main_layers.append(nn.Sequential(
                nn.Conv2d(channels + start_mask_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ))

            self.mask_layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1))
            start_mask_dim += 1
        
        self.class_proj = nn.Linear(hidden_dim, 1)

    def forward(self, embedding):
        masks = []
        outputs = []
        for i in range(self.num_layers):
            if len(masks) > 0:
                embedding = torch.cat([embedding,] + masks, dim=1)
            
            embedding = self.main_layers[i](embedding)

            mask_logits = self.mask_layers[i](embedding) / self.mask_temperature
            mask_weights = mask_logits.sigmoid()
            masks.insert(0, mask_weights) 

            out = {}

            mask_weights = mask_weights / mask_weights.sum(dim=[2, 3], keepdim=True)
            latent = (embedding * mask_weights).sum(dim=[2, 3])
            
            latent = self.dropout(latent)
            
            out['class'] = self.class_proj(latent)
            outputs.append(out)
            

        results = [o['class'] for o in outputs]

        if not self.training:
            results = results[-1]
        return results


class AttentionPooling(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        attention_scores = self.attention_weights(x)
        weighted_features = x * attention_scores
        aggregated_features = weighted_features.sum(dim=1)
        output = self.fc(aggregated_features)
        return output, attention_scores


class PrototypeAggregation(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.attention_pool = AttentionPooling(c_in)

    def forward(self, x):
        attention_prototype, attention_scores = self.attention_pool(x)
        mean_prototype = x.mean(dim=1)
        out = 0.5 * F.normalize(attention_prototype, dim=-1) + 0.5 * F.normalize(mean_prototype, dim=-1)
        out = F.normalize(out, dim=-1)
        return out, attention_scores


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks



def _log_classification_stats(pred_logits, gt_classes):
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    try:
        storage = get_event_storage()
        storage.put_scalar("cls_acc", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fg_cls_acc", fg_num_accurate / num_fg)
            storage.put_scalar("false_neg_ratio", num_false_negative / num_fg)
    except:
        pass


def focal_loss(inputs, targets, gamma=0.5, reduction="mean", bg_weight=0.2, num_classes=None):
    """Inspired by RetinaNet implementation"""
    if targets.numel() == 0 and reduction == "mean":
        return inputs.sum() * 0.0
    
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    p = F.softmax(inputs, dim=-1)
    p_t = p[torch.arange(p.size(0)).to(p.device), targets]
    p_t = torch.clamp(p_t, 1e-7, 1-1e-7)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if bg_weight >= 0:
        assert num_classes is not None
        loss_weight = torch.ones(loss.size(0)).to(p.device)
        loss_weight[targets == num_classes] = bg_weight
        loss = loss * loss_weight

    if reduction == "mean":
        loss = loss.mean()

    return loss


def distance_embed(x, temperature = 10000, num_pos_feats = 128, scale=10.0):
    x = x[..., None]
    scale = 2 * math.pi * scale
    dim_t = torch.arange(num_pos_feats)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    sin_x = x * scale / dim_t.to(x.device)
    emb = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
    return emb



def box_cxcywh_to_xyxy(bbox) -> torch.Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.unbind(-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(new_bbox, dim=-1)


@META_ARCH_REGISTRY.register()
class OpenSetDetectorWithExamples(nn.Module):

    @property
    def device(self):
        return self.pixel_mean.device

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [((x / 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images with the configured pixel mean and std.
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs):
            height = input_per_image["height"]
            width = input_per_image["width"]
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    @configurable
    def __init__(self,
                offline_backbone: Backbone,
                backbone: Backbone,
                offline_proposal_generator: nn.Module, 

                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],

                offline_pixel_mean: Tuple[float],
                offline_pixel_std: Tuple[float],
                offline_input_format: Optional[str] = None,

                class_prototypes_file="",
                bg_prototypes_file="",
                semantic_mask="",
                semantic_mask_threshold_smoothness=2.0,
                roialign_size=7,
                proposal_matcher = None,

                box2box_transform=None,
                smooth_l1_beta=0.0,
                test_score_thresh=0.001,
                test_nms_thresh=0.5,
                test_topk_per_image=100,
                cls_temp=0.1,
                
                num_sample_class=-1,
                seen_cids = [],
                all_cids = [],
                T_length=128,
                
                bg_cls_weight=0.2,
                batch_size_per_image=128,
                pos_ratio=0.25,
                mult_rpn_score=False,
                num_cls_layers=3,
                rp_eval_chunks=2,
                use_one_shot= False,
                one_shot_reference= '',
                vit_feat_name=None
                ):
        super().__init__()
        if ',' in class_prototypes_file:
            class_prototypes_file = class_prototypes_file.split(',')
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.backbone = backbone
        self.bg_cls_weight = bg_cls_weight

        if np.sum(pixel_mean) < 3.0:
            self.div_pixel = True
        else:
            self.div_pixel = False

        self.input_format = "RGB"
        self.offline_backbone = offline_backbone
        self.offline_proposal_generator = offline_proposal_generator        
        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0:
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False
        
        self.proposal_matcher = proposal_matcher
        
        if isinstance(class_prototypes_file, str):
            dct = torch.load(class_prototypes_file)
            prototypes = dct['prototypes']
            if 'label_names' not in dct:
                if len(prototypes) != len(all_cids):
                    raise ValueError(
                        "class prototype files without label_names must match the configured class count"
                    )
                warnings.warn(
                    "label_names not found in class_prototypes_file; using the configured class order"
                )
                prototype_label_names = list(all_cids)
            else:
                prototype_label_names = dct['label_names']
        elif isinstance(class_prototypes_file, list):
            p1, p2 = torch.load(class_prototypes_file[0]), torch.load(class_prototypes_file[1])
            if 'origin_label_names' in p1 or 'origin_label_names' in p2:
                assert 'origin_label_names' in p2
                oneshot_num_classes = len(p2['origin_label_names'])
                oneshot_sample_pool = len(p2['label_names']) // oneshot_num_classes
                embed_size = p2['prototypes'].shape[-1]
                oneshot_prototypes = p2['prototypes'].reshape(oneshot_num_classes, oneshot_sample_pool, -1, embed_size)

                oneshot_prototypes = oneshot_prototypes[
                    torch.arange(oneshot_num_classes), 
                    torch.randint(0, oneshot_sample_pool, (oneshot_num_classes,))]

                prototypes = torch.cat([p1['prototypes'], oneshot_prototypes], dim=0)
                prototype_label_names = p1['label_names'] + p2['origin_label_names']
            else:
                prototypes = torch.cat([p1['prototypes'], p2['prototypes']], dim=0)
                prototype_label_names = p1['label_names'] + p2['label_names']
        else:
            raise NotImplementedError()

        proto_instances = prototypes if len(prototypes.shape) == 3 else None
        self.semantic_mask_enabled = bool(semantic_mask)
        self.semantic_mask_threshold_smoothness = semantic_mask_threshold_smoothness
        self.semantic_mask_info = None
        self.semantic_mask_threshold_raw = None
        self._semantic_mask_seen_map = None
        if self.semantic_mask_enabled:
            self._init_semantic_mask(
                semantic_mask,
                seen_cids,
                all_cids,
                prototypes.shape[-1],
            )
        else:
            self.semantic_mask_base_rank_max = None

        if proto_instances is not None:
            proto_vecs = proto_instances.mean(dim=1)
        else:
            proto_vecs = prototypes

        class_weights = F.normalize(proto_vecs, dim=-1)

        if proto_instances is not None:
            self.register_buffer("prototype_instances", proto_instances)
            self.num_proto_classes = proto_instances.shape[0]
            self.prototype_vecs = None
        else:
            self.prototype_instances = None
            self.num_proto_classes = proto_vecs.shape[0]
            self.register_buffer("prototype_vecs", proto_vecs)

        if self.semantic_mask_enabled and proto_instances is not None:
            self.prototype_aggregation = PrototypeAggregation(proto_vecs.shape[-1])
        else:
            self.prototype_aggregation = None
        
        self.num_classes = len(all_cids)

        for c in all_cids:
            if c not in prototype_label_names:
                prototype_label_names.append(c)
                class_weights = torch.cat([class_weights, torch.zeros(1, class_weights.shape[-1])], dim=0)
        self.num_missing_proto = len(prototype_label_names) - self.num_proto_classes
        
        train_class_order = [prototype_label_names.index(c) for c in seen_cids]
        test_class_order = [prototype_label_names.index(c) for c in all_cids]

        self.label_names = prototype_label_names

        assert -1 not in train_class_order and -1 not in test_class_order

        self.train_class_order = train_class_order
        self.register_buffer("train_class_weight", class_weights[torch.as_tensor(train_class_order)])
        self.register_buffer("test_class_weight", class_weights[torch.as_tensor(test_class_order)])
        self.test_class_order = test_class_order

        self.ndim = prototypes.shape[-1]
        
        self.all_labels = all_cids
        self.seen_labels = seen_cids

        bg_protos = torch.load(bg_prototypes_file)
        if isinstance(bg_protos, dict):
            bg_protos = bg_protos['prototypes']
        if len(bg_protos.shape) == 3:
            bg_protos = bg_protos.flatten(0, 1)

        self.bg_tokens = nn.Embedding(num_embeddings=len(bg_protos), embedding_dim=bg_protos.ndim).cuda()
        self.bg_tokens.weight = nn.Parameter(bg_protos)
        self.num_bg_tokens = len(bg_protos)




        self.roialign_size = roialign_size
        self.roi_align = ROIAlign(roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)

        self.app_dim = 128
        self.cls_app_compress = nn.Conv2d(self.ndim, self.app_dim, kernel_size=1, stride=1, padding=0)
        self.reg_app_compress = nn.Conv2d(self.ndim, self.app_dim, kernel_size=1, stride=1, padding=0)


        self.T = T_length
        self.Tpos_emb = 128
        self.Temb = 128
        self.Tbg_emb = 128
        hidden_dim = 256
        self.fc_intra_class = nn.Linear(self.Tpos_emb, self.Temb)
        self.fc_other_class = nn.Linear(self.T, self.Temb)
        self.fc_back_class = nn.Linear(self.num_bg_tokens, self.Tbg_emb)

        cls_input_dim = self.Temb * 2 + self.Tbg_emb + self.app_dim

        bg_input_dim = self.Temb + self.Tbg_emb
        
        self.per_cls_cnn = PropagateNet(cls_input_dim, hidden_dim, num_layers=num_cls_layers)
        self.bg_cnn = PropagateNet(bg_input_dim, hidden_dim, num_layers=num_cls_layers)

        self.fc_bg_class = nn.Linear(self.T, self.Temb)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        self.reg_roialign_size = 20
        self.reg_roi_align = ROIAlign(self.reg_roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)

        reg_in_dim = self.Temb * 2 + self.app_dim
        reg_hidden_dim = 256

        self.rp1 = GatedMultiScaleConvBlock(reg_in_dim + 1, reg_hidden_dim)
        self.rp1_out = nn.Conv2d(reg_hidden_dim, 1, 3, 1, 1)

        self.rp2 = GatedMultiScaleConvBlock(reg_hidden_dim + 2, reg_hidden_dim)
        self.rp2_out = nn.Conv2d(reg_hidden_dim, 1, 3, 1, 1)

        self.rp3 = GatedMultiScaleConvBlock(reg_hidden_dim + 3, reg_hidden_dim)
        self.rp3_out = nn.Conv2d(reg_hidden_dim, 1, 3, 1, 1)

        self.rp4 = GatedMultiScaleConvBlock(reg_hidden_dim + 4, reg_hidden_dim)
        self.rp4_out = nn.Conv2d(reg_hidden_dim, 1, 3, 1, 1)

        self.rp5 = GatedMultiScaleConvBlock(reg_hidden_dim + 5, reg_hidden_dim)
        self.rp5_out = nn.Conv2d(reg_hidden_dim, 1, 3, 1, 1)

        self.reg_semantic_proj0 = nn.Conv2d(self.ndim, reg_in_dim, kernel_size=1, stride=1, padding=0)
        self.reg_semantic_proj_rest = nn.ModuleList([
            nn.Conv2d(self.ndim, reg_hidden_dim, kernel_size=1, stride=1, padding=0)
            for _ in range(4)
        ])
        self.reg_semantic_dcfm0 = AdaptiveDifferenceFusionBlock(reg_in_dim + 1, reg_in_dim)
        self.reg_semantic_dcfm_rest = nn.ModuleList([
            AdaptiveDifferenceFusionBlock(reg_hidden_dim + 1, reg_hidden_dim)
            for _ in range(len(self.reg_semantic_proj_rest))
        ])
        self.reg_semantic_scale = nn.Parameter(torch.ones(5))

        self.reg_semantic_bins = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 23)]
        self.reg_semantic_bin_order = [4, 3, 2, 1, 0]
        bin_sizes = [(b1 - b0 + 1) for (b0, b1) in self.reg_semantic_bins]
        self.reg_semantic_w = nn.ParameterList([
            nn.Parameter(torch.zeros(s)) for s in bin_sizes
        ])
        self.reg_semantic_bn0 = nn.BatchNorm2d(reg_in_dim)
        self.reg_semantic_bn_rest = nn.ModuleList([nn.BatchNorm2d(reg_hidden_dim) for _ in range(4)])
        self.reg_semantic_ln_eps = 1e-6


        self.r2c = SpatialIntegral(self.reg_roialign_size)

        self.reg_intra_dist_emb = nn.Linear(self.Tpos_emb, self.Temb)
        self.reg_bg_dist_emb = nn.Linear(self.num_bg_tokens, self.Temb)

        self.cls_temp = cls_temp
        self.num_sample_class = num_sample_class
        self.batch_size_per_image = batch_size_per_image
        self.pos_ratio = pos_ratio
        self.mult_rpn_score = mult_rpn_score

        self.use_one_shot = use_one_shot

        self.one_shot_ref = None

        if use_one_shot:
            self.one_shot_ref = torch.load(one_shot_reference)
        
        self.vit_feat_name = vit_feat_name
        self.rp_eval_chunks = max(int(rp_eval_chunks), 1)

    def _load_semantic_mask(self, semantic_mask, seen_cids, all_cids):
        try:
            with open(semantic_mask, "r") as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"failed to load semantic_mask: {semantic_mask}") from exc

        if "base" not in data or "novel" not in data:
            raise ValueError(f"semantic_mask missing base/novel sections: {semantic_mask}")

        def parse_section(section, section_name):
            if not isinstance(section, dict):
                raise ValueError(f"semantic_mask.{section_name} must be a dict: {semantic_mask}")
            parsed = {}
            for cls_name, cls_info in section.items():
                if not isinstance(cls_info, dict):
                    raise ValueError(f"semantic_mask.{section_name}.{cls_name} must be a dict")
                indices = cls_info.get("indices", {})
                items = []
                if isinstance(indices, dict):
                    for k, v in indices.items():
                        idx = int(k)
                        if isinstance(v, dict):
                            score_val = v.get("score", 0.0)
                        else:
                            score_val = v
                        try:
                            score = float(score_val)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"semantic_mask.{section_name}.{cls_name}.indices[{k}] has invalid score"
                            ) from exc
                        items.append((idx, score))
                elif isinstance(indices, list):
                    items = [(int(v), 0.0) for v in indices]
                else:
                    raise ValueError(
                        f"semantic_mask.{section_name}.{cls_name}.indices must be a dict or list"
                    )
                parsed[cls_name] = items
            return parsed

        base_map = parse_section(data["base"], "base")
        novel_map = parse_section(data["novel"], "novel")

        seen_set = set(seen_cids)
        all_set = set(all_cids)
        novel_set = all_set - seen_set

        missing_base = [c for c in seen_cids if c not in base_map]
        missing_novel = [c for c in novel_set if c not in novel_map]
        extra_base = [c for c in base_map if c not in seen_set]
        extra_novel = [c for c in novel_map if c not in novel_set]

        if missing_base or missing_novel or extra_base or extra_novel:
            parts = [
                "semantic_mask class mismatch",
                f"path={semantic_mask}",
            ]
            if missing_base:
                parts.append(f"missing_base={missing_base}")
            if missing_novel:
                parts.append(f"missing_novel={missing_novel}")
            if extra_base:
                parts.append(f"extra_base={extra_base}")
            if extra_novel:
                parts.append(f"extra_novel={extra_novel}")
            raise ValueError("; ".join(parts))

        return base_map, novel_map

    def _init_semantic_mask(
        self,
        semantic_mask,
        seen_cids,
        all_cids,
        embed_dim,
    ):
        base_map, novel_map = self._load_semantic_mask(semantic_mask, seen_cids, all_cids)
        self.semantic_mask_info = {}

        def build_info(label, items):
            if not items:
                return {
                    "indices": None,
                    "ranks": None,
                    "max_rank": 0.0,
                    "count": 0,
                    "init_rank": 0.0,
                }
            items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))
            indices = [idx for idx, _ in items_sorted]
            if len(indices) != len(set(indices)):
                raise ValueError(f"semantic_mask has duplicate indices for class {label}")
            bad = [i for i in indices if i < 0 or i >= embed_dim]
            if bad:
                raise ValueError(
                    f"semantic_mask index out of range for class {label}: {bad}"
                )
            init_rank = len(items_sorted) - 1

            return {
                "indices": torch.as_tensor(indices, dtype=torch.long),
                "ranks": torch.arange(len(indices), dtype=torch.float32),
                "max_rank": float(len(indices) - 1),
                "count": len(indices),
                "init_rank": float(init_rank),
            }

        for label, items in base_map.items():
            self.semantic_mask_info[label] = build_info(label, items)
        for label, items in novel_map.items():
            self.semantic_mask_info[label] = build_info(label, items)

        base_rank_max = []
        for label in seen_cids:
            info = self.semantic_mask_info.get(label)
            if info is None:
                raise ValueError(f"semantic_mask missing class {label}")
            base_rank_max.append(info["max_rank"] if info["count"] > 0 else 0.0)

        self.register_buffer(
            "semantic_mask_base_rank_max",
            torch.as_tensor(base_rank_max, dtype=torch.float32),
        )

        threshold_raw = []
        eps = 1e-6
        for label, max_rank in zip(seen_cids, base_rank_max):
            if max_rank <= 0:
                threshold_raw.append(0.0)
                continue
            info = self.semantic_mask_info[label]
            threshold_init = min(info["init_rank"], max_rank)
            ratio = threshold_init / max_rank
            ratio = min(max(ratio, eps), 1.0 - eps)
            threshold_raw.append(math.log(ratio / (1.0 - ratio)))

        self.semantic_mask_threshold_raw = nn.Parameter(
            torch.as_tensor(threshold_raw, dtype=torch.float32)
        )
        self._semantic_mask_seen_map = {label: i for i, label in enumerate(seen_cids)}

    def _semantic_mask_base_threshold(self, device, dtype):
        if self.semantic_mask_threshold_raw is None:
            return None, None
        base_rank_max = self.semantic_mask_base_rank_max.to(device=device, dtype=dtype)
        threshold_values = base_rank_max * torch.sigmoid(
            self.semantic_mask_threshold_raw.to(device=device, dtype=dtype)
        )
        valid = base_rank_max > 0
        if valid.any():
            base_mean = threshold_values[valid].mean()
        else:
            base_mean = threshold_values.new_tensor(0.0)
        return threshold_values, base_mean

    def _sanitize_class_name(self, name):
        safe = []
        for ch in name:
            if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch in ("_", "-"):
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)

    def _log_semantic_mask_threshold(self):
        if not self.training or not self.semantic_mask_enabled or self.semantic_mask_info is None:
            return
        storage = get_event_storage()
        if storage is None:
            return
        if getattr(self, "_semantic_mask_threshold_log_iter", None) == storage.iter:
            return
        self._semantic_mask_threshold_log_iter = storage.iter

        threshold_values, base_mean = self._semantic_mask_base_threshold(
            self.semantic_mask_base_rank_max.device, self.semantic_mask_base_rank_max.dtype
        )
        if threshold_values is None:
            return
        threshold_values_cpu = threshold_values.detach().float().cpu()
        base_mean_val = float(base_mean.detach().float().cpu().item())
        storage.put_scalar("semantic_mask/threshold/base_mean", base_mean_val)

        for label in self.seen_labels:
            info = self.semantic_mask_info.get(label)
            if not info or info["count"] == 0:
                continue
            idx = self._semantic_mask_seen_map[label]
            key = f"semantic_mask/threshold/base/{self._sanitize_class_name(label)}"
            storage.put_scalar(key, float(threshold_values_cpu[idx].item()))

        for label in self.all_labels:
            if label in self._semantic_mask_seen_map:
                continue
            info = self.semantic_mask_info.get(label)
            if not info or info["count"] == 0:
                continue
            threshold_value = min(base_mean_val, info["max_rank"])
            key = f"semantic_mask/threshold/novel/{self._sanitize_class_name(label)}"
            storage.put_scalar(key, float(threshold_value))

    def _apply_semantic_mask(
        self,
        proto_vecs,
        prototype_label_names,
    ):
        if not self.semantic_mask_enabled or self.semantic_mask_info is None:
            return proto_vecs
        device = proto_vecs.device
        dtype = proto_vecs.dtype
        threshold_values, base_mean = self._semantic_mask_base_threshold(device, dtype)
        threshold_smoothness = self.semantic_mask_threshold_smoothness
        embed_dim = proto_vecs.shape[-1]

        mask_rows = []
        for label in prototype_label_names:
            info = self.semantic_mask_info.get(label)
            if not info or info["count"] == 0:
                mask_rows.append(torch.ones(embed_dim, dtype=dtype, device=device))
                continue
            indices = info["indices"].to(device=device)
            ranks = info["ranks"].to(device=device, dtype=dtype)
            if label in self._semantic_mask_seen_map:
                threshold = threshold_values[self._semantic_mask_seen_map[label]]
            else:
                threshold = torch.minimum(base_mean, proto_vecs.new_tensor(info["max_rank"]))
            mask_vals = torch.sigmoid((threshold - ranks) / threshold_smoothness)
            row = torch.zeros(embed_dim, dtype=dtype, device=device)
            row = row.scatter(0, indices, mask_vals)
            mask_rows.append(row)

        mask = torch.stack(mask_rows, dim=0)
        if proto_vecs.dim() == 3:
            mask = mask[:, None, :]
        return proto_vecs * mask

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_semantic_keys = {
            f"{prefix}content_mask_theta_raw": f"{prefix}semantic_mask_threshold_raw",
            f"{prefix}content_mask_base_rank_max": f"{prefix}semantic_mask_base_rank_max",
        }
        for old_key, new_key in legacy_semantic_keys.items():
            if old_key in state_dict:
                if self.semantic_mask_enabled and new_key not in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]
        if not self.semantic_mask_enabled:
            state_dict.pop(f"{prefix}semantic_mask_threshold_raw", None)
            state_dict.pop(f"{prefix}semantic_mask_base_rank_max", None)

        old_prefix = f"{prefix}learnable_threshold_module."
        new_prefix = f"{prefix}prototype_aggregation."
        for key in list(state_dict):
            if key.startswith(old_prefix):
                if self.prototype_aggregation is not None:
                    suffix = key[len(old_prefix):]
                    if suffix.startswith("soft_attn."):
                        suffix = f"attention_pool.{suffix[len('soft_attn.'):]}"
                    new_key = f"{new_prefix}{suffix}"
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict[key]
                del state_dict[key]
            elif self.prototype_aggregation is None and key.startswith(new_prefix):
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    @classmethod
    def from_config(cls, cfg):
        offline_cfg = get_cfg()
        offline_cfg.merge_from_file(cfg.DE.OFFLINE_RPN_CONFIG)
        if cfg.DE.OFFLINE_RPN_LSJ_PRETRAINED:
            offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0
            offline_cfg.MODEL.RESNETS.NORM = "BN"
            offline_cfg.MODEL.FPN.NORM = "BN"
            offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1]
        if cfg.DE.OFFLINE_RPN_NMS_THRESH:
            offline_cfg.MODEL.RPN.NMS_THRESH = cfg.DE.OFFLINE_RPN_NMS_THRESH
        if cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST:
            offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST

        offline_backbone = build_backbone(offline_cfg)
        offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())

        for p in offline_backbone.parameters(): p.requires_grad = False
        for p in offline_rpn.parameters(): p.requires_grad = False
        offline_backbone.eval()
        offline_rpn.eval()

        backbone = build_backbone(cfg)
        for p in backbone.parameters(): p.requires_grad = False
        backbone.eval()

        if cfg.DE.OUT_INDICES:
            out_indices = cfg.DE.OUT_INDICES
            if isinstance(out_indices, str):
                out_indices = [int(x) for x in out_indices.split(",")]
            vit_feat_name = f"res{out_indices[-1]}" if out_indices else f"res{backbone.n_blocks-1}"
        else:
            vit_feat_name = f'res{backbone.n_blocks - 1}'

        return {
            "backbone": backbone,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "class_prototypes_file": cfg.DE.CLASS_PROTOTYPES,
            "bg_prototypes_file": cfg.DE.BG_PROTOTYPES,
            "semantic_mask": cfg.DE.SEMANTIC_MASK,
            "semantic_mask_threshold_smoothness": cfg.DE.SEMANTIC_MASK_THRESHOLD_SMOOTHNESS,

            "roialign_size": cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,

            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
            
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),

            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,

            "cls_temp": cfg.DE.TEMP,
            
            "num_sample_class": cfg.DE.TOPK,
            
            
            "seen_cids": SEEN_CLS_DICT[cfg.DATASETS.TRAIN[0]],
            "all_cids": ALL_CLS_DICT[cfg.DATASETS.TRAIN[0]],
            "T_length": cfg.DE.T,
            
            "bg_cls_weight": cfg.DE.BG_CLS_LOSS_WEIGHT,
            "batch_size_per_image": cfg.DE.RCNN_BATCH_SIZE,
            "pos_ratio": cfg.DE.POS_RATIO,
            
            "mult_rpn_score": cfg.DE.MULTIPLY_RPN_SCORE,

            "num_cls_layers": cfg.DE.NUM_CLS_LAYERS,
            "rp_eval_chunks": cfg.DE.RP_EVAL_CHUNKS,
            
            "use_one_shot": cfg.DE.ONE_SHOT_MODE,
            "one_shot_reference": cfg.DE.ONE_SHOT_REFERENCE,
            
            "vit_feat_name": vit_feat_name
        }
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        bs = len(batched_inputs)
        loss_dict = {}
        if not self.training: assert bs == 1
        self._log_semantic_mask_threshold()

        class_weights_train = None
        class_weights_test = None
        if self.semantic_mask_enabled:
            proto_instances = self.prototype_instances
            if proto_instances is not None:
                proto_instances = self._apply_semantic_mask(
                    proto_instances, self.label_names[:self.num_proto_classes]
                )
                proto_vecs = proto_instances.mean(dim=1)
            else:
                proto_vecs = self.prototype_vecs
                proto_vecs = self._apply_semantic_mask(
                    proto_vecs, self.label_names[:self.num_proto_classes]
                )

            if self.prototype_aggregation is not None:
                aggregated_all, _ = self.prototype_aggregation(proto_instances)
                class_weights_all = F.normalize(aggregated_all, dim=-1)
            else:
                class_weights_all = F.normalize(proto_vecs, dim=-1)

            if self.num_missing_proto > 0:
                zeros = class_weights_all.new_zeros(self.num_missing_proto, class_weights_all.shape[-1])
                class_weights_all = torch.cat([class_weights_all, zeros], dim=0)
            train_idx = torch.as_tensor(self.train_class_order, device=class_weights_all.device)
            test_idx = torch.as_tensor(self.test_class_order, device=class_weights_all.device)
            class_weights_train = class_weights_all[train_idx]
            class_weights_test = class_weights_all[test_idx]
        else:
            class_weights_train = self.train_class_weight
            class_weights_test = self.test_class_weight

        if self.training:
            class_weights = class_weights_train
        else:
            if self.use_one_shot:
                class_weights = []
                for c in self.all_labels:
                    if c in self.seen_labels:
                        class_weights.append(class_weights_train[self.seen_labels.index(c)].cpu())
                    else:
                        token = random.choice(self.one_shot_ref[c])[1]
                        if self.semantic_mask_enabled:
                            token = self._apply_semantic_mask(token[None, :], [c])[0]
                        class_weights.append(token)

                class_weights = F.normalize(torch.stack(class_weights), dim=-1)
                class_weights = class_weights.to(self.device)
            else:
                class_weights = class_weights_test

        num_classes = len(class_weights)

        with torch.no_grad():
            if self.offline_backbone.training or self.offline_proposal_generator.training:  
                self.offline_backbone.eval() 
                self.offline_proposal_generator.eval()  
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            proposals, _ = self.offline_proposal_generator(images, features, None)     
            images = self.preprocess_image(batched_inputs)
        
        with torch.no_grad():
            if self.backbone.training: self.backbone.eval()
            with autocast(enabled=True):
                all_patch_tokens = self.backbone(images.tensor)
                patch_tokens = all_patch_tokens[self.vit_feat_name]

        if self.training or self.use_one_shot: 
            with torch.no_grad():
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

                boxes = [x.proposal_boxes.tensor for x in proposals]

                class_labels = []
                matched_gt_boxes = []
                resampled_proposals = []

                num_bg_samples, num_fg_samples = [], []

                for proposals_per_image, targets_per_image in zip(boxes, gt_instances):
                    match_quality_matrix = box_iou(
                        targets_per_image.gt_boxes.tensor, proposals_per_image
                    )
                    matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                    if len(targets_per_image.gt_classes) > 0:
                        class_labels_i = targets_per_image.gt_classes[matched_idxs]
                    else:
                        assert torch.all(matched_labels == 0)
                        class_labels_i = torch.zeros_like(matched_idxs)
                    class_labels_i[matched_labels == 0] = num_classes
                    class_labels_i[matched_labels == -1] = -1
                    
                    if self.training:
                        positive = ((class_labels_i != -1) & (class_labels_i != num_classes)).nonzero().flatten()
                        negative = (class_labels_i == num_classes).nonzero().flatten()

                        batch_size_per_image = self.batch_size_per_image
                        num_pos = int(batch_size_per_image * self.pos_ratio)
                        num_pos = min(positive.numel(), num_pos)
                        num_neg = batch_size_per_image - num_pos
                        num_neg = min(negative.numel(), num_neg)

                        perm1 = torch.randperm(positive.numel(), device=self.device)[:num_pos]
                        perm2 = torch.randperm(negative.numel())[:num_neg].to(self.device)
                        pos_idx = positive[perm1]
                        neg_idx = negative[perm2]
                        sampled_idxs = torch.cat([pos_idx, neg_idx], dim=0)
                    else:
                        sampled_idxs = torch.arange(len(proposals_per_image), device=self.device).long()

                    proposals_per_image = proposals_per_image[sampled_idxs]
                    class_labels_i = class_labels_i[sampled_idxs]
                    
                    if len(targets_per_image.gt_boxes.tensor) > 0:
                        gt_boxes_i = targets_per_image.gt_boxes.tensor[matched_idxs[sampled_idxs]]
                    else:
                        gt_boxes_i = torch.zeros(len(sampled_idxs), 4, device=self.device)

                    resampled_proposals.append(proposals_per_image)
                    class_labels.append(class_labels_i)
                    matched_gt_boxes.append(gt_boxes_i)

                    num_bg_samples.append((class_labels_i == num_classes).sum().item())
                    num_fg_samples.append(class_labels_i.numel() - num_bg_samples[-1])
                
                if self.training:
                    storage = get_event_storage()
                    storage.put_scalar("fg_count", np.mean(num_fg_samples))
                    storage.put_scalar("bg_count", np.mean(num_bg_samples))

                class_labels = torch.cat(class_labels)
                matched_gt_boxes = torch.cat(matched_gt_boxes)
                
                rois = []
                for bid, box in enumerate(resampled_proposals):
                    batch_index = torch.full((len(box), 1), fill_value=float(bid)).to(self.device) 
                    rois.append(torch.cat([batch_index, box], dim=1))
                rois = torch.cat(rois)
        else:
            boxes = proposals[0].proposal_boxes.tensor 
            rois = torch.cat([torch.full((len(boxes), 1), fill_value=0).to(self.device) , 
                            boxes], dim=1)

        roi_features_4d = self.roi_align(patch_tokens, rois)
        roi_bs = len(roi_features_4d)

        roi_features = roi_features_4d.flatten(2)

        bs, spatial_size = roi_features.shape[0], roi_features.shape[-1]
        roi_features_proj = F.normalize(roi_features, dim=-1)
        class_weights_proj = F.normalize(class_weights, dim=-1)
        feats = roi_features_proj.transpose(-2, -1) @ class_weights_proj.T

        class_topk = self.num_sample_class
        class_indices = None
        if class_topk < 0:
            class_topk = num_classes
            sample_class_enabled = False
        else:
            if class_topk == 0:
                class_topk = num_classes
            sample_class_enabled = True

        if sample_class_enabled:
            num_active_classes = class_topk
            init_scores = F.normalize(roi_features_proj.flatten(2).mean(2), dim=1) @ class_weights_proj.T
            topk_class_indices = torch.topk(init_scores, class_topk, dim=1).indices

            if self.training:
                class_indices = []
                for i in range(roi_bs):
                    curr_label = class_labels[i].item()
                    topk_class_indices_i = topk_class_indices[i].cpu()
                    if curr_label in topk_class_indices_i or curr_label == num_classes:
                        curr_indices = topk_class_indices_i
                    else:
                        curr_indices = torch.cat([torch.as_tensor([curr_label]),
                                            topk_class_indices_i[:-1]])
                    class_indices.append(curr_indices)
                class_indices = torch.stack(class_indices).to(self.device)
            else:
                class_indices = topk_class_indices
            
            class_indices = torch.sort(class_indices, dim=1).values
        else:
            num_active_classes = num_classes

        other_classes = []
        if sample_class_enabled:
            indexes = torch.arange(0, num_classes, device=self.device)[None, None, :].repeat(bs, spatial_size, 1)
            for i in range(class_topk):
                cmask = indexes != class_indices[:, i].view(-1, 1, 1)
                _ = torch.gather(feats, 2, indexes[cmask].view(bs, spatial_size, num_classes - 1))
                other_classes.append(_[:, :, None, :])
        else:
            for c in range(num_classes):
                cmask = torch.ones(num_classes, device=self.device, dtype=torch.bool)
                cmask[c] = False
                _ = feats[:, :, cmask]
                other_classes.append(_[:, :, None, :])

        other_classes = torch.cat(other_classes, dim=2)
        other_classes = other_classes.permute(0, 2, 1, 3)
        other_classes = other_classes.flatten(0, 1)
        other_classes, _ = torch.sort(other_classes, dim=-1)
        other_classes = interpolate(other_classes, self.T, mode='linear')
        other_classes = self.fc_other_class(other_classes)
        other_classes = other_classes.permute(0, 2, 1)
        inter_dist_emb = other_classes.reshape(bs * num_active_classes, -1, self.roialign_size, self.roialign_size)

        intra_feats = torch.gather(feats, 2, class_indices[:, None, :].repeat(1, spatial_size, 1)) if sample_class_enabled else feats
        intra_dist_emb = distance_embed(intra_feats.flatten(0, 1), num_pos_feats=self.Tpos_emb)
        intra_dist_emb = self.fc_intra_class(intra_dist_emb)
        intra_dist_emb = self.fc_intra_class(intra_dist_emb)
        intra_dist_emb = intra_dist_emb.reshape(bs, spatial_size, num_active_classes, -1)

        intra_dist_emb = intra_dist_emb.permute(0, 2, 3, 1).flatten(0, 1).reshape(bs * num_active_classes, -1,
                                                                                self.roialign_size, self.roialign_size)

        bg_feats = roi_features_proj.transpose(-2, -1) @ self.bg_tokens.weight.t()
        bg_dist_emb = self.fc_back_class(bg_feats)
        bg_dist_emb = bg_dist_emb.permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)

        bg_dist_emb_c = bg_dist_emb[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)

        roi_app = self.cls_app_compress(roi_features_4d)
        roi_app_c = roi_app[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)

        per_cls_input = torch.cat([intra_dist_emb, inter_dist_emb, bg_dist_emb_c, roi_app_c], dim=1)


        cls_logits = self.per_cls_cnn(per_cls_input)

        if isinstance(cls_logits, list):
            cls_logits = [v.reshape(bs, num_active_classes) for v in cls_logits]
        else:
            cls_logits = cls_logits.reshape(bs, num_active_classes)

        cls_dist_feats = interpolate(torch.sort(feats, dim=2).values, self.T, mode='linear')
        bg_cls_dist_emb = self.fc_bg_class(cls_dist_feats)
        bg_cls_dist_emb = bg_cls_dist_emb.permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)
        bg_logits = self.bg_cnn(torch.cat([bg_cls_dist_emb, bg_dist_emb], dim=1))

        if isinstance(bg_logits, list):
            logits = []
            for c,b in zip(cls_logits, bg_logits):
                logits.append(torch.cat([c, b], dim=1) / self.cls_temp)
        else:
            logits = torch.cat([cls_logits, bg_logits], dim=1)
            logits = logits / self.cls_temp
        H,W = images.tensor.shape[2:]
        if self.training:
            fg_indices = class_labels != num_classes
            matched_gt_boxes = matched_gt_boxes[fg_indices]
            fg_proposals = rois[fg_indices, 1:]
            fg_batch_inds = rois[fg_indices, :1]
            fg_class_labels = class_labels[fg_indices]

            reg_bs = len(fg_proposals)
            aug_rois, pred_roi_mask, gt_roi_mask, covered_flag = augment_rois(fg_proposals, matched_gt_boxes, img_h=H, img_w=W, pooler_size=self.reg_roialign_size,
                        min_expansion=0.4, expand_shortest=True)
            aug_rois = torch.cat([fg_batch_inds, aug_rois], dim=1)
            gt_region_coords = abs_coord_2_region_coord(aug_rois[:, 1:], matched_gt_boxes, self.reg_roialign_size)

            storage = get_event_storage()
            storage.put_scalar("roi_cover_ratio", covered_flag.sum().item() / covered_flag.numel())
        else:
            reg_bs = len(rois)
            aug_rois, pred_roi_mask, _, _ = augment_rois(rois[:, 1:], None, img_h=H, img_w=W, pooler_size=self.reg_roialign_size,
                        min_expansion=0.4, expand_shortest=True)
            aug_rois = torch.cat([rois[:, :1], aug_rois], dim=1)

        aroi_feats_4d = self.reg_roi_align(patch_tokens, aug_rois)
        aroi_feats = aroi_feats_4d.flatten(2)
        aroi_app = self.reg_app_compress(aroi_feats_4d)
        aroi_feats_proj = F.normalize(aroi_feats, dim=-1)
        class_weights_proj = F.normalize(class_weights, dim=-1)

        agg_maps = []
        for bi, (b0, b1) in enumerate(self.reg_semantic_bins):
            feats = []
            for li in range(b0, b1 + 1):
                key = f"res{li}"
                x = all_patch_tokens[key]

                x_nhwc = x.permute(0, 2, 3, 1)
                x_nhwc = F.layer_norm(x_nhwc, (x_nhwc.shape[-1],), eps=self.reg_semantic_ln_eps)
                x = x_nhwc.permute(0, 3, 1, 2)
                feats.append(x)
            stack = torch.stack(feats, dim=0)

            w = F.softmax(self.reg_semantic_w[bi], dim=0).view(-1, 1, 1, 1, 1)
            agg = (stack * w).sum(dim=0)
            agg_maps.append(agg)

        inj_maps = [agg_maps[j] for j in self.reg_semantic_bin_order]
        sem_roi_feats_4d = [self.reg_roi_align(m, aug_rois) for m in inj_maps]
        sem_roi_mask = pred_roi_mask[:, None, :, :].float()
        sem_cond_0 = self.reg_semantic_proj0(sem_roi_feats_4d[0])
        sem_cond_0 = self.reg_semantic_dcfm0(torch.cat([sem_cond_0, sem_roi_mask], dim=1))
        sem_cond_0 = self.reg_semantic_bn0(sem_cond_0)

        sem_cond_rest = []
        for f, proj, dcfm, bn in zip(sem_roi_feats_4d[1:], self.reg_semantic_proj_rest, self.reg_semantic_dcfm_rest, self.reg_semantic_bn_rest):
            y = proj(f)
            y = dcfm(torch.cat([y, sem_roi_mask], dim=1))
            y = bn(y)
            sem_cond_rest.append(y)

        sem_conds = [sem_cond_0] + sem_cond_rest

        if not self.training:
            sem_conds = [
                c[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)
                for c in sem_conds
            ]
        bg_aroi_feats = aroi_feats_proj.transpose(-2, -1) @ self.bg_tokens.weight.t()
        bg_aroi_emb = self.reg_bg_dist_emb(bg_aroi_feats)

        fg_aroi_feats = aroi_feats_proj.transpose(-2, -1) @ class_weights_proj.T
        K2 = self.reg_roialign_size ** 2

        if self.training:
            bg_aroi_emb = bg_aroi_emb.permute(0, 2, 1).reshape(reg_bs, self.Temb, self.reg_roialign_size, self.reg_roialign_size)
            fg_aroi_feats = torch.gather(fg_aroi_feats, 2, fg_class_labels[..., None, None].repeat(1, K2, 1))[:, :, 0]

            fg_aroi_emb = distance_embed(fg_aroi_feats, num_pos_feats=self.Tpos_emb)
            fg_aroi_emb = self.reg_intra_dist_emb(fg_aroi_emb)
            fg_aroi_emb = fg_aroi_emb.permute(0, 2, 1).reshape(reg_bs, self.Temb,
                                                            self.reg_roialign_size, self.reg_roialign_size)
            aroi_emb = torch.cat([fg_aroi_emb, bg_aroi_emb], dim=1)
        else:
            fg_aroi_dist_feats = torch.gather(fg_aroi_feats, 2, class_indices[:, None, :].repeat(1, K2, 1)) if sample_class_enabled else fg_aroi_feats
            fg_aroi_emb = distance_embed(fg_aroi_dist_feats.flatten(0, 1), num_pos_feats=self.Tpos_emb)
            fg_aroi_emb = self.reg_intra_dist_emb(fg_aroi_emb)
            fg_aroi_emb = fg_aroi_emb.reshape(reg_bs, K2, num_active_classes, -1)
            fg_aroi_emb = fg_aroi_emb.permute(0, 2, 3, 1).flatten(0, 1).reshape(reg_bs * num_active_classes, -1,
                                            self.reg_roialign_size, self.reg_roialign_size)
            
            bg_aroi_emb = bg_aroi_emb.permute(0, 2, 1).reshape(reg_bs, self.Temb,
                                self.reg_roialign_size, self.reg_roialign_size)[:, None, :, :, :].repeat(
                                    1, num_active_classes, 1, 1, 1).flatten(0, 1)
            aroi_emb = torch.cat([fg_aroi_emb, bg_aroi_emb], dim=1)
            pred_roi_mask = pred_roi_mask[:, None, :, :].repeat(1, num_active_classes, 1, 1).flatten(0, 1)

        masks = [pred_roi_mask[:, None, :, :].float(), ]

        num_masks = len(pred_roi_mask)

        if not self.training:
            aroi_app = aroi_app[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)
        embedding = torch.cat([aroi_emb, aroi_app], dim=1)

        if not self.training:
            aug_rois = aug_rois[:, None, :].repeat(1, num_active_classes, 1).flatten(0, 1)
            

        rp_blocks = [
            (self.rp1, self.rp1_out),
            (self.rp2, self.rp2_out),
            (self.rp3, self.rp3_out),
            (self.rp4, self.rp4_out),
            (self.rp5, self.rp5_out),
        ]

        for i, ((rp, rp_out), sem_cond) in enumerate(zip(rp_blocks, sem_conds)):
            gate = masks[0]
            gate = gate.detach()

            sem_cond_gated = sem_cond * gate
            embedding = embedding + self.reg_semantic_scale[i] * sem_cond_gated

            all_mask_tensor = torch.cat(masks, dim=1)
            x_in = torch.cat([embedding, all_mask_tensor], dim=1)

            if (not self.training) and (self.rp_eval_chunks > 1) and (x_in.size(0) > 1):
                chunk_size = (x_in.size(0) + self.rp_eval_chunks - 1) // self.rp_eval_chunks
                emb_chunks = []
                mask_chunks = []
                for start in range(0, x_in.size(0), chunk_size):
                    x_part = x_in[start:start + chunk_size]
                    if x_part.numel() == 0:
                        continue
                    emb_part = rp(x_part)
                    mask_part = rp_out(emb_part) / 0.1
                    emb_chunks.append(emb_part)
                    mask_chunks.append(mask_part)
                embedding = torch.cat(emb_chunks, dim=0)
                pred_mask_logits = torch.cat(mask_chunks, dim=0)
            else:
                embedding = rp(x_in)
                pred_mask_logits = rp_out(embedding) / 0.1

            masks.insert(0, pred_mask_logits.sigmoid())
            pred_region_coords = self.r2c(pred_mask_logits)
            if self.training:
                gt_roi_mask = gt_roi_mask.float()

                loss_dict[f"aux_bce_loss_{i}"] = sigmoid_ce_loss(pred_mask_logits.flatten(1), gt_roi_mask.flatten(1), num_masks)
                loss_dict[f"aux_dice_loss_{i}"] = dice_loss(pred_mask_logits.flatten(1), gt_roi_mask.flatten(1), num_masks)

                loss_dict[f'rg_l1_loss_{i}'] = F.l1_loss(pred_region_coords, gt_region_coords)
                try:
                    loss_dict[f'rg_giou_loss_{i}'] = (1 - torch.diag(generalized_box_iou(
                                    box_cxcywh_to_xyxy(pred_region_coords),
                                    box_cxcywh_to_xyxy(gt_region_coords)))).mean()
                except Exception as e:
                    print(f'skip exception in giou losses: {e}')

        pred_abs_boxes = region_coord_2_abs_coord(aug_rois[:, 1:], pred_region_coords, self.reg_roialign_size)
        fg_pred_deltas = pred_deltas = self.box2box_transform.get_deltas    (
            fg_proposals if self.training else rois[:, None, 1:].repeat(1, num_active_classes, 1).flatten(0, 1), pred_abs_boxes)

        if not self.training:
            pred_deltas = pred_deltas.reshape(reg_bs, num_active_classes, 4)
            pred_deltas = pred_deltas.flatten(1)
        if self.training:
            class_labels = class_labels.long()
            if sample_class_enabled:
                bg_indices = class_labels == num_classes
                fg_indices = class_labels != num_classes

                class_labels[fg_indices] = (class_indices == class_labels.view(-1, 1)).nonzero()[:, 1]
                class_labels[bg_indices] = num_active_classes

            if isinstance(logits, list):
                _log_classification_stats(logits[-1].detach(), class_labels)

                for i, l in enumerate(logits):
                    loss = focal_loss(l, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight)
                    loss_dict[f'focal_loss_{i}'] = loss
            else:
                _log_classification_stats(logits.detach(), class_labels)
                loss = focal_loss(logits, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight)
                loss_dict['focal_loss'] = loss

            gt_pred_deltas = self.box2box_transform.get_deltas(
                fg_proposals,
                matched_gt_boxes,
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )

            box_loss = loss_box_reg / max(class_labels.numel(), 1.0)
            if not torch.isinf(box_loss).any():
                loss_dict['bbox_loss'] = box_loss
            else:
                loss_dict['bbox_loss'] = torch.zeros(1, device=self.device)

            return loss_dict
        else:
            assert len(proposals) == 1
            image_size = proposals[0].image_size

            scores = F.softmax(logits, dim=-1)
            output = {'scores': scores[:, :-1] }

            predict_boxes = self.box2box_transform.apply_deltas(
                pred_deltas,
                rois[:, 1:],
            )

            if self.use_one_shot:
                gt_classes = gt_instances[0].gt_classes
                target_class_ids = torch.unique(gt_classes).tolist()
                all_scores = []
                all_boxes = []

                for target_cid in target_class_ids:
                    indices = (class_indices == target_cid).nonzero()
                    roi_inds = indices[:, 0]
                    cls_inds = indices[:, 1]

                    _scores = torch.zeros(len(roi_inds), self.num_classes + 1, device=self.device)
                    _scores[:, target_cid] = scores[roi_inds, cls_inds]
                    _boxes = predict_boxes.reshape(bs, class_topk, 4)[roi_inds, cls_inds]

                    all_scores.append(_scores)
                    all_boxes.append(_boxes)
                
                if len(all_scores) == 0:
                    return []
                else:
                    scores = torch.cat(all_scores)
                    predict_boxes = torch.cat(all_boxes)
            else:
                if sample_class_enabled:
                    full_scores = torch.zeros(len(scores), num_classes + 1, device=self.device)
                    full_scores.scatter_(1, class_indices, scores)
                    full_scores[:, -1] = scores[:, -1]

                    full_boxes = torch.zeros(len(scores), num_classes, 4, device=self.device)
                    predict_boxes = predict_boxes.view(len(scores), num_active_classes, 4)
                    full_boxes.scatter_(1, class_indices[:, :, None].repeat(1, 1, 4), predict_boxes)
                    full_boxes = full_boxes.flatten(1)

                    scores = full_scores
                    output['scores'] = full_scores[:, :-1]
                    predict_boxes = full_boxes
            
            
            if self.mult_rpn_score:
                rpn_scores = [x.objectness_logits for x in proposals][0]
                rpn_scores[rpn_scores < 0] = 0
                scores = (scores * rpn_scores[:, None]) ** 0.5
            
            instances, _ = fast_rcnn_inference(
                    [predict_boxes],
                    [scores],
                    [image_size],
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_topk_per_image,
                ) 

            results = self._postprocess(instances, batched_inputs)
            output['instances'] = results[0]['instances']
            return [output, ]
