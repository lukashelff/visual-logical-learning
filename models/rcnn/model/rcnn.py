import warnings
from collections import OrderedDict
from typing import Any, Mapping
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.utils import _log_api_usage_once


class GeneralizedMultiHeadRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads List((nn.Module)): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: List[nn.Module], transform: nn.Module) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for roi_head in self.roi_heads:
            roi_head.to(*args, **kwargs)
        return self

    def eval(self):
        super().eval()
        for roi_head in self.roi_heads:
            roi_head.eval()
        return self

    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        for roi_head in self.roi_heads:
            roi_head.train(mode)
        return self

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict_rcnn = state_dict.copy()
        state_dict_roid_head = state_dict.copy()
        # if metadata is not None:
        #     state_dict._metadata = metadata

        # get number of roi_heads from the checkpoint
        # num_roi_heads = len(self.roi_heads)

        # if the checkpoint was saved with a single roi_head, we want to expand it
        for key in list(state_dict.keys()):
            if key.startswith("roi_heads."):
                del state_dict_rcnn[key]
                # rename key for roi_heads
                new_key = key.replace("roi_heads.", "")
                state_dict_roid_head[new_key] = state_dict_roid_head[key]
                del state_dict_roid_head[key]

            else:
                del state_dict_roid_head[key]
        for roi_head in self.roi_heads:
            roi_head.load_state_dict(state_dict_roid_head, strict=strict)

        super().load_state_dict(state_dict_rcnn, strict=strict)

    def load_multi_head_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict_rcnn = state_dict.copy()
        state_dict_roid_head = state_dict.copy()
        # if metadata is not None:
        #     state_dict._metadata = metadata

        # get number of roi_heads from the checkpoint
        # num_roi_heads = len(self.roi_heads)

        # if the checkpoint was saved with a single roi_head, we want to expand it
        for key in list(state_dict.keys()):
            if key.startswith("roi_heads."):
                del state_dict_rcnn[key]
                # rename key for roi_heads
                new_key = key.replace("roi_heads.", "")
                state_dict_roid_head[new_key] = state_dict_roid_head[key]
                del state_dict_roid_head[key]

            else:
                del state_dict_roid_head[key]
        for roi_head in self.roi_heads:
            roi_head.load_state_dict(state_dict_roid_head, strict=strict)

        super().load_state_dict(state_dict_rcnn, strict=strict)

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        losses = {}
        detections = [{}] * images.tensors.shape[0]
        for head_id, roi_head in enumerate(self.roi_heads):
            head_targets = []
            if targets is not None:
                for target in targets:
                    head_targets.append({
                        "boxes": target["boxes"][target["labels_ids"] == head_id],
                        "labels": target["labels"][target["labels_ids"] == head_id],
                        "image_id": target["image_id"],
                        "area": target["area"][target["labels_ids"] == head_id],
                        "iscrowd": target["iscrowd"][target["labels_ids"] == head_id],
                        "masks": target["masks"][target["labels_ids"] == head_id],
                    })

            detection, detector_losses = roi_head(features, proposals, images.image_sizes, head_targets)
            detection = self.transform.postprocess(detection, images.image_sizes, original_image_sizes)
            for d_id, d in enumerate(detection):
                for key, value in d.items():
                    if key in detections[d_id]:
                        detections[d_id][key] = torch.cat([detections[d_id][key], value])
                    else:
                        detections[d_id][key] = value

            for key, value in detector_losses.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value
            # losses.update(detector_losses)

        # detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
