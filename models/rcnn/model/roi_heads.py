from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops


class MultiLabelRoIHeads(RoIHeads):
    def __init__(self, labels_per_segment, *args, **kwargs):
        self.labels_per_segment = labels_per_segment
        super().__init__(*args, **kwargs)

    def multi_label_proposals(self, proposals, gt_boxes, gt_labels):
        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)

        return proposals, matched_idxs, labels, sampled_inds

    def select_training_samples(
            self,
            proposals,  # type: List[Tensor]
            targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # proposals, matched_idxs, labels, sampled_inds = self.multi_label_proposals(proposals, gt_boxes, gt_labels)

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        proposals, matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs = []
        labels = []
        multi_l_proposals = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # match quality matrix between ground-truth and proposals
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # index of the proposals with the highest IoU overlap with a ground-truth box
                color_idx1 = torch.logical_or((gt_labels_in_image == 1), (gt_labels_in_image == 2))
                color_idx2 = torch.logical_or((gt_labels_in_image == 3), (gt_labels_in_image == 4))
                color_idx = torch.logical_or((gt_labels_in_image == 5), torch.logical_or(color_idx1, color_idx2))
                length_idx = torch.logical_or((gt_labels_in_image == 6), (gt_labels_in_image == 7))

                if color_idx.size()[0] != match_quality_matrix.size()[0]:
                    raise ValueError(
                        f'color_idx size {color_idx.size()}, match_quality_matrix size {match_quality_matrix.size()}, '
                        f'gt_labels_in_image size {gt_labels_in_image.size()}, gt_boxes_in_image size'
                        '{gt_boxes_in_image.size()}, proposals_in_image size {proposals_in_image.size()}')
                color_match_quality_matrix = (color_idx * match_quality_matrix.T).T
                length_match_quality_matrix = (length_idx * match_quality_matrix.T).T

                color_matched_idxs_in_image = self.proposal_matcher(color_match_quality_matrix)
                length_matched_idxs_in_image = self.proposal_matcher(length_match_quality_matrix)
                # matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                if self.labels_per_segment == 3:
                    other_idx = torch.logical_not(torch.logical_or(color_idx, length_idx))
                    roof_matched_idxs_in_image = torch.empty((0,), dtype=torch.int64,
                                                             device=color_matched_idxs_in_image.device)
                    multi_l_proposals.append(
                        torch.cat((proposals_in_image, proposals_in_image, proposals_in_image)))
                elif self.labels_per_segment == 4:
                    roof_idx1 = torch.logical_or((gt_labels_in_image == 10), (gt_labels_in_image == 11))
                    roof_idx2 = torch.logical_or((gt_labels_in_image == 12), (gt_labels_in_image == 13))
                    roof_idx = torch.logical_or(roof_idx1, roof_idx2)
                    roof_match_quality_matrix = (roof_idx * match_quality_matrix.T).T
                    roof_matched_idxs_in_image = self.proposal_matcher(roof_match_quality_matrix)
                    other_idx = torch.logical_not(torch.logical_or(color_idx, torch.logical_or(length_idx, roof_idx)))
                    multi_l_proposals.append(
                        torch.cat((proposals_in_image, proposals_in_image, proposals_in_image, proposals_in_image)))
                else:
                    raise ValueError("labels_per_segment should be 3 or 4")

                other_match_quality_matrix = (other_idx * match_quality_matrix.T).T
                other_matched_idxs_in_image = self.proposal_matcher(other_match_quality_matrix)
                matched_idxs_in_image = torch.cat(
                    (color_matched_idxs_in_image, length_matched_idxs_in_image, roof_matched_idxs_in_image,
                     other_matched_idxs_in_image), dim=0)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)

        return multi_l_proposals, matched_idxs, labels
