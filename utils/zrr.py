from typing import List
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

class ZeroResultRate:
    def __init__(self, num_classes:int, iou_threshold:float=0.8):

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        self.zrr = 0
        self.zrr_per_class = {
            cls_id : []
            for cls_id in range(self.num_classes)
        }
        self.num_samples = 0
        self.result = None

    def get_matched_count(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> int:
    
        iou_matrix = box_iou(pred_boxes, target_boxes).cpu().numpy()
        cost_matrix = 1.0 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_count = 0
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > self.iou_threshold:
                matched_count += 1

        return matched_count

    def update(self, preds: List, targets: List):
        batch_size = len(preds)
        
        for i in range(batch_size):
            pred_boxes = preds[i]["boxes"]
            pred_labels = preds[i]["labels"]
            target_boxes = targets[i]["boxes"]
            target_labels = targets[i]["labels"]
            
            if pred_boxes.numel() == 0:
                self.zrr += 1
                for cls_id in target_labels.unique():
                    self.zrr_per_class[cls_id.item()].append(0)
            
            else:
                for cls_id in target_labels.unique():
                    
                    pred_cls_count = (pred_labels == cls_id).sum()
                    if pred_cls_count == 0:
                        self.zrr_per_class[cls_id.item()].append(0)

                    else:
                        pred_class_indices = torch.nonzero(pred_labels == cls_id).flatten()
                        target_class_indices = torch.nonzero(target_labels == cls_id).flatten()

                        pred_class_boxes = pred_boxes[pred_class_indices]
                        target_class_boxes = target_boxes[target_class_indices]

                        matched_count = self.get_matched_count(pred_class_boxes, target_class_boxes)
                        if matched_count == len(target_class_indices):
                            self.zrr_per_class[cls_id.item()].append(0)
                        else:
                            self.zrr_per_class[cls_id.item()].append(1)
                            
        self.num_samples += batch_size

    def reset(self):
        self.zrr = 0
        self.zrr_per_class = {
            cls_id : []
            for cls_id in range(self.num_classes)
        }
        self.num_samples = 0
        self.result = None

    def compute(self):
        
        if self.result is not None:
            return self.result

        for cls_id in range(self.num_classes):
            zrr_cls = self.zrr_per_class[cls_id]
            if len(zrr_cls) == 0:
                self.zrr_per_class[cls_id] = -1
            else:
                res = sum(zrr_cls) / len(zrr_cls)
                self.zrr_per_class[cls_id] = res
        
        self.result = {
            "zrr": self.zrr / self.num_samples,
            "zrr_per_class": self.zrr_per_class
        }
        return self.result
