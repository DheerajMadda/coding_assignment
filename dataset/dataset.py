import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from typing import List, Callable, Dict, Literal

class DetectionDataset(Dataset):
    def __init__(
        self,
        inputs: List[str],
        targets: List[str],
        transform: Callable,
        class_map: Dict,
        image_height: int=480,
        image_width: int=640,
        mean: List[float]=[0., 0., 0.],
        std: List[float]=[1., 1., 1.],
        return_normalized_image: bool=True,
        return_bbox_format: Literal["xyxy", "xywhn"]="xywhn",
    ):
        assert return_bbox_format in ["xyxy", "xywhn"], "`return_bbox_format` must be one of ['xyxy', 'xywhn']"
        
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.class_map = class_map
        self.image_height = image_height
        self.image_width = image_width
        self.return_normalized_image = return_normalized_image
        self.return_bbox_format = return_bbox_format

        self.norm_transform = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

    def read_image(self, path):
        return cv2.cvtColor(
            cv2.imread(path, cv2.IMREAD_COLOR), 
            cv2.COLOR_BGR2RGB
        )
    
    def read_label(self, path):
        with open(path, 'r') as f:
            labels = f.read().splitlines()
        return labels
    
    def get_class_ids_and_bboxes(self, labels):
        # Convert the list to a NumPy array
        arr = np.array([line.split() for line in labels])

        # Extract class indices using class_map
        class_ids = np.array([self.class_map[class_name] for class_name in arr[:, 0]], dtype=int)

        # Extract xyxy coordinates
        bboxes = arr[:, 1:5].astype(float)
                                 
        return class_ids.tolist(), bboxes.tolist()

    def pascal_voc_to_yolo(self, bboxes):
        """
        convert pascal_voc bboxes (xyxy) to yolo bboxes (xn,yn,wn,hn : normalized)
        """
        
        for i in range(len(bboxes)):
            xmin, ymin, xmax, ymax = bboxes[i]

            # Calculate center coordinates
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Calculate width and height
            width = xmax - xmin
            height = ymax - ymin

            # Normalize coordinates and dimensions
            x_center /= self.image_width
            y_center /= self.image_height
            width /= self.image_width
            height /= self.image_height

            bboxes[i] = [x_center, y_center, width, height]

    def load_image_and_labels(self, idx):
        
        # Load image and label
        image = self.read_image(self.inputs[idx])
        labels = self.read_label(self.targets[idx])

        if not labels:
            random_idx = np.random.randint(0, len(self.inputs))
            return self.load_image_and_labels(random_idx)
        
        # get class_ids, bboxes
        class_ids, bboxes = self.get_class_ids_and_bboxes(labels)

        # preprocess
        aug = self.transform(image=image, bboxes=bboxes, category_ids=class_ids)
        image = aug["image"]
        bboxes = aug["bboxes"]
        class_ids = aug["category_ids"]
        
        if not bboxes:
            # after preprocess; if image has 0 bboxes, get random image and labels
            random_idx = np.random.randint(0, len(self.inputs))
            return self.load_image_and_labels(random_idx)
        
        if self.return_normalized_image:
            # norm transform
            image = self.norm_transform(image=image)["image"]

        if self.return_bbox_format == "xywhn":
            # convert pascal_voc bboxes (xyxy) to yolo bboxes (xn,yn,wn,hn : normalized)
            self.pascal_voc_to_yolo(bboxes)

        return image, bboxes, class_ids

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        
        image, bboxes, class_ids = self.load_image_and_labels(idx)
        
        # Combine class indices and bboxes coordinates
        target = torch.column_stack([
            torch.tensor(class_ids, dtype=torch.float32),
            torch.tensor(bboxes, dtype=torch.float32)
        ]) # shape: (num_bboxes, 5); [class_index, xn, yn, wn, hn]

        return image, target
