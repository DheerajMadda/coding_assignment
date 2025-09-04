import os
import time
from glob import glob

import numpy as np
import fiftyone as fo

dataset = fo.Dataset("BDD100K_Dataset_Samples_Visualizations")

def read_label(path):
    with open(path, 'r') as f:
        labels = f.read().splitlines()
    return labels

def get_class_names_and_bboxes(labels):
    # Convert the list to a NumPy array
    arr = np.array([line.split() for line in labels])

    class_names = [class_name for class_name in arr[:, 0]]

    # Extract xyxy coordinates
    bboxes = arr[:, 1:5].astype(float)
                                
    return class_names, bboxes.tolist()

def pascal_voc_to_fiftyone(bboxes, image_width=1280, image_height=720):
    
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]

        # Calculate width and height
        width = xmax - xmin
        height = ymax - ymin

        # Normalize coordinates and dimensions
        xmin /= image_width
        ymin /= image_height
        width /= image_width
        height /= image_height

        bboxes[i] = [xmin, ymin, width, height]

data_dir = os.path.join("data", "bdd100k_samples")

splits = ["train", "val"]
attributes = {
    "weather": [
        "clear", "rainy", "undefined", "snowy", "overcast","partly cloudy", "foggy"
    ],
    "scene": [
        "city street", "highway", "residential", "parking lot",
        "undefined", "tunnel", "gas stations"
    ],
    "timeofday": [
        "daytime", "dawn_dusk", "night", "undefined"
    ],
    "label": [
        "occluded", "truncated", "small", "medium", "large", "uncertain"
    ]
}


for split in splits:
    split_dir = os.path.join(data_dir, split)

    for category in attributes:
        for sub_category in attributes[category]:
            image_paths = sorted(
                glob(os.path.join(split_dir, "images", category, sub_category, ".*jpg"))
            )
            label_paths = sorted(
               glob(os.path.join(split_dir, "labels", category, sub_category, ".*txt"))
            )

            for image_path, label_path in zip(image_paths, label_paths):

                labels = read_label(label_path)
        
                # get class_names, bboxes
                class_names, bboxes = get_class_names_and_bboxes(labels)
                pascal_voc_to_fiftyone(bboxes)
   
                detections = []
                for label, bbox in zip(class_names, bboxes):
                    detection = fo.Detection(
                        label=label,
                        bounding_box=bbox,
                        tags=[f"{category}_{sub_category}"]
                    )
                    detections.append(detection)

                sample = fo.Sample(filepath=image_path)
                sample["detections"] = fo.Detections(detections=detections)
                sample["split"] = split

                dataset.add_sample(sample)

if __name__ == "__main__":

    session = fo.launch_app(dataset, remote=True, port=5151)

    try:
        print("FiftyOne is running at http://localhost:5151")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

    session.close()
