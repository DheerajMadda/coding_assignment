![bosch_logo](https://github.com/user-attachments/assets/50202174-1ce6-4957-a25f-5c687bbbb51f)

# <p align="center"><strong> Applied Computer Vision Coding Assignment</strong></p>
### <p align="center">Version: 1.1.2</p>

### Repository Structure

- *[data]* -> The BDD100K dataset to be downloaded and extracted to this directory. <br/> (Note, part of this directory content can be downloaded from [here](https://drive.google.com/drive/folders/1KoL0lm4LqDEySaSwQG6uU-CMoK_hKF8U))

- *[dataset]* -> Contains the PyTorch Dataset class definition.

- *[docker]* -> Contains the docker application. 

- *[eval_results]* -> Contains the evaluation metric results.

- *[experiments]* -> Contains the trained model weights.

- *[losses]* -> Contains the Loss function definitions.
  
- *[model]* -> Contains the model definitions.

- *[notebooks]* -> Contains the jupyter notebooks for various tasks.

- *[plots]* -> Contains the BDD100K dataset distribution plots.
  
- *[postprocess]* -> Contains the model postprocess definitions. 

- *[utils]* -> Contains the utility function definitions.

- *[requirements.txt]* -> Contains all the required libraries to be installed.

</br>
</br>

<hr>

# <p align="center"><strong>Task 1 - Dataset Stats and Visulaizations</strong></p>

## 1.1 Dataset 
### [Notebooks]
- *notebooks/Task_1a_Stats.ipynb* -> A jupyter notebook that parses BDD100K dataset label (.json), creates metadata informations and generates and saves distribution plots. </br>
- *notebooks/Task_1b_FiftyOne.ipynb* -> A jupyter notebook that saves samples to disk which is later required for FiftyOne Docker application. </br>

### [Scripts]
- *dataset/  init  .py* -> A package declaration file. </br>
- *dataset/collate.py* -> Contains the collate function which is used to overwrite the dataloader's default collate function as per the required of dataloader's output format. </br>
- *dataset/dataset.py* -> Contains the DetectionDataset class definition. </br>

</br>

### [Details]
- The source label files "bdd100k_labels_images_train.json" and "bdd100k_labels_images_val.json, that contains labels for each image. It contains information such as filename, labelnames, bbox coordinates, various attributes such as weather, occluded, etc. (Note that not all are attributes are considered for this task, only attributes -> (label, boxes, weather, scene, timeofday, occluded, truncated) are considered.
- Using these metadata informations about the image, created 5 additional attributes -> (area, small, medium, large, uncertain).
- area => bbox area
- small => Objects that have an area; < 32^2
- medium => Objects that have an area; (32^2 < area < 96^2)
- large => Objects that have an area; (area > 96^2)
- uncertain => a) All Objects which have an area < 8^2. b) Objects ["traffic-sign", "traffic-light", "person", "bike", "rider", "motor"] which have an area > 512^2. Note that, the meaning of `uncertain` here is not that the ground truth labels might be wrong but the also model predictions with respect to such objects might be wrong, given how pixalated these crops could be and hence, can be considered as `uncertain`.

Why these 4 additional attributes are created? -> They are created to understand how model performs across objects that are very small (uncertain), small, medium, large.

- The following dataframe for both, train and val is created where each row provides information about the single object. Thus, a filename having 4 objects -> 4 rows with same filename, each for per object or bbox. 
- Note, for a given filename, there can be no labels at all, duplicate labels, overlapped boxes for same obejct -> These all edge-cases are filtered out.
- The dataframe looks like the following
</br>
<img width="1638" height="394" alt="image" src="https://github.com/user-attachments/assets/fa7c8135-32d0-4fe1-ae6b-080922cc17c9" />

</br>
</br>

## 1.2 Docker

### [Scripts]
- *docker/docker-compose.yaml* -> A Docker-Compose to spin up both containers. </br>
- *docker/task_1_stats/Dockerfile* -> A Dockerfile for the Dataset distribution visualization task. </br>
- *docker/task_1_stats/requirements.txt* -> A requirements definition file required for the docker app. </br>
- *docker/task_1_stats/app/main.py* -> A Streamlit application. </br>
- *docker/task_1_stats/app/plots* -> A directory that contains train and val dataset distribution images. </br>
- *docker/task_1_viz/Dockerfile* -> A Dockerfile for the FiftyOne visualization for Samples. </br>
- *docker/task_1_viz/requirements.txt* -> A requirements definition file required for the docker app. </br>
- *docker/task_1_viz/app/main.py* -> A FiftyOne application. </br>
- *docker/task_1_viz/app/data* -> A directory that contains train and val sample images across various categories (Note, part of this directory content can be downloaded from [here](https://drive.google.com/drive/folders/1KoL0lm4LqDEySaSwQG6uU-CMoK_hKF8U). </br>

</br>

### [Details]
- There are two docker applications.
- a) Streamlit App => For visualizing dataset distributions of train and val splits across various attributes. (Runs on [http://localhost:8501](http://localhost:8501/))
- b) FiftyOne App => For visualizing few train and val samples across various attributes. (Runs on [http://localhost:5151](http://localhost:5151/))

</br>

Please run the following shell commands to run these docker applications. **Note this will pull the images from dockerhub, no need to build!**
Containers with name "bdd100k_stats_app" and bdd100k_dataviz_app" respectively for Streamlit Dataset Stats and FiftyOne Sample Visualizations. 

</br>

[Recommended] Build and Run

```
cd docker
docker compose up -d
```
</br>
[Optional] Build and Run

```
cd docker
docker compose up -d --build
```
#### Note Build and Tested the docker image on Linux System!

</br>
</br>

<hr>

# <p align="center"><strong>Task 2 - Model Architecture and Training</strong></p>

### [Notebooks]
- *notebooks/Task_2a_Training_Prerequisite.ipynb* -> A jupyter notebook that computes Mean and Std of the dataset and saves labels in the format that is efficient for training the model. </br>
- *notebooks/Task_2b_Model_Profiling.ipynb* -> A jupyter notebook performs model profiling. </br>
- *notebooks/Task_2c_Model_Training.ipynb* -> A jupyter notebook that trains the model and save at experiments directory. </br>

### [Scripts]
#### Modeling
- *model/  init  .py* -> A package declaration file. </br>
- *model/arch.py* -> Contains the main model class definition. </br>
- *model/configs.py* -> Contains the main model configs for various model sizes ("n", "s", "m", "l", "x"). </br>
- *model/network/  init  .py* -> A package declaration file. </br>
- *model/network/backbone.py* -> Contains the model Backbone class definition. </br>
- *model/network/neck.py* -> Contains the model Neck class definition. </br>
- *model/network/head.py* -> Contains the model Head class definition. </br>
- *model/network/modules.py* -> Contains the various modules required for the model. </br>

#### Losses
- *losses/  init  .py* -> A package declaration file. </br>
- *losses/detection_loss.py* -> Contains the main Detection Loss class definition. </br>
- *losses/bbox_loss.py* -> Contains the Bbox Loss class definition. </br>
- *losses/tal.py* -> Contains the Task-Aligned Assigner module. </br>
- *losses/utils.py* -> Contains the various utility functions required for the Loss Function. </br>

#### PostProcess
- *postprocess/  init  .py* -> A package declaration file. </br>
- *postprocess/drawing.py* -> Contains the drawing bbox function definition. </br>
- *postprocess/nms.py* -> Contains the NMS function definition. </br>
- *postprocess/pipeline.py* -> Contains the PostProcess Class definition that takes raw model outputs as inputs. </br>
- *postprocess/utils.py* -> Contains the various utility functions required for the Postprocessing. </br>

</br>
</br>

### [Details]

#### Model Selection
- There are various pretrained models trained on BDD100K Dataset which can be found at [here](https://github.com/SysCV/bdd100k-models/tree/main/det)
- All these models are 2-Stage Detectors where ConvNext models have the highest validation metrics.
- These 2-Stage Detectors requires more computational cost for Training and have higher latencies than 1-Stage Detectors.
- Note that, selection of a Detection Model falls mainly under 2 following requirements; </br>
  -> 1) Model to be used in the Data/Annotation pipeline where we can afford to use heavy models having higher latencies and better performance. </br>
  -> 2) Model to be used in the Real-time scenario where we **cannot** afford to use such heavy models. Instead 1-Stage Models like YOLO models are preferred. 
- As this assignment **did not** specify any of such requirement, which gave freedom to assume any one of these use-case -> Hence, assuming **Real-Time Scenario**.
- Most widely used 1-Stage models for Real-time scenario are of YOLO family, mainly YOLOv8. We have now YOLOv12 as well.
- Is there anything better than YOLOv8 or YOLOv12, in any of the parameters like be it model_size, performance, latency, etc ?
- Thus, explored and came across a 2025 Paper that has an interesting model architecture which seems promising!


Paper: [Improvement of Yolov8 Object Detection Based on Lightweight Neck Model for Complex Images](https://www.ias-iss.org/ojs/IAS/article/view/3514) </br>
Published Date: 2025-03-24 </br>
Source Code: **Unavailable** </br>

This paper replaces various modules of original YOLOv8 with various light-weight and efficient modules while achieving better performance than YOLOv8 for real-time applications. <br/>
*Unfortunately*, the author's have **not** made their souce code publicly available yet as of this writing (2025-09-03).

#### Model Architecture

<img width="1301" height="705" alt="image" src="https://github.com/user-attachments/assets/35db3b7b-1cd3-4ff7-8c0b-227ebd35e22b" />

**Blocks**
- *GhostConv:* CBS -> x - > Concat(x, CBS(x))
- *LDG_Conv: Light weight Depth Ghost Conv:* x-> GhostConv -> y -> Concat(y, DepthWiseConv(y)) -> ChannelShuffle
- *CBS Block:* Convolution(stride = 2) + Batch Normalization + SiLU activation
- *Bottleneck:* Residual -> CBS -> CBS -> x + Residual
- *LDG_Bottleneck:* Residual -> LDG_Conv -> LDG_Conv -> x + Residual
- *C2F Block:* CBS -> split(x) -> [x1, x2] -> x2 -> (n)Bottleneck [y1, y2, y3]-> Concat(x1, y1, y2, y3) -> CBS
- *C2F_LDG Block:*  CBS -> split(x) -> [x1, x2] -> x2 -> (n)LDG_Bottleneck [y1, y2, y3]-> Concat(x1, y1, y2, y3) -> CBS
- *SPPF Block: Spatial Pyramid Pooling Fast:* CBS -> x -> Maxpools [y1, y2, y3]-> Concat(x, y1, y2, y3) -> CBS
- *CA Block: CoordinateAttention:* Residual -> [AVGPool_H, AVGPool_W] -> ConvBN(Concat(AVGPool_H, AVGPool_W)).sigmoid() -> Split-> [F_H, F_W] -> Conv(F_H).sigmoid() * Conv(F_W).sigmoid() * Residual

**a) Backbone**
  [Forward Pass]
- Assume an input image of shape (1, 3, 480, 640) where height = 480, width = 640, in_channel = 3.
- The very first, stem CBS block -> reduces spatial dimensions while increasing channels. Thus (1, 3, 480, 640) -> (1, 32, 240, 320)
- Now next (CBS + C2f) is considered as a stage and there are 4 such sequential stages.
- Stage 1: (1, 32, 240, 320) -> CBS (1, 64, 120, 160) -> C2F (1, 64, 120, 160)
- Stage 2: (1, 64, 120, 160) -> CBS (1, 128, 60, 80) -> C2F (1, 128, 60, 80)
- Stage 3: (1, 128, 60, 80) -> CBS (1, 256, 30, 40) -> C2F (1, 256, 30, 40)
- Stage 4: (1, 256, 30, 40) -> CBS (1, 512, 15, 20) -> C2F (1, 512, 15, 20)
- The outputs from Stage 2, Stage 3, Stage 4 (x2, x3, x4) are passed to the Neck Block

**b) Neck**
  [Forward Pass]
- Inputs to this Module is x2, x3, x4 from Backbone
- SPPF Block: x4 (1, 512, 15, 20) -> y4  (1, 512, 15, 20)
- UP1 = y4 -> Concat(x3, Upsample(CA(y4))) -> C2F_LDG -> y3 (1, 256 , 30, 40)
- UP2 = y3 -> Concat(x2, Upsample(CA(y3))) -> C2F_LDG -> y2 (1, 128 , 60, 80)
- Now we have y2, y3, y4
- DOWN1 = Concat(y3, LDG_Conv(CA(Downsample(y2)))) -> C2F_LDG -> x3 (1, 256 , 30, 40)
- DOWN2 = Concat(y4, LDG_Conv(CA(Downsample(x3)))) -> C2F_LDG -> x4 (1, 512 , 15, 20)
- Now we have y2, x3, x4
- This is then passes to the Head Block

Note, that in the above architecture diagram (from the paper), 2 Downsamples blocks are missing which are used to downsample the resolution of feature maps! CBS is used as a Downsampling Block with stride=2.
</br>

Note: c) **Head Block** is a Standard Yolov8 Head Block with 3 decoupled BBox Conv Blocks and Classification Conv Blocks 

</br>
</br>

#### Model Comparisons
</br>
Following model comparisons are of model variant/size - "m"; Batch size = 16 on RTX 3060
</br>

<img width="1475" height="189" alt="image" src="https://github.com/user-attachments/assets/f22e7631-844e-4a03-8441-6a7eff5d8078" />

</br>

With respect to the above comparisons, selecting the **Implemented YOLOv8I model**!

</br>
</br>

#### Model Implementation and Training
- Implemented the model architecture from scratch defined in the mentioned paper. Named it as YOLOv8I (`I` for Improved).
- Created various model sizes ("n", "s", "m", "l", "x") similar to original YOLOv8.
- Tested its forward pass and performed profiling.
- Trained "m" variant of the model on entire training dataset for 50 epochs on RTX 3080 with FP16-Mixed precision. Each epoch took around 30 minutes!
- Note that, the Head Module, is reused from the official ultralytics repository.
- Instead of using pytorch-lightning for training, wrote custom training scripts.

#### PyTorch Dataset
- Created custom PyTorch Dataset class that takes an index => reads images and labels => augmentations on images => converts bbox format from xyxy to xywh (normalized) => maps class_labels to class_ids.

#### Loss Function
- Reused Loss Function from the official ultralytics repository.

</br>
</br>

<hr>

# <p align="center"><strong>Task 3 - Evaluation and Analysis</strong></p>
 
### [Notebooks]
- *notebooks/Task_3a_Evaluation.ipynb* -> A jupyter notebook computes metrics for evaluation. </br>
- *notebooks/Task_3a_Evaluation_analysis.ipynb* -> A jupyter notebook that performs analysis of the evaluation results. </br>

</br>

### [Details]
- Metrics used are MeanAveragePrecision (mAP) and ZeroResultsRate (ZRR).
- While mAP, provides a comprehensive score by averaging the Average Precision (AP) values across multiple classes and is widely used object detection model metric; ZRR is a metric that tells how many data points yielded "No results" i.e No detections at all!
- ZRR: The lower the metric value, the better.
- ZRR e.g a) for an image containing objects of various classes, if model detected no objects at all; then ZRR = 1. <br/>
   b) for an image containing 2 objects of 2 classes each (total 4 objects); <br/>
   b.1) if model detected only 2 objects of class_A then; ZRR|Class_A = 0, ZRR|Class_B = 1 <br/>
   b.2) if model detected only 1 object of each classes then; ZRR|Class_A = 0, ZRR|Class_B = 0
- ZRR metric is a performance metric that tells us that out of the dataset, for how many samples and classes the model is not able to detect objects.
- This can be used to backtrack to images and helps in finding out any anomolies in the image/ scene which helps in understanding why and where the model is failing!
- Computed mAP and ZRR across all classes and class-wise too.
- `torchmetrics` library is used for computing mAP while for ZRR, the code has been written from scratch.

#### Validation set: Achieved mAP@50 = 0.4114

<br/>
<br/>

#### Drawing connections between Dataset Distribution and Computed Metrics

<br/>

Considering the "All" attribute -> meaning all class-wise crops across entire dataset; please find below the dataset distribution (from Task 1) and metric plots for both mAP and ZRR.
<br/>
<br/>
<img width="720" height="620" alt="image" src="https://github.com/user-attachments/assets/2a630b3d-8a5c-4155-86f8-ce93a49b4d3c" />
<br/>
<img width="1187" height="589" alt="image" src="https://github.com/user-attachments/assets/98e49ca8-73f2-41f2-a612-e42746ca6ea7" />
<br/>
<img width="1196" height="561" alt="image" src="https://github.com/user-attachments/assets/a7197e75-4388-42b7-ae1b-54bbfbfb7438" />
<br/>
<br/>
<img width="1378" height="161" alt="image" src="https://github.com/user-attachments/assets/8ea4f36b-144d-41a4-a099-15b3d83d9011" />
<br/>

Insights derived from above plots:

```

[Traffic Sign]
Second most frequent class with approximately 35,000 samples. Demonstrates solid performance metrics with mAP of 0.313095. The substantial sample size supports effective sign recognition across various conditions.

[Traffic Light]
Third most common class with around 27,000 samples. Shows the highest mAP score (0.332901), indicating excellent detection accuracy which suggests reliable identification of traffic signals.

[Person]
Moderately represented with approximately 14,000 samples. Achieves strong performance with mAP of 0.336629 and despite fewer samples than vehicle classes, maintains high accuracy for human detection.

[Bike]
Minimal representation in the dataset with very low mAP (0.232744) and needs substantial sample increase for effective detection.

[Rider]
Least frequent among transportation objects with minimal samples. Shows competitive mAP (0.144686). Limited data availability poses challenges for reliable rider detection.

[Motor]
Very small sample size with decent mAP (0.210383). Poses challenge for the model's performance.

[Car] 
The most dominant class in the dataset with over 100,000 samples. Shows decent performance with high mAP (0.188658). This large sample size likely contributes to robust model training for vehicle detection but the low mAP is due to the fact that there are many Cars with bbox area < 32^2. Poses challenge for the model's performance.

[Truck]
Lower frequency class with around 5,000 samples. Shows reasonable performance with mAP of 0.208412. The smaller sample size may impact model robustness for truck detection.

[Bus]
Similar sample count to truck (~2,000-3,000). Demonstrates good mAP performance (0.260390). May benefit from additional training samples.

[Train]
Extremely rare class with practically no samples in the dataset. Shows low mAP (0.15802), indicating insufficient training data for recognition.

```

Similarly, for the "Uncertain" attribute -> please find below the dataset distribution (from Task 1) and metric plots for mAP .
<br/>
<br/>
<img width="1346" height="563" alt="image" src="https://github.com/user-attachments/assets/c6a16070-c655-45f6-b3f9-156f93b62041" />
<br/>
<img width="1343" height="650" alt="image" src="https://github.com/user-attachments/assets/f15a83f4-a934-4c82-8aab-7c3d639dc27f" />
<br/>

<br/>
<br/>

<hr>

# <p align="center"><strong> Thank You</strong></p>

<hr>

