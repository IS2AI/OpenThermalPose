# OpenThermalPose
## An Open-Source Annotated Thermal Human Pose Dataset and Initial YOLOv8-Pose Baselines 
- [Preprint on TechRxiv](https://www.techrxiv.org/users/682600/articles/741508-openthermalpose-an-open-source-annotated-thermal-human-pose-dataset-and-initial-yolov8-pose-baselines)
- [Published on IEEE](https://ieeexplore.ieee.org/document/10581992)
  
The OpenThermalPose dataset provides 6,090 images and 14,315 annotated human instances. Annotations include bounding boxes and 17 anatomical keypoints, following the conventions used in the benchmark MS COCO Keypoint dataset. The dataset covers various fitness exercises, multiple-person activities, and outdoor walking in different locations under different weather conditions. 

#### Examples of images for sports exercises and two-person activities in an indoor environment
<img src="https://github.com/IS2AI/OpenThermalPose/blob/main/session_1_2.png"> 

#### Examples of images for random walking in outdoor environments
<img src="https://github.com/IS2AI/OpenThermalPose/blob/main/3370.png"> <img src="https://github.com/IS2AI/OpenThermalPose/blob/main/3471.png">


The OpenThermalPose dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1C5ThcFZm1twYtEta8GWUe1SENc9ER_0t/view?usp=sharing).  

As a baseline, we trained and evaluated YOLOv8-pose models (nano, small, medium, large, and x-large) on our dataset. For more information about training, validation, and running YOLOv8-pose models, please visit [Ultralytics Docs](https://docs.ultralytics.com/tasks/pose/).

## Install Ultralytics
```
pip install ultralytics
```

## Train
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='open_thermal_pose.yaml', epochs=100, imgsz=640)
```
## Val
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('path/to/best.pt')  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
```

## Predict
### Pre-trained models
Download a zip file with all checkpoints from [Google Drive](https://drive.google.com/file/d/1BS2AB6wGRjZ8Tvz44_jkURR6moRpVS1j/view?usp=sharing). 

### Python 
```
from ultralytics import YOLO
import cv2

# Load a model (.pt)
model = YOLO('path/to/model')  

# Load an image
image = cv2.imread('path/to/image')

# Predict with the model
results = model.predict(image, save=True)  
```
## Citation
Please cite this work if you use the dataset or pre-trained models in your research.
```
@INPROCEEDINGS{10581992,
  author={Kuzdeuov, Askat and Taratynova, Darya and Tleuliyev, Alim and Varol, Huseyin Atakan},
  booktitle={2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG)}, 
  title={OpenThermalPose: An Open-Source Annotated Thermal Human Pose Dataset and Initial YOLOv8-Pose Baselines}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Privacy;Annotations;Source coding;Pose estimation;Lighting;Medical services;Motion capture},
  doi={10.1109/FG59268.2024.10581992}}

```
## References
1. https://github.com/ultralytics/ultralytics
2. https://docs.ultralytics.com/tasks/pose/ 
