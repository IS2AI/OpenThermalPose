# OpenThermalPose
## An Open-Source Annotated Thermal Human Pose Dataset and Initial YOLOv8-Pose Baselines
The OpenThermalPose dataset provides 6,090 images and 14,315 annotated human instances. Annotations include bounding boxes and 17 anatomical keypoints, following the conventions used in the benchmark MS COCO Keypoint dataset. The dataset covers various fitness exercises, multiple-person activities, and outdoor walking in different locations under different weather conditions. 

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
Download a zip file with all checkpoints from [Google Drive](https://drive.google.com/file/d/1PHyHSuM-n2XgqNgpl55Nn7fAqhukVEXw/view?usp=sharing). 

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

## References
1. https://github.com/ultralytics/ultralytics
2. https://docs.ultralytics.com/tasks/pose/ 
