# OpenThermalPose
## An Open-Source Annotated Thermal Human Pose Dataset and Initial YOLOv8-Pose Baselines
The OpenThermalPose dataset provides 6,090 images and 14,315 annotated human instances. Annotations include bounding boxes and 17 anatomical keypoints, following the conventions used in the benchmark MS COCO Keypoint dataset. The dataset covers various fitness exercises, multiple-person activities, and outdoor walking in different locations under different weather conditions. 

The OpenThermalPose dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1C5ThcFZm1twYtEta8GWUe1SENc9ER_0t/view?usp=sharing).  

As a baseline, we trained and evaluated YOLOv8-pose models (nano, small, medium, large, and x-large) on our dataset. For more information about training, validation, and running YOLOv8 models, please visit [Ultralytics Docs](https://docs.ultralytics.com/tasks/pose/).

## Inference
### Pre-trained models
You can download a zip file with all checkpoints from [Google Drive](https://drive.google.com/file/d/1PHyHSuM-n2XgqNgpl55Nn7fAqhukVEXw/view?usp=sharing). 

### Install Ultralytics
```
pip install ultralytics
```
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
### CLI
```
yolo pose predict model=path/to/model source='path/to/image' 
```

## References
1. https://github.com/ultralytics/ultralytics
2. https://docs.ultralytics.com/tasks/pose/ 
