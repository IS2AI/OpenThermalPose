# OpenThermalPose
## An Open-Source Annotated Thermal Human Pose Dataset and Initial YOLOv8-Pose Baselines 
The OpenThermalPose dataset provides 6,090 images of 31 subjects and 14,315 annotated human instances. Annotations include bounding boxes and 17 anatomical keypoints, following the conventions used in the benchmark MS COCO Keypoint dataset. The dataset covers various fitness exercises, multiple-person activities, and outdoor walking in different locations under different weather conditions. As a baseline, we trained and evaluated the YOLOv8-pose models (nano, small, medium, large, and x-large) on this dataset. 
- [Preprint on TechRxiv](https://www.techrxiv.org/users/682600/articles/741508-openthermalpose-an-open-source-annotated-thermal-human-pose-dataset-and-initial-yolov8-pose-baselines)
- [Published on IEEE](https://ieeexplore.ieee.org/document/10581992)
- [Download the dataset from Google Drive](https://drive.google.com/file/d/1C5ThcFZm1twYtEta8GWUe1SENc9ER_0t/view?usp=sharing)
- [Download the dataset from HuggingFace](https://huggingface.co/datasets/issai/OpenThermalPose)
- [Download the pre-trained YOLOv8-pose models](https://drive.google.com/file/d/1BS2AB6wGRjZ8Tvz44_jkURR6moRpVS1j/view?usp=sharing)

# OpenThermalPose2
## Extending the Open-Source Annotated Thermal Human Pose Dataset With More Data, Subjects, and Poses
We extended our OpenThermalPose dataset with more data, subjects, and poses. The new OpenThermalPose2 dataset contains 11,391 images of 170 subjects and 21,125 annotated human instances. The dataset covers various fitness exercises, multiple-person activities, persons sitting in an indoor environment, and persons walking in outdoor locations under different weather conditions. We trained and evaluated the YOLOv8-pose and YOLO11-pose models (nano, small, medium, large, and x-large) on this dataset. 
- [Preprint on TechRxiv](https://www.techrxiv.org/users/682600/articles/1231799-openthermalpose2-extending-the-open-source-annotated-thermal-human-pose-dataset-with-more-data-subjects-and-poses)
- [Published on IEEE](https://ieeexplore.ieee.org/document/11020744)
- [Download the dataset from Google Drive](https://drive.google.com/file/d/1BDVprz9NtenCp3wDovA2lfKVBzJDCTt2/view?usp=sharing)
- [Download the dataset from HuggingFace](https://huggingface.co/datasets/issai/OpenThermalPose2)
- [Download the pre-trained YOLOv8-pose and YOLO11-pose models](https://drive.google.com/file/d/19bvKSNKs3Z-8EFSJcTMaI-MdkNwVKh1f/view?usp=sharing)

#### Sports exercises and two-person activities in an indoor environment
<img src="https://github.com/IS2AI/OpenThermalPose/blob/main/session_1_2.png"> 

#### Walking in outdoor environments
<img src="https://github.com/IS2AI/OpenThermalPose/blob/main/3370.png"> 

#### Sitting in an indoor environment
<img src="https://github.com/IS2AI/OpenThermalPose/blob/main/102_1_2_4_15_101_1.png"> 

We trained and evaluated the YOLOv8-pose and YOLO11-pose models (nano, small, medium, large, and x-large) on our datasets. For more information about training, validation, and running of these models, please visit [Ultralytics Docs](https://docs.ultralytics.com/tasks/pose/).

## Install Ultralytics
```
pip install ultralytics
```

## Train
```
from ultralytics import YOLO

# Load a model
# you can choose any version of yolov8-pose/yolo11-pose
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
# you can choose any version of yolov8-pose/yolo11-pose
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
```
from ultralytics import YOLO
import cv2

# Load a model (.pt)
# you can choose any pre-trained yolov8-pose/yolo11-pose
model = YOLO('path/to/model')  

# Load an image
image = cv2.imread('path/to/image')

# Predict with the model
results = model.predict(image, save=True)  
```
## Citation
Please cite our work if you use our datasets/pre-trained models in your research.
#### OpenThermalPose
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
#### OpenThermalPose2
```
@ARTICLE{11020744,
  author={Kuzdeuov, Askat and Zakaryanov, Miras and Tleuliyev, Alim and Varol, Huseyin Atakan},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={OpenThermalPose2: Extending the Open-Source Annotated Thermal Human Pose Dataset With More Data, Subjects, and Poses}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Cameras;Training;Annotations;Lighting;Image resolution;Pose estimation;Accuracy;Testing;Indoor environment;Biological system modeling;Thermal human pose estimation;thermal human pose dataset;YOLOv8-pose;YOLO11-pose},
  doi={10.1109/TBIOM.2025.3575499}}
```
## References
1. https://github.com/ultralytics/ultralytics
2. https://docs.ultralytics.com/tasks/pose/ 
