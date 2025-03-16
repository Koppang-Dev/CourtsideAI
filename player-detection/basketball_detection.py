from roboflow import Roboflow
from ultralytics import YOLO
import shutil
import torch


# Dataset download
rf = Roboflow(api_key="mjeDosQ77ikJz5K5h84J")
project = rf.workspace("test-datset").project("player_detect-0spfb")
version = project.version(1)
dataset = version.download("yolov5")

# Print the dataset location to check the folder path
print(f"Dataset location: {dataset.location}")

# Moving the splits into another folder with the same name
# This is because the system expects it

# Moving the test/train/validation sets into this new folder
if not os.path.exists('Player_detect-1/Player_detect-1/train'):
    shutil.move('Player_detect-1/train', 'Player_detect-1/Player_detect-1/train')

if not os.path.exists('Player_detect-1/Player_detect-1/test'):
    shutil.move('Player_detect-1/test', 'Player_detect-1/Player_detect-1/test')

if not os.path.exists('Player_detect-1/Player_detect-1/valid'):
    shutil.move('Player_detect-1/valid', 'Player_detect-1/Player_detect-1/valid')

# Training the model - Using Yolo Medium - Moderate accuracy and speed
model = YOLO("yolov5m.pt")

# Explicitly set the device to GPU (cuda) or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.train(data=f"{dataset.location}/data.yaml", epochs=100, imgsz=640, device=device, workers=2)
