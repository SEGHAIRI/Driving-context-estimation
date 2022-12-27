from config import Config
from hydranet import HydraFusion
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch
from torch.utils.data import random_split
from radiate import Sequence
import torch.nn as nn
import torch.optim as optim

#Import Hydrafusion driving context module
args= ''
configuration = Config(args)

Hydranet = HydraFusion(configuration)

print(Hydranet)

#Context driving classes
classe_id = {'city': 0, 'night' : 1, 'fog' : 2, 'snow' : 3, 'rain' : 4}

#Class to import and prepare data
class Radiate(Dataset):

    def __init__(self, root_dir, sequence):
        self.root = root_dir
        self.seq=sequence
        self.Navtech_Cartesian = root_dir + '/Navtech_Cartesian'
        with open(root_dir + '/Navtech_Cartesian.txt') as f:
            self.lines = f.readlines()
        dell = []
        for i in range(len(self.lines)):
            Time_radar = self.lines[i][self.lines[i].find('Time')+6:]
            outputs = self.seq.get_from_timestamp(t=float(Time_radar), get_sensors=True, get_annotations=True)
        
            if outputs == {}:
                dell.append(i)

        for j in range(len(dell)):
            del self.lines[dell[j]-j]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):

        annotation = np.array([0,0,0,0,0])
        annotation[classe_id[self.root[14:-4]]] = 1
        #Radar data
        radar_cart = os.path.join(self.Navtech_Cartesian,
                                self.lines[idx][self.lines[idx].find(' ')+1:self.lines[idx].rfind(' ')-6]+'.png')
        radar_cart = plt.imread(radar_cart)
        radar_cart = radar_cart.reshape(1,radar_cart.shape[0],radar_cart.shape[1])
        # Annotation transformation
        Time_radar = self.lines[idx][self.lines[idx].find('Time')+6:]
        outputs = self.seq.get_from_timestamp(t=float(Time_radar), get_sensors=True, get_annotations=True)
        
        #Right camera data
        image_camera_right = np.array(outputs['sensors']['camera_right_rect'], dtype = float)
        image_camera_right = image_camera_right.reshape(image_camera_right.shape[2],image_camera_right.shape[0],image_camera_right.shape[1])
        image_camera_left = np.array(outputs['sensors']['camera_left_rect'], dtype = float)
        image_camera_left = image_camera_left.reshape(image_camera_left.shape[2],image_camera_left.shape[0],image_camera_left.shape[1])
        #LiDAR data
        lidar_bev_image = np.array(outputs['sensors']['lidar_bev_image'], dtype = float)
        lidar_bev_image = lidar_bev_image.reshape(lidar_bev_image.shape[2],lidar_bev_image.shape[0],lidar_bev_image.shape[1])
            
        return image_camera_right, image_camera_left, lidar_bev_image, radar_cart, annotation

#In this case I used 5 radiate data sequances, 
path = "/content/data"
dir_list = os.listdir(path)

root_dir = "/content/data/" + dir_list[0]
sequence = Sequence(sequence_path=root_dir, config_file='/content/config.yaml')
data = Radiate(root_dir,sequence)
for bb in  range(1,len(dir_list)):
    root_dir = "/content/data/" + dir_list[bb]
    sequence = Sequence(sequence_path=root_dir, config_file='/content/config.yaml')
    data = torch.utils.data.ConcatDataset([data, Radiate(root_dir,sequence)])

#Devide the dataset in training, validation and test data 
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.2)
train_size = len(data) - val_size - test_size

generator = torch.Generator()
generator.manual_seed(0)

train_ds, val_ds, test_ds = random_split(data, [train_size, val_size, test_size], generator=generator)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
valid_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

#Prepare the model to training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Hydranet
model = model.to(device)
checkpoint = torch.load('/content/Best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (image_camera_right, image_camera_left, lidar_bev_image, radar_cart, targets) in enumerate(test_loader):
            image_camera_right = image_camera_right.to(device=device)
            image_camera_left = image_camera_left.to(device=device)
            lidar_bev_image = lidar_bev_image.to(device=device)
            radar_cart = radar_cart.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            scores = model.forward(leftcamera_x=image_camera_left, rightcamera_x=image_camera_right, radar_x=radar_cart, bev_lidar_x=lidar_bev_image)
            val, predictions = scores.reshape(1,5).max(1, keepdim=True)
            val, gnd = targets.max(1, keepdim=True)
            num_correct += np.array((gnd==predictions).cpu())[0][0]*1
            num_samples += predictions.size(0)
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )




