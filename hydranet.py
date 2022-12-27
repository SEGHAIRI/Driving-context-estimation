import torch
import torch.nn as nn
import torch.nn.functional as F
from stem import RadarStem, CameraStem, LidarStem
from torchvision.models.resnet import BasicBlock
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from typing import List
from gate import KnowledgeBasedGateModule, AttentionGatingModule, DeepGatingModule


'''This file defines our HydraNet-based sensor fusion architecture.'''
class HydraFusion(nn.Module):

    def __init__(self, config):
        super(HydraFusion, self).__init__()
        self.config = config
        self.dropout = config.dropout
        self.activation = F.relu if config.activation == 'relu' else F.leaky_relu
        self.initialize_transforms()
        self.initialize_stems()
        self.initialize_Gate()
    '''initializes the normalization/resizing transforms applied to input images.'''
    def initialize_transforms(self):
        if self.config.use_custom_transforms:
            self.image_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[88.12744903564453,90.560546875,90.5104751586914], image_std=[66.74466705322266,74.3885726928711,75.6873779296875])
            self.radar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[15.557413101196289,15.557413101196289,15.557413101196289], image_std=[18.468725204467773,18.468725204467773,18.468725204467773])
            self.lidar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[2.1713976860046387,2.1713976860046387,2.1713976860046387], image_std=[20.980266571044922,20.980266571044922,20.980266571044922])
            self.fwd_lidar_transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[0.0005842918762937188,0.0005842918762937188,0.0005842918762937188], image_std=[0.10359727591276169,0.10359727591276169,0.10359727591276169])
        else:
            self.transform = GeneralizedRCNNTransform(min_size=376, max_size=1000, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]) #from ImageNet
            self.image_transform = self.transform
            self.radar_transform = self.transform
            self.lidar_transform = self.transform
            self.fwd_lidar_transform = self.transform


    '''initializes the stem modules as the first blocks of resnet-18.'''
    def initialize_stems(self):
        if self.config.enable_radar:
            self.radar_stem = RadarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)  # TODO:define these config values in config.py
        if self.config.enable_camera:  
            self.camera_stem = CameraStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)
        if self.config.enable_lidar:
            self.lidar_stem = LidarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)
        if self.config.enable_cam_lidar_fusion:
            self.fwd_lidar_stem = LidarStem(BasicBlock, [2, 2, 2, 2], pretrained=self.config.pretrained).to(self.config.device)


    '''initializes the Gate modules.'''
    def initialize_Gate(self):
        if self.config.gate == 'KnowledgeBasedGateModule':
            self.Gate = KnowledgeBasedGateModule()
        if self.config.gate == 'AttentionGatingModule': 
            self.Gate = AttentionGatingModule(64, 5, self.dropout) 
        if self.config.gate == 'DeepGatingModule':
            self.Gate = DeepGatingModule(64, 5, self.dropout)
    '''
    <sensor>_x is in the input image/sensor data from each modality for a single frame. 
    radar_y, cam_y contains the target bounding boxes for training BEV and FWD respectively.
    Currently. all enabled branches are executed for every input.
    '''
    def forward(self, leftcamera_x=None, rightcamera_x=None, radar_x=None, bev_lidar_x=None):
        
        if self.config.enable_camera:
            rightcamera_x, _ = self.image_transform(rightcamera_x)
            leftcamera_x, _ = self.image_transform(leftcamera_x)
            l_camera_output = F.dropout(self.camera_stem((leftcamera_x.tensors).float().to(self.config.device)), self.dropout, training=self.training)
            r_camera_output = F.dropout(self.camera_stem((rightcamera_x.tensors).float().to(self.config.device)), self.dropout, training=self.training)
    
        if self.config.enable_lidar:
            bev_lidar_x, _ = self.lidar_transform(bev_lidar_x)
            bev_lidar_output = F.dropout(self.lidar_stem((bev_lidar_x.tensors).float().to(self.config.device)), self.dropout, training=self.training)
        
        if self.config.enable_radar:
            radar_x, _ = self.radar_transform(radar_x)
            radar_output = F.dropout(self.radar_stem((radar_x.tensors).to(self.config.device)), self.dropout, training=self.training)
        inputt  = torch.cat((l_camera_output,l_camera_output,bev_lidar_output, radar_output), 3)
        output = self.Gate(inputt)
        
        return output





