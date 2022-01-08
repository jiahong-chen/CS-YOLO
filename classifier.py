import os
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from dataclasses import dataclass

@dataclass
class Classifier:
    arch: str = 'vgg19'
    checkpoint_path: str = './model/model_best.pth.tar'
    label_path: str = './label/obj.txt'

    def __post_init__(self):
        self.model = self.load_checkpoint()
        self.label = self.load_label()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def load_checkpoint(self):
        print("=> creating model '{}'".format(self.arch))
        model = models.__dict__[self.arch]()
        model.to(self.device)

        if os.path.exists(self.checkpoint_path):
            print('loading checkpoint...')
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("please provide a valid checkpoint!")
        return model
    
    def load_label(self):
        label = list()
        with open(self.label_path, 'r') as f:
            for line in f:
                label.append(line.rstrip())
        return label
    
    @torch.no_grad()
    def classification(self, img):
        torch.eval()

        transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_ = transform(img)
        img_ = img_.squuze(0)

        input = img_.to(self.device)
        output = self.model(input)

        _, indices = torch.max(output, 1)
        percentage = nn.function.softmax(output, dim=1)[0]*100

        confidence = percentage[int(indices)].items()
        predict = self.label[int(indices)]
        return predict, confidence