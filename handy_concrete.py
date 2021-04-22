from handy_core import IHandyNetwork
import cv2
import torch
from torchvision import transforms
from utils.transforming import ToTensor, Normalize
from model import Handy
import numpy as np
import torch.nn.functional as F
#from PIL import Image

"""
This is the network that should be implemented
"""
class HandyConcrete(IHandyNetwork):
    def __init__(self,):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])
        self.seq_length = 128
        self.probs = []
        self.images = [] # ??? IDK if we should store it in memory
        self.model = Handy(pretrain=True,
                        width_mult=1.,
                        lstm_layers=1,
                        lstm_hidden=256,
                        device = self.device,
                        bidirectional=True,
                        dropout=False)
        try:
            save_dict = torch.load('models/net_v2_1600.pth.tar')
        except:
            print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print("Loaded model weights")

    def get_probs(self, images):
        batch = 0
        while batch * self.seq_length < images.shape[1]:
            if (batch + 1) * self.seq_length > images.shape[1]:
                image_batch = images[:, batch * self.seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * self.seq_length:(batch + 1) * self.seq_length, :, :, :]
            logits = self.model(image_batch.to(self.device))
            if batch == 0:
                self.probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                self.probs = np.append(self.probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1


    def predict_handwashing_time_live(self, frames:list) -> (bool, int):

        return False, 14

    def predict_handwashing_time(self, frames:list) -> int:
        images = []
        for i, frame in enumerate(frames):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(img)
        labels = np.zeros(len(images)) # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        sample = self.transform(sample)        
        images = sample['images']
        images = images.view(1, images.shape[0], images.shape[1], images.shape[2], images.shape[3]) 
        self.get_probs(frames)
        events = np.argmax(probs, axis=0)[:-1]
        print('Predicted event frames: {}'.format(events)) 
        confidence = []
        for i, e in enumerate(events):
            confidence.append(probs[e, i])
        print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

        time = (events[-1]-events[0])//10        
        return time


class HandyConcreteFake(IHandyNetwork):

    def predict_handwashing_time_live(self, frames:list) -> (bool, int):
        return False, 14

    def predict_handwashing_time(self, frames:list) -> int:
        return 21


# This module should have the object initialized
handy = HandyConcrete()