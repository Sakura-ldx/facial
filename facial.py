import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model.resnet import resnet50
from dataset.CelebA import CelebA


class FacialPredictor(object):
    """
    facail attributes and landmarks
    """

    def __init__(self, args):
        self.batch_num = args.batch_num
        self.device = args.device
        
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 4)
        self.model.load_state_dict(torch.load(args.checkpoint_path))
        self.model.eval()
        self.model.to(torch.device(device))
        
    def transform_image(self, image):
        my_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return my_transforms(image).unsqueeze(0)
                                                                                                                                             
    def __call__(self, imgs):
        img_num = len(imgs)
        batch_num = self.batch_num
        # pre_vals_list, pre_ids_list = [], []
        for begin_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, begin_img_no + batch_num)
            imgs_tf = [
                self.transform_image(imgs[ino])
                for ino in range(begin_img_no, end_img_no)
            ]
            tensor = torch.cat(imgs_tf).cuda()
            with torch.no_grad():
                outputs = self.model(tensor)
            # if len(imgs_tf) == 1:
            #     outputs = outputs.unfold(0, 1, 2).mean(dim=1)
            # else:
            #     outputs = outputs.unfold(0, 2, 2).mean(dim=2)

            outputs = (outputs > 0).cpu().numpy()
            outputs = pd.DataFrame(outputs, columns=['glasses', 'gender', 'smile', 'age'])
        return outputs
