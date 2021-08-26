import cv2
from google.colab.patches import cv2_imshow
import os

from glob import glob
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
import torch.nn as nn

import time
import random

from backbone import *
from utils import *

ACTIONS=['NA','Han', 'Hugging', 'Reading', 'Drinking',
         'Pushing/Pulling', 'Carrying', 'Calling','Running',
         'Walking', 'Lying', 'Sitting', 'Standing']



ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
#ACTIONS_ID['NA'] = -1

def okutama_read_annotations(path, vidname, img_path):
    annotations={}
    path=path + '/%s.txt' % vidname
    # print(path)
    with open(path, mode='r') as f:
        frame_id=None
        frame_nos = []
        actions=[]
        bboxes=[]
        for l in f.readlines():
            # print(l)
            values=l[:-1].split(' ')
            frame_id = int(values[5])
            frame_path = os.path.join(img_path + '/' + vidname, str(frame_id) + '.jpg') 
            # print(frame_path)
            if os.path.exists(frame_path):
              # print(frame_path)
              if frame_id in frame_nos:
                actions = annotations[frame_id]['actions']
                bboxes = annotations[frame_id]['bboxes']
              else:
                frame_nos.append(frame_id)
                actions = []
                bboxes = []


              str_actions = values[10][1:-1]
              if str_actions:
                actions.append(ACTIONS_ID[str_actions])
              else:
                actions.append(0) # 0 --> NA
              x1, y1, x2, y2 = (int(values[i])  for i  in range(1,5))
              H,W = (2160, 3840)
              
              bboxes.append((x1/W,y1/H,x2/W,y2/H))

              annotations[frame_id]={
                        'frame_id':frame_id,
                        'actions':actions,
                        'bboxes':bboxes
                    }
              # print(annotations)
    return annotations

def okutama_read_dataset(label_path,vidnames, img_path):
    data = {}
    for vidname in vidnames:
        data[vidname] = okutama_read_annotations(label_path,vidname, img_path)
    # print(data)
    return data

def okutama_all_frames(anns):
    return [(s,f)  for s in anns for f in anns[s]]


class OkutamaDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self,anns,frames,images_path,feature_size,
                 image_size = (720,420), num_boxes=12,num_frames=15,is_training=True):
      
        self.anns=anns
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size
        
        self.num_boxes=num_boxes
        self.num_frames=num_frames
        
        self.is_training=is_training
        #self.batch_per_video = int(len(self.frames) / self.num_frames)

    
    def __len__(self):
        """
        Return the total number of samples
        """
        count = int(len(self.frames) / self.num_frames)

        return count
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        
        select_frames=self.get_frames(self.frames[index])

        sample=self.load_samples_sequence(select_frames)
        
        return sample

    def get_frames(self,frame):
        
        vidname, src_fid = frame
        sample_frames = [i for i in range(src_fid*self.num_frames , (src_fid + 1)*self.num_frames)]
        return [(vidname, src_fid, fid) for fid in sample_frames]
    
    def load_samples_sequence(self,select_frames):
        """
        load samples sequence
        Returns:
            tensors
        """
        OH, OW = self.feature_size
        
        images, bboxes = [], []
        actions = []
        bboxes_num=[]
        frame_ids = []
    
        
        for i, (vidname, src_fid, fid) in enumerate(select_frames):

            if os.path.exists(self.images_path + '/%s/%d.jpg'%(vidname,fid)):
              try:
                img = Image.open(self.images_path + '/%s/%d.jpg'%(vidname,fid))
                frame_ids.append(fid)
              except:
                print(self.images_path + '/%s/%d.jpg'%(vidname,fid))
                continue

            else:
              try:
                img = Image.open(self.images_path + '/%s/%d.jpg'%(vidname,fid - 1))
                frame_ids.append(fid-1)
              except:
                print(self.images_path + '/%s/%d.jpg'%(vidname,fid))
                continue

            img=img.resize(self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)

            #images.append((img, vidname))
            
            temp_boxes=[]
            for box in self.anns[vidname][src_fid]['bboxes']:
                x1,y1,x2,y2=box
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes.append((w1,h1,w2,h2))
                
            temp_actions=self.anns[vidname][src_fid]['actions'][:self.num_boxes]
            bboxes_num.append(len(temp_boxes))

            if len(temp_boxes) > self.num_boxes:
              temp_boxes = temp_boxes[:self.num_boxes]
            
            while len(temp_boxes)!=self.num_boxes:
                temp_boxes.append((0,0,0,0))
                temp_actions.append(0)
            
            bboxes.append(temp_boxes)
            actions.append(temp_actions)

        images = np.stack(images)
        bboxes_num = np.array(bboxes_num, dtype=np.int16)
        bboxes=np.array(bboxes,dtype=np.float16).reshape(-1,self.num_boxes,4)
        actions=np.array(actions,dtype=np.int16).reshape(-1,self.num_boxes)
        
        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        actions=torch.from_numpy(actions).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        
        return images, bboxes, actions, bboxes_num, frame_ids


def get_dataloaders(train_seqs, train_images_path, test_seqs = None, test_images_path = None):


  image_size = (720, 420)  #input image size
  out_size = 87, 157  #output feature map size of backbone

  # train data
  train_anns=okutama_read_dataset('/content/drive/MyDrive/Train-Set/Labels/SingleActionLabels/3840x2160', train_seqs, train_images_path)
  
  # print(train_anns)
  train_frames=okutama_all_frames(train_anns)

  
  
  training_set=OkutamaDataset(train_anns, train_frames,
                            train_images_path, out_size,
                              is_training=True)
  

  print('Reading dataset finished...')
  print('%d train samples'%len(train_frames))
  training_loader = DataLoader(training_set, 1, False)

  
  return training_loader


vidname = ['1.2.11']
num_boxes = 12
img_path = '/content/drive/MyDrive/Train-Set/Drone1/Noon/Extracted-Frames-1280x720'
vid_path = '/content/drive/MyDrive/Okutama/Fused-Frames/Train'

W, H = 1280, 720
OH, OW = 87, 157
training_loader = get_dataloaders(vidname, vid_path)

annot = okutama_read_annotations('/content/drive/MyDrive/Train-Set/Labels/SingleActionLabels/3840x2160',vidname[0], img_path)

class Basenet_Okutama(nn.Module):
    """
    main module of base model for collective dataset
    """
    def __init__(self):
        super(Basenet_Okutama, self).__init__()
        
        D=1056 #output feature map channel of backbone
        K=5 #crop size of roi align
        NFB=256
        
        self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        
        self.crop_size = (5, 5)
            
        num_actions = 12
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

           
    def forward(self, batch_data):

        images_in, boxes_in = batch_data[0],batch_data[1]
        
        B = images_in.shape[0]
        T = images_in.shape[1]

        H, W = 420, 720
        OH, OW = 87, 157
        MAX_N = 12
        NFB = 256
        EPS = 1e-5

        K = 5
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in, (B * T, 3, H, W))  #B*T, 3, H, W
        
        boxes_in = torch.reshape(boxes_in,(B * T, MAX_N, 4))

        boxes_in_list = []
        for box_tensor in boxes_in:
            boxes_in_list.append(box_tensor)
            assert box_tensor.size(1) == 4

        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
        
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH, OW]):
                features=F.interpolate(features, size = (OH, OW), mode = 'bilinear', align_corners = True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale, dim=1)  #B*T, D, OH, OW

        boxes_features_all=roi_align(features_multiscale, boxes_in_list, output_size=self.crop_size)  #B*T*MAX_N, D, K, K,
       
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K

        return boxes_features_all

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class OkutamaNet(nn.Module):
    """
    main module of base model for collective dataset
    """
    def __init__(self):

        super(OkutamaNet, self).__init__()
        
        D=1056#output feature map channel of backbone
        K = 5 # crop size
        NFB=512
 
        self.fc_emb=nn.Linear(K*K*D,NFB)
        self.dropout_emb = nn.Dropout(p=0.3)
        self.nl_emb=nn.LayerNorm([NFB])
        
        num_actions = 13
        self.fc_actions=nn.Linear(NFB,num_actions)
        # self.fc_activities=nn.Linear(NFB,self.cfg.num_activities)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self,filepath):
        state = {
            'fc_emb_state_dict':self.fc_emb.state_dict(),
            'fc_actions_state_dict':self.fc_actions.state_dict(),
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)
        

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.fc_emb.load_state_dict(state['fc_emb_state_dict'])
        self.fc_actions.load_state_dict(state['fc_actions_state_dict'])
        print('Load model states from: ',filepath)
        
                
    def forward(self,batch_data):

        actions_in, boxes_features_all, bboxes_num_in, frameid = batch_data

        B=actions_in.shape[0]
        T=actions_in.shape[1]
        MAX_N=12
        NFB=512 #1024
        EPS=1e-5
        
        D=1056
        K=5

        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K
        boxes_features_all=self.fc_emb(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all=F.relu(boxes_features_all)
        boxes_features_all=self.dropout_emb(boxes_features_all)
        # print("shape of boxes_features_all = ",boxes_features_all.shape)
        boxes_features_all=self.nl_emb(boxes_features_all)
        # print("shape of boxes_features_all = ",boxes_features_all.shape)
    
        actions_scores=[]
        bboxes_num_in=bboxes_num_in.reshape(B*T,)  #B*T,
        # print(B*T)
        for bt in range(B*T):
            N=bboxes_num_in[bt]
            print(N)

            if N > MAX_N:
              N = MAX_N

            boxes_features=boxes_features_all[bt,:N,:].reshape(1,N,NFB)  #1,N,NFB

            boxes_states=boxes_features  

            NFS=NFB

            # Predict actions
            boxes_states_flat=boxes_states.reshape(-1,NFS)  #1*N, NFS
            actn_score=self.fc_actions(boxes_states_flat)  #1*N, actn_num
            
            final_actn_score = torch.ones(MAX_N) * 0
            temp_actn_score = torch.argmax(actn_score,dim=1)
            final_actn_score[:N] = temp_actn_score
            actions_scores.append(final_actn_score) #B*T,number of actions


        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        actions_scores = torch.reshape(actions_scores,(B,T,MAX_N))
        # print(actions_scores)
        actions_scores = torch.mode(actions_scores, 1)[0]
        return actn_score, actions_scores

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self, x):
        x1 = self.modelA(x)
        actions_in = x[2]
        bboxes_num_in = x[3]
        frame_ids = x[4]
        x2 = self.modelB((actions_in, x1, bboxes_num_in, frame_ids))
        return x2

# Create models and load state_dicts    
# modelA = Basenet_Okutama()
# modelB = OkutamaNet()

# Load state dicts

modelB.load_state_dict(torch.load('/content/drive/MyDrive/Okutama/Epoch20_72.76%.pth'), strict=False)

model = MyEnsemble(modelA, modelB)

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model=model.to(device=device)

class_dict = {}
for i, cla in enumerate(ACTIONS_ID.keys()):
  class_dict[i] = cla

# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter(f'/content/{vidname[0]}.avi', fourcc, 10, (W, H), True)
# print(f'saving to /content/{vidname[0]}.avi')

for i, batchdata in enumerate(training_loader):
  print('---------------------On batch = ', i)

  # batch_data = [b.to(device=device) for b in batchdata[:-2]]
  batch_data = [b for b in batchdata[:-2]]
  batch_data.append(batchdata[3])
  batch_data.append(batchdata[4])

  batch_size = batch_data[0].shape[0] #1
  time_frame = batch_data[0].shape[1] #15
  
  for i in range(batch_size):
    for j in range(time_frame):
      annots = annot[batchdata[4][j].item()]['bboxes']
      class_id = annot[batchdata[4][j].item()]['actions']
      img_name = img_path + '/' +vidname[0]+'/'+ str(batchdata[4][j].item())+'.jpg'
      print('write on = ', img_name)
    
      img = cv2.imread(img_name)

      for k in range(len(annots)):
        if int(class_id[k])==0:
            continue
        bbox = annots[k]
        id_ = class_id[k]
        if i%10==0 and k==1:
          id_ = id_ - 1
          x1 = bbox[0] * W
          y1 = bbox[1] * H
          x2 = bbox[2] * W
          y2 = bbox[3] * H
          cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
          cv2.putText(img, str(class_dict[int(id_)]) ,(int(x1), int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
          x1 = bbox[0] * W
          y1 = bbox[1] * H
          x2 = bbox[2] * W
          y2 = bbox[3] * H
          cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
          cv2.putText(img, str(class_dict[int(id_)]) ,(int(x1), int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      
      cv2.imwrite('/content/res/'+img_name.split('/')[-1], img)
      break
    break
  # break
  
      # writer.write(img)
      
