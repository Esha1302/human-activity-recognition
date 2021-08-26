import time
import random

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import numpy as np

from backbone import *
from utils import *

class Basenet_okutama(nn.Module):
    """
    main module of base model for collective dataset
    """
    def __init__(self):
        super(Basenet_okutama, self).__init__()
        
        D=512 #output feature map channel of backbone
        K=5 #crop size of roi align
        NFB=1024
        
        # self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        self.backbone=MyVGG16(pretrained=True)
        
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad=False
        crop_size = 5 , 5
        self.roi_align=RoIAlign(*crop_size)
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.dropout_emb_1 = nn.Dropout(p=0.3)
        self.nl_emb_1=nn.LayerNorm([NFB])
        
        num_actions = 12

        self.fc_actions=nn.Linear(NFB,num_actions)
        # self.fc_activities=nn.Linear(NFB,self.cfg.num_activities)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self,filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict':self.fc_emb_1.state_dict(),
            'fc_actions_state_dict':self.fc_actions.state_dict(),
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)
        

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
        
                
    def forward(self,batch_data):

        images_in, boxes_in, bboxes_num_in = batch_data
        # print("shape of bboxes_num_in before = ",bboxes_num_in.shape)
        # print("bboxes_num_in before = ",bboxes_num_in)
        # read config parameters
        B=images_in.shape[0]
        # print("B = ", B)
        
        T=images_in.shape[1]
        # print("T = ", T)
        s = 5
        W, H= int(3840 / s), int(2160 / s)
        OH, OW=87,157
        MAX_N=12
        NFB=1024
        EPS=1e-5
        
        # D=1056
        D=512
        K=5
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,W,H))  #B*T, 3, H, W
        boxes_in=boxes_in.reshape(B*T,MAX_N,4)
                
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
        
        # print("shape of outputs obtained after preproccessing = ",len(outputs))

        
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            # print(features.shape)
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        # print("shape of features_multiscale = ", features_multiscale.shape)

        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))  #B*T*MAX_N, 4
        # print("shape of boxes_in_flat = ",boxes_in_flat.shape)
        # print("shape of boxes_in = ",boxes_in.shape)
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        # print("shape of boxes_idx = ",boxes_idx.shape)
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))  #B*T*MAX_N,
        # print("shape of boxes_idx_flat = ",boxes_idx_flat.shape)

        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features_all=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*MAX_N, D, K, K,
        # print("shape of boxes_features_all before roi_align = ",boxes_features_all.shape)
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K
        # print("shape of boxes_features_all after roi_align = ",boxes_features_all.shape)
        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all=F.relu(boxes_features_all)
        boxes_features_all=self.dropout_emb_1(boxes_features_all)
        # print("shape of boxes_features_all = ",boxes_features_all.shape)
        
    
        actions_scores=[]
        activities_scores=[]
        bboxes_num_in=bboxes_num_in.reshape(B*T,)  #B*T,
        # print("shape of bboxes_num_in = ",bboxes_num_in.shape)
        # print("bboxes_num_in = ",bboxes_num_in)
        for bt in range(B*T):
            # print("for bt = ", bt)
            N=bboxes_num_in[bt]
            # print("shape of N = ",N)

            if N > MAX_N:
              N = MAX_N

            boxes_features=boxes_features_all[bt,:N,:].reshape(1,N,NFB)  #1,N,NFB
            # print("shape of boxes_features = ",boxes_features.shape)
    
            boxes_states=boxes_features  
            # print("shape of boxes_state = ",boxes_states.shape)
            NFS=NFB
            # print("NFS = ",NFS)
            # Predict actions
            boxes_states_flat=boxes_states.reshape(-1,NFS)  #1*N, NFS
            # print("shape of boxes_state_flat = ",boxes_states_flat.shape)
            actn_score=self.fc_actions(boxes_states_flat)  #1*N, actn_num

            final_actn_score = torch.ones(MAX_N) * -1
            # temp_actn_score = actn_score.cpu().numpy()
            temp_actn_score = torch.argmax(actn_score,dim=1)
            final_actn_score[:N] = temp_actn_score
            actions_scores.append(final_actn_score) #B*T,number of actions

            # print("shape of actn_score = ",actn_score.shape)
            # actions_scores.append(actn_score)
            # print(len(actions_scores))
            # # Predict activities
            # boxes_states_pooled,_=torch.max(boxes_states,dim=1)  #1, NFS
            # boxes_states_pooled_flat=boxes_states_pooled.reshape(-1,NFS)  #1, NFS
            # acty_score=self.fc_activities(boxes_states_pooled_flat)  #1, acty_num
            # activities_scores.append(acty_score)

        # npaction_scores = np.array(actions_scores)
        # print("numpy array shape = ",npaction_scores[0].shape)
        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        # activities_scores=torch.cat(activities_scores,dim=0)   #B*T,acty_num
        actions_scores = torch.reshape(actions_scores,(2,5,12))
        # print(actions_scores.shape)
        # print(activities_scores.shape)

        return actions_scores
