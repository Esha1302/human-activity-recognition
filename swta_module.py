import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os

from google.colab.patches import cv2_imshow

path=Path('/content/drive/MyDrive/Drone-Action/all-frames/1.1.6')

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def get_dense_optical_flow(arr1,arr2):
    prvs=rgb2gray(arr1)
    next=rgb2gray(arr2)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv = np.zeros_like(prvs_rgb)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb

def normalize_image_array(arr): 
    return (arr-arr.min())*255/arr.max()

def show_im(rgb,prvs_rgb,mult=1): 
    arr=np.array(np.multiply(prvs_rgb,rgb)+mult*prvs_rgb,dtype=np.uint8)
    return arr,Image.fromarray(arr)

def get_wta(rgb_frames, tsn_frames, bboxes, 
            org_size = (3840, 2160), target_size = (1280, 720)):
    """
    This function fuses OF frames with each RGB Frame
    """
    frame1, frame2, frame3 = tsn_frames
    crop1 = np.ones_like(frame1)*10
    crop2 = np.ones_like(frame2)*10
    crop3 = np.ones_like(frame3)*10

    for bbox in bboxes:
        x1 = int(np.ceil(bbox[0] * target_size[0] / org_size[0]))
        x2 = int(np.ceil(bbox[2] * target_size[0] / org_size[0]))
        y1 = int(np.ceil(bbox[1] * target_size[0] / org_size[0]))
        y2 = int(np.ceil(bbox[3] * target_size[0] / org_size[0]))

        crop1[y1-40:y2+40, x1-40:x2+40] = frame1[y1-40:y2+40, x1-40:x2+40]
        
        crop2[y1-40:y2+40, x1-40:x2+40] = frame2[y1-40:y2+40, x1-40:x2+40]
        
        crop3[y1-40:y2+40, x1-40:x2+40] = frame3[y1-40:y2+40, x1-40:x2+40]

    OF_prev=get_dense_optical_flow(crop1,crop2)
    OF_next=get_dense_optical_flow(crop2,crop2)

    fused_frames = np.zeros_like(rgb_frames)
    for i, frame in enumerate(rgb_frames):
        fused_frames[i], _ =show_im(OF_prev/250 + OF_next/250,frame)

    return fused_frames
