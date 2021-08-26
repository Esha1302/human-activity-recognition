from glob import glob
import os
import random
from PIL import Image
import numpy as np

# create a dictionary containg frame num corresponding to video names
FRAMES_NUM = {}
for name in glob('/content/drive/MyDrive/Train-Set/*/*/Extracted-Frames-1280x720/*'):
# for name in glob('/content/drive/MyDrive/Drone-Action/datasets/Train-Set/*/*/Extracted-Frames-1280x720/*'):
  key = name.split('/')[-1]
  FRAMES_NUM[key] = len(os.listdir(name))

ACTIONS=['NA','Han', 'Hugging', 'Reading', 'Drinking',
         'Pushing/Pulling', 'Carrying', 'Calling','Running',
         'Walking', 'Lying', 'Sitting', 'Standing']



ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
ACTIONS_ID['NA'] = -1

def okutama_read_annotations(path,vidname, img_path):
    annotations={}
    path=path + '/%s.txt' % vidname

    with open(path,mode='r') as f:
       
        frame_id=None
        frame_nos = []
        actions=[]
        bboxes=[]
        for l in f.readlines():
            values=l[:-1].split(' ')
           
            
            frame_id = int(values[5])
            
            frame_path = os.path.join(img_path + '/' + vidname, str(frame_id) + '.jpg') 
            
            str_actions = values[10][1:-1]
            if os.path.exists(frame_path):
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
                actions.append(-1) # -1 --> NA
              # x,y,w,h = (int(values[i])  for i  in range(1,5))
              x1, y1, x2, y2 = (int(values[i])  for i  in range(1,5))
              W,H=(3840, 2160)
              
              # bboxes.append((y/H,x/W,(y+h)/H,(x+w)/W))
              bboxes.append((y1/H, x1/W, y2/H, x2/H))

              annotations[frame_id]={
                        'frame_id':frame_id,
                        'actions':actions,
                        'bboxes':bboxes
                    }
              # print("annotations = ", annotations)
              

    return annotations

def okutama_read_dataset(label_path,vidnames, img_path):
    data = {}
    for vidname in vidnames:
        data[vidname] = okutama_read_annotations(label_path,vidname, img_path)
        # print(vidname)
    return data

def okutama_all_frames(anns):
    return [(s,f)  for s in anns for f in anns[s]]

class OkutamaDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self,anns,frames,images_path,feature_size,
                 image_size = (512,512), num_boxes=12,num_frames=5,is_training=True):
      
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
        # print(count)
        return count
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        if self.frames[index * self.num_frames][0] == self.frames[(index+1)*self.num_frames][0]:
          select_frames=self.get_frames(self.frames[index * self.num_frames])
          
          sample=self.load_samples_sequence(select_frames)
          
          return sample

        else:
          vidname, src_fid = self.frames[index * self.num_frames]

          diff = self.num_frames - (FRAMES_NUM[self.frames[index * self.num_frames][0]] - src_fid) 
          # 10 - (2272 -2270) = 8 frames 
          # print("no. of frames: ", FRAMES_NUM[self.frames[index * self.num_frames][0]])
          # print(index*self.num_frames)

          # print(diff)

          vidname, src_fid = self.frames[index * self.num_frames]


          sample_frames = [i for i in range(src_fid - diff, src_fid -diff + self.num_frames)]
          # print(sample_frames)
          select_frames = [(vidname, src_fid, fid) for fid in sample_frames]

          # print(select_frames)
          
          
          sample=self.load_samples_sequence(select_frames)
          
          return sample


    
    def get_frames(self,frame):
        
        vidname, src_fid = frame

        # print(frame)

        sample_frames = [i for i in range(src_fid , src_fid + self.num_frames)]
        return [(vidname, src_fid, fid) for fid in sample_frames]

    
    
    def load_samples_sequence(self,select_frames):
        """
        load samples sequence
        Returns:
            tensors
        """
        OH, OW=self.feature_size
        
        images, bboxes = [], []
        actions = []
        bboxes_num=[]
    
        
        for i, (vidname, src_fid, fid) in enumerate(select_frames):

            if os.path.exists(self.images_path + '/%s/%d.jpg'%(vidname,fid)):
              img = Image.open(self.images_path + '/%s/%d.jpg'%(vidname,fid))

            else:
              img = Image.open(self.images_path + '/%s/%d.jpg'%(vidname,fid - 1))

            img=img.resize(self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)
            
            temp_boxes=[]
            for box in self.anns[vidname][src_fid]['bboxes']:
                y1,x1,y2,x2=box
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes.append((w1,h1,w2,h2))
                
            temp_actions=self.anns[vidname][src_fid]['actions'][:self.num_boxes]
            bboxes_num.append(len(temp_boxes))

            if len(temp_boxes) > self.num_boxes:
              #print('Error: More than 12 actions present')
              temp_boxes = temp_boxes[:self.num_boxes]
            
            while len(temp_boxes)!=self.num_boxes:
                temp_boxes.append((0,0,0,0))
                temp_actions.append(-1)
            
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            # print(np.array(actions))
        
        # print(np.array(actions).shape)
        images = np.stack(images)
        bboxes_num = np.array(bboxes_num, dtype=np.int16)
        bboxes=np.array(bboxes,dtype=np.float16).reshape(-1,self.num_boxes,4)
        actions=np.array(actions,dtype=np.int16).reshape(-1,self.num_boxes)
        
        #convert to pytorch tensor
        images=torch.from_numpy(images, ).float()
        bboxes=torch.from_numpy(bboxes).float()
        actions=torch.from_numpy(actions).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        
        return images, bboxes, actions, bboxes_num

def get_dataloaders(train_seqs, test_seqs, train_images_path, test_images_path):

  s = 5
  image_size = int(3840 / s), int(2160 / s)  #input image size
  out_size = 87, 157  #output feature map size of backbone


  # train data
  train_anns=okutama_read_dataset('/content/drive/MyDrive/Train-Set/Labels/SingleActionLabels/3840x2160', train_seqs, train_images_path)
  # train_anns=okutama_read_dataset('/content/drive/MyDrive/Drone-Action/datasets/Train-Set/Labels/SingleActionLabels/3840x2160', train_seqs, train_images_path)
  train_frames=okutama_all_frames(train_anns)
  
  
  training_set=OkutamaDataset(train_anns,train_frames,
                            train_images_path,out_size,image_size,
                              is_training=True)
  
  # # #test data
  test_anns=okutama_read_dataset('/content/drive/MyDrive/Test-Set/Labels/SingleActionLabels/3840x2160', test_seqs,test_images_path)
  # # test_anns=okutama_read_dataset('/content/drive/MyDrive/Drone-Action/datasets/Test-Set/Labels/SingleActionLabels/3840x2160', 
  # #                                test_seqs, test_images_path)
  test_frames=okutama_all_frames(test_anns)

  testing_set=OkutamaDataset(test_anns,test_frames,
                            test_images_path,out_size,image_size,
                              is_training=True)
  

  print('Reading etdataset finished...')
  print('%d train samples'%len(train_frames))
  print('%d test samples'%len(test_frames))
  
  #get dataLoaders
  training_loader = DataLoader(training_set, 2, True)
  testing_loader = DataLoader(testing_set, 2, True)
  
  return training_loader, testing_loader

train_seqs = ['1.1.1', '1.1.2', '1.1.3', '1.1.4', '1.1.5', '1.1.6', '1.1.7', '1.1.11']
# train_seqs = ['1.1.11']
train_images_path = '/content/drive/MyDrive/Train-Set/Drone1/Morning/Extracted-Frames-1280x720'
# train_images_path = '/content/drive/MyDrive/Drone-Action/datasets/Train-Set/Drone1/Morning/Extracted-Frames-1280x720'

#test_seqs = ['1.1.8', '1.1.9']
test_seqs = ['1.1.9']
test_images_path = '/content/drive/MyDrive/Test-Set/Drone1/Morning/Extracted-Frames-1280x720'
# test_images_path = '/content/drive/MyDrive/Drone-Action/datasets/Test-Set/Drone1/Morning/Extracted-Frames-1280x720'

training_loader, testing_loader = get_dataloaders(train_seqs, test_seqs, train_images_path, test_images_path)
