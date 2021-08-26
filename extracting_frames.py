import cv2  
from glob import glob
import os


TARGET_DIR = '/content/drive/MyDrive/MOD20/ExtractedFrames'
def extract_frames(src, target):
    """
    'src': contains path to video
    'target': dir path to save extracted frames
    """
  
    # Read the video from specified path 
    cam = cv2.VideoCapture(src) 
    
    try: # creating a folder named data 
        if not os.path.exists(target): 
            os.makedirs(target) 
            print('Creating: {}'.format(target))

        else:
            print('Frames already extracted')
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 
    
    # frame 
    currentframe = 0
    
    while(True): 
        
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            # if video is still left continue creating images 
            name = f'{target}/{str(currentframe)}.jpg'
            # print ('Creating...' + name) 
            
            # writing the extracted images 
            cv2.imwrite(name, frame) 
            
            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows()


for act in a:
    ROOT_DIR = f'/content/drive/MyDrive/Drone-Action/datasets/MOD20/mod20/{act}/*.mp4'
    # print(len(glob(ROOT_DIR)))
    # print(len(os.listdir(os.path.join(TARGET_DIR, activity))))
    # print('------------------------------------')
    for vid in glob(ROOT_DIR):
        #loop through all activity directories
        activity = vid.split('/')[-2]
        ACT_DIR = os.path.join(TARGET_DIR, activity)
        if not os.path.exists(ACT_DIR):
            os.mkdir(ACT_DIR)
            print('Creating: {}'.format(ACT_DIR))

        vidname = vid.split('/')[-1][:-4]
        VID_TARGET_DIR = os.path.join(ACT_DIR, vidname)

        if os.path.exists(VID_TARGET_DIR):
            continue

        extract_frames(vid, VID_TARGET_DIR)
