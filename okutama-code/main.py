import time
import random

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train_okutama(data_loader, model, device, optimizer, epoch):
    
    actions_meter=AverageMeter()
    # activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()

    #parameters
    B = 2
    T = 5
    num_boxes = 12
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        # forward
        actions_scores=model((batch_data[0],batch_data[1],batch_data[3]))
        actions_scores = torch.reshape(actions_scores, (B*T,num_boxes)).to(device=device)
        # print(actions_scores.shape)

        # actions_scores = actions_scores.unsqueeze(0)
        # actions_scores = torch.zeros(actions_scores.size(0), 15).scatter_(1, actions_scores, 1.)
        
        actions_in=batch_data[2].reshape((batch_size,num_frames,num_boxes))
        # print(actions_in.shape)
        # activities_in=batch_data[3].reshape((batch_size,num_frames))
        bboxes_num=batch_data[3].reshape(batch_size,num_frames)

        actions_in_nopad=[]
        # if cfg.training_stage==1:
        actions_in=actions_in.reshape((batch_size*num_frames,num_boxes,))
        bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
        for bt in range(batch_size*num_frames):
            N=bboxes_num[bt]
            actions_in_nopad.append(actions_in[bt,:N])
        # else:
        #     for b in range(batch_size):
        #         N=bboxes_num[b][0]
        #         actions_in_nopad.append(actions_in[b][0][:N])
        # actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
        # if cfg.training_stage==1:
        #     activities_in=activities_in.reshape(-1,)
        # else:
        #     activities_in=activities_in[:,0].reshape(batch_size,)
        
        # Predict actions
        # print("shape of actions_scores = ", actions_scores.shape)
        # print("shape of actions_in = ", actions_in.shape)
        # actions_in = torch.reshape(actions_in, (B,T,num_boxes)).to(device=device)
        # print("actions_in = ", actions_in)
        # print("actions_scores = ",actions_scores)
        # actions_scores = Variable(actions_scores.float(), requires_grad = True)
        # actions_in = Variable(actions_in.float(), requires_grad = True)
        loss = nn.MultiLabelMarginLoss()
        
        actions_loss = loss(actions_scores, actions_in)
        actions_loss = Variable(actions_loss, requires_grad = True)
        # actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)  
        # actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
        # print("actions_labels = ",actions_labels)
        actions_correct=torch.sum(torch.eq(actions_scores.int(),actions_in.int()).float())

        # # Predict activities
        # activities_loss=F.cross_entropy(activities_scores,activities_in)
        # activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
        # activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
        
        
        # Get accuracy
        actions_accuracy=actions_correct.item()/(actions_scores.shape[0] * num_boxes)
        # activities_accuracy=activities_correct.item()/activities_scores.shape[0]
        
        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        # activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        # total_loss=actions_loss
        loss_meter.update(actions_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        actions_loss.backward()
        optimizer.step()
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'actions_acc':actions_meter.avg*100
    }

   return train_info


def test_okutama(data_loader, model, device, epoch):
    model.eval()
    
    actions_meter=AverageMeter()
    # activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    num_boxes = 12
    B = 2
    T = 5
    epoch_timer=Timer()
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            batch_data=[b.to(device=device) for b in batch_data]
            batch_size=batch_data[0].shape[0]
            num_frames=batch_data[0].shape[1]
            
            actions_in=batch_data[2].reshape((batch_size,num_frames, num_boxes))
            # activities_in=batch_data[3].reshape((batch_size,num_frames))
            bboxes_num=batch_data[3].reshape(batch_size,num_frames)

            # forward
            actions_scores=model((batch_data[0],batch_data[1],batch_data[3]))
            actions_scores = torch.reshape(actions_scores, (B*T,num_boxes)).to(device=device)
            actions_in_nopad=[]
            
            # if cfg.training_stage==1:
            actions_in=actions_in.reshape((batch_size*num_frames,num_boxes,))
            bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
            for bt in range(batch_size*num_frames):
                N=bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt,:N])
            # else:
            #     for b in range(batch_size):
            #         N=bboxes_num[b][0]
            #         actions_in_nopad.append(actions_in[b][0][:N])
            # actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
            # if cfg.training_stage==1:
            #     activities_in=activities_in.reshape(-1,)
            # else:
            #     activities_in=activities_in[:,0].reshape(batch_size,)
            loss = nn.MultiLabelMarginLoss()
            actions_loss = loss(actions_scores, actions_in)
            actions_loss = Variable(actions_loss, requires_grad = True)

            # actions_loss=F.cross_entropy(actions_scores,actions_in)  
            # actions_labels=torch.argmax(actions_scores,dim=1)  #ALL_N,
            actions_correct=torch.sum(torch.eq(actions_scores.int(),actions_in.int()).float())

            # # Predict activities
            # activities_loss=F.cross_entropy(activities_scores,activities_in)
            # activities_labels=torch.argmax(activities_scores,dim=1)  #B,
            # activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            
            # Get accuracy
            actions_accuracy=actions_correct.item()/(actions_scores.shape[0] * num_boxes)
            # activities_accuracy=activities_correct.item()/activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            # activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            
            loss_meter.update(actions_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'actions_acc':actions_meter.avg*100
    }

    return test_info

def train_net(training_loader, validation_loader, epochs):
    """
    training
    """
    
    # Set random seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Set data position
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    
    model=Basenet_okutama()
    # if cfg.use_multi_gpu:
    #     model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.train()
    model.apply(set_bn_eval)

    train_learning_rate = 2e-4  #initial learning rate 
    lr_plan = {41:1e-4, 81:5e-5, 121:1e-5}  #change learning rate in these epochs 
    train_dropout_prob = 0.3  #dropout probability
    weight_decay = 0
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=train_learning_rate,weight_decay=weight_decay)
    
    # if cfg.test_before_train:
    #     test_info=test(validation_loader, model, device, 0, cfg)
    #     print(test_info)

    # Training iteration
    best_result={'epoch':0, 'actions_acc':0}
    start_epoch=1
    max_epoch = epochs
    for epoch in range(start_epoch, start_epoch+max_epoch):
        print("Epoch number = ", epoch)
        if epoch in lr_plan:
            adjust_lr(optimizer, lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train_okutama(training_loader, model, device, optimizer, epoch)
        # show_epoch_info('Train', cfg.log_path, train_info)
        print(train_info)

        # Test
        test_interval_epoch = 2

        if epoch % test_interval_epoch == 0:
            test_info=test_okutama(validation_loader, model, device, epoch)
            # show_epoch_info('Test', cfg.log_path, test_info)
            print(test_info)
            
            if test_info['actions_acc']>best_result['actions_acc']:
                best_result=test_info
            # print_log(cfg.log_path, 
            #           'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
            print('Best accuracy: %.2f%% at epoch #%d.'%(best_result['actions_acc'], best_result['epoch']))
            
            # Save model
            # if cfg.training_stage==2:
            #     state = {
            #         'epoch': epoch,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #     }
            #     filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
            #     torch.save(state, filepath)
            #     print('model saved to:',filepath)   
            # elif cfg.training_stage==1:
#             for m in model.modules():
#                 if isinstance(m, Basenet):

#                     filepath=cfg.result_path+'/epoch%d_%.2f%%.pth'%(epoch,test_info['actions_acc'])
#                     m.savemodel(filepath)
# #                         print('model saved to:',filepath)
            # else:
            #     assert False
