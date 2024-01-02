#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trimble / Bilberry : AI Engineer technical exercise
Part 1
"""

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.data_loader import Dataset
import argparse
import os
import numpy as np
import random
from utils.model import Resnet,MobileNet
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Field & Road classifier')

# Benchmark specific args
parser.add_argument('--classes', default=['fields','roads'], type=list,help='Names of the different classes')
parser.add_argument('--class_stats', default=[36,85], type=list,help='The cardinality of each class in the training set')
parser.add_argument('--model', default='resnet', type=str,help='model architecture')
parser.add_argument('--weight', default='uniform', type=str,help='parameter that decides which weight to assign to the positive class')
parser.add_argument('--optimizer', default='sgd', type=str,help='Decide if optmizer is Adam or SGD')
parser.add_argument('--learning_rate', default=1e-2, type=float,help='learning step')
parser.add_argument('--weight_decay', default=1e-3, type=float,help='regularization')
parser.add_argument('--batch_size', default=8, type=int,help='number of samples per batch')
parser.add_argument('--epochs', default=20, type=int,help='number of epochs')
parser.add_argument('--num_workers', default=0, type=int,help="number of workers")
parser.add_argument('--step_size', type=int, default=4, help='step size of scheduler of learning rate (default: 4)')
parser.add_argument('--gamma', type=float, default=0.4, help='learning rate is multiplied by gamma')
parser.add_argument('--threshold', type=float, default=0.4 , help='threhsold to discriminate between negative and positive classes')
parser.add_argument('--data_path', default='./dataset', type=str, help="path of samples")
parser.add_argument('--split_files_path', default='./data_splits', type=str, help="path of  folder of split files")
parser.add_argument('--args_save_path', default='./Results', type=str,help="folder to save arguments into a file")
parser.add_argument('--test_id', default='test_0', type=str,help="namefile for the running test")
parser.add_argument('--random_train', default=False, type=bool,help="Decide if the training process will be random or not")
parser.add_argument('--random_seed', default=42, type=int,help="random seed to guarantee results reproducibility")


def main():     
    args = parser.parse_args()
    # Create a string representation of the argparse arguments
    arg_str = ''
    for arg in vars(args):
        arg_str += f'{arg}: {getattr(args, arg)}\n'
    
    # Create the test folder if it does not exist
    if not(os.path.exists(os.path.join(args.args_save_path,args.test_id))):
        os.mkdir(os.path.join(args.args_save_path,args.test_id))
    ## Save the arguments into a text file
    with open(os.path.join(args.args_save_path,args.test_id,'args.txt'), 'w') as f:
        f.write(arg_str)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

        def _init_fn(worker_id):
            np.random.seed(args.random_seed + worker_id)
    
    # Define the model
    if args.model == 'resnet':
        model = Resnet(version=18)
    else:
        model = MobileNet()
    
    # Define Optimizer 
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,betas=(0.9, 0.99))   

    # Define Scheduler 
    scheduler = lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=args.gamma)

    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(Dataset(os.path.join(args.split_files_path,'train.csv'),args.data_path,True),
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,shuffle=True, drop_last=False)
    
    validloader = torch.utils.data.DataLoader(Dataset(os.path.join(args.split_files_path,'valid.csv'),args.data_path,False),
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,shuffle=False, drop_last=False)
    
    testloader = torch.utils.data.DataLoader(Dataset(os.path.join(args.split_files_path,'test.csv'),args.data_path,False),
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,shuffle=False, drop_last=False)
      
    # Calculate the weights to scale class erros
    def weights(w):
        s = np.array(w).sum()
        return w[0]/s        
    
    # Define the loss function
    if args.weight=='unifrom':
        pos_weight = torch.tensor([1.])
    else: 
        pos_weight = torch.tensor(weights(args.class_stats)).float()
                                
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    sigmoid = torch.nn.Sigmoid()
    
    best_val_f1 = 0
    best_epoch = 0
    best_params = model.state_dict()
    
    for epoch in range(args.epochs):  
        model.train()
        for i, dataset in enumerate(trainloader,0): 
            optimizer.step()
            
            inputs, labels = dataset   
            preds = model(inputs)
                
            loss  = criterion(preds, labels.float().view(-1, 1))
            # backward pass
            loss.backward()
            
            optimizer.zero_grad()

        scheduler.step()
        
        # Print the current learning rate
        print(f"Epoch {epoch}: Learning rate: {scheduler.get_last_lr()[0]:.9f}")
                            
        with open(os.path.join(args.args_save_path,args.test_id,'output.txt'), 'a') as f:
            print( """ Evaluation on validation dataset """, file=f )        
        print( """ Evaluation on validation dataset """ )
        
        model.eval()
        correct = 0 
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():        
            for i, dataset in enumerate(validloader,0): 
                inputs, labels = dataset
                outputs = model(inputs)
                total += labels.size(0)
                
                preds = sigmoid(outputs).squeeze()
                preds = torch.where(preds >= args.threshold, torch.tensor(1), torch.tensor(0))
                correct += (preds == labels).sum().item()
                
                all_labels += labels.tolist()
                all_preds += preds.tolist()

    
        val_accuracy = 100 * correct / total
        val_f1 = 100*f1_score(all_labels, all_preds)
        with open(os.path.join(args.args_save_path,args.test_id,'output.txt'), 'a') as f:
            print('F1 score on validation set at epoch {}: %.2f %%'.format(epoch) % val_f1, file=f ) 
            print('preds : ',all_preds, file=f )
            print('labels : ',all_labels, file=f )
            
        print('F1 score : ',val_f1)
        print('preds : ',all_preds)
        print('labels : ',all_labels)
        with open(os.path.join(args.args_save_path,args.test_id,'output.txt'), 'a') as f:
            print('Accuracy on validation set at epoch {}: %.2f %%'.format(epoch) % val_accuracy, file=f )
        print('Accuracy on validation set at epoch {}: %.2f %%'.format(epoch) % val_accuracy)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_params = model.state_dict()   
            """ save the best model parameters to a file """
            torch.save(best_params, os.path.join(args.args_save_path,args.test_id,"best_params.pth"))
                            
        with open(os.path.join(args.args_save_path,args.test_id,'output.txt'), 'a') as f:
            print( """ Evaluation on testing dataset """, file=f )        
        print( """ Evaluation on testing dataset """ )
        
        model.eval()
        correct = 0 
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():           
            for i, dataset in enumerate(testloader,0): 
                inputs, labels = dataset
                outputs = model(inputs)
                total += labels.size(0)
                
                preds = sigmoid(outputs).squeeze()
                preds = torch.where(preds >= args.threshold, torch.tensor(1), torch.tensor(0))
                  
                correct += (preds == labels).sum().item()
                all_labels += labels.tolist()
                all_preds += preds.tolist()
    
        test_accuracy = 100 * correct / total
        test_f1 = 100*f1_score(all_labels, all_preds)
        with open(os.path.join(args.args_save_path,args.test_id,'output.txt'), 'a') as f:
            print('F1 score on testing set at epoch {}: %.2f %%'.format(epoch) % test_f1, file=f ) 
            print('preds : ',all_preds, file=f )
            print('labels : ',all_labels, file=f )
            
        print('F1 score : ',test_f1)
        print('preds : ',all_preds)
        print('labels : ',all_labels)
        
        with open(os.path.join(args.args_save_path,args.test_id,'output.txt'), 'a') as f:
            print('Accuracy on testing set at epoch {}: %.2f %%'.format(epoch) % test_accuracy, file=f )
        print('Accuracy on testing set at epoch {}: %.2f %%'.format(epoch) % test_accuracy)
            
    # Print best results at the end of the training
    with open(os.path.join(args.args_save_path,args.test_id,'output.txt'), 'a') as f:
        print(" best_epoch : ",best_epoch,file=f)
        print(" best_val_f1 : ",best_val_f1,file=f)
    print(" best_epoch : ",best_epoch)
    print(" best_val_f1 : ",best_val_f1)
    
    print("Finished Training")

    
if __name__ =="__main__":
    main()
            
