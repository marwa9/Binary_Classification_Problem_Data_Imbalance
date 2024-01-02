#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from utils.data_loader import Dataset
import argparse
import os
from utils.model import Resnet,MobileNet
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Field & Road classifier')

# Benchmark specific args
parser.add_argument('--classes', default=['fields','roads'], type=list,help='Names of the different classes')
parser.add_argument('--model', default='resnet', type=str,help='model architecture')
parser.add_argument('--batch_size', default=8, type=int,help='number of samples per batch')
parser.add_argument('--num_workers', default=0, type=int,help="number of workers")
parser.add_argument('--threshold', type=float, default=0.65, help='threshold to discriminate between negative and positive classes')
parser.add_argument('--data_path', default='./dataset', type=str, help="path of samples")
parser.add_argument('--split_files_path', default='./data_splits', type=str, help="path of  folder of split files")
parser.add_argument('--args_save_path', default='./Results', type=str,help="folder to save arguments into a file")
parser.add_argument('--test_id', default='test_0', type=str,help="namefile for the running test")


def main():     
    args = parser.parse_args()
    
    # Define the model
    if args.model == 'resnet':
        model = Resnet(version=18)
    else:
        model = MobileNet()
    
    model.load_state_dict(torch.load(os.path.join(args.args_save_path,args.test_id,"best_params.pth")))
    

    # Define dataloader 
    validloader = torch.utils.data.DataLoader(Dataset(os.path.join(args.split_files_path,'valid.csv'),args.data_path,False),
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,shuffle=False, drop_last=False)
    
    testloader = torch.utils.data.DataLoader(Dataset(os.path.join(args.split_files_path,'test.csv'),args.data_path,False),
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,shuffle=False, drop_last=False)
    
    sigmoid = torch.nn.Sigmoid()
    
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

    test_accuracy =  correct / total
    test_f1 = f1_score(all_labels, all_preds)

    print('Accuracy : ',test_accuracy)
    print('F1 score : ',test_f1)
    # print('preds : ',all_preds)
    # print('labels : ',all_labels)
        
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

    test_accuracy = correct / total
    test_f1 = f1_score(all_labels, all_preds)

    print('Accuracy : ',test_accuracy)
    print('F1 score : ',test_f1)
    # print('preds : ',all_preds)
    # print('labels : ',all_labels)

        

    
if __name__ =="__main__":
    main()
            
