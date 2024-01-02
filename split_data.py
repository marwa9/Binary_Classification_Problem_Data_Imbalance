# -*- coding: utf-8 -*-


import pandas as pd
import os
import random
random.seed(42)

dataset_path = "./dataset"

#=================================================================#
#             Create training and validation splits               #
#=================================================================#

fields_samples = os.listdir(os.path.join(dataset_path,'fields'))
roads_samples = os.listdir(os.path.join(dataset_path,'roads'))

training_rate = 0.8 # => validation_rate = 0.3

def split(samples,rate):
    training = random.sample(samples, round(rate*len(samples)))
    for el in training:
        samples.remove(el)
    return training,samples

training_fields, validation_fields = split(fields_samples,training_rate) 
training_roads, validation_roads = split(roads_samples,training_rate) 


def set_dataframe(set_negative,set_positive,oversample):
    set_negative = [os.path.join('fields',el) for el in set_negative]
    if oversample:
        set_negative +=set_negative
    set_positive = [os.path.join('roads',el) for el in set_positive]
    df_set = pd.DataFrame(set_negative+set_positive,columns=['sample'])
    df_set['label'] = len(set_negative)*[0] + len(set_positive)*[1]
    return df_set


df_training = set_dataframe(training_fields,training_roads,False)
df_validation = set_dataframe(validation_fields,validation_roads,False)

df_training.to_csv('./data_splits_non_oversampled/train.csv')
df_validation.to_csv('./data_splits_non_oversampled/valid.csv')

#===================================================#
#              Create testing split                 #
#===================================================#

testing_samples = os.listdir(os.path.join(dataset_path,'test_images'))
testing_samples = [os.path.join('test_images',el) for el in testing_samples]
df_testing = pd.DataFrame(testing_samples,columns=['sample'])
df_testing['label'] = [0,0,1,1,0,1,1,1,1,0]
df_testing.to_csv('./data_splits_non_oversampled/test.csv')