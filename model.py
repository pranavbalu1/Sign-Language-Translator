import os
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import tensorflow as tf



PATH_TRAINING= 'sign_mnist/sign_mnist_train.csv'
PATH_TESTING = 'sign_mnist/sign_mnist_test.csv'

IMG_SHAPE = (28,28,1)
BATCH_SIZE = 32
label_dict = {
    0:'A' , 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',10:'K',11:'L',12:'M',
    13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'       
}



pd_training = pd.read_csv(PATH_TRAINING)
pd_testing = pd.read_csv(PATH_TESTING)



pd_training.head()

