import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, Input, Lambda,LSTM,Dropout,Bidirectional
from sklearn.model_selection import train_test_split
import keras.optimizers as ko
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import freqz
import os
import glob
import copy
import seaborn as sns
import re
import pickle
import operator
import IPython.display as ipd
import itertools
import numpy.random as rng
import random
import pandas as pd

#Define Keras Model
def LSTM_branch(input_shape):
    input_seq = Input(shape=input_shape)
    x = Bidirectional(LSTM(128,return_sequences=True),merge_mode='ave')(input_seq)
    x = Bidirectional(LSTM(128,return_sequences=True),merge_mode='ave')(x)
    x = Bidirectional(LSTM(128))(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(128)(x)
    x = Dropout(0.1)(x)
    encoded = Activation("linear")(x)
    return Model(input_seq,encoded,name="BiLSTM")

# Loss and metrics
def euclidean_distance(vects):
    x, y = vects
    #return K.sqrt(K.sum(K.square(x - y), axis=-1, keepdims=True))
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def filter_by_freq(target,threshold):
    filtered = dict()
    for key in target:
        if len(target[key]) >= threshold:
            filtered[key] = target[key]
    return filtered

#Load support set from selection
def create_support_set(sel_keys,selection,total,filtered,full):
    support_set = dict()
    for i in range(0, len(sel_keys)):
        support_set[sel_keys[i]] = filtered[sel_keys[i]][selection[i]]

    #If true adds rare phrases (those with less than 12 instances)
    if full:
        #load support set for rare phrases (with less than 12 tokens)
        rare_phrases = { k : total[k] for k in set(total) - set(filtered) }
        for key in rare_phrases.keys():
            support_set[key]= librosa.load(rare_phrases[key][0]) #Choose the first one
    return support_set 

def remv_support_set(sel_keys,selection,filtered):
    #Remove support set instances from filtered set. 
    new_filtered = copy.deepcopy(filtered)
    for i in range(len(sel_keys)):
        a = new_filtered[sel_keys[i]]
        del a[selection[i]]
        new_filtered[sel_keys[i]] = a
    return new_filtered

def split_set(new_filtered,train_size):
  #Returns train and test set
    train = dict()
    test = dict()
    for k in new_filtered.keys():
        #train[k],test[k] = train_test_split(new_filtered[k],train_size=train_size, random_state=rand_state)
        train[k],test[k] = train_test_split(new_filtered[k],train_size=train_size)
    return train, test

#Generate train set for k-shot learning
def get_batch(dataset,k,n):
    """Create batch of 2*n pairs per class using up to k examples, n same class, n different class"""
    pairs = []
    labels = []
    categories = dataset.keys()

    #Create subset of dataset with only k elements per class
    k_set = dict()
    for cat in categories:
        k_set[cat] = random.sample(dataset[cat],k) #Take k samples with no replacement per class

    for i in range(n):
        for cat in categories:
            z1, z2 = random.choice(k_set[cat]), random.choice(k_set[cat])
            pairs += [[z1,z2]] #Same class pair
            
            #Pick a a different category than current "cat"
            while True:   
                notcat = random.choice(list(categories))
                if(notcat != cat):  
                    break  
            z1, z2 = random.choice(k_set[cat]), random.choice(k_set[notcat])
            pairs += [[z1,z2]] #different class pair
            labels += [1, 0] #1 to same pairs, 0 to contrastive
    return np.array(pairs), np.array(labels)


# # Load features from all phrases

# In[4]:


with open("features_total.pkl", "rb") as input_file:
    total_features = pickle.load(input_file)

#Transpose vectors and compute decibels
total_features_db = dict()
for k in total_features.keys():
    for i in range(len(total_features[k])):
        total_features[k][i] = lb.amplitude_to_db(total_features[k][i],top_db=65.0)
        total_features[k][i] = total_features[k][i].astype('int8')

#Get most common phrases
filt_features = filter_by_freq(total_features,12)
total_features = 0


#Create support set from averages
support_set = dict()
for k in filt_features.keys():
    support_set[k] = np.mean(filt_features[k],axis=0)
support_set_array = np.array([s for s in list(support_set.values())])


#Create classification set
def create_classif_task(test_set):
    classif_test = []
    classif_labels = []

    #use the full test set
    for k in test_set.keys():
        for a in test_set[k]:
            classif_test.append(a)
            classif_labels.append(k)
    return (np.array(classif_test),classif_labels)

def get_predictions(support_set,classif_test,model):
    predictions = []
    support_set_array = np.array([s for s in list(support_set.values())])
    classif_test_repeated = np.repeat(classif_test,len(support_set_array),axis=0)
    I, L = pd.factorize(list(support_set.keys()))
    for k in range(len(classif_test)):
        pred_support = model.predict([classif_test_repeated[32*k:32+32*k],support_set_array]).ravel()
        pred_class = np.where(pred_support == np.min(pred_support))[0][0]
        predictions.append(L[pred_class])
    return predictions


def train_model(x,y,labels,epochs):
    "Creates, trains and returns trained model"
    input_shape = (64,128) #(Timesteps,n_features)
    lstm = LSTM_branch(input_shape)
    inputA = Input(shape=input_shape,name="InputA")
    inputB = Input(shape=input_shape,name="InputB")
    encodedA = lstm(inputA)
    encodedB = lstm(inputB)
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape,name="distance")([encodedA, encodedB])
    model = Model(input=[inputA,inputB],output=distance)
    model.compile(optimizer='adam', loss=contrastive_loss)
    history = model.fit([x, y],labels,epochs=epochs,batch_size=256,shuffle=True)
    return model, history.history['loss']

def generate_sets(k):
#Generate train_test set
    train_set, test_set = split_set(filt_features,k)
    train_pairs, train_labels =  get_batch(train_set,k,1000)
    min_phrases_test = min([len(i) for i in test_set.values()])
    test_pairs, test_labels = get_batch(test_set,min_phrases_test,100)
    te1 = test_pairs[:,0,:,:]
    te2 = test_pairs[:,1,:,:]
    tr1 = train_pairs[:,0,:,:]
    tr2 = train_pairs[:,1,:,:]
    return tr1,tr2,train_labels,train_set,te1,te2,test_labels,test_set

def compute_one_run(k,epochs):
    tr1,tr2,train_labels,train_set,te1,te2,test_labels,test_set = generate_sets(k)
    model, history = train_model(tr1,tr2,train_labels,epochs)

    #Verification task evaluation (test)
    v_pred_te = model.predict([te1,te2])
    v_acc_te = compute_accuracy(test_labels,v_pred_te)

    #Verification task evaluation (train)
    v_pred_tr = model.predict([tr1,tr2])
    v_acc_tr = compute_accuracy(train_labels,v_pred_tr)

    #Classification task evaluation (test)
    classif_test, classif_labels_test = create_classif_task(test_set)
    predictions_test = get_predictions(support_set,classif_test,model)
    c_acc_te = np.mean([predictions_test[i] == classif_labels_test[i] for i in range(len(predictions_test))])
    
    #Classification task evaluation (train)
    classif_train, classif_labels_train = create_classif_task(train_set)
    predictions_train = get_predictions(support_set,classif_train,model)
    c_acc_tr = np.mean([predictions_train[i] == classif_labels_train[i] for i in range(len(predictions_train))])

    #Accuracy per class (test)
    acc_c_class_test = dict()
    for k in test_set.keys():
        k_indices = list(filter(lambda x: classif_labels_test[x] == k, range(len(classif_labels_test))))
        acc_c_class_test[k] = np.mean([predictions_test[i] == classif_labels_test[i] for i in k_indices])
    
    #Accuracy per class (train)
    acc_c_class_train = dict()
    for k in train_set.keys():
        k_indices = list(filter(lambda x: classif_labels_train[x] == k, range(len(classif_labels_train))))
        acc_c_class_train[k] = np.mean([predictions_train[i] == classif_labels_train[i] for i in k_indices])

    return (v_acc_tr,v_acc_te,c_acc_tr,c_acc_te,acc_c_class_train,acc_c_class_test,history)


H = []
n = 30
shots = 7
for i in range(n):
    print("Experiment: " + str(i+1) + " from " + str(n))
    X = compute_one_run(k=shots,epochs=5)
    H.append(X)
    K.clear_session()


with open('k'+ str(shots) +'.pickle', 'wb') as f:
    pickle.dump(H, f)


#with open(directory + '/k1.pickle', 'rb') as f:
#    x = pickle.load(f)

