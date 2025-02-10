import os
os.environ['KERAS_BACKEND'] = "tensorflow"
import sys 
sys.path.insert(1, '../') 
sys.path.insert(1, '../../')

from training_classroom import TrainingClassroom 
from data_generation import StanfordCarsDataloader as SCDL

import keras 
import tensorflow as tf 
import numpy as np

import matplotlib.pyplot as plt 


acc_metric = keras.metrics.CategoricalAccuracy()
top5_acc_metric = keras.metrics.TopKCategoricalAccuracy(k=5)

tc = TrainingClassroom("./students/tcs/test_0", n_broad_labels=16) # to speed up the test 
epochs = [0, 6, 12, 18, 21, 24, 27, 30, 33, 36, 39] 


# evaluating on validation set: 
print("EVALUATING ON VALIDATION SET") 
data = SCDL('valid', batch_size=1, augment_func=lambda x:x) # batch size = 1 since we're going to have to split the items based on their broad label anyways 

valid_accs = {}
valid_top5_accs = {} 
for epoch in epochs: 
    tc.load_from_name("epoch_{}.keras".format(epoch))
    print("EPOCH", epoch) 
    #tc.evaluate(data)

    # test it out
    for batch in range(len(data)):
        x, y = data[batch]
        y_true = y[0][1].reshape((1, -1))
        broad_label = np.argmax(y[0][0])
        logits = tc.studentlst[broad_label].model(x).numpy() 
        #print("TARGETS", y_true) 
        #print("LOGITS", logits)
        acc_metric.update_state(y_true, logits)
        #print("AGAIN") 
        #print("TARGETS", y_true) 
        #print("LOGITS", logits)
        top5_acc_metric.update_state(y_true, logits)

    accuracy = float(tf.squeeze(acc_metric.result()).numpy()) 
    top5_acc = float(tf.squeeze(top5_acc_metric.result()).numpy()) 

    print("EPOCH {} VALID ACC: {:.4f} , TOP5: {:.4f}".format(epoch, accuracy, top5_acc)) 

    valid_accs[epoch] = accuracy 
    valid_top5_accs[epoch] = top5_acc 
    
    acc_metric.reset_state()
    top5_acc_metric.reset_state()

print("valid_accs =", valid_accs) 
print("valid_top5_accs =", valid_top5_accs) 




# evaluating on test set: 
print("TESTING ON TEST SET") 
data = SCDL('test', batch_size=1, augment_func=lambda x:x) # batch size = 1 since we're going to have to split the items based on their broad label anyways 

test_accs = {}
test_top5_accs = {} 
for epoch in epochs: 
    tc.load_from_name("epoch_{}.keras".format(epoch))
    print("EPOCH", epoch) 
    #tc.evaluate(data)

    # test it out
    for batch in range(len(data)):
        x, y = data[batch]
        y_true = y[0][1].reshape((1, -1))
        broad_label = np.argmax(y[0][0])
        logits = tc.studentlst[broad_label].model(x).numpy() 
        #print(y_true, logits) 
        acc_metric.update_state(y_true, logits)
        top5_acc_metric.update_state(y_true, logits)

    accuracy = float(tf.squeeze(acc_metric.result()).numpy()) 
    top5_acc = float(tf.squeeze(top5_acc_metric.result()).numpy()) 

    print("EPOCH {} TEST ACC: {:.4f} , TOP5: {:.4f}".format(epoch, accuracy, top5_acc)) 

    test_accs[epoch] = accuracy 
    test_top5_accs[epoch] = top5_acc 
    
    acc_metric.reset_state()
    top5_acc_metric.reset_state()

print("test_accs =", test_accs) 
print("test_top5_accs =", test_top5_accs) 


    





