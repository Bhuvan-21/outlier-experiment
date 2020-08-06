from Face_recognition import FR_model
import os
from PIL import Image
import numpy as np
from Options.options import  options
import tensorflow as tf
from model import model
import utils

add_samples=utils.add_samples
normalize=utils.normalize

opt =options().parser
directory_name='crop_part1'
image_names=os.listdir(directory_name)

y_l=opt.y_l
y_u=opt.y_u
o_l=opt.o_l
o_u=opt.o_u
n_samples=opt.n_samples
mid_l=y_u+1
mid_u=o_l-1

test_set=[]
train_set=[]
y_train=[]
y_test=[]

# add from y_l to y_u to train set
add_samples(train_set,y_train,n_samples,y_l,y_u,image_names)
# add from y_l to y_u to train set
add_samples(train_set,y_train,n_samples,o_l,o_u,image_names)
# add mid_l to mid_u
add_samples(test_set,y_test,2*n_samples,mid_l,mid_u,image_names)

# converting to np array
train_set=np.asarray(train_set)
test_set=np.asarray(test_set)
y_train=np.asarray(y_train)/100
y_test=np.asarray(y_test)/100
y_train=y_train.reshape((-1,1))
y_test=y_test.reshape((-1,1))

FR=FR_model()
# normalizing the dataset
train_set=normalize(train_set)
test_set=normalize(test_set)

# getting corresponding embeddings
train_set=FR(train_set)
test_set=FR(test_set)




model=model()

model.compile(tf.keras.optimizers.Adam(),tf.keras.losses.MeanAbsoluteError())

model.fit(train_set,y_train,batch_size=opt.b_size,epochs=opt.epochs,validation_data=(test_set,y_test))