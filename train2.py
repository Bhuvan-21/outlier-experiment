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


FR=FR_model()
# normalizing the dataset
train_set=normalize(train_set)
test_set=normalize(test_set)

# getting corresponding embeddings
train_set=FR(train_set)
test_set=FR(test_set)





model=model()



optimizer=tf.keras.optimizers.Adam()
batch_size=opt.b_size
n_batches = int(len(train_set) / opt.b_size)

for i in range(opt.epochs):
    loss_t=0
    loss_vt=0
    it=0
    for j in range(n_batches):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            curr=train_set[it:it+batch_size]
            forward=model(curr,True)
            loss=tf.keras.losses.MeanAbsoluteError()(y_train[it:it+batch_size].reshape(forward.shape),forward)
            loss_t += loss


        grads=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        it+=batch_size

    index = np.arange(0, len(test_set))
    np.random.shuffle(index)
    test_set = test_set[index]
    y_test = y_test[index]
    loss_t=loss_t.numpy()

    forward_v = model(test_set[:batch_size], False)
    loss_v = tf.keras.losses.MeanAbsoluteError()(y_test[:batch_size].reshape(forward_v.shape), forward_v)
    loss_v=loss_v.numpy()
    loss_t /= n_batches
    print("Loss: {} Validation loss: {} ".format( round(loss_t,4) , round(loss_v,4)))









