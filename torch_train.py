from Face_recognition import FR_model
import os
from PIL import Image
import numpy as np
from Options.options import  options
from torch_model import classifier
import utils
import torch

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
mid_l=opt.mid_l
mid_u=opt.mid_u

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

model=classifier()
batch_size=opt.b_size
optimizer=torch.optim.Adam(model.parameters())
n_batches = int(len(y_train) / batch_size)
MAE=torch.nn.L1Loss()
if torch.cuda.is_available():
    device=torch.device('cuda:0')
    print("cuda capable device detected")
else:
    device=torch.device('cpu')
    print("No cuda capable device detected")


model=model.to(device)
for i in range(opt.epochs):
    loss_t=0
    loss_vt=0
    it=0
    model.train()
    for j in range(n_batches):

        curr=train_set[it:it+batch_size]
        curr=torch.from_numpy(curr).to(device)
        forward=model.forward(curr)
        curr_y=torch.from_numpy(y_train[it:it+batch_size].reshape(forward.shape)).to(device)
        loss=MAE(forward,curr_y)
        loss_t += loss
        model.zero_grad()
        loss.backward()
        optimizer.step()
        it+=batch_size

    # validation loss
    index = np.arange(0, len(test_set))
    np.random.shuffle(index)
    test_set = test_set[index]
    y_test = y_test[index]
    loss_t=loss_t.detach().cpu().numpy()

    model.eval()
    forward_v = model(torch.from_numpy(test_set[:batch_size]).to(device))
    loss_v=MAE(forward_v,torch.from_numpy(y_test[:batch_size].reshape(forward_v.shape)).to(device))
    loss_v=loss_v.detach().cpu().numpy()
    loss_t /= n_batches
    print("Epoch no: {} Loss: {} Validation loss: {} ".format( i,loss_t, loss_v))


