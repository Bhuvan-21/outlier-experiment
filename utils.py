from PIL import Image
import numpy as np


def get_image(i):
    directory_name='crop_part1/'
    im = Image.open(directory_name + i)
    im=im.resize((160,160))
    im = np.asarray(im)
    return im

def add_samples(X,Y,n_samples,y_l,y_u,image_names):
    # adding samples from y_l to y_u
    for i in image_names:
        if len(X) >= n_samples:
            break;
        age = int(i.split('_')[0])
        if (age >= y_l and age <= y_u):
            image = get_image(i)
            X.append(image)
            Y.append(age)

def normalize(data):
    mean=data.mean()
    stddev=data.std()
    data=(data-mean)/stddev
    return data