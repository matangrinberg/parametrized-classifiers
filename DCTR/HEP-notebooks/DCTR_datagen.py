import os
import sys
import numpy as np
from DCTR_evgen import generate_dataset, generate_dataset_ref
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('/Users/matangrinberg/Library/CloudStorage/GoogleDrive-matan.grinberg@gmail.com/My Drive/(21-24) University of California, Berkeley/ML HEP/parametrized-classifiers/data')

n_points, n_mult, nx = 10000, 10, sys.argv[1]
print('Running DCTR data generator with (n_points, n_mult, nx) of ' + str(n_points) + ', ' + str(n_mult) + ', ' + str(nx))

x0 = generate_dataset_ref(n_points, n_mult, nx)
x1 = generate_dataset(n_points, n_mult, nx)
y0 = np.zeros(n_points * n_mult)
y1 = np.ones(n_points * n_mult)

data_dir = '/global/home/users/mgrinberg/parametrized-classifiers/data/'
run_name = 'interpolate_standard_n' + str(n_points*n_mult) + 'nx' + str(nx)

x = np.concatenate((x0, x1), axis=0)
y = np.concatenate((y0, y1), axis=0)

np.savez(data_dir + run_name, x, y)