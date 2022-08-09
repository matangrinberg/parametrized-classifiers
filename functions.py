import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import tensorflow as tf

############
# Functions
############

# Our 3d parameter space is the surface of a sphere centered at (1, 1, 1), with radius 1/sqrt(2)
# x, y coordinates are the x and y variances of our 2d Gaussian
# z coordinate determines the x-coordinate of the mean, mu = (z, 0)

def mean_gen(theta, phi):
    mu = 0.5 - (np.sqrt(2)/2) * np.cos(phi + np.pi / 4)
    return mu


def varx_gen(theta, phi):
    vx = 1.5 - (np.sqrt(2)/2) * np.cos(theta + np.pi / 4) * np.sin(phi + np.pi / 2)
    return vx


def vary_gen(theta, phi):
    vy = 1.5 - (np.sqrt(2)/2) * np.sin(theta + np.pi / 4) * np.sin(phi + np.pi / 2)
    return vy


# Generate n data
def spherical_data(n, thetas, phis, rand=1234):
    
    mx1, my1 = np.zeros(n), np.zeros(n)
    vx1, vy1 = np.ones(n), np.ones(n)
    
    mx2, my2 = mean_gen(thetas, phis), np.zeros(n)
    vx2, vy2 = varx_gen(thetas, phis), vary_gen(thetas, phis)
    
    x1, y1 = np.transpose(np.array([np.random.normal(mx1, vx1, size=n), 
                                    np.random.normal(my1, vy1, size=n), thetas, phis])), np.zeros(n)
    x2, y2 = np.transpose(np.array([np.random.normal(mx2, vx2, size=n), 
                                    np.random.normal(my2, vy2, size=n), thetas, phis])), np.ones(n)
    
    x, y = np.append(x1, x2, axis=0), np.append(y1, y2, axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = rand)
    
    return x_train, x_test, y_train, y_test


def discrete_angles(n, m, s, rand = 1234):
    angles = np.random.uniform(0, s * np.pi, m)
    xk = np.arange(m)
    pk = (1 / m) * np.ones(int(m))
    discrete_distr = stats.rv_discrete(name='discrete_distr', values=(xk, pk), seed=rand)
    thetas = angles[discrete_distr.rvs(size=n)]
    return thetas


def test_on_integers(x_train, y_train):
    names = []
    names.append('discrete_model_th5_ph3')
#     names.append('discrete_model_th4_ph3')
    names.append('discrete_model_th3_ph3')
#     names.append('discrete_model_th2_ph3')
    names.append('discrete_model_th2_ph2')
    names.append('discrete_model_th1_ph1')
    names.append('discrete_model_th0_ph0')
#     names.append('discrete_model_mth15_mph15')

    l = len(names)

    models = []
    predictions = []
    
    for i in range(l):
        models.append(tf.keras.models.load_model('3dmodels/' + names[i]))
        xr = models[i](x_train).numpy().transpose()[0]
        if xr.shape[0] > 0:
            loss = np.sum(-y_train * np.log(xr) - (1-y_train) * np.log(1-xr)) / xr.shape[0]
            print(names[i], 'has loss', loss)
        else:
            print("xr shape is", xr.shape)
            
    return
 

def learn_parameters(model_pf, x_train, y_train, iterations=20, epochs=1, batch_size=100):
    e, b = epochs, batch_size
    x_inputs = x_train[:, 0:2]
    
    for i in range(iterations):
        model_pf.fit(x_inputs, y_train, epochs=e, batch_size=b)
        print(i,"Fitted result: ", model_pf.trainable_weights[:][0][0])
        x = model_pf(x_inputs).numpy().transpose()[0]
#         print(np.sum(-y_train * np.log(x) - (1-y_train) * np.log(1-x)) / x.shape[0])

    return model_pf.trainable_weights[:][0][0]
    
 
