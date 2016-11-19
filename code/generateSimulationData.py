import numpy as np

# simulation data
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.core import K
import tensorflow as tf

# random numbers
from random import randint

# Dougal code imports
from skl_groups.features import Features
from sklearn.pipeline import Pipeline
from skl_groups.divergences import KNNDivergenceEstimator
from skl_groups.kernels import PairwisePicker, Symmetrize, RBFize, ProjectPSD

"""
The purpose of this file is to generate a set of simulated data for the lung 
image analysis project. The data will come from the MNIST dataset, and abnormal 
nodes will be a combination of 2 images of different digits.

Functions:
- trainModel: train a simple, deep neural network model on MNIST data
- generateAbnormalNode: choose 2 random images from the MNIST data, combine them,
  and return the combined image

Notes:
- When extracting features and building the nn model, should the learning phase
  be the same or different? (0 for test and 1 for train)
"""

def trainModel(X_train, y_train, X_test, y_test):
    """
    Train a simple deep neural network on the MNIST dataset.
    Taken from a Keras tutorial (https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)

    Gets to 98.40% test accuracy after 20 epochs
    (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a K520 GPU.

    Inputs:

    Returns:
    - model: the trained NN model
    """

    batch_size = 128
    nb_classes = 10
    nb_epoch = 5  # supposed to be 20 - for real

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return model  # or return the weights of the second to last layer?


def generateAbnormalNode(mnistOnes, mnistZeros):
    """
    Choose 2 images from the MNIST dataset to combine into an "abnormal" node

    Inputs:
    - mnistOnes: vectorized images of "1"s 
    - mnistZeros: vectorized images of "0"s

    Returns:
    - abnormal: the abnormal node 
    """
    # generate a random number to select a 1 image
    idx1 = randint(0, len(mnistOnes)-1)
    # generate a random number to select a 0 image
    idx0 = randint(0, len(mnistZeros)-1)
    # select a 1 image
    v1 = mnistOnes[idx1]
    # select a 0 image
    v0 = mnistZeros[idx0]
    # combine the 2 images into 1 (add them, values are btwn 0 and 1)
    abnormal = v1+v0
    # threshold values above 1.0
    idx = [i for i, v in enumerate(abnormal) if v > 1.0]
    abnormal[idx] = 1.0
    return abnormal


def simulateSinglePatient(y, totalNodes, digits, mnistOnes, mnistZeros, model):
    """
    Generate the abnormal and normal nodes for a single patient

    Inputs:
    - totalAbnormal: the number of abnormal nodes to generate
    - totalNodes: the total number of nodes to generate
    - digits: MNIST data
    - mnistOnes, mnistZeros: data used to generate abnormal nodes
    - model to use for generating features

    Returns:
    - nodes: list of normal and abnormal nodes
    """
    nodes = [[] for i in xrange(totalNodes)]
    # create y abnormal nodes
    for i in xrange(y):
        # generate a single abnormal node
        abnormal = generateAbnormalNode(mnistOnes, mnistZeros)
        # extract the feature
        feats = extractFeatures(abnormal.reshape((1, 784)), model)
        # add the feature for that node to the list
        nodes[i] = feats

    # create totalNodes-y normal nodes
    for i in xrange(y, totalNodes):
        # generate the random number
        num = randint(0, len(digits)-1)
        # extract the feature
        feats = extractFeatures(digits[num].reshape((1, 784)), model)
        # add the feature for that node to the list
        nodes[i] = feats

    return np.vstack(nodes)


def extractFeatures(X, model):
    """
    Extract the features of X using the activation layer of the model

    Inputs:
    - X: data sample to extract features for
    - model: model to use to get the features

    Returns: the np array of features (output from the last layer of the model)
    """
    # https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # https://github.com/fchollet/keras/issues/1641
    get_last_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-2].output])
    layer_output = get_last_layer_output([X, 0])[0]

    return layer_output


def computePairwiseSimilarities(patients, y):
    """
    Compute the pairwise similarity between bags using Dougal code

    Inputs:
    - patients: the collection of patient features
    - y: labels (number of abnormal nodes) for each patient. Used to fit the
         KNNDivergenceEstimator

    Returns: 
    - sims: the pairwise similarities between each patient
    * Note: sims is a NxN symmetric matrix, where N is the number of patients
    """

    # pass the features and labels to scikit-learn Features
    feats = Features(patients, labels=y) # directly from Dougal
    # note: learning methods won't use the labels, this is for conveinence

    # estimate the distances between the bags (patients) using KNNDivergenceEstimator
    # details: use the kl divergence, find 3 nearest neighbors
    #          not sure what the pairwise picker line does?
    #          rbf and projectPSD help ensure the data is separable?
    distEstModel = Pipeline([
        ('divs', KNNDivergenceEstimator(div_funcs=['kl'], Ks=[3])),
        ('pick', PairwisePicker((0, 0))),
        ('symmetrize', Symmetrize()),
        ('rbf', RBFize(gamma=1, scale_by_median=True)),
        ('project', ProjectPSD())
    ])

    # return the pairwise similarities between the bags (patients)
    sims = distEstModel.fit_transform(feats)
    return sims


#--------------------------------------------------------------------------
# Actually do stuff...
#--------------------------------------------------------------------------

# Load MNIST data
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Train the model
print "Training the knn model..."
K._LEARNING_PHASE = tf.constant(0)
model = trainModel(X_train, y_train, X_test, y_test)
print "KNN model trained!"

# get the data for generating the abnormal nodes
mnistOneIndices = [i for i in xrange(len(y_test)) if y_test[i]==1 ]
mnistZeroIndices = [i for i in xrange(len(y_test)) if y_test[i]==0 ]
mnistOnes = X_test[mnistOneIndices]
mnistZeros = X_test[mnistZeroIndices]

# Generate the simulated patients
N = 100  # number of patients - should be 7292
totalNodes = 50  # total number of nodes for each patient - should be 500
patients = [[] for i in xrange(N)]
y = [ 0 for i in xrange(N)]
print "Generating simulated patients..."
for i in xrange(N):
    # generate y
    y[i] = randint(0, totalNodes)
    # generate the nodes
    patients[i] = simulateSinglePatient(y[i], totalNodes, X_test, mnistOnes, mnistZeros, model)

print "Patients have been simulated!"

# Compute the pairwise similarity between patients using Dougal code
print "Calculating similarities..."
sims = computePairwiseSimilarities(patients, y)
print "Similarities calculated!"

# Build the nearest neighbor graph for the patients