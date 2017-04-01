from __future__ import print_function
import numpy as np
import argparse

# building the neural network
from keras.models import *
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.core import K

# building the SVM
from sklearn.svm import SVC

# building the random forest
from sklearn.ensemble import RandomForestClassifier

# cross-validation
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# saving/loading data
import h5py  # for saving/loading features
import pickle as pk
import pandas as pd

#----------------------------------------------------------------------------
# Loading/Saving Data
#----------------------------------------------------------------------------
def loadAnnotationData(filename):
    """
    Load the fhog/histogram features extracted from the labeled patches.

    Inputs:
    - filename: name of the file to extract data from

    Returns:
    - featuresList: list of patch image feature vectors
    - data: raw data loaded from the pickle file
    """
    print('Going to load pickle file:', filename)

    # read the pickle file
    print("Reading pickle file...")
    fid = open(filename, 'rb')
    data = pk.load(open(filename, 'rb'))
    fid.close()
    print("Finished reading the pickle file!")

    # Now convert the data from a list of lists for each subject
    # to a single list of features
    featuresList = []
    featureCount = 0
    # since the data is a list of subjects, iterate through them
    for subj in data:
        featureCount += len(subj['I'])
        # for every subject, add each of its feature vectors to the list
        for patch in subj['I']:
            featuresList.append(patch)

    print('Extracted', featureCount, 'feature vectors from the data.')
    return featuresList


def loadAnnotationClasses(filename):
    """
    Load the classes associated with the labeled patches.

    Inputs:
    - filename: name of the file to extract labels from

    Returns:
    - rawClasses: raw classes
    """
    # load the .csv file using pandas
    dataframe = pd.read_csv(filename)
    print('Loaded the patch classes.')
    # extract the column header associated with the labels
    col = list(dataframe)[1]
    print('Looking at column with header', col)
    # extract the label column and convert it to a list
    classes = dataframe[col].values.tolist()
    # return the list of labels
    print('Classes for ',len(classes), 'patches have been extracted from the file!')
    return classes
    

def saveModel(filename, model):
    """
    Save the trained model for use later.

    Inputs:
    - filename: name of the file where the model will be save to
    - model: the trained model that needs to be saved

    Effects:
    - saves the model to the specified filename
    """
    model.save(filename)
    print('Model saved to file', filename)


def loadModel(filename):
    """
    Load a previously trained model.

    Inputs:
    - filename: name of the file where the model was previously saved

    Returns:
    - model: the loaded model
    """
    model = load_model(filename)
    print("Model loaded!")
    return model

#-------------------------------------------------------------------------------------
# Helper functions to train and test the models
#-------------------------------------------------------------------------------------

def trainNeuralNetwork(X_train, Y_train, printFeedback=0):
    """
    Train a simple neural network on the training data.

    Inputs:
    - X_train: training data (fhog/histogram features)
    - y_train: class for each point of the training data
    - printFeedback: tells keras whether or not to be verbose when training the model

    Returns:
    - model: the trained model
    """
    # set up variables for the model
    batch_size = 100 # since the dataset has ~ 1530 points, make a batch ~ 100
    nb_classes = 12
    nb_epoch = 20 # supposed to be 20, according to past work

    print(X_train.shape[0], 'training data samples')

    # set up the structure of the neural network
    model = Sequential()
    model.add(Dense(512, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(20)) # this is the layer I want to get features from (-4)
    model.add(Activation('relu'))
    model.add(Dense(output_dim=nb_classes))
    model.add(Activation('softmax'))

    # compile the model (tell keras how to evaluate it)
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # train the model
    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=printFeedback)
    # should I include some validation? Probably not?
    return model


def testNeuralNetwork(model, X_test, Y_test):
    """
    Test a previously trained neural network model.

    Inputs:
    - model: previously trained model
    - X_test: test data (fhog/histogram features)
    - Y_test: class for each point of the test data

    Returns:
    - accuracy: the accuracy of the model

    Effects:
    - prints the accuracy of the model in classifying the test data
    """
    score = model.evaluate(X_test, Y_test)
    print('Test data score:', score[0])
    print('Test data accuracy:', score[1])

    return score


def trainSVM(X_train, Y_train):
    """
    Train a SVM to classify the patch features.

    Inputs:
    - X_train: training data
    - Y_train: classes for training data

    Returns:
    - model: trained model
    """
    # make the model object
    model = SVC() 
    model.fit(X_train, Y_train)
    print(model.score(X_train, Y_train ))

    return model


def testSVM(model, X_test, Y_test):
    """
    Test a previously trained SVM.

    Inputs:
    - model: the previously trained model
    - X_test: the test data
    - Y_test: the classes for the test data

    Effects:
    - Prints the accuracy of the model in classifying the test data

    Returns:
    - score: the average accuracy of the model in classifying the test data
    """
    score = model.score(X_test, Y_test)
    return score


def trainForest(X_train, Y_train):
    """
    Train a random forest classifier to identify class labels

    Inputs:
    - X_train: training data
    - Y_train: classes for training data

    Returns:
    - model: a trained model
    """
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    return model


def testForest(model, X_test, Y_test):
    """
    See if a previously trained random forest classifier can identify
    patch labels for test data.

    Inputs:
    - model: the previously trained model
    - X_test: test data
    - Y_test: categorical classes of the test data

    Effects:
    - prints the accuracy of the model in classifying the test data

    Returns:
    - model: a trained model
    """
    score = model.score(X_test, Y_test)

    return score


def convertClassesToCategorial(y):
    """
    Since the original classes are strings, convert them first to numbers
    and then specify that they are categorial numbers (not numerical values).

    Inputs:
    - y: the original classes

    Returns:
    - y_new: the categorical classes

    Does it make sense to make this a hardcoded mapping?
    """
    # get information for mapping the classes to numbers
    numClasses = len(np.unique(y))
    classes = np.unique(y)
    y_new = np.zeros(len(y))

    # iterate through the class labels to see the mapping
    for i in xrange(numClasses):
        orig = classes[i]
        print("Class", i, "was originally", orig)

    # now actually map the class labels to numbers
    for i in xrange(len(y)):
        newClass, = np.where(classes==y[i])
        y_new[i] = newClass

    return y_new


#-------------------------------------------------------------------------------------
# Primary functions to do stuff
#-------------------------------------------------------------------------------------

def runCrossValidation(data, classes, nFolds=10, modelType='nn'):
    """
    Run n-fold cross validation to evaluate the average accuracy of the model.

    Inputs:
    - data: all data X
    - classes: all classes y
    - nFolds: the number of folds for the cross-validation
    - modelType: which type of model to cross-validation
                nn = neural network
                svm = support vector machine
                rf = random forest

    Effects:
    - prints the accuracy of each fold of cross-validation and the average accuracy

    Returns:
    - scores: score and accuracy of each fold
    """
    # make a cross-validation object
    cv = KFold(n=len(classes), n_folds=nFolds, shuffle=True, random_state=0)

    scores = []
    # iterate through the different folds
    for i, (trainIdx, testIdx) in enumerate(cv):
        if modelType == 'nn': # use the neural network framework
            # train the model on the training data
            m = trainNeuralNetwork(data[trainIdx], classes[trainIdx])
            # test the model on the test data
            scores.append(testNeuralNetwork(m, data[testIdx], classes[testIdx]))
        elif modelType == 'svm': # use the support vector machine framework
            # train the model on the training data
            m = trainSVM(data[trainIdx], classes[trainIdx])
            # test the model on the test data
            scores.append(testSVM(m, data[testIdx], classes[testIdx]))
        elif modelType == 'rf': # use the random forest framework
            # train the model on the training data
            m = trainForest(data[trainIdx], classes[trainIdx])
            # test the model on the test data
            scores.append(testForest(m, data[testIdx], classes[testIdx]))

    scores = np.asarray(scores)
    if modelType == 'nn':
        print('\nAverage accuracy of the model:', np.mean(scores[:, 1]))
    else:
        print('\nAverage accuracy of the model:', np.mean(scores))
    return scores


def extractLearnedFeatures(model, newData):
    """
    Using the previously trained model, extract a learned set of features for the new
    data from the second to last layer of the model.

    Inputs:
    - model: the trained model
    - newData: the new data to extract features from

    Returns:
    - learnedFeats: features extracted from the model
    """
    # https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # https://github.com/fchollet/keras/issues/1641
    # extract layer
    get_last_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-4].output])
    layer_output = get_last_layer_output([X, 0])[0]

    return layer_output


# filenames
rootPath = '/home/jenna/Research/COPDImageAnalysis/annotations/'
annotationDataFn = rootPath + 'data/histFHOG_largeRange_setting1.data.p'
annotationClassesFn = rootPath + 'data/annotationClasses.csv'
neuralNetworkModelFn = rootPath + 'models/keras_neural_network'

# load the data and the classes for the data
features = np.asarray(loadAnnotationData(annotationDataFn))
classes = loadAnnotationClasses(annotationClassesFn)

# convert the classes to categorical labels
numericalClasses = np.asarray(convertClassesToCategorial(classes))
categoricalClasses = np_utils.to_categorical(numericalClasses, len(np.unique(numericalClasses)))

# TESTING FUNCTIONS HERE
# Neural Network
# m = trainNeuralNetwork(features, categoricalClasses)
# s1 = testNeuralNetwork(m, features, categoricalClasses)
# print('Classification accuracy for NN:', s1[1])
# # SVM
# m = trainSVM(features, numericalClasses)
# s2 = testSVM(m, features, numericalClasses)
# print('Classification accuracy for SVM:',s2)
# # Random forest (could potentially look at the N features that contribute most to the classification)
# m = trainForest(features, numericalClasses)
# s3 = testForest(m, features, numericalClasses)
# print('Classification accuracy for RF:', s3)


# Run cross-validation on each model type
nnScores = runCrossValidation(features, categoricalClasses, nFolds=50, modelType='nn')[:, 1]
svmScores = runCrossValidation(features, numericalClasses, nFolds=50, modelType='svm')
rfScores = runCrossValidation(features, numericalClasses, nFolds=50, modelType='rf')

nnAvgAcc = np.mean(nnScores)
svmAvgAcc = np.mean(svmScores)
rfAvgAcc = np.mean(rfScores)

print("\nSummary of cross-validation results")
print("Neural network avg accuracy:", nnAvgAcc)
print("           SVM avg accuracy:", svmAvgAcc)
print(" Random forest avg accuracy:", rfAvgAcc)

# # Train the neural network
# neuralNetworkModel = trainNeuralNetwork(features, categoricalClasses, printFeedback=1)
# # save the neural network
# saveModel(neuralNetworkModelFn, neuralNetworkModel)