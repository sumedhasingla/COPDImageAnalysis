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

# AUROC
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# calculating the confidence interval
from scipy.stats import chi2

# saving/loading data
import h5py  # for saving/loading features
import pickle as pk
import shelve
import pandas as pd

#------------------------------------------------------------------------------------
# Loading/Saving Annotation Data
#------------------------------------------------------------------------------------
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
    - classes: class labels for each patch
    - ids: subject ids
    - indices: patch mask ids
    """
    # load the .csv file using pandas
    dataframe = pd.read_csv(filename)
    print('Loaded the patch classes.')
    # extract the column header associated with the labels
    col = list(dataframe)[1]
    print(list(dataframe))
    print('Looking at column with header', col)
    # extract the label column and convert it to a list
    classes = dataframe[col].values.tolist()
    # extract the column header associated with the ids
    col = list(dataframe)[0]
    # extract the id column and convert it to a list
    ids = dataframe[col].values.tolist()
    # and the patch count
    col = list(dataframe)[2]
    # extract the count column and convert it to a list
    indices = dataframe[col].values.tolist()
    # return the list of labels
    print('Classes for',len(classes), 'patches have been extracted from the file!')
    return classes, ids, indices
    

def loadPatchesByIndices(filename, indices):
    """
    Load only patches specified by the indices parameter.
    
    Inputs:
    - filename: path to the pickle file to load data from
    - indices: list of patch indices to load
    
    Returns:
    - featuresList: features for the patches specified by the indices 
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
    patchIndexList = []
    featureCount = 0
    # since the data is a list of subjects, iterate through them
    for subj in data:
        # iterate through the patches
        for i in xrange(len(subj['L'])):
            # check to see if the index is in the list of indices to load
            if subj['L'][i] in indices:
                featureCount += 1
                featuresList.append(subj['I'][i])
                patchIndexList.append(subj['L'][i])

    print('Extracted', featureCount, 'feature vectors from the data.')
    
    if patchIndexList == indices:
        return featuresList
    else:
        print("*** The patch indices loaded don't match the requested patch indices! Please check me.")
        return 0


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
    model.add(Dense(63, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(63))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(63))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
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


def predictNeuralNetwork(model, X_test):
    """
    Predict the y values for the test set using a previously trained 
    neural network model.

    Inputs:
    - model: previously trained model
    - X_test: test data (fhog/histogram features)
    - Y_test: class for each point of the test data

    Returns:
    - accuracy: the accuracy of the model

    Effects:
    - prints the accuracy of the model in classifying the test data
    """
    y_hats = model.predict(X_test)
    # print('Test data score:', score[0])
    # print('Test data accuracy:', score[1])
    return y_hats #np.argmax(y_hats, axis=1)


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
    model = SVC(probability=True) 
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


def predictSVM(model, X_test):
    """
    Test a previously trained SVM.

    Inputs:
    - model: the previously trained model
    - X_test: the test data

    Returns:
    - y_hats: predictions for the y corresponding to X_test
    """
    y_hats = model.predict_proba(X_test)
    print(y_hats.shape)
    return y_hats


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


def predictForest(model, X_test):
    """
    See if a previously trained random forest classifier can identify
    patch labels for test data.

    Inputs:
    - model: the previously trained model
    - X_test: test data

    Returns:
    - model: a trained model
    """
    y_hats = model.predict_proba(X_test)
    print(y_hats.shape)
    return y_hats


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
    # if modelType == 'nn':
    #     print('\nAverage accuracy of the model:', np.mean(scores[:, 1]))
    # else:
    #     print('\nAverage accuracy of the model:', np.mean(scores))
    return scores


def runCrossValidationAll(data, catClasses, numClasses, nFolds=10):
    """
    Run n-fold cross validation to evaluate the average accuracy of the model.

    Inputs:
    - data: all data X
    - catClasses: all classes y (categorical)
    - numClasses: all classes y (numerical)
    - nFolds: the number of folds for the cross-validation

    Effects:
    - prints the accuracy of each fold of cross-validation and the average accuracy

    Returns:
    - scores: score and accuracy of each fold
    """
    # make a cross-validation object
    cv = KFold(n=len(classes), n_folds=nFolds, shuffle=True, random_state=0)

    scoresNN = []
    scoresSVM = []
    scoresRF = []

    # AUROC
    nClasses = 12
    nn_mean_tpr = 0.0
    svm_mean_tpr = 0.0
    rf_mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    # iterate through the different folds
    for i, (trainIdx, testIdx) in enumerate(cv):
        # train the model on the training data
        m1 = trainNeuralNetwork(data[trainIdx], catClasses[trainIdx])
        # test the model on the test data
        # scoresNN.append(testNeuralNetwork(m1, data[testIdx], catClasses[testIdx]))
        nnYhat = predictNeuralNetwork(m1, data[testIdx])
        # "threshold" data
        nnYhatThresholded = np.zeros_like(nnYhat)
        nnYhatThresholded[np.arange(len(nnYhat)), nnYhat.argmax(1)] = 1
        # testing the AUROC stuff
        nn_class_mean_tprs = 0.0
        # iterate through all of the classes
        for c in xrange(nClasses):
            # print(catClasses[testIdx].shape)
            # print(nnYhat.shape)
            # print(nnYhat[0,:])
            fpr, tpr, thresholds = roc_curve(catClasses[testIdx][:, c], nnYhatThresholded[:, c])
            if np.isnan(tpr).any():
                tpr = np.array([0.0, 1.0])
                fpr = np.array([0.0, 1.0])
            # we want to get the average fpr/tpr rate for all of the classes
            classCount = np.count_nonzero(catClasses[testIdx][:, c])
            nn_class_mean_tprs += classCount*interp(mean_fpr, fpr, tpr)/len(testIdx)
            # print(nn_class_mean_tprs)
            nn_class_mean_tprs[0] = 0.0

        nn_mean_tpr += nn_class_mean_tprs

        # train the model on the training data
        print(trainIdx.shape)
        m2 = trainSVM(data[trainIdx], numClasses[trainIdx])
        # # test the model on the test data
        # scoresSVM.append(testSVM(m2, data[testIdx], numClasses[testIdx]))
        svmYhat = predictSVM(m2, data[testIdx])

        # "threshold" data
        svmYhatThresholded = np.zeros_like(svmYhat)
        svmYhatThresholded[np.arange(len(svmYhat)), svmYhat.argmax(1)] = 1
        # testing the AUROC stuff
        svm_class_mean_tprs = 0.0
        # iterate through all of the classes
        for c in xrange(nClasses):
            # print(catClasses[testIdx].shape)
            # print(nnYhat.shape)
            # print(nnYhat[0,:])
            fpr, tpr, thresholds = roc_curve(catClasses[testIdx][:, c], svmYhatThresholded[:, c])
            if np.isnan(tpr).any():
                tpr = np.array([0.0, 1.0])
                fpr = np.array([0.0, 1.0])
            # we want to get the average fpr/tpr rate for all of the classes
            classCount = np.count_nonzero(catClasses[testIdx][:, c])
            svm_class_mean_tprs += classCount*interp(mean_fpr, fpr, tpr)/len(testIdx)
            # print(nn_class_mean_tprs)
            svm_class_mean_tprs[0] = 0.0

        svm_mean_tpr += svm_class_mean_tprs


        # train the model on the training data
        m3 = trainForest(data[trainIdx], numClasses[trainIdx])
        # # test the model on the test data
        # scoresRF.append(testForest(m3, data[testIdx], numClasses[testIdx]))
        rfYhat = predictForest(m3, data[testIdx])

        # "threshold" data
        rfYhatThresholded = np.zeros_like(rfYhat)
        rfYhatThresholded[np.arange(len(rfYhat)), rfYhat.argmax(1)] = 1
        # testing the AUROC stuff
        rf_class_mean_tprs = 0.0
        # iterate through all of the classes
        for c in xrange(nClasses):
            # print(catClasses[testIdx].shape)
            # print(nnYhat.shape)
            # print(nnYhat[0,:])
            fpr, tpr, thresholds = roc_curve(catClasses[testIdx][:, c], rfYhatThresholded[:, c])
            if np.isnan(tpr).any():
                tpr = np.array([0.0, 1.0])
                fpr = np.array([0.0, 1.0])
            # we want to get the average fpr/tpr rate for all of the classes
            classCount = np.count_nonzero(catClasses[testIdx][:, c])
            rf_class_mean_tprs += classCount*interp(mean_fpr, fpr, tpr)/len(testIdx)
            # print(nn_class_mean_tprs)
            rf_class_mean_tprs[0] = 0.0

        rf_mean_tpr += rf_class_mean_tprs

    # scoresNN = np.asarray(scoresNN)[:,1]
    # scoresSVM = np.asarray(scoresSVM)
    # scoresRF = np.asarray(scoresRF)
    # if modelType == 'nn':
    #     print('\nAverage accuracy of the model:', np.mean(scores[:, 1]))
    # else:
    #     print('\nAverage accuracy of the model:', np.mean(scores))

    nn_mean_tpr /= nFolds
    nn_mean_tpr[-1] = 1.0
    nn_mean_auc = auc(mean_fpr, nn_mean_tpr)
    svm_mean_tpr /= nFolds
    svm_mean_tpr[-1] = 1.0
    svm_mean_auc = auc(mean_fpr, svm_mean_tpr)
    rf_mean_tpr /= nFolds
    rf_mean_tpr[-1] = 1.0
    rf_mean_auc = auc(mean_fpr, rf_mean_tpr)

    # plot the ROC curve
    plt.plot(mean_fpr, nn_mean_tpr, color='b', label='Mean ROC for Neural Network (area = %0.2f)' % nn_mean_auc)
    plt.plot(mean_fpr, svm_mean_tpr, color='r', linestyle="-.", lw='2', label='Mean ROC for SVM (area = %0.2f)' % svm_mean_auc)
    plt.plot(mean_fpr, rf_mean_tpr, color='g', linestyle="--", label='Mean ROC for Random Forest (area = %0.2f)' % rf_mean_auc)

    # plot for random chance
    plt.plot([0, 1], [0, 1], linestyle=':', color='k', label='Luck')

    # Taken from example, setting up plot to look pretty
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Label Classifiers')
    plt.legend(loc="lower right")

    plt.show()

    # return scoresNN, scoresSVM, scoresRF


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
    learnedFeats = get_last_layer_output([newData, 0])[0]

    return learnedFeats


def calculateConfidenceInterval(accuracies):
    """
    Given a list of accuracies, calculate the mean and confidence interval for
    the accuracy of that model.

    Inputs:
    - accuracies: a list of accuracies of one of the cross-validated models

    Returns:
    - sampleMean: the mean of the input list
    - ci: the upper and lower bounds for the confidence interval
    """
    # calculate the sample mean of the accuracies
    sampleMean = np.mean(accuracies)

    # calculate the sample standard deviation of the accuracies
    sumOfDiffs = np.sum((accuracies-sampleMean)**2)

    # get the bounds of the confidence interval (ci)
    ciBound1 = chi2.ppf(0.95, len(accuracies)-1) 
    ciBound2 = chi2.ppf(0.05, len(accuracies)-1)

    # calculate the confidence interval itself
    ci = [sumOfDiffs/ciBound1, sumOfDiffs/ciBound2]

    print("Mean:", sampleMean)
    print(ci)
    # print("StdDev:", sampleStdDev)

    return sampleMean, ci



#----------------------------------------------------------------------------
# MAIN SECTION
#----------------------------------------------------------------------------

# Set up argparser
# Arguments to add
# - root path
# - annotation data filename
# - annotaiton classes filename
# - neural network model "dump" filename
# - file to pass through the feature extraction function
parser = argparse.ArgumentParser()
parser.add_argument("--test-functions", help="Test the 3 model training functions", 
                    dest="testFunctions", action='store_true')
parser.add_argument("--cross-validate", help="Run cross-validation on all 3 models: neural network, support\nvector machine, and random forest",
                    dest="crossValidate", action='store_true')
parser.add_argument("--save-model", help="Train and save the neural network model",
                    dest="saveModel", action='store_true')
parser.add_argument("--extract-features", help="Load a learned neural network and use it to extract learned features for a single subject",
                    dest="extractFeatures", action='store_true')
parser.set_defaults(testFunctions=False, crossValidation=False, saveModel=False, extractFeatures=False)

# FILENAMES
rootPath = '/home/jenna/Research/COPDImageAnalysis/annotations/'
# rootPath = "/pylon2/ms4s88p/jms565/projects/COPDGene/"
annotationDataFn = rootPath + 'data/histFHOG_largeRange_setting1.data.p'
annotationClassesFn = rootPath + 'data/annotationClasses.csv'
# annotationClassesFn = rootPath + 'data/goodPatchClasses.csv'
neuralNetworkModelFn = rootPath + 'models/keras_neural_network_3layers'
subjFeatureFn = '/home/jenna/Research/10002K_INSP_STD_BWH_COPD_BSpline_Iso1.0mm_SuperVoxel_Param30mm_fHOG_Hist_Features.csv.gz'
# unannotated data
featuresFn = rootPath+"unannotated/histFHOG_largeRange_setting1.data.p"
shelfFn = rootPath+'unannotated/histFHOG_largeRange_setting1.shelve'
newFeaturesPickleFn = rootPath+'unannotated/learnedFeatures_3layers.data.p'
newFeaturesShelfFn = rootPath+'unannotated/learnedFeatures_3layers.shelve'

# DO THIS PART EVERY TIME
# load the data and the classes for the data
classes, annotatedIds, patchIndices = loadAnnotationClasses(annotationClassesFn)
features = np.asarray(loadAnnotationData(annotationDataFn))
# features = np.asarray(loadPatchesByIndices(annotationDataFn, patchIndices))

# convert the classes to categorical labels
numericalClasses = np.asarray(convertClassesToCategorial(classes))
categoricalClasses = np_utils.to_categorical(numericalClasses, len(np.unique(numericalClasses)))

args = parser.parse_args()

# TESTING FUNCTIONS HERE
if args.testFunctions:
    # Neural Network
    m = trainNeuralNetwork(features, categoricalClasses)
    s1 = testNeuralNetwork(m, features, categoricalClasses)
    print('Classification accuracy for NN:', s1[1])
    # SVM
    m = trainSVM(features, numericalClasses)
    s2 = testSVM(m, features, numericalClasses)
    print('Classification accuracy for SVM:',s2)
    # Random forest (could potentially look at the N features that contribute most to the classification)
    m = trainForest(features, numericalClasses)
    s3 = testForest(m, features, numericalClasses)
    print('Classification accuracy for RF:', s3)

# Build and evaluate 3 models
if args.crossValidate:
    # Run cross-validation on each model type
    # nnScores = runCrossValidation(features, categoricalClasses, nFolds=50, modelType='nn')[:, 1]
    # svmScores = runCrossValidation(features, numericalClasses, nFolds=50, modelType='svm')
    # rfScores = runCrossValidation(features, numericalClasses, nFolds=50, modelType='rf')

    # nnScores, svmScores, rfScores = runCrossValidationAll(features, categoricalClasses, numericalClasses, nFolds=10)
    runCrossValidationAll(features, categoricalClasses, numericalClasses, nFolds=10)

    # nnAvgAcc, nnConfInt = calculateConfidenceInterval(nnScores)
    # svmAvgAcc, svmConfInt = calculateConfidenceInterval(svmScores)
    # rfAvgAcc, rfConfInt = calculateConfidenceInterval(rfScores)

    # print("\nSummary of cross-validation results")
    # print("Neural network avg evaluation:")
    # print("               Mean:", nnAvgAcc)
    # print("Confidence Interval:", nnConfInt)
    # print("SVM avg accuracy:")
    # print("               Mean:", svmAvgAcc)
    # print("Confidence Interval:", svmConfInt)
    # print("Random forest avg accuracy:")
    # print("               Mean:", rfAvgAcc)
    # print("Confidence Interval:", rfConfInt)

# Build the neural network and save it
if args.saveModel:
    # Train the neural network
    neuralNetworkModel = trainNeuralNetwork(features, categoricalClasses, printFeedback=1)
    # save the neural network
    saveModel(neuralNetworkModelFn, neuralNetworkModel)

# Extract the features for non-annotated images
if args.extractFeatures:
    # load the trained neural network
    m = loadModel(neuralNetworkModelFn)
    # start by opening the feature file for the histogram/fhog features for all subjects
    with open(featuresFn, 'rb') as f:
        data = np.asarray(pk.load(f))

    # and open the shelve file with the metadata to get the list of all subject ids
    shelfData = shelve.open(shelfFn)
    allSubjIds = shelfData['subjList']
    # we already have the list of subjects who have usable annotation labels
    # now find the indices of the unannotated subjects who we have features for
    unannotatedIdxs = [i for i in xrange(len(allSubjIds)) if allSubjIds[i] not in annotatedIds]
    print("There are",len(unannotatedIdxs),"subjects who have not been annotated.")

    # the metadata needs to be changed if it is a pickle file
    unannotatedIds = [sid for sid in allSubjIds if sid not in annotatedIds]
    # structure of the metadata taken from Kayhan's code
    fid = shelve.open(newFeaturesShelfFn, 'n')
    fid['dataConfigDict'] = shelfData['dataConfigDict']
    fid['subjList'] = unannotatedIds
    fid['metaVoxelDict'] = list(np.asarray(shelfData['metaVoxelDict'])[unannotatedIdxs]),
    fid['phenotypeDB_clean'] = shelfData['phenotypeDB_clean'][shelfData['phenotypeDB_clean']['sid'].isin(list(unannotatedIds))],
    fid['shelveFn'] = shelfData['shelveFn'],
    fid['imgFeatureFn'] = shelfData['imgFeatureFn'],
    fid['pickleFn'] = shelfData['pickleFn'],
    fid['snpCSVDataFn'] = shelfData['snpCSVDataFn']
    fid.close()

    # close the shelve file
    shelfData.close()

    unannotatedDataFeats = []
    # for each good subject (by index)
    for idx in unannotatedIdxs:
        # give the features to the informed feature extraction function 
        learnedFeatures = extractLearnedFeatures(m, data[idx]['I'])
        # put the new features into a dictionary for that subject
        subjDict = {'I': learnedFeatures}
        # add that subject's dictionary to the list of unannotated data features
        unannotatedDataFeats.append(subjDict)

    print("Informed features have been extracted for", len(unannotatedDataFeats), "subjects")

    # save the new data structure as a pickle file
    fid = open(newFeaturesPickleFn, 'wb')
    pk.dump(unannotatedDataFeats, fid)
    fid.close()
