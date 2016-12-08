import numpy as np
import argparse
from joblib import Parallel, delayed

# generating simulated data
from keras.datasets import mnist
from keras.models import *
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.core import K
import tensorflow as tf

# saving/loading simulated data
import h5py  # for saving/loading features
import pickle as pk

# Dougal code imports
from skl_groups.features import Features
from sklearn.pipeline import Pipeline
from skl_groups.divergences import KNNDivergenceEstimator
from skl_groups.kernels import PairwisePicker, Symmetrize, RBFize, ProjectPSD

# from buildGraphSingleSubj.py (builing knn graphs)
from cyflann import *

# from compileGraph.py (sparse graphs)
import scipy.sparse as sp
from scipy import *

# for on Bridges
import os

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
#--------------------------------------------------------------------------
# New Functions Which Do Cool Math/ML Stuff
#--------------------------------------------------------------------------

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
    nb_epoch = 20  # supposed to be 20 - for real

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20))
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

    # fn = "simulatedData/keras-model"
    # model.save(fn)
    return model  # or return the weights of the second to last layer?

def generateAbnormalNode(zeroImgs, oneImgs, n, model, normalMax=255):
    """
    Choose 2 images from the MNIST dataset to combine into an "abnormal" node

    Inputs:
    - mnistOnes: vectorized images of "1"s 
    - mnistZeros: vectorized images of "0"s
    - n:
    - model: keras model

    Returns:
    - abnormal: the abnormal node 
    """
    # generate a random number to select a 1 image
    idx1 = np.random.randint(0, len(oneImgs[0])-1)
    # generate a random number to select a 0 image
    idx0 = np.random.randint(0, len(zeroImgs[0])-1)
    # select a 1 image
    i1 = oneImgs[idx1]
    # select a 0 image
    i0 = zeroImgs[idx0]
    # combine the 2 images into 1 (add them, values are btwn 0 and 1)
    abImg = i0+i1
    # normalize the image
    abImg[abImg > normalMax] = normalMax
    # get the features
    abFeat = extractFeatures(abImg.reshape((n, 784)), model)
    # abFeat = extractFeatures(abImg)
    return [abFeat, abImg]

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
    # extract layer
    get_last_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-4].output])
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
    distEstModel = Pipeline([ # div_funcs=['kl'], rewrite this to actually use PairwisePicker correctly next time
        ('divs', KNNDivergenceEstimator(div_funcs=['hellinger'], Ks=[3], n_jobs=-1, version='fast')),
        ('pick', PairwisePicker((0, 0))),
        ('symmetrize', Symmetrize())#,
        # ('rbf', RBFize(gamma=1, scale_by_median=True)),
        # ('project', ProjectPSD())
    ])

    # return the pairwise similarities between the bags (patients)
    sims = distEstModel.fit_transform(feats)
    return sims

#--------------------------------------------------------------------------
# Imported Functions from Previous Work (WARNING: MIGHT BE MODIFIED)
#--------------------------------------------------------------------------

def buildSubjectGraph(subj, patients, neighbors=3):
    """
    Find the numNodes nodes of each subject that are closest to N nodes
    in every other subject.

    Inputs:
    - subj: index of the subject being database'd
    - patients: collection of data to be graphed
    - neighbors: the number of nearest nodes to save

    Returns:
    - subjDBs: list of lists of dictionaries of lists
        - first layer = first subject
        - second layer = second subject
        - third layer = dictionary accessed by keys
        - "nodes": list of N nearest nodes
        - "dists": list of dists for N nearest nodes
    """
    flann = FLANNIndex()
    subjDB = []
    # vectorize data
    # rowLengths = [ s['I'].shape[0] for s in data ]
    rowLengths = [ p.shape[0] for p in patients ]
    # X = np.vstack( [ s['I'] for s in data] )
    X = np.vstack( [ p for p in patients])
    # build the graph for a subject
    print "Now building subject-subject mini databases..."
    # flann.build_index(data[subj]['I'])
    flann.build_index(patients[subj])
    nodes, dists = flann.nn_index(X, neighbors+1)
    # decode the results
    idx = 0
    for i in rowLengths:
        # save the numNodes number of distances and nodes
        if (dists[idx:idx+i, 0] == 0.0).all():
            # shift the nodes
            temp = {
                "nodes": nodes[idx:idx+i, 1:],
                "dists": dists[idx:idx+i, 1:]
            }
        else: 
            # no shift needed
            temp = {
                "nodes": nodes[idx:idx+i, :neighbors],
                "dists": dists[idx:idx+i, :neighbors]
            }
        idx = idx + i
        subjDB.append(temp)
    print "Subject level database complete for subject " + str(subj) + "!"
    return subjDB

def compileGraphSingleSubj(superMetaData, subjIdx, subjGraph, numSimNodes=3):
    """ 
    Extract the data from a single subject into a massive matrix
    Matrix size is # elements in DB subject x # elements in query subject

    Inputs:
    - subjGraph: data loaded from h5 files
    - superMetaData: data about the size and index of the query subjects
                    (from getSubjectSizes())
    - subjIdx: number for identifying the current subject
    - numSimNodes (opt): how many similar nodes will be placed in the matrix

    Returns:
    """
    # set up initial graph:
    fn = str(subjIdx).zfill(4)
    # subjGraph = loadDenseSubjectGraph("/pylon2/ms4s88p/jms565/subjectGraphs/"+fn)
    # subjGraph = loadDenseSubjectGraph("./individualSubjectGraphs/"+fn)

    # make initial sparse matrix here
    j = 0
    sparseSubj = buildBlock(subjGraph[j], subjIdx, j, superMetaData)
    # for each query subject in the h5 file for one db subject
    for j in xrange(len(subjGraph)-1):
        # make the block for subjects i-j
        sparseJ = buildBlock(subjGraph[j+1], subjIdx, j+1, superMetaData)
        # concatenate w/ row matrix
        sparseSubj = sp.hstack((sparseSubj, sparseJ), format='csr')
        
    print "Finished building single graph for DB subject " + str(subjIdx) + "!"
    return sparseSubj

def buildBlock(graphIJ, i, j, superMD, numSimNodes=3):
    """
    Build a single block (subj_i-subj_j) of the subj_i matrix

    Inputs:
    - graphIJ: 
    - i: index of subj i
    - j: index of subj j
    - superMetaData: metadata of the dataset
    - numSimNodes: 3 by default, can be changed for different dataset

    Returns:
    - sparse matrix full of the nonzero values in that block
    """

    # set up shape of sparse matrix
    shapeIJ = (superMD["subjectSuperPixels"][i], superMD["subjectSuperPixels"][j])

    # if i and j are the same, don't take the distance between the node and itself
    # if graphIJ['dists'][0][0]==0:
    #     rows = graphIJ["nodes"][:, 1:numSimNodes+1]
    #     dists = graphIJ["dists"][:, 1:numSimNodes+1]
    # else:
    rows = graphIJ["nodes"][:, 0:numSimNodes]
    dists = graphIJ["dists"][:, 0:numSimNodes]
    # set up values for the columns of the matrix (total # cols = # pix in subj j)
    cols = np.matrix([[k] * numSimNodes for k in xrange(superMD["subjectSuperPixels"][j])])

    return sp.csr_matrix((list(dists.flat),(list(rows.flat), list(cols.flat))), shape=shapeIJ)

#--------------------------------------------------------------------------
# Saving and Loading Files
#--------------------------------------------------------------------------

def saveSimFeats(fn, features, ids, y):
    """
    Function to save the generated patient features using pickle

    Inputs:
    - fn: filename/directory to save to (extensionless)
    - features: single patient's features/nodes
    - ids: subject ids
    - y: the label (number of abnormal nodes)

    Returns: nothing
    """
    cucumber = {
        "ids": ids,
        "y": y,
        "features": features
    }
    with open(fn+"-feats.data.p", "wb") as f:
        pk.dump(cucumber, f)
    f.close()
    print "Saved features data for the simulated patients using pickle."

def loadSimFeats(fn):
    """
    Load a previously saved simulated subject from a .npz file.

    Inputs:
    - fn: filename/directory to load from (extensionless)

    Returns:
    - features: single patient's features/nodes
    - ids: subject ids
    - y: the label (number of abnormal nodes)
    """
    with open(fn+"-feats.data.p", "rb") as f:
        loader = pk.load(f)
    f.close()
    print "Simluated patient features and metadata loaded!"
    return loader['ids'], np.asarray(loader['y']), loader['features']

def saveSimImg(fn, img):
    """
    Function to save the generated patient imgs using pickle

    Inputs:
    - fn: filename/directory to save to (extensionless)
    - img: single patient's image collection

    Returns: nothing
    """
    np.savez(fn, image=img)
    # print "Saved another patient image."

def loadSimImg(fn):
    """
    Load a previously saved simulated subject from a .npz file.

    Inputs:
    - fn: filename/directory to load from (extensionless)

    Returns:
    - images: single patient's features/nodes
    """
    loader = np.load(fn+".npz")
    print "Image " + fn + " loaded!"
    return loader['image']

def saveSimilarities(fn, sims):
    """
    Save the similarity matrix to a .npz file.

    Inputs:
    - fn: directory/filename to save the file to (extension will be provided by func.)
    - sims: similarity matrix

    Returns: nothing
    """
    np.savez(fn, similarities=sims)
    print "Saved the similarities to a file."

def loadSimilarities(fn):
    """
    Load the previously saved similarity matrix from a .npz file

    Inputs:
    - fn: directory/filename (minus extension) to load the file from

    Returns:
    - loadedSims: similarity matrix
    """
    loader = np.load(fn+".npz")
    print "Similarities loaded!"
    return loader['similarities']

def saveSparseGraph(graph, fn):
    """
    IMPORTED!
    Try to save the graph using numpy.save
    
    Inputs:
    - graph: the csr_matrix to save
    - fn: the filename base (no extensions)
    
    Returns: none
    """
    np.savez(fn, data=graph.data, indices=graph.indices, indptr=graph.indptr, shape=graph.shape)
    print "Saved the files"
        
def loadSparseGraph(fn):
    """
    IMPORTED
    Try to load the previously saved graph. Uses a different format to build the 
    sparse matrix. 

    NOTE: Initial building uses the format

        csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])

    The row and col indices are not easily extractable from the csr_matrix, so the
    loader uses the format

        csr_matrix((data, indices, indptr), [shape=(M, N)])

    to recreate the matrix. The loaded and original matrices have been compared 
    and are the same.
    
    Inputs:
    - fn: the file name/path base (no extensions)
    
    Returns: 

    """
    loader = np.load(fn+".npz")
    print "Sparse graph loaded"
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

def saveSimMetadata(fn, numSubjs, numNodes):
    """
    Generate and save the metadata needed for the simulated patient dataset.

    Metadata:
    - totalSuperPixels: the total number of nodes in all of the patients
    - subjectSuperPixels: the number of nodes in each patient
    - superPixelIndexingStart: 

    Inputs: 
    - fn: output filename
    - numSubjs: number of subjects to generate
    - numNodes: number of nodes for each subject

    Returns: nothing
    """
    # generate the metadata
    totalSP = numSubjs*numNodes
    subjSP = [numNodes] * numSubjs
    superPixelIndexingStart = np.zeros(numSubjs)
    superPixelIndexingEnd = np.zeros(numSubjs)

    for i in xrange(numSubjs):
        if i == 0 :
            superPixelIndexingStart[i] = 0
            superPixelIndexingEnd[i] = subjSP[i]-1
        else:
            superPixelIndexingStart[i] = superPixelIndexingStart[i-1] + subjSP[i-1]
            superPixelIndexingEnd[i] = superPixelIndexingStart[i] + subjSP[i-1]

    np.savez(fn, totalSP=totalSP, subjSP=subjSP, indStart=superPixelIndexingStart, indEnd=superPixelIndexingEnd)

def loadSimMetadata(fn):
    """
    Load metadata for simulated dataset from file.

    Inputs: 
    - fn: filename to read from (extensionless)

    Returns:
    - md: metadata
        - totalSuperPixels: total number of nodes in set
        - subjectSuperPixels: the number of nodes in each subject
        - superPixelIndexingStart: the start index of each subject in the list
        - superPixelIndexingEnd: the end index of each subject in the list
    """
    loader = np.load(fn+".npz")
    md = {
        "totalSuperPixels": loader['totalSP'],
        "subjectSuperPixels": loader['subjSP'],
        "superPixelIndexingStart": loader['indStart'],
        "superPixelIndexingEnd": loader['indEnd']
    }
    return md

def saveFeatures(fn, feats, digitNumbers):
    """
    After extracting the features of the MNIST test set, save them
    to a .h5 file.

    Inputs:
    - fn: filename to save to (extensionless)
    - feats: list/nparray of features to save

    Returns: nothing
    """
    fn = fn + ".h5"
    print fn
    with h5py.File(fn, "w") as hf:
        hf.create_dataset("feats", data=feats, compression='gzip', compression_opts=7)
        hf.create_dataset("numbers", data=digitNumbers, compression='gzip', compression_opts=7)

    print "Features file saved!"

def loadFeatures(fn):
    """
    Load the features of the MNIST test set.

    Inputs:
    - fn: the filename to load from (extensionless)

    Returns:
    - feats: list/nparray of features
    """
    fn = fn + ".h5"
    print fn
    with h5py.File(fn, "r") as hf:
        a = hf.get('feats')
        feats = np.array(a)
        b = hf.get('numbers')
        numbers = np.array(b)

    return feats, numbers
    print "Features loaded!"

#--------------------------------------------------------------------------
# Actually do stuff...
#--------------------------------------------------------------------------

# Argument parsing
parser = argparse.ArgumentParser()
# Adding arguments to parse
parser.add_argument("-s", "--subject", help="the index of the subject in the list", type=int)
parser.add_argument("-c", "--cores", help="the number of cores/jobs to use in parallel", type=int, default=2)
# parser.add_argument("-d", "--debug", help="flag to print out helpful output", type=int, default=0)  # 1 for debugging messages
runTypeHelp = "which type of program to run"
runTypeHelp += "    - 0: generate metadata"
runTypeHelp += "    - 1: test the functions to see if they run"
runTypeHelp += "    - 2: generate all 7292 simulated patients"
runTypeHelp += "    - 3: generate, save, build graph for, and sparsify graph for single subject"
runTypeHelp += "    - 4: generate kernel (similarities)"
parser.add_argument("-r", "--runtype", help=runTypeHelp, type=int, default=5)

args = parser.parse_args()

if args.runtype == 0: 
    # preprocessing
    N = 2000  # number of patients - should be 7292
    totalNodes = 400  # total number of nodes for each patient - should be 500
    mdFN = "metadata-simulated"
    saveSimMetadata(mdFN, N, totalNodes)
    loadSimMetadata(mdFN)

    # Load MNIST data
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    A = np.vstack((X_test, X_train[0:25000]))
    A_y = np.hstack((y_test, y_train[0:25000]))
    B = X_train[25000:]
    B_y = y_train[25000:]
    X_train = B.astype('float32')
    X_test = A.astype('float32')
    X_train /= 255
    X_test /= 255
    y_train = B_y
    y_test = A_y

    # # Train the model
    # print "Training the knn model..."
    # K._LEARNING_PHASE = tf.constant(0)
    # model = trainModel(X_train, y_train, X_test, y_test)
    # print "KNN model trained!"

    # Load a previously trained keras model
    kerasFN = "simulatedData/keras-model"
    # model.save(kerasFN)
    model = load_model(kerasFN)

    # Generate the features for the test data set
    feats = extractFeatures(X_test, model)
    print feats.shape
    # Save the features and their classes 
    featsFN = "simulatedData/node-features"
    saveFeatures(featsFN, feats, y_test) 
    # Load the feature and their classes
    loadedFeats, loadedDigitClass = loadFeatures(featsFN)

elif args.runtype == 1:
    # Load MNIST data
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    A = np.vstack((X_test, X_train[0:25000]))
    A_y = np.hstack((y_test, y_train[0:25000]))
    X_test = A.astype('float32')
    X_test /= 255
    y_test = A_y

    # Load the features for the test data set 
    featsFN = "simulatedData/node-features"
    loadedFeats, loadedY = loadFeatures(featsFN)

    # # Load a previously trained keras model
    kerasFN = "simulatedData/keras-model"
    model = load_model(kerasFN)

    # get the data for generating the abnormal nodes
    mnistOneIndices = [i for i in xrange(len(y_test)) if y_test[i]==1 ]
    mnistZeroIndices = [i for i in xrange(len(y_test)) if y_test[i]==0 ]
    onesImgs = X_test[mnistOneIndices]
    zerosImgs = X_test[mnistZeroIndices]
    
    # Generate the simulated patients
    N = 2000  # number of patients - should be 2000
    totalNodes = 400  # total number of nodes for each patient - should be 400
    patFeats = [None]*N
    ids = ["S"+str(i).zfill(4) for i in xrange(N)]
    mu = totalNodes/2
    sigma = 175
    img0 = None
    # generate the abnormal nodes
    yDist = np.random.normal(mu, sigma, N*5)
    yClipped = yDist[((yDist >= 0) & (yDist <= totalNodes))]
    y = yClipped[:N]
    yRound = np.floor(y).astype(int)
    totalY = np.sum(yRound)
    # select permutation of "1" images
    perm1 = np.hstack([np.random.permutation(len(onesImgs))]*200)
    # select permutation of "0" images
    perm0 = np.hstack([np.random.permutation(len(zerosImgs))]*200)
    # get the images
    subsetImgs0 = zerosImgs[perm0]
    subsetImgs1 = onesImgs[perm1]
    # combine the images
    abImgs = subsetImgs1+subsetImgs1
    abImgs[abImgs > 1] = 1.0
    # get the features for the abnormal images
    abFeats = extractFeatures(abImgs, model)
    # generate list of permuted indices
    permutations = [np.random.permutation(len(loadedFeats))]*100
    permutations = np.hstack(permutations) # this should be 35000*100 long (1D)
    print "Generating simulated patients..."
    idxN = 0
    idxA = 0
    for i in xrange(N):    
        patImgs = []
        # select subset of indices from list - normal nodes
        subset = permutations[idxN:idxN+totalNodes-yRound[i]]
        # generate the nodes and add some small (<= 1% of max feature value) Gaussian noise
        normalFeats = loadedFeats[subset] + np.random.rand(len(subset), len(loadedFeats[0]))*loadedFeats.max()/100.0
        # normalFeats = loadedFeats[subset]
        normalImgs = X_test[subset]
        if yRound[i] > 0.0: 
            abnormalFeats = abFeats[idxA:idxA+yRound[i]]
            abnormalImgs = abImgs[idxA:idxA+yRound[i]]
            # add the generated features to the list for that patient
            patFeats[i] = np.concatenate((normalFeats, abnormalFeats))
            # add the generated images to the list for that patient
            patImgs = np.concatenate((normalImgs, abnormalImgs))
            # Woo sanity check
            print "Iteration " + str(i)
            print "            yRound: " + str(yRound[i])
            print "  Len(normalNodes): " + str(len(normalFeats)) + " totalNodes-yRound: " + str(totalNodes-yRound[i])
            print "   Len(normalImgs): " + str(len(normalImgs))
            print "Len(abnormalNodes): " + str(len(abnormalFeats)) + " yRound: " + str(yRound[i])
            print " Len(abnormalImgs): " + str(len(abnormalImgs))
            print "     Len(patFeats): " + str(len(patFeats[i]))
        else: 
            patFeats[i] = normalFeats
            patImgs = normalImgs
            # Woo sanity check
            print "Iteration " + str(i)
            print "            yRound: " + str(yRound[i])
            print "  Len(normalNodes): " + str(len(normalFeats)) + " totalNodes-yRound: " + str(totalNodes-yRound[i])
            print "   Len(normalImgs): " + str(len(normalImgs))
            print "     Len(patFeats): " + str(len(patFeats[i]))

        if i == 0:
            img0 = patImgs

        # increment index counter
        idxN += totalNodes-yRound[i]
        idxA += yRound[i]
        imgFN = "./simulatedData/simulatedImages/" + ids[i]
        saveSimImg(imgFN, patImgs)

    print "Patients have been simulated!"

    # Save the patients
    patientsFN = "./simulatedData/simulatedSubjects"
    saveSimFeats(patientsFN, patFeats, ids, y)

    imgFN = "./simulatedData/simulatedImages/S0000"
    loadedImg = loadSimImg(imgFN)
    # print "Patient image list has been loaded: " 
    # print "      Images: " + str((loadedImg==img0).all())

    loadedIds, loadedYs, loadedPatches = loadSimFeats(patientsFN)
    print "Patient features have been loaded: " 
    print "    Features: " + str((np.asarray(loadedPatches)==np.asarray(patFeats)).all())
    print "         Ids: " + str((np.asarray(loadedIds)==np.asarray(ids)).all())
    print "           Y: " + str((loadedYs==y).all())

elif args.runtype == 2:
    # save the subject
    patientsFN = "./simulatedData/simulatedSubjects"
    loadedIds, numAbnormalNodes, loadedSubjs = loadSimFeats(patientsFN)

    # Compute the pairwise similarity between patients using Dougal code
    print "Calculating similarities..."
    sims = computePairwiseSimilarities(loadedSubjs, numAbnormalNodes)
    print "Similarities calculated!"
    # save the similarities
    kernelFN = "./simulatedData/kernel-2000-sym-he"
    # kernelFN = "./simulatedData/kernel-2000-sym-v2"
    saveSimilarities(kernelFN, sims)
    # load the similarities to check
    loadedK = loadSimilarities(kernelFN)
    # check the similarities
    print (loadedK==sims).all()

elif args.runtype == 3:
    # parallelized part
    # generate, save, build graph for, and sparsify graph for single subject
    # read in subject graph
    # patientsFN = './simulatedData/simulatedSubjects'
    patientsFN = '/pylon1/ms4s88p/jms565/simulatedData/simulatedSubjects'
    loadedIds, loadedYs, loadedSubjs = loadSimFeats(patientsFN)    
    data = loadedSubjs
    # build the graph for the subject
    print "About to start building the graph in parallel..."
    subjGraph = Parallel(n_jobs=args.cores, backend='threading')(delayed(buildSubjectGraph)(args.subject, data) for i in xrange(1))
    # subjGraph = buildSubjectGraph(args.subject, data)
    print "Finished building the graph!"
    # sparsify the graph for the subject
    print "Now sparsifying the graph..."
    # mdFN = "simulatedData/metadata-simulated"
    mdFN = "/pylon1/ms4s88p/jms565/simulatedData/metadata-simulated"
    md = loadSimMetadata(mdFN)
    sparseGraph = Parallel(n_jobs=args.cores, backend='threading')(delayed(compileGraphSingleSubj)(md, args.subject, subjGraph[0], numSimNodes=3) for i in xrange(1))
    # sparseGraph = compileGraphSingleSubj(md, args.subject, subjGraph[0], numSimNodes=3)
    print "Finished sparsifying the graph!"
    # save the sparse graph for the subject
    sparseFN = "./simulatedData/sparseGraphs" + str(args.subject).zfill(4)
    # sparseFN = os.environ['LOCAL'] + '/S' + str(args.subject).zfill(4)
    saveSparseGraph(sparseGraph[0], sparseFN)
    print "Sparse graph for subject " + str(args.subject) + "!"
    loaded = loadSparseGraph(sparseFN)
    print "Checking sparse graph:"
    print "  Same data: " + str((loaded.data==sparseGraph[0].data).all())
    print "  Same indices: " + str((loaded.indices==sparseGraph[0].indices).all())
    print "  Same indptrs: " + str((loaded.indptr==sparseGraph[0].indptr).all())
    print "  Same shapes: " + str(loaded.shape==sparseGraph[0].shape)

elif args.runtype == 4:
    # generate kernel
    # load the patient node information
    patientsFN = "./simulatedData/simulatedSubjects"
    loadedSubjs, numAbnormalNodes = loadSimSubject(patientsFN)
    # Compute the pairwise similarity between patients using Dougal code
    print "Calculating similarities..."
    sims = computePairwiseSimilarities(loadedSubjs, numAbnormalNodes)
    print "Similarities calculated!"
    # Save the similarities
    kernelFN = "./simulatedData/kernel-matrix"
    saveSimilarities(kernelFN, sims)
    # Load the similarities to test
    loadedKernel = loadSimilarities(kernelFN)
