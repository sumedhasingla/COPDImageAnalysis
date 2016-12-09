#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import argparse
from argparse import RawTextHelpFormatter

import numpy as np
import cPickle as pickle
import pickle as pk

from plot_ave_roc import plotAve
import sklearn.metrics as slm

"""
Notes:
- If generalizing, need to change:
    - number of subjects (runtype 0)
    - paths for the files

"""
#-----------------------------------------------------------------------------------------
# Homemade version of matlab tic and toc functions
#-----------------------------------------------------------------------------------------

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

#-----------------------------------------------------------------------------------------
# Load/Save functions
#-----------------------------------------------------------------------------------------

def loadSparseGraph(fn):
    """
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
    import numpy as np
    import scipy.sparse as sp
    
    loader = np.load(fn)
    print("Sparse graph loaded")
    return sp.csr_matrix((loader['data'], 
                          loader['indices'], 
                          loader['indptr']), 
                          shape=loader['shape'])

def loadPredictions(fn):
    """
    Load the previously saved predictions

    Inputs:
    - fn: filename to load from

    Returns:
    - pred: the loaded predictions
    """
    with h5py.File(fn, 'r') as hf:
        predictions = np.array(hf.get("predictions"))
    return predictions

def loadKernel(fn):
    """
    Load the similarity/kernel matrix.

    Inputs:
    - fn: file to load from

    Returns:
    - the loaded kernel matrix
    """
    loader = np.load(fn+".npz")
    print("Kernel loaded!")
    return loader["similarities"]

def saveCoeffs(fn, coeffs):
    """
    Save the coefficients (nus) to a hdf5 file.

    Inputs:
    - fn: filename to save to
    - coeffs: nus

    Returns: nothing
    """
    with h5py.File(fn,'w') as f:
        ds = f.create_dataset("coeff", coef.shape, dtype=coef.dtype, compression='gzip', compression_opts=9, data=coef)

def loadCoeffs(fn):
    """
    Load the coefficients from a file.

    Inputs:
    - fn: filename to read from

    Returns:
    - coeffs: the loaded coefficients
    """
    with h5py.File(fn, 'r') as hf:
        coeffs = np.array(hf.get("coeff"))
    return coeffs

def saveInterpModel(fn, model):
    """
    Save the model used for the interpolation into a pickle file

    Inputs:
    - fn: filename
    - model: the model

    Returns: nothing
    """
    fid = open(fn,'wb')
    pickle.dump(model, fid, pickle.HIGHEST_PROTOCOL )
    fid.close()

def loadInterpModel(fn):
    """
    Load the pickle file containing the model.

    Inputs:
    - fn: filename to load from

    Returns:
    - model: the interpolation model loaded from the pickle file
    """
    with open(fn, "rb") as f:
        model = pk.load(f)
    return model

#-----------------------------------------------------------------------------------------
# Interpretation functions
#-----------------------------------------------------------------------------------------

def createPreProcFile(sid, subjList, sample_weights, predictions, fnRoot):
    """
    Create a compiled H5 file

    Inputs:
    - sid: subject id
    - subjList: list of subject ids
    - sample_weights: row of weights for that subject from the kernel
    - predictions:
    - fn: 

    Returns: nothing

    Effects: saves a H5 file containing information about a single subject
             needed for the interpretation step.
    """
    print('working on: ' + str(sid))
    # set negative weights in kernel to zero here   
    sample_weights = sample_weights/sample_weights.max()
    sample_weights[sample_weights < 0.0] = 0.0

    # open the sparse graph file
    f = h5py.File(fnRoot+sid+".hdf5", "w")

    # save the values to the dataset
    hdfPath = '/sid'
    dset = f.create_dataset(hdfPath, np.str_(sid).shape , 
                                     dtype=np.str_(sid).dtype)
    dset[...] = sid


    hdfPath = '/subjList'
    dset = f.create_dataset(hdfPath, (len(subjList),), 
                            dtype=np.array(subjList).dtype)
    dset[...] = np.array(subjList)


    hdfPath = '/sample_weights'
    dset = f.create_dataset(hdfPath, sample_weights.shape, 
                            dtype=sample_weights.dtype)
    dset[...] = sample_weights


    hdfPath = '/predicted'
    dset = f.create_dataset(hdfPath, predictions.shape, 
                            dtype=predictions.dtype)
    dset[...] = predictions

    f.close()

def interpretSubject(targets, edgeMatrix, numNodesList, simWeight):
    """
    This function interpret given y locally for a subject 

    Inputs:
    - targets:
    - edgeMatrix:
    - numNodesList:
    - simWeight: weights for subject i (row of subj i from kernel)

    Returns
    - coef: nu (weights)
    - clf: name of the model
    """
    from scipy.sparse import csr_matrix
    #from sklearn.feature_selection import SelectFromModel
    #from sklearn.linear_model import LassoCV
    #from sklearn.linear_model import LassoLarsCV
    from sklearn.linear_model import LarsCV


    # first make a binary matrix the same size as edgeMatrix, it has one if there is knn edge
    # find the node farthest away from the each node for each block
    rows = np.array( (edgeMatrix - (edgeMatrix.max(axis=0)).todense()).argmax(axis=0)  ).squeeze()
    cols = np.array(range( len(rows) ))
    data = np.ones(len(cols),)
    knnMatrix =  csr_matrix( (data,(rows,cols)), shape=edgeMatrix.shape )
    knnMatrix = knnMatrix.todense()

    # print(knnMatrix.shape)
    
    # Make a popularity vector for each subject
    startIndex = 0
    popularityVec = []
    for n in numNodesList: # all 400
        popularityVec.append( np.array(knnMatrix[:,startIndex:(startIndex+n)].sum(axis=1)) )
        startIndex += n
        
    # make the popularity feature and normalize it    
    popularityVec = np.hstack(popularityVec)    
    # normalize vector -- similar to idf idea
    popularityVecMean = popularityVec.mean(axis=1)
    popularityVecMean[popularityVecMean==0.0] = 1.0   # make sure there is no devision by zero
    popularityVecNormalized = popularityVec/popularityVecMean[:,np.newaxis]
    
    
    # approximate the target using sparse linear model
    X = popularityVecNormalized.T
    # print("simweight: ", simWeight.shape)
    # print("shape of X: ", X.shape)
    # bringing the important weight inside of the loss function b/c it is not implemented in 
    X2 = np.sqrt(simWeight[:,np.newaxis])*X 
    y2 = np.sqrt(simWeight)*targets

    #clf = LassoCV(cv=10,n_jobs=-1)
    #clf = LassoLarsCV(cv=10,n_jobs=-1)
    clf = LarsCV(cv=10, normalize=False)
    #sfm = SelectFromModel(clf, threshold=0.25)
    #sfm.fit(X2, y2)
    #n_features = sfm.transform(X2).shape[1]
    #if not(maxNumRegions==None):
    #    while n_features > maxNumRegions:
    #        sfm.threshold += 0.1
    #        n_features = sfm.transform(X2).shape[1]
    #        
    #print("number of features : ", n_features)
    #indices = sfm.get_support(indices=True)
    #coef = sfm.get_params()
    clf.fit(X2, y2)
    coef = clf.coef_
    return coef, clf 
   


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description=""" Subject Graph interpretor. """, formatter_class=RawTextHelpFormatter ) 

    # sparseFileRoot = "./simulatedData/sparseGraphs/"
    # sparseFileRoot = "/pylon2/ms4s88p/jms565/simulatedData/sparseGraphs/"
    # parser.add_argument('--sparseFileRoot', type=str, help='input sparse graph file', required=True, default=sparseFileRoot)
    # preProcFileRoot = "./simulatedData/interpretation/preProc/"
    # preProcFileRoot = "/pylon2/ms4s88p/jms565/simulatedData/interpretation/preProc_"
    # preProcFileRoot = os.environ['LOCAL']+"pre-"
    # parser.add_argument('--preProcFileRoot', type=str, help='input HDF5 file', required=True, default=preProcFileRoot)
    # coefFileRoot = "./simulatedData/interpretation/coeffs/"
    # coefFileRoot = "/pylon2/ms4s88p/jms565/simulatedData/interpData/coeffs/"
    # coefFileRoot = os.environ['LOCAL']+"coeff-"
    # parser.add_argument('--outCoefFileRoot', type=str, help='output HDF5 file for coefficients', required=True, default=coefFileRoot)
    # pickleFileRoot = "./simulatedData/interpretation/models/"
    # pickleFileRoot = "/pylon2/ms4s88p/jms565/simulatedData/interpData/models/"
    # pickleFileRoot = os.environ['LOCAL']+"model-"
    # parser.add_argument('--outPickFile', type=str, help='output Pickle file for LARS object', required=True, default=pickleFileRoot)
    runHelp = "Select which functions to run in the code:"
    runHelp += "    - 0: set up the h5py files containing info for a single subject's interpretation"
    runHelp += "    - 1: run the interpretation code"
    runHelp += "    - 2: processing of the interpretation (ROC)"
    parser.add_argument('-r', '--runtype', type=int, help=runHelp, default=1)
    parser.add_argument('-s', '--subjectId', type=str, help='id of subject', default="S0000")

    args = vars(parser.parse_args())
    print( args )
    # print(args['sid'])
    # sparseFileRoot = args['sparseFileRoot']
    # preProcFileRoot = args['preProcFileRoot']
    # coefFileRoot = args['outCoefFileRoot']
    # pickleFileRoot = args['outPickleFileRoot']

    # filepaths for local
    sparseFileRoot = "./simulatedData/sparseGraphs/"
    preProcFileRoot = "./simulatedData/interpretation/preProc/"
    coefFileRoot = "./simulatedData/interpretation/coeffs/"
    pickleFileRoot = "./simulatedData/interpretation/models/"

    # filepaths for cluster
    # sparseFileRoot = "./sparseGraphs/"
    # preProcFileRoot = "./interpretation/preProc/"
    # coefFileRoot = "./interpretation/coeffs/"
    # pickleFileRoot = "./interpretation/models/"

    from scipy.sparse import csr_matrix
    import h5py

    if args['runtype'] == 0:
        # generate the files
        subjList = ['S'+str(i).zfill(4) for i in xrange(2000)]

        for i in xrange(2000):
            # sid = args['subjectId']
            sid = "S" + str(i).zfill(4)
            kernelFN = "./simulatedData/kernel-kl"
            # kernelFN = "/pylon1/ms4s88p/jms565/simulatedData/kernel-kl"
            kernelWeights = loadKernel(kernelFN)[i]
            # print("Checking the shape of the kernel weights: ", kernelWeights.shape)
            predictionsFN = "./simulatedData/interpretation/predictions.hdf5"
            # predictionsFN = "/pylon2/ms4s88p/jms565/simulatedData/predictions.hdf5"
            predictions = loadPredictions(predictionsFN)
            createPreProcFile(sid, subjList, kernelWeights, predictions, preProcFileRoot)
        print("Finished building the preprocessing files for all subjects!")

        # to test and make sure files aren't the same
        fn0 = "./simulatedData/interpretation/preProc/S0000.hdf5"
        fn1 = "./simulatedData/interpretation/preProc/S0001.hdf5"
        f0 = h5py.File(fn0, "r")
        f1 = h5py.File(fn1, "r")
        print("Checking some files...")
        print("sample weights (false): ", (f1['/sample_weights'].value==f0['/sample_weights'].value).all())
        print("predictions (true): ", (f1['/predicted'].value==f0['/predicted'].value).all())
        f0.close()
        f1.close()

    elif args['runtype'] == 1: 
        f = h5py.File(preProcFileRoot+args['subjectId']+'.hdf5', "r")
        # read environment variables 
        # sid = f['/sid'].value #subject id
        sid = args['subjectId']
        # knnGraphsRoot = f['/knnGraphsRoot'].value # not needed if provided as an input (--inputFile)
        # knnGraphsRoot = args['sparseFile']+sid
        knnGraphsRoot = sparseFileRoot+sid
        # subjList = list(f['/subjList'].value) # Used to find the index of the id -- replaced subjList.index(sid)
        sample_weights = f['/sample_weights'].value 
        predicted = f['/predicted'].value # from regression, need this
        # numNodesList_clean = list(f['/numNodesList_clean'].value) # probably don't need
        f.close()

        # Read the graph edge 
        print( "working on: ",sid)
        EdgeMatrix_clean = loadSparseGraph(knnGraphsRoot+'.npz')
        numNodesList_clean = [400 for i in xrange(2000)]  # mal - hardcoding
        # need to both be (2000,)
        # print(predicted.shape)
        # print(sample_weights.shape)

        coef, clf = interpretSubject(predicted, EdgeMatrix_clean, numNodesList_clean, sample_weights)

        # saving the results
        saveCoeffs(coefFileRoot+sid+".hdf5", coef)
        saveInterpModel(pickleFileRoot+sid+".p.data", clf)

        # checking
        # cl = loadCoeffs(coefFileRoot+sid+".hdf5")        
        # ml = loadInterpModel(pickleFileRoot+sid+".p.data")

        # print("Checking save/load...")
        # print("  coeffs: ", (cl==coef).all())
        # print("  model: ", ml==clf)
        print("Finished interpreting subject: ", sid)

    elif args['runtype'] == 2:
        # not yet implemented
        N = 2000
        xFPR = [[]*N]
        yTPR = [[]*N]
        # generate list of ROC curves
        for i in xrange(N):
            #load the known labels (predictions)
            predFN = "./simulatedData/interpretation/predictions.hdf5"
            yTrue = loadPredictions(predFN)
            #load the coefficients
            coeffFN = coefFileRoot+"S"+str(i).zfill(4)+".hdf5"
            yPred = loadCoeffs(coeffFN)
            #generate the ROC
            fpr, tpr = slm.roc_curve()
            # append new tpr and fpr values to existing lists
            xFPR[i] = fpr
            yTPR[i] = tpr
        plotAve(xss, yss)
        print("Finished generating average ROC!")
