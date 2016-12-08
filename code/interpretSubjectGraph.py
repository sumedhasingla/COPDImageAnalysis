#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import argparse
from argparse import RawTextHelpFormatter

import numpy as np
import cPickle as pickle

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

#-----------------------------------------------------------------------------------------
# Interpretation functions
#-----------------------------------------------------------------------------------------

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

    print(knnMatrix.shape)
    
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
   
def createH5File(sid, subjList, sample_weights, predictions, fnRoot):
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



if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description=""" Subject Graph interpretor. """, formatter_class=RawTextHelpFormatter ) 

    sparseFileRoot = "./simulatedData/sparseGraphs/"
    # sparseFileRoot = "/pylon2/ms4s88p/jms565/simulatedData/sparseGraphs/"
    # parser.add_argument('--sparseFile', type=str, help='input sparse graph file', default=sparseFileRoot)
    preProcFileRoot = "./simulatedData/interpretation/preProc_"
    # preProcFileRoot = "/pylon2/ms4s88p/jsm565/simulatedData/interpretation/preProc_"
    # parser.add_argument('--inputFile', type=str, help='input HDF5 file', required=False, default=preProcFileRoot)
    coefFileRoot = "./simulatedData/interpData/coeffs/"
    # coefFileRoot = "/pylon2/ms4s88p/jms565/simulatedData/interpData/coeffs/"
    # parser.add_argument('--outCoefFile', type=str, help='output HDF5 file for coefficients', required=False, default=coefFileRoot)
    pickleFileRoot = "./simulatedData/interpData/coeffs/"
    # pickleFileRoot = "/pylon2/ms4s88p/jms565/simulatedData/interpData/coeffs/"
    # parser.add_argument('--outPickFile', type=str, help='output Pickle file for LARS object', required=False, default=pickleFileRoot)
    runHelp = "Select which functions to run in the code:"
    runHelp += "    - 0: set up the h5py files containing info for a single subject's interpretation"
    runHelp += "    - 1: run the interpretation code"
    runHelp += "    - 2: visualization of the interpretation"
    parser.add_argument('-r', '--runtype', type=int, help=runHelp, default=0)
    parser.add_argument('-s', '--subjectId', type=str, help='id of subject', default="S0000")

    args = vars(parser.parse_args())
    # print( args )

    from scipy.sparse import csr_matrix
    import h5py

    if args['runtype'] == 0:
        # generate the files

        # TESTING
        sid = 'S0000'
        subjList = ['S'+str(i).zfill(4) for i in xrange(2000)]
        kernelFN = "./simulatedData/kernel-kl"
        kernelWeights = loadKernel(kernelFN)
        predictionsFN = "./simulatedData/interpretation/predictions.hdf5"
        predictions = loadPredictions(predictionsFN)
        sparseFileRoot = "./simulatedData/interpretation/preProc_"
        createH5File(sid, subjList, kernelWeights, predictions, sparseFileRoot)

    elif args['runtype'] == 1: 
        f = h5py.File(preProcFileRoot+args['subjectId']+'.hdf5', "r")
        # read environment variables 
        sid = f['/sid'].value #subject id
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

        coef, clf = interpretSubject(predicted, EdgeMatrix_clean, 
                         numNodesList_clean, sample_weights)

        # saving the results
        f = h5py.File(args['outCoefFile'],'w')
        #hdfPath = '/idx'
        #dset = f.create_dataset(hdfPath, idx.shape, 
        #                         dtype=idx.dtype)
        #dset[...] = idx

        hdfPath = '/coef'
        dset = f.create_dataset(hdfPath, coef.shape, 
                                dtype=coef.dtype)
        dset[...] = coef

        f.close()

        fid = open(args['outPickFile'],'wb')
        pickle.dump(clf, fid, pickle.HIGHEST_PROTOCOL )
        fid.close()

    elif args['runtype'] == 2:
        # not yet implemented
        # select subject
        # read in the weights
        # make a series of subplots (1 per nonzero node)
        # for each node
        # visualize the subject's node in a single subplot
        k = 1
