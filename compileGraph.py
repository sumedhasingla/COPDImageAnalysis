#!/usr/bin/env python
import sys
import os
import numpy as np

# handling different types of data
import pandas as pd
import pickle as pk
import shelve
import h5py
from joblib import Parallel, delayed  # conda install -c anaconda joblib=0.9.4
from cyflann import *
import argparse
import scipy.sparse as sp
from scipy import *

#--------------------------------------------------------------------------
# Saving/Loading Single Graphs, etc.
#--------------------------------------------------------------------------

def loadPickledData():     

    # pickleFn =  '%(pickleRootFolder)s/%(pickleSettingName)s/%(pickleSettingName)s.data.p'%\
        # {'pickleSettingName':pickleSettingName, 'pickleRootFolder':pickleRootFolder}

    # On Bridges for job
    pickleFn =  '/pylon2/ms4s88p/jms565/COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p'

    # desktop or laptop or home
    # pickleFn = "COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p"
    print "pickleFn : ", pickleFn
    
    # reading pickle file
    print "Reading pickle file ...."
    fid = open(pickleFn,'rb')
    data = pk.load(open(pickleFn,'rb'))
    fid.close()
    print "Done !"
    return data


def loadSubjectGraph(fn):
    """
    Load the subject graphs from an HDF5 file.

    Inputs:
    - fn: filename to load from 

    Returns:
    - graphs: loaded subject graphs
    """
    print "Loading the subject graph..."
    fn = fn + ".h5"
    with h5py.File(fn, 'r') as hf:
        # print("List of arrays in this file: \n" + str(hf.keys()))
        metadata = hf.get('metadata').shape
        print metadata
        graph = []
        for j in xrange(metadata[0]):
            # get the name of the group
            dsName = str(j).zfill(4)
            # extract the group and the items from the groups
            g = hf.get(dsName)
            nodes = g.get("nodes")
            dists = g.get("dists")
            # put the items into the data structure
            temp = {
                "nodes": np.array(nodes),
                "dists": np.array(dists)
            }
            graph.append(temp)
    print "Graph loaded!"
    return graph

#--------------------------------------------------------------------------
# Building Complete Sparse Graph
#--------------------------------------------------------------------------

def getSubjectSizes():
    """
    Get the total number of superpixels, the number of superpixels for each
    subject, and the start/end indices for each subject in the combined graph

    Inputs: 
    None

    Outputs:
    - superMetaData: dictionary containing
        - totalSuperPixels: the total number of superpixels of all subjects combined
        - subjectSuperPixels: the nubmer of superpixels in each subject
        - superPixelIndexingStart: the index indicating the start of each subject
        - superPixelIndexingEnd: the index indicating the end of each subject
    """
    # read data from original files
    data = loadPickledData()
    # look at size of each file
    numSubjSuperPixels = [ len(s['I']) for s in data ]
    totalSuperPixels = sum(numSubjSuperPixels)
    # create list/dictionary for storing sizes of each subject, start index, and end index?
    superPixelIndexingStart = np.zeros(len(numSubjSuperPixels))
    superPixelIndexingEnd = np.zeros(len(numSubjSuperPixels))
    # subj1: 0 - len(subj1)-1
    # subj2: len(subj1) - len(subj1)+len(subj2)-1
    # is there a more condensed way to write this?
    for i in xrange(len(numSubjSuperPixels)):
        if i == 0 :
            superPixelIndexingStart[i] = 0
            superPixelIndexingEnd[i] = numSubjSuperPixels[i]-1
        else:
            superPixelIndexingStart[i] = numSubjSuperPixels[i-1] + superPixelIndexingStart[i-1]
            superPixelIndexingEnd[i] = numSubjSuperPixels[i] + superPixelIndexingEnd[i-1]

    # return the list/dictionary
    superMetaData = {
        "totalSuperPixels": totalSuperPixels,
        "subjectSuperPixels": numSubjSuperPixels,
        # add both start and end and figure out which one to use later
        "superPixelIndexingStart": superPixelIndexingStart,
        "superPixelIndexingEnd": superPixelIndexingEnd
    }
    return superMetaData


def compileGraphSingleSubj(subjGraph, superMetaData, subjIdx, numSimNodes=5):
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
    # Uses the superMetaData, subjIdx, numSimNodes
    numSubjPix = superMetaData["subjectSuperPixels"][subjIdx]

    # set up initial sparse matrix
    subjJShape = (superMetaData["subjectSuperPixels"][0], numSubjPix)
    # get 3 closest distances for all elements in subj
    cols = subjGraph[0]["nodes"][:, 0:numSimNodes] 
    dists = subjGraph[0]["dists"][:, 0:numSimNodes]
    rows = np.matrix([[i] * numSimNodes for i in xrange(superMetaData["subjectSuperPixels"][0])])
    # make sparse matrix here
    sparseSubj = sp.csr_matrix( (list(dists.flat),(list(rows.flat), list(cols.flat))), shape=subjJShape)

    # for each query subject in the h5 file for one db subject
    for j in xrange(len(subjGraph)-1):
        subjJShape = (superMetaData["subjectSuperPixels"][j+1], numSubjPix)
        # get 3 closest distances 
        cols = subjGraph[j+1]["nodes"][:, 0:numSimNodes] 
        dists = subjGraph[j+1]["dists"][:, 0:numSimNodes]
        rows = np.matrix([[i] * numSimNodes for i in xrange(superMetaData["subjectSuperPixels"][j+1])])
        # make sparse matrix here
        sparseJ = sp.csr_matrix( (list(dists.flat),(list(rows.flat), list(cols.flat))), shape=subjJShape)
        # concatenate w/ row matrix?
        sparseSubj = sp.vstack((sparseSubj, sparseJ), format='csr')
        
    print "Finished building single graph for DB subject " + str(subjIdx) + "!"
    return sparseSubj


def compileGraphAllSubj(superMetaData, numSimNodes=5 ):
    """
    Compile the distances from all subject graphs to a single matrix.

    Inputs:
    - subjGraph: data loaded from h5 files
    - superMetaData: data about the size and index of the query subjects
                    (from getSubjectSizes())
    - subjIdx: number for identifying the current subject
    - numSimNodes (opt): how many similar nodes will be placed in the matrix

    Outputs:
    - sparseGraph: the sparseGraph
    """
    # set up initial graph:
    fn = str(0).zfill(4)
    subjGraph = loadSubjectGraph("/pylon2/ms4s88p/jms565/subjectGraphs/"+fn)
    # subjGraph = loadSubjectGraph("./individualSubjectGraphs/"+fn)
    # compileGraphSingleSubj()
    sparseGraph = compileGraphSingleSubj(subjGraph, superMetaData, 0, numSimNodes=3)

    # for each subject
    for s in xrange(len(superMetaData["subjectSuperPixels"])-1):
    # for s in xrange(1):
        fn = str(s+1).zfill(4)
        subjGraph = loadSubjectGraph("/pylon2/ms4s88p/jms565/subjectGraphs/"+fn)
        # subjGraph = loadSubjectGraph("./individualSubjectGraphs/"+fn)
        # compileGraphSingleSubj()
        sparseSubjI = compileGraphSingleSubj(subjGraph, superMetaData, s+1, numSimNodes=3)
        sparseGraph = sp.hstack((sparseGraph, sparseSubjI), format='csr')
    # return the massive joint graph matrix
    return sparseGraph
    print "Finished compiling complete sparse graph!"


#--------------------------------------------------------------------------
# Saving/Loading Complete Sparse Graph
#--------------------------------------------------------------------------

def saveSparseGraph(graph, fn):
    """
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
    Try to load the previously saved graph
    
    Inputs:
    - fn: the file name/path base (no extensions)
    
    Returns: 
    - the loaded sparse matrix
    """
    loader = np.load(fn+".npz")
    print "Sparse graph loaded"
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


# Actually do things
# fn = "test_results/"+str(args.subject).zfill(4)
# fn = "/pylon1/ms4s88p/jms565/test_results/"+str(args.subject).zfill(4)
# Load the graph for a single subject
# fn = "0000"
# subjGraph = loadSubjectGraph(fn) 
# Get the metadata - should this actually be called metadata?
superMetaData = getSubjectSizes()
# # compile the graph for a single subject
# subjIdx = 0
# numSimNodes = 5
# graph = compileGraphSingleSubj(subjGraph, superMetaData, subjIdx)

# Create the compiled graph
simNodes = 3
sparseGraph = compileGraphAllSubj(superMetaData, numSimNodes=simNodes)
# Save the compiled graph
outFN = "/pylon2/ms4s88p/jms565/compiledSparseGraph"
# outFN = "compiledSparseGraph"

saveSparseGraph(sparseGraph, outFN)

# confirmation
# loadedGraph = loadSparseGraph(outFN)
# A = sparseGraph.todense()
# B = loadedGraph.todense()
# print A.all()==B.all()