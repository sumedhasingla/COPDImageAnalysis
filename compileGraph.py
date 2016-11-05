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
    # pickleFn =  '/pylon2/ms4s88p/jms565/COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p'

    # desktop or laptop or home
    pickleFn = "COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p"
    print "pickleFn : ", pickleFn
    
    # reading pickle file
    print "Reading pickle file ...."
    fid = open(pickleFn,'rb')
    data = pk.load(open(pickleFn,'rb'))
    fid.close()
    print "Done !"
    return data


def loadDenseSubjectGraph(fn):
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

def loadMetadata(filename):
    """
    Load the metadata.

    Inputs:
    - filename: the name of the metadata file to load

    Returns: the loaded lung dataset metadata
    """
    loader = np.load(filename+".npz")
    md = {
        "totalSuperPixels": loader['totalSP'],
        "subjectSuperPixels": loader['subjSP'],
        "superPixelIndexingStart": loader['indStart'],
        "superPixelIndexingEnd": loader['indEnd']
    }
    return md


def compileGraphSingleSubj(superMetaData, subjIdx, numSimNodes=3):
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
    # subjGraph = loadSubjectGraph("/pylon2/ms4s88p/jms565/subjectGraphs/"+fn)
    subjGraph = loadDenseSubjectGraph("./individualSubjectGraphs/"+fn)

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
    if i == j:
        rows = graphIJ["nodes"][:, 1:numSimNodes+1]
        dists = graphIJ["dists"][:, 1:numSimNodes+1]
    else:
        rows = graphIJ["nodes"][:, 0:numSimNodes]
        dists = graphIJ["dists"][:, 0:numSimNodes]
    # set up values for the columns of the matrix (total # cols = # pix in subj j)
    cols = np.matrix([[k] * numSimNodes for k in xrange(superMD["subjectSuperPixels"][j])])

    return sp.csr_matrix((list(dists.flat),(list(rows.flat), list(cols.flat))), shape=shapeIJ)


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

#-------------------------------------------------------------------------------
# Actually do things
#-------------------------------------------------------------------------------

# Argument parsing stuff
parser = argparse.ArgumentParser()
# Adding arguments to parse
# parser.add_argument("-n", "--neighbors", help="the number of nearest neighbors", type=int, default=5)
parser.add_argument("-s", "--subject", help="the index of the subject in the list", type=int)
parser.add_argument("-c", "--cores", help="the number of cores/jobs to use in parallel", type=int, default=1)
parser.add_argument("-t", "--jobtype", help="the type of job to run: build a single graph (0) or compile all graphs (1)", type=int, default=0)
# Parsing the arguments
args = parser.parse_args()

# Get the metadata - should this actually be called metadata?
metadataFN = "test-metadata"
superMetaData = loadMetadata(metadataFN)

# ----------------------------------------------------------------------
# Testing revisions
# Create the compiled graph for a single subject i
# simNodes = 3
# print "About to start building the graph in parallel"
# subjGraph0 = compileGraphSingleSubj(superMetaData, 0)
# print "Finished building first graph"

# outFN = "sparseGraphs/"+str(0).zfill(4)
# saveSparseGraph(subjGraph0, outFN)

# print "About to start building the graph in parallel"
# subjGraph1 = compileGraphSingleSubj(superMetaData, 1)
# print "Finished building second graph"

# sparseGraph = sp.vstack((subjGraph0, subjGraph1), format="csr")
# print "Stacked the graphs vertically"

# outFN = "sparseGraphs/"+str(1).zfill(4)
# saveSparseGraph(subjGraph1, outFN)
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# PARALLELIZED 
print "About to start building the graph in parallel"
subjGraph = Parallel(n_jobs=args.cores, backend='threading')(delayed(compileGraphSingleSubj)(superMetaData, args.subject) for i in xrange(1))
print "Finished building individual graphs"

outFN = "sparseGraphs/"+str(args.subject).zfill(4)
saveSparseGraph(subjGraph, outFN)

#----------------------------------------------------------------------

#----------------------------------------------------------------------
# # Testing 
# sparseGraph = compileGraphSingleSubj(superMetaData, 0)
# for i in xrange(3):
#     subjGraph = compileGraphSingleSubj(superMetaData, i+1)
#     fn = "sparseGraphs/"+str(i+1).zfill(4)
#     saveSparseGraph(subjGraph, fn)
#     sparseGraph = sp.vstack((sparseGraph, subjGraph), format='csr')

#----------------------------------------------------------------------

# Compile the subject graphs
if args.jobtype == 1:
    sparseGraph2 = loadSparseGraph("sparseGraphs/0000")
    # for i in xrange(len(superMetaData["numSubjSuperPixels"])-1):
    for i in xrange(3):
        fn = str(i+1).zfill(4)
        loadedGraph = loadSparseGraph("sparseGraphs/"+fn)
        sparseGraph2 = sp.vstack((sparseGraph2, loadedGraph), format="csr")

    outFN = "sparseGraphs/compositeSparseGraph"
    saveSparseGraph(sparseGraph, outFN)

#---------------------------------------------------------------------
# Checking to see if it worked
# confirmation
# loadedGraph = loadSparseGraph(outFN)
# A = sparseGraph.todense()
# # B = loadedGraph.todense()
# C = sparseGraph2.todense()
# print A.all()==C.all()

# print "------------------ Subject " + str(0) + " ----------------"
# mati = subjGraph0
# # check to see if min/max bot > 0
# print "  Min >= 0: " + str((mati.min())>= 0.0)
# print "  Max >= 0: " + str((mati.max())>= 0.0)
# # check to see the number of elements in each col 
# print "  Min # elements in each col: " + str((mati>0).sum(axis=0).min())
# print "  Max # elements in each col: " + str((mati>0).sum(axis=0).max())
# # check to see the number of elements in each row
# print "  Min # elements in each row: " + str((mati>0).sum(axis=1).min())
# print "  Max # elements in each row: " + str((mati>0).sum(axis=1).max())
# # check to see if the number of cols w/ 4 == shape of mat
# print "  Shape of the matrix: " + str(mati.shape)
# # check to see how many nonzero values are in the matrix
# print "  Number of nonzero values: " + str((mati>0).sum())
# print "  According to the matrix: " + str(mati.nnz)
# print "  Should be: " + str(3*superMetaData["totalSuperPixels"])

# print "------------------ Subject " + str(0) + " (Loaded) ----------------"
# mati = loadedGraph
# # check to see if min/max bot > 0
# print "  Min >= 0: " + str((mati.min())>= 0.0)
# print "  Max >= 0: " + str((mati.max())>= 0.0)
# # check to see the number of elements in each col 
# print "  Min # elements in each col: " + str((mati>0).sum(axis=0).min())
# print "  Max # elements in each col: " + str((mati>0).sum(axis=0).max())
# # check to see the number of elements in each row
# print "  Min # elements in each row: " + str((mati>0).sum(axis=1).min())
# print "  Max # elements in each row: " + str((mati>0).sum(axis=1).max())
# # check to see if the number of cols w/ 4 == shape of mat
# print "  Shape of the matrix: " + str(mati.shape)
# # check to see how many nonzero values are in the matrix
# print "  Number of nonzero values: " + str((mati>0).sum())
# print "  According to the matrix: " + str(mati.nnz)
# print "  Should be: " + str(3*superMetaData["totalSuperPixels"])

# print "------------------ Subject " + str(1) + " ----------------"
# mati = subjGraph1
# # check to see if min/max bot > 0
# print "  Min >= 0: " + str((mati.min())>= 0.0)
# print "  Max >= 0: " + str((mati.max())>= 0.0)
# # check to see the number of elements in each col 
# print "  Min # elements in each col: " + str((mati>0).sum(axis=0).min())
# print "  Max # elements in each col: " + str((mati>0).sum(axis=0).max())
# # check to see the number of elements in each row
# print "  Min # elements in each row: " + str((mati>0).sum(axis=1).min())
# print "  Max # elements in each row: " + str((mati>0).sum(axis=1).max())
# # check to see if the number of cols w/ 4 == shape of mat
# print "  Shape of the matrix: " + str(mati.shape)
# # check to see how many nonzero values are in the matrix
# print "  Number of nonzero values: " + str((mati>0).sum())
# print "  According to the matrix: " + str(mati.nnz)
# print "  Should be: " + str(3*superMetaData["totalSuperPixels"])