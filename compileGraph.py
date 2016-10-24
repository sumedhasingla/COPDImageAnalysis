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

def loadPickledData():     

    # pickleFn =  '%(pickleRootFolder)s/%(pickleSettingName)s/%(pickleSettingName)s.data.p'%\
        # {'pickleSettingName':pickleSettingName, 'pickleRootFolder':pickleRootFolder}

    # On Bridges for job
    # pickleFn =  '/pylon1/ms4s88p/jms565/COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p'
    # shelveFn = '/pylon1/ms4s88p/jms565/COPDGene_pickleFiles/histFHOG_largeRange_setting1.shelve'

    # desktop or laptop or home
    pickleFn = "COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p"
    shelveFn = 'COPDGene_pickleFiles/histFHOG_largeRange_setting1.shelve'
    print "pickleFn : ", pickleFn
    print "shelveFn :", shelveFn
    
    # reading pickle and shelve files
    print "Reading the shelve file ..."
    fid = shelve.open(shelveFn,'r')
    metaVoxelDict = fid['metaVoxelDict']
    subjList = fid['subjList']
    phenotypeDB_clean = fid['phenotypeDB_clean']
    fid.close()
    print "Done !"
    print "Sample of the metadata: "
    print "IDs of a few subjects : " , metaVoxelDict[0]['id']
    print "labelIndex of the meta data (a few elements): " , metaVoxelDict[0]['labelIndex'][1:10]   
    print "Reading pickle file ...."
    fid = open(pickleFn,'rb')
    data = pk.load(open(pickleFn,'rb'))
    fid.close()
    print "Done !"
    return metaVoxelDict,subjList, phenotypeDB_clean, data


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
    metaVoxelDict, subjList, phenotypeDB_clean, data = loadPickledData()
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
    subjIdx = 0
    numSimNodes = 5
    # set up matrix: number of elements in DB subject x total number of elements in query subjects
    numSubjPix = superMetaData["subjectSuperPixels"][subjIdx]
    singleSubjGraph = np.zeros((numSubjPix, superMetaData["totalSuperPixels"]))
    # for each query subject in the h5 file
    for j in xrange(1):
        for k in xrange(len(subjGraph[j]["nodes"])):
            # get 3 closest distances 
            nodes = subjGraph[j]["nodes"][k][0:numSimNodes]
            dists = subjGraph[j]["dists"][k][0:numSimNodes]
            # put dists at the location (db subj node, query subj nodes)
            shiftedK = int(superMetaData["superPixelIndexingStart"][j]+k)
            for i in xrange(numSimNodes):
                singleSubjGraph[nodes[i]][shiftedK] = dists[i]
    # * make sure to adjust the query subj nodes wrt the offset from the prev subjs
    print "Finished building this graph"
    return singleSubjGraph


def compileGraphAllSubj(subjGraph, superMetaData, numSimNodes=5 ):
    """
    Compile the distances from all subject graphs to a single matrix.

    Inputs:
    - subjGraph: data loaded from h5 files
    - superMetaData: data about the size and index of the query subjects
                    (from getSubjectSizes())
    - subjIdx: number for identifying the current subject
    - numSimNodes (opt): how many similar nodes will be placed in the matrix

    Outputs:
    - ???
    """
    # set up joint graph?
    # for each subject
    # compileGraphSingleSubj()
    # ^ can this be a single liner?
    # return the massive joint graph matrix
    

# Actually do things
# fn = "test_results/"+str(args.subject).zfill(4)
# fn = "/pylon1/ms4s88p/jms565/test_results/"+str(args.subject).zfill(4)
# Load the graph for a single subject
fn = "0000"
subjGraph = loadSubjectGraph(fn) 
# Get the metadata - should this actually be called metadata?
superMetaData = getSubjectSizes()
# compile the graph for a single subject
subjIdx = 0
numSimNodes = 5
graph = compileGraphSingleSubj(subjGraph, superMetaData, subjIdx)