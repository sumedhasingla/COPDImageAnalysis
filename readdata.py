import sys
import os
import numpy as np

# handling different types of data
import pandas as pd
import pickle as pk
import shelve
import h5py
from pyflann import *


# this is a helper function to set the configuration
def loadPickledData(useHarilick=False):     
   
    pickleFn =  'COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p'
    # pickleFn =  '%(pickleRootFolder)s/%(pickleSettingName)s/%(pickleSettingName)s.data.p'%\
            # {'pickleSettingName':pickleSettingName, 'pickleRootFolder':pickleRootFolder}
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

metaVoxelDict, subjList, phenotypeDB_clean, data = loadPickledData()

# see data_looking.ipynb for structure information

# READ/PLAY WITH DATA FIRST

# for each subject, create an approximate NN graph using pyflann (cyflann)
#  *http://www.cs.ubc.ca/research/flann/
#  * install it using pip or conda
#  https://github.com/primetang/pyflann
#  https://github.com/dougalsutherland/cyflann

def buildSubjectTrees(subjects, data, neighbors=5):
    """
    Find the numNodes nodes of each subject that are closest to N nodes
    in every other subject.

    Inputs:
    - subjects: included for size (hackish programming)
    - data: collection of data to be tree'ed
    - neighbors: the number of nearest nodes to save

    Returns:
    - subjDBs: list of lists of dictionaries of lists
        - first layer = first subject
        - second layer = second subject
        - third layer = dictionary accessed by keys
        - "nodes": list of 

    """
    flann = FLANN()
    subjDBs = []
    # build the tree for each subject
    print "Now building subject-subject mini databases..."
    for i in xrange(len(subjects)-1):
        results = []
        for j in xrange(len(subjects[i+1:])):
            # print "i: " + str(i) + " j: " + str(j)
            nodes, dists = flann.nn(data[i]['I'], data[j]['I'], neighbors, algorithm='kmeans')
            # save the numNodes number of distances and nodes
            temp = {
                "nodes": nodes,
                "dists": dists
            }
            results.append(temp)
        subjDBs.append(results)

    print "Subject level databases complete!"
    return subjDBs

neighbors = 5
subjTrees = buildSubjectTrees(subjList, data, neighbors)

# digression : http://www.theverge.com/google-deepmind

#------------ NEXT STEP (WARNING: TAKE IT WITH HUGE GRAIN OF SALT)
#  I installed this package : https://github.com/dougalsutherland/skl-groups
# obs_knnObj_Kayhan = knnDiv.indices_[0]   # knn object for the observed data which is the first element in the list
# knnDiv_Kayhan.features_.make_stacked()
# X_feats_Jenna = knnDiv.features_.stacked_features
# obs_knnIdx = obs_knnObj_Kayhan.nn_index(X_feats_Jenna, 3)[0][:,2]
# obs_knnDist = obs_knnObj.nn_index(X_feats_Jenna, 3)[1][:,2]

# # interpretation
# # isolate a subject tree
# knnObj1 = subjTrees[0]['results']
# # get its features
# knnObj1.featureus_.make_stacked()
# # collect features of all other trees
# allSubjFeatures = subjTrees.features_.stacked_features
# # not sure what this is
# indices = knnObj1.nn_index(allSubjFeatures, 3)[0][:, 2]
# distances = subjTrees.nn_index(allSubjFeatures, 3)[1][:, 2]


# -------------------------------
# WE WILL BUILD SPARSE MATRIX representing the connectivity between nodes


# -----------------------
# We will try igraph to detect communities:
# http://igraph.org/redirect.html

