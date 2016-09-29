import sys
import os
import numpy as np

# handling different types of data
import pandas as pd
import pickle as pk
import shelve
import h5py
from joblib import Parallel, delayed  # conda install -c anaconda joblib=0.9.4
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
    # for i in xrange(len(subjects)-1):
    for i in xrange(20):  # for testing only
        results = []
        # for j in xrange(len(subjects[i+1:])):
        for j in xrange(20):  # for testing only
            # print "i: " + str(i) + " j: " + str(j)
            nodes, dists = flann.nn(data[i]['I'], data[j]['I'], neighbors, algorithm='kmeans')
            # save the numNodes number of distances and nodes
            temp = {
                "nodes": nodes,
                "dists": dists
            }
            results.append(temp)
        results = buildBranches(i, subjects, data, neighbors, flann)
        subjDBs.append(results)
    # [subjDBs.append(buildBranches(i, subjects, data, neighbors, flann)) for i in xrange(2)]
    # subjDBs=Parallel(n_jobs=6)(delayed(buildBranches)(i, subjects, data, neighbors, flann) for i in xrange(8))

    print "Subject level databases complete!"
    print subjDBs
    return subjDBs

def buildSubjectTreesParallel(subjects, data, neighbors=5):
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
    # for i in xrange(len(subjects)-1):
    # for i in xrange(2):  # for testing only
        # results = []
        # for j in xrange(len(subjects[i+1:])):
        # # for j in xrange(3):  # for testing only
        #     # print "i: " + str(i) + " j: " + str(j)
        #     nodes, dists = flann.nn(data[i]['I'], data[j]['I'], neighbors, algorithm='kmeans')
        #     # save the numNodes number of distances and nodes
        #     temp = {
        #         "nodes": nodes,
        #         "dists": dists
        #     }
        #     results.append(temp)
        # results = buildBranches(i, subjects, data, neighbors, flann)
        # subjDBs.append(results)
    # [subjDBs.append(buildBranches(i, subjects, data, neighbors, flann)) for i in xrange(2)]
    subjDBs=Parallel(n_jobs=4)(delayed(buildBranches)(i, subjects, data, neighbors, flann) for i in xrange(8))

    print "Subject level databases complete!"
    print subjDBs
    return subjDBs

def buildBranches(i, subjects, data, neighbors, flann):
    """
    Inner loop for buildSubjectTrees()

    Inputs:
    - i: current index to start at
    - subjects: for determining how many items to iterate through
    - data: data to cluster
    - neighbors: the number of nearest nodes to save
    - flann: from the containing function

    Returns:
    - results: a single branch of tree'ed data
    """
    results = []
    # for j in xrange(len(subjects[i+1:])):
    for j in xrange(3):  # for testing only
        # print "i: " + str(i) + " j: " + str(j)
        nodes, dists = flann.nn(data[i]['I'], data[j]['I'], neighbors, algorithm='kmeans')
        # save the numNodes number of distances and nodes
        temp = {
            "nodes": nodes,
            "dists": dists
        }
        results.append(temp)
    return results


#----------------------------------- ^ trying to parallelize

def saveSubjectTrees(trees, fn):
    """
    Save the subject trees in an HDF5 file.

    Inputs:
    - trees: subject trees
    - fn: filename to save to

    Returns:
    nothing
    """
    fn = fn + ".h5"
    with h5py.File(fn, 'w') as hf:
        # metadata storage
        tableDims = [len(trees), len(trees[0])]
        hf.create_dataset("metadata", tableDims, compression='gzip', compression_opts=7)
        for i in xrange(len(trees)):
            for j in xrange(len(trees[0])):
                dsName = str(i).zfill(4)+"_"+str(j).zfill(4)
                g = hf.create_group(dsName)
                g.create_dataset("nodes", data=trees[i][j]['nodes'], compression='gzip', compression_opts=7)
                g.create_dataset("dists", data=trees[i][j]['dists'], compression='gzip', compression_opts=7)


def loadSubjectTrees(fn):
    """
    Load the subject trees from an HDF5 file.

    Inputs:
    - fn: filename to load from 

    Returns:
    - trees: loaded subject trees
    """
    fn = fn + ".h5"

    with h5py.File(fn, 'r') as hf:
        print("List of arrays in this file: \n" + str(hf.keys()))
        metadata = hf.get('metadata').shape
        print metadata
        trees = []
        for i in xrange(metadata[0]):
            branch = []
            for j in xrange(metadata[1]):
                # get the name of the group
                dsName = str(i).zfill(4)+"_"+str(j).zfill(4)
                # extract the group and the items from the groups
                g = hf.get(dsName)
                nodes = g.get("nodes")
                dists = g.get("dists")
                # put the items into the data structure
                temp = {
                    "nodes": np.array(nodes),
                    "dists": np.array(dists)
                }
                branch.append(temp)
            trees.append(branch)

    return trees

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

def main():
    """ Main function """
    metaVoxelDict, subjList, phenotypeDB_clean, data = loadPickledData()
    neighbors = 5
    subjTrees = buildSubjectTrees(subjList, data, neighbors)
    fn = "subjTrees"
    saveSubjectTrees(subjTrees, fn)
    data = loadSubjectTrees(fn)

if __name__ == '__main__':
    main()