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
import argparse


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


def buildSubjectTree(subj, subjects, data, neighbors=5):
    """
    Find the numNodes nodes of each subject that are closest to N nodes
    in every other subject.

    Inputs:
    - subj: index of the subject being database'd
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
    subjDB = []
    # build the tree for a subject
    print "Now building subject-subject mini databases..."
    flann.build_index(data[subj]['I'])
    for j in xrange(len(subjects)):
    # for j in xrange(100):  # for testing only
        nodes, dists = flann.nn_index(data[j]['I'], neighbors, algorithm='kmeans')
        # save the numNodes number of distances and nodes
        temp = {
            "nodes": nodes,
            "dists": dists
        }
    	subjDB.append(temp)
    print "Subject level databases complete!"
    return subjDB

#----------------------------------- ^ trying to parallelize

def saveSubjectTree(trees, fn):
    """
    Save the subject trees in an HDF5 file.

    Inputs:
    - trees: subject trees
    - fn: filename to save to

    Returns:
    nothing
    """
    print "Saving subject trees to HDF5 file..."
    fn = fn + ".h5"
    with h5py.File(fn, 'w') as hf:
        # metadata storage
        tableDims = [len(trees), len(trees[0])]
        hf.create_dataset("metadata", tableDims, compression='gzip', compression_opts=7)
        for j in xrange(len(trees)):
            dsName = str(j).zfill(4)
            g = hf.create_group(dsName)
            g.create_dataset("nodes", data=trees[j]['nodes'], compression='gzip', compression_opts=7)
            g.create_dataset("dists", data=trees[j]['dists'], compression='gzip', compression_opts=7)
    print "Subject tree file saved!"


def loadSubjectTree(fn):
    """
    Load the subject trees from an HDF5 file.

    Inputs:
    - fn: filename to load from 

    Returns:
    - trees: loaded subject trees
    """
    print "Loading the subject tree..."
    fn = fn + ".h5"
    with h5py.File(fn, 'r') as hf:
        # print("List of arrays in this file: \n" + str(hf.keys()))
        metadata = hf.get('metadata').shape
        # print metadata
        tree = []
        for j in xrange(metadata[1]):
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
            tree.append(temp)
    print "Tree loaded!"
    return tree


def main():
	# Argument parsing stuff
	parser = argparse.ArgumentParser()
	# Adding arguments to parse
	parser.add_argument("-n", "--neighbors", help="the number of nearest neighbors", type=int)
	parser.add_argument("-s", "--subject", help="the index of the subject in the list", type=int)
	# parser.add_argument("-f", "--filepath", help="the file path for the output file")
	# Parsing the arguments
	args = parser.parse_args()

	# Actually do things
	metaVoxelDict, subjList, phenotypeDB_clean, data = loadPickledData()
	subjTree = buildSubjectTree(args.subject, subjList, data, args.neighbors)
	fn = "test_results/"+str(args.subject).zfill(4)
	saveSubjectTree(subjTree, fn)
	# data2 = loadSubjectTree(fn)

if __name__ == "__main__":
	main()