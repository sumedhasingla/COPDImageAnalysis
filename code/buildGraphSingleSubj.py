#!/home/jms565/anaconda2/bin/python
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

"""

Notes:
- The distance generated in the graph is the Euclidean distance squared.
"""

# this is a helper function to set the configuration
def loadShelvePickledData(useHarilick=False):     

    # pickleFn =  '%(pickleRootFolder)s/%(pickleSettingName)s/%(pickleSettingName)s.data.p'%\
        # {'pickleSettingName':pickleSettingName, 'pickleRootFolder':pickleRootFolder}

    # On Bridges for job
    pickleFn =  '/pylon1/ms4s88p/jms565/COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p'
    shelveFn = '/pylon1/ms4s88p/jms565/COPDGene_pickleFiles/histFHOG_largeRange_setting1.shelve'

    # desktop or laptop or home
    # pickleFn = "COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p"
    # shelveFn = 'COPDGene_pickleFiles/histFHOG_largeRange_setting1.shelve'
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



# this is a helper function to set the configuration
def loadPickledData():     

    # pickleFn =  '%(pickleRootFolder)s/%(pickleSettingName)s/%(pickleSettingName)s.data.p'%\

    # On Bridges for job
    pickleFn =  '/pylon1/ms4s88p/jms565/COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p'

    # desktop or laptop or home
    # pickleFn = "COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p"
    print "pickleFn : ", pickleFn
    
    # reading pickle and shelve files
    print "Reading pickle file ...."
    fid = open(pickleFn,'rb')
    data = pk.load(open(pickleFn,'rb'))
    fid.close()
    print "Done !"
    return data


def buildSubjectGraph(subj, data, neighbors=5):
    """
    Find the numNodes nodes of each subject that are closest to N nodes
    in every other subject.

    Inputs:
    - subj: index of the subject being database'd
    - data: collection of data to be graph'ed
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
    rowLengths = [ s['I'].shape[0] for s in data ]
    X = np.vstack( [ s['I'] for s in data] )
    # build the graph for a subject
    print "Now building subject-subject mini databases..."
    flann.build_index(data[subj]['I'])
    nodes, dists = flann.nn_index(X, neighbors)
    # decode the results
    idx = 0
    for i in rowLengths:
        # save the numNodes number of distances and nodes
        temp = {
            "nodes": nodes[idx:idx+i],
            "dists": dists[idx:idx+i]
        }
        idx = idx + i
        subjDB.append(temp)
    print "Subject level databases complete!"
    return subjDB

#----------------------------------- ^ trying to parallelize

def saveSubjectGraph(graphs, fn):
    """
    Save the subject graphs in an HDF5 file.

    Inputs:
    - graphs: subject graphs
    - fn: filename to save to

    Returns:
    nothing
    """
    print "Saving subject graphs to HDF5 file..."
    # print "len(graphs): " + str(len(graphs))
    fn = fn + ".h5"
    with h5py.File(fn, 'w') as hf:
        # metadata storage
        # print "dimensions of the table: " + str(len(graphs))
        hf.create_dataset("metadata", [len(graphs)], compression='gzip', compression_opts=7)
        for j in xrange(len(graphs)):
            # print "    current j: " + str(j)
            dsName = str(j).zfill(4)
            g = hf.create_group(dsName)
            g.create_dataset("nodes", data=graphs[j]['nodes'], compression='gzip', compression_opts=7)
            g.create_dataset("dists", data=graphs[j]['dists'], compression='gzip', compression_opts=7)
    print "Subject graph file saved!"


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


# def main():
# Argument parsing stuff
parser = argparse.ArgumentParser()
# Adding arguments to parse
parser.add_argument("-n", "--neighbors", help="the number of nearest neighbors", type=int, default=5)
parser.add_argument("-s", "--subject", help="the index of the subject in the list", type=int)
parser.add_argument("-c", "--cores", help="the number of cores/jobs to use in parallel", type=int, default=1)
# parser.add_argument("-f", "--filepath", help="the file path for the output file")
# Parsing the arguments
args = parser.parse_args()

# grapher = buildGraphs()

# Actually do things
data = loadPickledData()
# subjGraph = buildSubjectGraph(args.subject, subjList, data, args.neighbors)
print "About to start building the graphs in parallel..."
subjGraph = Parallel(n_jobs=args.cores, backend='threading')(delayed(buildSubjectGraph)(args.subject, data, args.neighbors) for i in xrange(1))
print "Finished buliding the graphs!"
# fn = "test_results/"+str(args.subject).zfill(4)
fn = "/pylon1/ms4s88p/jms565/test_results/"+str(args.subject).zfill(4)
saveSubjectGraph(subjGraph[0], fn)
# data2 = loadSubjectGraph(fn)

#if __name__ == "__main__":
#    main()
