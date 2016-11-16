#!/usr/bin/env python
import numpy as np
import scipy.sparse as sp


loader = np.load("test-metadata.npz")
superMetaData = {
    "totalSuperPixels": loader['totalSP'],
    "subjectSuperPixels": loader['subjSP'],
    "superPixelIndexingStart": loader['indStart'],
    "superPixelIndexingEnd": loader['indEnd']
}


for i in xrange(4):
    # load subj i graph
    subi = np.load("./sparseGraphs/"+str(i).zfill(4)+".npz")
    # subi = np.load("/pylon1/ms4s88p/jms565/sparseGraphs/"+str(i+1).zfill(4)+".npz")
    mati = sp.csr_matrix((subi['data'], subi['indices'], subi['indptr']), shape=subi['shape'])
    print "------------------ Subject " + str(i) + " ----------------"
    # check to see if min/max bot > 0
    print "  Min >= 0: " + str((mati.min())>= 0.0)
    print "  Max >= 0: " + str((mati.max())>= 0.0)
    # check to see the number of elements in each col 
    print "  Min # elements in each col: " + str((mati>0).sum(axis=0).min())
    print "  Max # elements in each col: " + str((mati>0).sum(axis=0).max())
    # check to see the number of elements in each row
    print "  Min # elements in each row: " + str((mati>0).sum(axis=1).min())
    print "  Max # elements in each row: " + str((mati>0).sum(axis=1).max())
    # check to see if the number of cols w/ 4 == shape of mat
    print "  Shape of the matrix: " + str(mati.shape)
    print "  Number sp in subj i: " + str(superMetaData["subjectSuperPixels"][i])
    # check to see how many nonzero values are in the matrix
    print "  Number of nonzero values: " + str((mati>0).sum())
    print "  According to the matrix: " + str(mati.nnz)
    print "  Should be: " + str(3*superMetaData["totalSuperPixels"])
    # check to make sure there's the same number of values in the subi lists
    print "  Number of data points in file: " + str(len(subi["data"]))
    print "  Number of index points in file: " + str(len(subi["indices"]))
    print "  Number of index pointer points in file: " + str(len(subi["indptr"]))
    print "  First index pointer points: " + str(subi['indptr'][0])
