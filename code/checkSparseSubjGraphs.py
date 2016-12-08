#!/usr/bin/env python
import numpy as np
import scipy.sparse as sp


# loader = np.load("test-metadata.npz")
# superMetaData = {
#     "totalSuperPixels": loader['totalSP'],
#     "subjectSuperPixels": loader['subjSP'],
#     "superPixelIndexingStart": loader['indStart'],
#     "superPixelIndexingEnd": loader['indEnd']
# }

sub = [None]*10
mat = [None]*10

for i in xrange(10):
    # load subj i graph
    sub[i] = np.load("./simulatedData/S"+str(i).zfill(4)+".npz")
    # sub[i] = np.load("/pylon1/ms4s88p/jms565/sparseGraphs/"+str(i+1).zfill(4)+".npz")
    mat[i] = sp.csr_matrix((sub[i]['data'], sub[i]['indices'], sub[i]['indptr']), shape=sub[i]['shape'])
    print "------------------ Subject " + str(i) + " ----------------"
    # check to see if min/max bot > 0
    print "  Min >= 0: " + str((mat[i].min())>= 0.0)
    print "  Max >= 0: " + str((mat[i].max())>= 0.0)
    # check to see the number of elements in each col 
    print "  Min # elements in each col: " + str((mat[i]>0).sum(axis=0).min())
    print "  Max # elements in each col: " + str((mat[i]>0).sum(axis=0).max())
    # check to see the number of elements in each row
    print "  Min # elements in each row: " + str((mat[i]>0).sum(axis=1).min())
    print "  Max # elements in each row: " + str((mat[i]>0).sum(axis=1).max())
    # check to see if the number of cols w/ 4 == shape of mat
    print "  Shape of the matrix: " + str(mat[i].shape)
    # print "  Number sp in subj i: " + str(superMetaData["subjectSuperPixels"][i])
    # check to see how many nonzero values are in the matrix
    print "  Number of nonzero values: " + str((mat[i]>0).sum())
    print "  According to the matrix: " + str(mat[i].nnz)
    # print "  Should be: " + str(3*superMetaData["totalSuperPixels"])
    # check to make sure there's the same number of values in the sub[i] lists
    print "  Number of data points in file: " + str(len(sub[i]["data"]))
    print "  Number of index points in file: " + str(len(sub[i]["indices"]))
    print "  Number of index pointer points in file: " + str(len(sub[i]["indptr"]))
    print "  First index pointer points: " + str(sub[i]['indptr'][0])

for i in xrange(9):
    print "Comparing subj " + str(i) + " to subj " + str(i+1)
    print "  Data: " + str((mat[i].data==mat[i+1].data).all())
    print "  Indptr: " + str((mat[i].indptr==mat[i+1].indptr).all())
    print "  Indices: " + str((mat[i].indices==mat[i+1].indices).all())
    print "  Shape (should be same): " + str(mat[i].shape==mat[i+1].shape)

