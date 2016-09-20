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
def getConfig(useHarilick=False):
    #config = {}
    #config['pickleSettingName'] = pickleSettingName
    
    #if useHarilick:
    #    pickleConfig_dbFn = databaseRoot + '/pickleConfig_Harilick_Param30_setting1.csv'
    #else:  # use FHOG
    #    pickleConfig_dbFn = databaseRoot + '/pickleConfig_fHOG_Hist_Param30_thr3000_Bin2049.csv'        
    
    #pickleConfig_db = pd.read_csv(pickleConfig_dbFn,comment='#')
    
    #row = pickleConfig_db[ pickleConfig_db['pickleSettingName'] == pickleSettingName ]
    #pickleRootFolder = row['pickleRootFolder'].values[0]
    #respState=row['respState'].values[0]
    #snpCSVDataFn = row['snpCSVDataFn'].values[0] 
    #spacing = row['spacing'].values[0]
    #interpolation = row['interpolation'].values[0]
    #numSP = row['numSP'].values[0]
    #sigma_smooth = row['sigma_smooth'].values[0]
    #max_band = row['sigma_smooth'].values[0]
    #nTheta = row['nTheta'].values[0]
    #nPhi = row['nPhi'].values[0]

    ## put the variables in the config dictionary
    #config['pickleRootFolder'] = pickleRootFolder
    #config['respState'] = respState
    #config['snpCSVDataFn'] = snpCSVDataFn
    #config['spacing'] = spacing
    #config['interpolation'] = interpolation
    #config['numSP'] = numSP
    #config['sigma_smooth'] = sigma_smooth
    #config['max_band'] = max_band
    #config['nTheta'] = nTheta
    #config['nPhi'] = nPhi
    
    
    ## name of databases
    #nii_dbFn = databaseRoot+\
    #'/niiConvert_Iso%(spacing).1fmm_%(interp)s_%(respState)s.csv'%\
    # {'spacing':spacing,
    #  'interp':interpolation,
    #  'respState':respState}

    #sprVxl_dbFn = databaseRoot+\
    #'/superVxl_Iso%(spacing).1fmm_%(interp)s_%(respState)s_Param%(param)d.csv'%\
    #{'spacing':spacing,
    #'interp':interpolation,
    #'respState':respState,
    #'param':numSP}

    #if useHarilick:
    #    harilick_dbFn = databaseRoot +\
    #    '/harilick_Iso%(spacing).1fmm_%(interp)s_%(respState)s_numSP%(numSP)dmm_%(harilickSettingName)s.csv'%\
    #    {'spacing':spacing,
    #     'interp':interpolation,
    #     'respState':respState,
    #     'harilickSettingName':harilickSettingName,
    #     'numSP':numSP}
    #else:   
    #    fhog_dbFn = databaseRoot+\
    #       '/fhog_Iso%(spacing).1fmm_%(interp)s_%(respState)s_sigmaSmooth%(sigma_smooth).1f_maxBand%(max_band)d_nTheta%(nTheta)d_nPhi%(nPhi)d.csv'%\
    #         {'spacing':spacing,
    #          'interp':interpolation,
    #          'respState':respState,'sigma_smooth':sigma_smooth,
    #          'max_band':max_band,'nTheta':nTheta,'nPhi':nPhi}


    #    fhogHistFeature_dbFn = databaseRoot+\
    #     '/fhogHistFeatures_Iso%(spacing).1fmm_%(interp)s_%(respState)s_numSP%(numSP)dmm.csv'%\
    #     {'spacing':spacing,
    #      'interp':interpolation,
    #      'respState':respState,
    #      'numSP':numSP}              
   

    pickleFn =  'COPDGene_pickleFiles/histFHOG_largeRange_setting1.data.p'
    # pickleFn =  '%(pickleRootFolder)s/%(pickleSettingName)s/%(pickleSettingName)s.data.p'%\
            # {'pickleSettingName':pickleSettingName, 'pickleRootFolder':pickleRootFolder}
    shelveFn = 'COPDGene_pickleFiles/histFHOG_largeRange_setting1.shelve'
    #config['pickleFn'] = pickleFn
    #config['shelveFn'] = shelveFn
    print "pickleFn : ", pickleFn
    print "shelveFn :", shelveFn

    # read the databases in
    #nii_db = pd.read_csv(nii_dbFn)
    #sprVxl_db = pd.read_csv(sprVxl_dbFn)
    
    #if useHarilick:
    #    harilick_db = pd.read_csv(harilick_dbFn)
    #else:
    #    fhog_db = pd.read_csv(fhog_dbFn)
    #    fhogHistFeature_db = pd.read_csv(fhogHistFeature_dbFn,comment='#')

        
    # reading pickle and shelve files
    print "Reading the shelve file ..."
    # tic()
    fid = shelve.open(shelveFn,'r')
    metaVoxelDict = fid['metaVoxelDict']
    #pca = fid['pcaHist']
    subjList = fid['subjList']
    phenotypeDB_clean = fid['phenotypeDB_clean']
    fid.close()
    print "Done !"
    # toc()

    print "Here is how meta data look like: "
    print "IDs of a few subjects : " , metaVoxelDict[0]['id']
    print "Here is the labelIndex of the meta data (a few elements): " , metaVoxelDict[0]['labelIndex'][1:10]   

    print "Reading pickle file ...."
    # tic()
    fid = open(pickleFn,'rb')
    data = pk.load(open(pickleFn,'rb'))
    fid.close()
    print "Done !"
    # toc()
    
    #if useHarilick:
    #    return config, (nii_db, sprVxl_db), (harilick_db), (metaVoxelDict,subjList, phenotypeDB_clean, data)  
    #else:
    return metaVoxelDict,subjList, phenotypeDB_clean, data

metaVoxelDict, subjList, phenotypeDB_clean, data = getConfig()

# see data_looking.ipynb for structure information

# READ/PLAY WITH DATA FIRST

# for each subject, create an approximate NN graph using pyflann (cyflann)
#  *http://www.cs.ubc.ca/research/flann/
#  * install it using pip or conda
#  https://github.com/primetang/pyflann
#  https://github.com/dougalsutherland/cyflann

def buildSubjectTrees(data, numNodes=3):
    """
    Find the numNodes nodes of each subject that are closest to N nodes
    in every other subject.

    Inputs:
    - data: collection of data to be tree'ed
    - numNodes: the number of nearest nodes to save

    Returns:
    - subjDBs: list of lists of dictionaries of lists
        - first layer = first subject
        - second layer = second subject
        - third layer = dictionary accessed by keys
        - "nodes": list of 

    """
    numNeighbors = 1 # how many nodes to compare the initial node to
    numNodes = 3 # how many pairs to nodes to store for each bag
    flann = FLANN()
    subjDBs = []
    # build the tree for each subject
    print "Now building subject-subject mini databases..."
    for i in xrange(len(subjList)-1):
        results = []
        for j in xrange(len(subjList[i+1:])):
            print "i: " + str(i) + " j: " + str(j)
            nodes, dist = flann.nn(data[i]['I'], data[j]['I'], numNeighbors, algorithm='kmeans')
            # sort the results based on distance
            dist, nodes = (list(t) for t in zip(*sorted(zip(dist.tolist(), nodes.tolist()))))
            # save the numNodes number of distances and nodes
            temp = {
                "nodes": nodes[0:numNodes],
                "distances": dist[0:numNodes]
            }
            results.append(temp)
        subjDBs.append(results)

    print "Subject level databases complete!"
    return subjDBs

neighbors=5
subjTrees = buildSubjectTrees(data, neighbors)

# digression : http://www.theverge.com/google-deepmind

#------------ NEXT STEP (WARNING: TAKE IT WITH HUGE GRAIN OF SULT)
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

