#!/usr/bin/env python
import numpy as np
import pickle as pk

def loadPickledData():     
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


def writeMetadata(filename, metadata):
    """
    Write the metadata to a file.

    Input:
    - filename: location and name to save the metadata file
    - metadata: the data to save
    """
    np.savez(fn, totalSP=metadata["totalSuperPixels"], subjSP=metadata["subjectSuperPixels"], indStart=metadata["superPixelIndexingStart"], indEnd=metadata["superPixelIndexingEnd"])

def loadMetadata(filename):
    """
    Load the metadata.

    Inputs:
    - filename: the name of the metadata file to load

    Returns: the loaded lung dataset metadata:
    - md: dictionary containing
        - totalSuperPixels: the total number of superpixels of all subjects combined
        - subjectSuperPixels: the number of superpixels in each subject
        - superPixelIndexingStart: the index indicating the start of each subject in the block
        - superPixelIndexingEnd: the index indicating the end of each subject in the block
    """
    loader = np.load(filename+".npz")
    md = {
        "totalSuperPixels": loader['totalSP'],
        "subjectSuperPixels": loader['subjSP'],
        "superPixelIndexingStart": loader['indStart'],
        "superPixelIndexingEnd": loader['indEnd']
    }
    return md



md = getSubjectSizes()
fn = "./copd-metadata"
writeMetadata(fn, md)
md2 = loadMetadata(fn)
print md['totalSuperPixels']==md2['totalSuperPixels']
print (md['subjectSuperPixels']==md2['subjectSuperPixels']).all()
print (md['superPixelIndexingStart']==md2['superPixelIndexingStart']).all()
print (md['superPixelIndexingEnd']==md2['superPixelIndexingEnd']).all()