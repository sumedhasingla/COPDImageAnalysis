from __future__ import print_function
import numpy as np

# imports for MMD
from sklearn.metrics.pairwise import euclidean_distances
import mmd
from sklearn import cross_validation as cv
from sklearn.kernel_ridge import KernelRidge

# imports for file handling
import pickle as pk


#---------------------------------------------------------------------------------------------
# Functions for Saving/Loading Data
#---------------------------------------------------------------------------------------------

def loadFeatures(filename):
    """
    Load a set of features from the specified file.

    Inputs:
    - filename: path to the file with the features

    Returns:
    - features: the loaded features
    """
    # load the data from the file
    fid = open(filename, 'rb')
    data = pk.load(fid)
    fid.close()
    print("Data loaded from file.")

    # format the data for easier use
    feats = []
    for subj in data:
        feats.append(subj['I'])
    print("Finished formatting the loaded data!")

    return feats

def saveSimilarityKernel(fn, sims):
    """
    Save the similarity matrix to a .npz file.

    Inputs:
    - fn: directory/filename to save the file to (extension will be provided by func.)
    - sims: similarity matrix

    Returns: nothing
    """
    np.savez(fn, similarities=sims)
    print "Saved the similarities to a file."


def loadSimilarityKernel(fn):
    """
    Load the previously saved similarity matrix from a .npz file

    Inputs:
    - fn: directory/filename (minus extension) to load the file from

    Returns:
    - loadedSims: similarity matrix
    """
    loader = np.load(fn+".npz")
    print "Similarities loaded!"
    return loader['similarities']


#---------------------------------------------------------------------------------------------
# Functions for doing stuff
#---------------------------------------------------------------------------------------------

def computeSubjSubjKernel(subjects, div='KL', numNeighbors=3):
    """
    Start by computing the pairwise similarities between subject
    using Dougal's code. Then, for HE and KL, symmetrize, RBFize,
    and project the similarities onto a positive semi-definite space.

    Inputs:
    - subjects: the collection of patient features
    - div: which divergence to use. Options are
            - 'KL': Kullback-Leibler divergence, 'kl' in the function (default)
            - 'HE': Hellinger divergence, 'hellinger' in the function
            - 'MMD': Maximum Mean Discrepancy, calls another function
    - numNeighbors: how many neighbors to look at. Default is 3

    Returns: 
    - kernel: the kernel calculated using the pairwise similarities between each subject
    * Note: kernel is a NxN symmetric matrix, where N is the number of subjects
    """

    # pass the features and labels to scikit-learn Features
    feats = Features(patients) # directly from Dougal

    # specify the divergence to use
    if div == 'KL':
        # estimate the distances between the bags (patients) using KNNDivergenceEstimator
        # details: use the kl divergence, find 3 nearest neighbors
        #          not sure what the pairwise picker line does?
        #          rbf and projectPSD help ensure the data is separable?
        distEstModel = Pipeline([ # div_funcs=['kl'], rewrite this to actually use PairwisePicker correctly next time
            ('divs', KNNDivergenceEstimator(div_funcs=['kl'], Ks=[numNeighbors], n_jobs=-1, version='fast')),
            ('pick', PairwisePicker((0, 0)))
            ('symmetrize', Symmetrize())
            # ('rbf', RBFize(gamma=1, scale_by_median=True)),
            # ('project', ProjectPSD())
        ])
        # return the pairwise similarities between the bags (patients)
        sims = distEstModel.fit_transform(feats)

        # Great, we have the similarities and they're symmetric
        # Now RBFize them, but do the scale by median by hand
        rbf = RBFize(gamma=1, scale_by_median=False)
        simsMedian = np.media(sims[np.triu_indices_from(sims)])
        medianScaledSims = sims/simsMedian
        rbfedSims = rbf.fit_transform(medianScaledSims)

        # Final step in building the kernel: project the rbf'ed similarities
        #   onto a positive semi-definite space
        psd = ProjectPSD()
        kernel = psd.fit_transform(rbfedSims)

    elif div == 'HE':
        # estimate the distances between the bags (patients) using KNNDivergenceEstimator
        # details: use the hellinger divergence, find 3 nearest neighbors
        #          not sure what the pairwise picker line does?
        #          rbf and projectPSD help ensure the data is separable?
        distEstModel = Pipeline([ # div_funcs=['kl'], rewrite this to actually use PairwisePicker correctly next time
            ('divs', KNNDivergenceEstimator(div_funcs=['hellinger'], Ks=[numNeighbors], n_jobs=-1, version='fast')),
            ('pick', PairwisePicker((0, 0))),
            ('symmetrize', Symmetrize())
            # ('rbf', RBFize(gamma=1, scale_by_median=True)),
            # ('project', ProjectPSD())
        ])

        # return the pairwise similarities between the bags (patients)
        sims = distEstModel.fit_transform(feats)

        # Great, we have the similarities and they're symmetric
        # Now RBFize them, but do the scale by median by hand
        rbf = RBFize(gamma=1, scale_by_median=False)
        simsMedian = np.media(sims[np.triu_indices_from(sims)])
        medianScaledSims = sims/simsMedian
        rbfedSims = rbf.fit_transform(medianScaledSims)

        # Final step in building the kernel: project the rbf'ed similarities
        #   onto a positive semi-definite space
        psd = ProjectPSD()
        kernel = psd.fit_transform(rbfedSims)

    elif div == 'MMD':
        # start by getting the median pairwise squared distance between subject,
        #   used as a heuristic for choosing the bandwidth of the inner RBF kernel
        subset = np.vstack(feats)
        subset = subset[np.random.choice(subset.shape[0], min(2000, subset.shape[0]), replace=False)]
        subsetSquaredDists = euclidean_distances(subset, squared=True)
        featsMedianSquaredDist = np.median(subsetSquaredDists[np.triu_indices_from(subsetSquaredDists, k=numNeighbors)], overwrite_input=True)

        # now we need to determine gamma (scaling factor, inverse of sigma)
        #   This was initially done in the library, but Kayhan believes there's
        #   a multiplication instead of a division, so it's being done by hand
        firstGamma = 1/featsMedianSquaredDist

        # calculate the mmds
        mmds, mmkDiagonals = mmd.rbf_mmd(feats, gammas=firstGamma, squared=True, ret_X_diag=True)

        # now let's turn the squared MMD distances into a kernel
        # get the median squared MMD distance
        mmdMedianSquaredDist = np.median(mmds[np.triu_indices_from(mmds, k=numNeighbors)])
        kernel = np.exp(np.multiply(mmds, -1/mmdMedianSquaredDist))

    return kernel





#---------------------------------------------------------------------------------------------
# Main code section
#---------------------------------------------------------------------------------------------

# set up argparser
# - divergence argument
# - number of neighbors argument

# specify file paths
rootPath = '/home/jenna/Research/COPDImageAnalysis/annotations/'
featureFn = rootPath + 'unannotated/learnedFeatures.data.p'
klKernelFn = rootPath + 'unannotated/kernel-kl'
heKernelFn = rootPath + 'unannotated/kernel-he'
mmdKernelFn = rootPath + 'unannotated/kernel-mmd'

# load the data
features = loadFeatures(featureFn)

# run the kernel computation
k = computeSubjSubjKernel(features)

# save the kernel
saveSimilarityKernel(klKernelFn, k)

# Test: load the kernel to check it
loadedK = loadSimilarityKernel(klKernelFn)

print((k == loadedK).all())
# probably should assert symmetricity too?