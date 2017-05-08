Pipeline Description (Simulated Data)
======================================
Start with generateSimulationData.py to generate the data, build the KNN graph, and extract the divergences


Preprocessing Annotation Pipeline Description
=============================================
This pipeline uses some code contributed to https://github.com/kayhan-batmanghelich/LungProject

Start by using the file LungProject/src/python/extractAnnotationFeatures.py to extract a set of features from each patch's mask. The file takes the subject's ID as a command line input.
- To run this file for multiple subjects on Bridges, use the file LungProject/src/scripts/runExtractAnnotationFeatures.sh

Use LungProject/src/python/convertPatchCsvToH5.py to convert the .csv files containing the really, really long feature vectors from .csv files (extracted in previous step) to .hdf5 files.

Then convert the directory of .hdf5 files with the really really long feature vectors to a single file using LungProject/src/python/scaleDownFhogFeatues2192To63.py file. This code is dependent on some files and did not come well documented. It runs once for all subjects, so if you're running it on Bridges, you should request an interactive node.

Now you have a set of pickle files with all of the extracted image features for all the patches present! The data has been preprocessed! From here, we return to using this repository.



Pipeline Description (Annotation Data)
======================================
Assumption: the labels from the annotation data are already converted into histogram and fhog features with a feature vector length of 63 for each patch, and the features are stored in a pickle file

Starting with the features, use python/learnAnnotationFeatures.py. This file can be used to 
- Build and evaluate a neural network, a SVM, and a random forest for classifying the annotated patches. The evaluation is done using N-fold cross validation on the patch features of the annotated data. To test the functions that train and test these three models, use the command `python python/learnAnnotationFeatures.py --test-functions`. To cross-validate these three models in order to evaluate their performance, use the command `python python/learnAnnotationFeatures.py --cross-validate`.
- Build a neural network based on all annotated patch features. Use the command `python python/learnAnnotationFeatures.py --save-model` to train the neural network on all of the available annotation features and save it.
- Load a previously saved, fully trained neural network. Use the loaded network to extract "learned" features from all patches in all unannotated subjects and save them (also in .hdf5 format). Use the command `python python/learnAnnotationFeatures.py --extract-features` to accomplish this task.

* Note: other arguments can easily be added to this file. I have made some notes of potential arguments that could make life easier if this code is used for another set of files. I did not add them myself due to time constraints.

After the learned features are extracted, use python/buildSubjSubjSimGraph.py to
- Build the KNN graph and extract the divergence between every K-nearest subjects
  - Possible divergences include Kullback-Leibler (KL), Hellinger (HE), and Mean Maximum Discrepancy (MMD)
- Symmetrize the divergences, project them onto a positive semidefinite space, and RBFize them to get a kernel

Then use the Jupyter notebook notebooks/kernelEvaluation3D.py to 
- Load the saved divergence graphs
- Symmetrize the divergences, project them onto a positive semidefinite space, and something else. This process ensures the divergences obey the triangle inequality and are symmetric, and makes them behave like similarity measures (higher number means the 2 subjects are more similar rather than more different).
- Perform Cholesky factorization
- Check: Use TSNE manifold embedding to transform the features to a 2D space for graphing in a scatterplot
- Use some other manifold embedding to transform the features to a 100D space
- Use 50-fold cross-validation to evaluate the performance of the features in predicting the subjects' FEV1 values using linear regression with LASSO (or ridge?) regularization
- Save the results of the cross-validation
- Graph the results of the cross-validation
