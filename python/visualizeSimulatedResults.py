import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


def loadSubjImg(subjNum, imgNum, display=True):
	"""
	Given the subject id number and the number of the image to load, 
	load and display the image.

	Inputs:
	- subjNum: the id number (integer) for the subject
	- imgNum: the number of the image within the subject

	Returns:
	- img: the loaded image

	Effect: displays the loaded image
	"""
	# select the subject
	imgsFN = "./simulatedData/simulatedImages/S" + str(subjNum).zfill(4)
	# load the collection of that subject's images
	imgsList = loadSimImg(imgsFN)
	# select the desired image from the list
	v = imgsList[imgNum]
	# reshape specific vector as matrix
	img = np.reshape(v, (28, 28))
	# show the image
	if display:
		plt.imshow(img, cmap="gray")
		plt.show()
	return img

#--------------------------------------------------------------------------
# Saving and Loading Files
#--------------------------------------------------------------------------

def loadSimImg(fn):
    """
    Load a previously saved simulated subject from a .npz file.

    Inputs:
    - fn: filename/directory to load from (extensionless)

    Returns:
    - images: single patient's features/nodes
    """
    loader = np.load(fn+".npz")
    print "Image " + fn + " loaded!"
    return loader['image']


subjNum = 0
imgIdx = 0
img = loadSubjImg(subjNum, imgIdx)

# for i in xrange(500):
# 	im = loadSubjImg(subjNum, i)
# 	print str(i) + " " + str(im.min()) + " " + str(im.max())

"""
To check:
- 22
- 24
- 56
- 67
- 96
- 106
"""

# Generate sample abnormal patches
# # Load MNIST data
# # the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
# A = np.vstack((X_test, X_train[0:25000]))
# A_y = np.hstack((y_test, y_train[0:25000]))
# X_test = A.astype('float32')
# X_test /= 255
# y_test = A_y
# # get the data for generating the abnormal nodes
# mnistOneIndices = [i for i in xrange(len(y_test)) if y_test[i]==1 ]
# mnistZeroIndices = [i for i in xrange(len(y_test)) if y_test[i]==0 ]
# onesImgs = X_test[mnistOneIndices]
# zerosImgs = X_test[mnistZeroIndices]

# for i in xrange(30):
# 	# generate a random number to select a 1 image
# 	idx1 = np.random.randint(0, len(onesImgs)-1)
# 	# generate a random number to select a 0 image
# 	idx0 = np.random.randint(0, len(zerosImgs)-1)
# 	# select a 1 image
# 	i1 = onesImgs[idx1]
# 	# select a 0 image
# 	i0 = zerosImgs[idx0]
# 	# combine the 2 images into 1 (add them, values are btwn 0 and 1)
# 	abImg = i0+i1
# 	# reshape to image
# 	im = np.reshape(abImg, (28, 28))
# 	# show image
# 	plt.imshow(im, cmap='gray')
# 	plt.show()
