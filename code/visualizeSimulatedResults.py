import numpy as np
import matplotlib.pyplot as plt


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


# select the subject
subjNum = 0
imgsFN = "./simulatedData/simulatedImages/S" + str(subjNum).zfill(4)
# load the collection of that subject's images
imgsList = loadSimImg(imgsFN)
# select the desired image from the list
imgIdx = 0
v = imgsList[imgIdx]
# reshape specific vector as matrix
img = np.reshape(v, (28, 28))
# show the image
plt.imshow(img)
plt.show()