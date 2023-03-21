import os
import cv2
import numpy as np
from keras.utils.image_utils import img_to_array
from sklearn.decomposition import PCA

def resize(data_path=None, width=100, height=100,final_data_path=None):
    """
    This function resizes the images by a chosen ratio using linear interpolation (weighted average)

    Args:
        data_path: original data directory
        width: final width of the images
        height: final height of the images
        final_data_path: final (processed) data directory

    Returns:
        np.array with all the images
    """
    data = []
    # run thought the directory
    for filename in os.listdir(data_path)[:]:
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # read each image
            image = cv2.imread(os.path.join(data_path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # resize the image
            image = cv2.resize(image, (width, height))
            # image to np.array
            image = img_to_array(image)
            data.append(image)
            np.savetxt(final_data_path + '/{}.txt'
                       .format(filename[:-4]),
                       image.ravel())
    # normalize the data
    X = np.array(data, dtype="float32") / 255.0
    return X


def resize_pca(data_path=None, width=100, height=100, var_exp=0.9,final_data_path=None):
    """
    This function resizes the images by a chosen ratio using linear interpolation (weighted average)

    Args:
        data_path: original data directory
        width: final width of the images
        height: final height of the images
        var_exp : % of the variance remaining
        final_data_path: final (processed) data directory

    Returns:
        (np.array with all the images with the pca decomposition, pca components to do the inverse transformation)
    """
    data = []
    pca_comp = []
    # run thought the directory
    for filename in os.listdir(data_path)[:]:
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # read each image
            image = cv2.imread(os.path.join(data_path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # resize the image
            image = cv2.resize(image, (width, height))
            # image to np.array
            image = img_to_array(image)
            # pca
            pca = PCA(n_components=var_exp)
            image_transformed = pca.fit_transform(image.reshape(width,height))
            # store compressed image
            data.append(image_transformed)
            # save compressed image
            np.savetxt(final_data_path + '/{}.txt'
                       .format('image' + filename[:-4] + str(image_transformed.shape)),
                       image_transformed.ravel())
            # save componentes for inverse transform
            np.savetxt(final_data_path + '/{}.txt'
                       .format('pca_comp' + filename[:-4] + str(pca.components_.shape)),
                       pca.components_.ravel())
    # Normalize data
    X = np.array(data, dtype="float32") / 255.0
    return (X,pca_comp)

