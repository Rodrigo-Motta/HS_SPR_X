import os
import cv2
import numpy as np
from keras.utils.image_utils import img_to_array
from sklearn.decomposition import PCA
import pandas as pd

def get_data(data_path=None, label_data_path=None, width=100, height=100,final_data_path=None):
    """
    This function resizes the images by a chosen ratio using linear interpolation (weighted average) and
    save the images in given directory

    Args:
        data_path: original data directory
        width: final width of the images
        height: final height of the images
        final_data_path: final (processed) data directory

    Returns:
        (np.array with the images, np.array with the labels)
    """
    data = []
    df = pd.read_csv(label_data_path)
    labels = df.gender.values
    progress = 0
    # run thought the directory
    for filename in os.listdir(data_path)[:]:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # read each image
            image = cv2.imread(os.path.join(data_path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # resize the image
            image = cv2.resize(image, (width, height))
            # image to np.array
            image = img_to_array(image)
            data.append(image)
            if final_data_path != None:
                np.savetxt(final_data_path + '/{}.txt'
                           .format(filename[:-4]),
                           image.ravel())
        progress += 1
        print('Progress: {:.2f}%'.format((1 - (len(os.listdir(data_path)[:]) - progress)/len(os.listdir(data_path)[:]))*100),
              end='\r')
    # normalize the data
    X = np.array(data, dtype="float32") / 255.0
    return (X, labels)


def get_data_pca(data_path=None,label_data_path=None, width=100, height=100, var_exp=0.9,final_data_path=None):
    """
    This function resizes the images by a chosen ratio using linear interpolation (weighted average)

    Args:
        data_path: original data directory
        width: final width of the images
        height: final height of the images
        var_exp : % of the variance remaining
        final_data_path: final (processed) data directory

    Returns:
        (np.array all the images with the pca decomposition, np.array all the labels,
         pca components to do the inverse transformation)
    """
    data = []
    pca_comp = []
    df = pd.read_csv(label_data_path)
    labels = df.gender.values
    progress = 0
    # run thought the directory
    for filename in os.listdir(data_path)[:]:
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
            if final_data_path != None:
                # save compressed image
                np.savetxt(final_data_path + '/{}.txt'
                           .format('image' + filename[:-4] + str(image_transformed.shape)),
                           image_transformed.ravel())
                # save componentes for inverse transform
                np.savetxt(final_data_path + '/{}.txt'
                           .format('pca_comp' + filename[:-4] + str(pca.components_.shape)),
                           pca.components_.ravel())
        progress += 1
        print('Progress: {:.2f}%'.format((1 - (len(os.listdir(data_path)[:]) - progress)/len(os.listdir(data_path)[:]))*100),
              end='\r')
    # Normalize data
    X = np.array(data, dtype="float32") / 255.0
    return (X,labels, pca_comp)

def get_data_svd(data_path=None, final_data_path=None, r=100):
    '''
    This function applies an SVD routine to compress the images passed here and saves
    the image in a given directory.

    Args:
        data_path: original data directory
        final_data_path: final (processed) data directory
        r: level of compression (the smaller the r, the bigger the compression)

    Returns:
        .png compressed image.
    '''
    progress = 0
    error_list = []
    for filename in os.listdir(data_path)[:]:
        if filename[:6] in os.listdir(final_data_path)[:6]:
            continue
        elif (filename.endswith(".jpg") or filename.endswith(".png")):
            try:
                # read each image
                image = cv2.imread(os.path.join(data_path, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # applies svd
                U, S, Vt = np.linalg.svd(image, full_matrices=False)
                S = np.diag(S)
                image_compressed = U[:,:r] @ S[:r,:r] @ Vt[:r,:]
                if final_data_path != None:
                    cv2.imwrite(filename=final_data_path + '/{}_comp.png'.format(filename[:6]),
                               img=image_compressed)
            except:
                print(f'An error occured in file {filename}')
                error_list.append(filename)
                continue
        progress += 1
        print('Progress: {:.2f}%'.format((1-(len(os.listdir(data_path)[:]) - progress)/len(os.listdir(data_path)[:]))*100),
              end='\r')
    return error_list