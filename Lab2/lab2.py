""" CS4243 Lab 2: Image Segmentations
See accompanying Jupyter notebook (lab2.ipynb) and PDF (lab2.pdf) for instructions.
"""
import cv2
import numpy as np
import random

from time import time
from skimage import color
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed


# Part 1 

def smoothing(img):
    """Smooth image using Guassain filter.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).

    Returns:
        img_smoothed (np.ndarray)   : Output smoothed image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    img_smoothed = cv2.GaussianBlur(img, (5, 5), 5.0)
    """ YOUR CODE ENDS HERE """

    return img_smoothed


def RGBtoLab(img):
    """Convert RGB image into L*a*b color space.

    Args:
        img (np.ndarray)            : Input RGB image  of size (H, W, 3).


    Returns:
        lab (np.ndarray)            : Converted L*a*b image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    lab = color.rgb2lab(img)
    """ YOUR CODE ENDS HERE """
   
    return lab


# Part 2
def k_means_clustering(data, k):
    """ Estimate clustering centers using k-means algorithm.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        k (int)                     : Number of centroids

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """
    start = time()

    """ YOUR CODE STARTS HERE """

    n_samples, n_features = data.shape
    centers = np.stack(list([data[i] for i in random.sample(range(n_samples), k)]))
    labels = np.stack(list([np.argmin(np.stack([np.linalg.norm(data[i] - centers[j]) for j in range(k)]))] for i in range(n_samples)))
    for epoch in range(100):
        new_centers_list = list()
        need_to_rand = False
        for i in range(k):
            index = np.equal(labels, np.stack([i] * n_samples).reshape(labels.shape))
            count = index.sum()
            total_sum = np.sum(data * index, axis=0)
            if count == 0:
                need_to_rand = True
                break
            new_centers_list.append(total_sum / count)

        if need_to_rand:
            new_centers = np.stack(list([data[i] for i in random.sample(range(n_samples), k)]))
        else:
            new_centers = np.stack(list(new_centers_list))

        if np.max(np.abs(new_centers - centers)) < 1e-6:
            break

        centers = new_centers
        labels = np.stack(list([np.argmin(np.stack([np.sqrt(np.sum((data[i] - centers[j]) ** 2)) for j in range(k)]))] for i in range(n_samples)))
    """ YOUR CODE ENDS HERE """

    end = time()
    kmeans_runtime = end - start
    print("K-means running time: %.3fs." % kmeans_runtime)
    return labels, centers


def get_bin_seeds(data, bin_size, min_bin_freq=1):
    """ Generate initial bin seeds for windows sampling.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bin_size (float)            : Bandwidth.
        min_bin_freq (int)          : For each bin_seed, number of the minimal points should cover.

    Returns:
        bin_seeds (List)            : Reprojected bin seeds. All bin seeds with total point number 
                                      bigger than the threshold.
    """

    """ YOUR CODE STARTS HERE """
    if bin_size == 0:
        return data

    bin_sizes = dict()
    for point in data:
        binned_point = np.round(point / bin_size)
        if bin_sizes.get(tuple(binned_point)) is None:
            bin_sizes[tuple(binned_point)] = 1
        else:
            bin_sizes[tuple(binned_point)] += 1

    bin_seeds = np.array([point for point, count in bin_sizes.items() if count >= min_bin_freq])
    bin_seeds = bin_seeds * bin_size
    """ YOUR CODE ENDS HERE """

    return bin_seeds


def mean_shift_single_seed(start_seed, data, nbrs, max_iter):
    """ Find mean-shift peak for given starting point.

    Args:
        start_seed (np.ndarray)     : Coordinate (x, y) of start seed. 
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        nbrs (class)                : Class sklearn.neighbors._unsupervised.NearestNeighbors.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        peak (tuple)                : Coordinate (x,y) of peak(center) of the attraction basin.
        n_points (int)              : Number of points in the attraction basin.
                              
    """

    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    current_mean = start_seed
    all_points = data
    """ YOUR CODE STARTS HERE """
    for _ in range(max_iter):
        idx = nbrs.radius_neighbors([current_mean], bandwidth, return_distance=False)[0]
        all_points = data[idx]
        if len(all_points) == 0:
            break  # Depending on seeding strategy this condition may occur
        old_mean = current_mean  # save the old mean
        current_mean = np.mean(all_points, axis=0)
        # If converged or at max_iter, adds the cluster
        if np.linalg.norm(current_mean - old_mean) < stop_thresh:
            break
    peak = tuple(current_mean)
    n_points = len(all_points)
    """ YOUR CODE ENDS HERE """

    return peak, n_points


def mean_shift_clustering(data, bandwidth=0.7, min_bin_freq=5, max_iter=300):
    """pipline of mean shift clustering.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bandwidth (float)           : Bandwidth parameter for mean shift algorithm.
        min_bin_freq(int)           : Parameter for get_bin_seeds function.
                                      For each bin_seed, number of the minimal points should cover.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)
    """
    start = time()
    n_jobs = None
    seeds = get_bin_seeds(data, bandwidth, min_bin_freq)
    n_samples, n_features = data.shape
    center_intensity_dict = {}

    # We use n_jobs=1 because this will be used in nested calls under
    # parallel calls to _mean_shift_single_seed so there is no need for
    # for further parallelism.
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(data)
    # execute iterations on all seeds in parallel
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(mean_shift_single_seed)
        (seed, data, nbrs, max_iter) for seed in seeds)

    # copy results in a dictionary
    for i in range(len(seeds)):
        if all_res[i] is not None:
            center_intensity_dict[all_res[i][0]] = all_res[i][1]

    if not center_intensity_dict:
        # nothing near seeds
        raise ValueError("No point was within bandwidth=%f of any seed."
                         " Try a different seeding strategy \
                         or increase the bandwidth."
                         % bandwidth)

    """ YOUR CODE STARTS HERE """
    sorted_by_intensity = sorted(center_intensity_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
    sorted_centers = np.array([i[0] for i in sorted_by_intensity])
    unique = np.ones(len(sorted_centers), dtype=np.bool)
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=n_jobs).fit(sorted_centers)
    for i, center in enumerate(sorted_centers):
        if unique[i]:
            idx = nbrs.radius_neighbors([center], return_distance=False)[0]
            unique[idx] = 0
            unique[i] = 1  # leave the current point as unique
    unique_centers = sorted_centers[unique]

    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(unique_centers)
    distances, idxs = nbrs.kneighbors(data)
    labels = idxs.flatten()
    centers = unique_centers

    """ YOUR CODE ENDS HERE """
    end = time()
    kmeans_runtime = end - start
    print("mean shift running time: %.3fs." % kmeans_runtime)
    return labels, centers


# Part 3:
def k_means_segmentation(img, k):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        k (int)                     : Number of centroids
    
    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    data = img.reshape(img.shape[0] * img.shape[1], -1)
    labels, centers = k_means_clustering(data, k)
    """ YOUR CODE ENDS HERE """

    return labels, centers


def mean_shift_segmentation(img, b):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        b (float)                     : Bandwidth.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    data = img.reshape(img.shape[0] * img.shape[1], -1)
    labels, centers = mean_shift_clustering(data, bandwidth=b)
    """ YOUR CODE ENDS HERE """

    return labels, centers


# Helper functions: You should not have to touch the following functions.
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """

    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def colors(k):
    """generate the color for the plt.scatter.

    Args:
        k (int): the number of the centroids

    Returns:
        ret (list): list of colors .

    """

    colour = ["coral", "dodgerblue", "limegreen", "deeppink", "orange", "darkcyan", "rosybrown", "lightskyblue", "navy"]
    if k <= len(colour):
        ret = colour[0:k]
    else:
        ret = []
        for i in range(k):
            ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret


def stack_seg(img, labels, centers):
    """stack segmentations for visualization.

    Args:
        img (np.ndarray): image
        labels(np.ndarray): lables for every pixel. 
        centers(np.ndarray): cluster centers.

    Returns:
        np.vstack(result) (np.ndarray): stacked result.

    """

    labels = labels.reshape((img.shape[:-1]))
    reduced = np.uint8(centers)[labels]
    result = [np.hstack([img])]
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        mask = np.dstack([mask]*3) # Make it 3 channel
        ex_img = cv2.bitwise_and(img, mask)
        result.append(np.hstack([ex_img]))

    return np.vstack(result)
