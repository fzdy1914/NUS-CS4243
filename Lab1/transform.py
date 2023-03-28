import cv2
import numpy as np
from skimage import io
import os.path as osp


def load_image(file_name):
    """
    Load image from disk
    :param file_name:
    :return: image: numpy.ndarray
    """
    if not osp.exists(file_name):
        print('{} not exist'.format(file_name))
        return
    image = np.asarray(io.imread(file_name))
    if len(image.shape)==3 and image.shape[2]>3:
        image = image[:, :, :3]
    # print(image.shape) #should be (x, x, 3)
    return image


def save_image(image, file_name):
    """
    Save image to disk
    :param image: numpy.ndarray
    :param file_name:
    :return:
    """
    io.imsave(file_name,image)


def cs4243_resize(image, new_width, new_height):
    """
    5 points
    Implement the algorithm of nearest neighbor interpolation for image resize,
    Please round down the value to its nearest interger, 
    and take care of the order of image dimension.
    :param image: ndarray
    :param new_width: int
    :param new_height: int
    :return: new_image: numpy.ndarray
    """
    new_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    if len(image.shape) == 2:
        new_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            height = np.math.floor(i / new_height * image.shape[0])
            if height == image.shape[0]:
                height = image.shape[0] - 1
            width = np.math.floor(j / new_width * image.shape[1])
            if width == image.shape[1]:
                width = image.shape[1] - 1

            if len(image.shape) == 2:
                new_image[i][j] = image[height][width]
            else:
                for k in range(3):
                    new_image[i][j][k] = image[height][width][k]

    return new_image


def cs4243_rgb2grey(image):
    """
    5 points
    Implement the rgb2grey function, use the
    weights for different channel: (R,G,B)=(0.299, 0.587, 0.114)
    Please scale the value to [0,1] by dividing 255
    :param image: numpy.ndarray
    :return: grey_image: numpy.ndarray
    """
    if len(image.shape) != 3:
        print('RGB Image should have 3 channels')
        return

    new_image = np.zeros((image.shape[0], image.shape[1]), dtype='float64')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = image[i][j][0] * 0.299 + image[i][j][1] * 0.587 + image[i][j][2] * 0.114

    return new_image/255.


def cs4243_histnorm(image, grey_level=256):
    """
    5 points
    Stretch the intensity value to [0, 255]
    :param image : ndarray
    :param grey_level
    :return res_image: hist-normed image
    Tips: use linear normalization here https://en.wikipedia.org/wiki/Normalization_(image_processing)
    """
    res_image = image.copy().astype("float64")

    min = res_image.min()
    max = res_image.max()

    res_image = (res_image - min) * (grey_level - 1) / (max - min)

    return res_image


def cs4243_histequ(image, grey_level=256):
    """
    10 points
    Apply histogram equalization to enhance the image.
    the cumulative histogram will aso be returned and used in the subsequent histogram normalization function.
    :param image: numpy.ndarray(float64)
    :return: ori_hist: histogram of original image
    :return: cum_hist: cumulated hist of original image, pls normalize it with image size.
    :return: res_image: image after being applied histogram equalization.
    :return: uni_hist: histogram of the enhanced image.
    Tips: use numpy buildin funcs to ease your work on image statistics
    """
    ori_hist = np.bincount(image.flatten(), minlength=grey_level)

    cum_hist = ori_hist.cumsum().astype("float64")
    for i in range(grey_level):
        cum_hist[i] = cum_hist[i] / image.size

    uniform_hist = cum_hist.copy()
    for i in range(grey_level):
        uniform_hist[i] = cum_hist[i] * 255

    # Set the intensity of the pixel in the raw image to its corresponding new intensity
    height, width = image.shape
    res_image = np.zeros(image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i, j] = uniform_hist[image[i, j]]

    uni_hist = np.bincount(res_image.flatten(), minlength=grey_level)
    return ori_hist, cum_hist, res_image, uni_hist


def cs4243_histmatch(ori_image, refer_image):
    """
    10 points
    Make ori_image have the similar intensity distribution as refer_image
    :param ori_image #image to be processed
    :param refer_image #image of target gray histogram
    :return: ori_hist: histogram of original image
    :return: ref_hist: histogram of reference image
    :return: res_image: image after being applied histogram normalization.
    :return: res_hist: histogram of the enhanced image.
    Tips: use cs4243_histequ to help you
    """

    ori_hist = np.bincount(ori_image.flatten(), minlength=256)
    ori_cum_hist = ori_hist.cumsum() / ori_image.size

    ref_hist = np.bincount(refer_image.flatten(), minlength=256)
    ref_cum_hist = ref_hist.cumsum() / refer_image.size

    map_value = np.zeros(256, dtype='uint8')
    for i in range(256):
        map_value[i] = np.argmin(np.abs(ref_cum_hist - ori_cum_hist[i]))

    # Set the intensity of the pixel in the raw image to its corresponding new intensity
    height, width = ori_image.shape
    res_image = np.zeros(ori_image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i, j] = map_value[ori_image[i, j]]

    res_hist = np.bincount(res_image.flatten(), minlength=256)

    return ori_hist, ref_hist, res_image, res_hist


def cs4243_rotate180(kernel):
    """
    Rotate the matrix by 180.
    Can utilize build-in Funcs in numpy to ease your work
    :param kernel:
    :return:
    """
    kernel = np.flip(np.flip(kernel, 0),1)
    return kernel


def cs4243_gaussian_kernel(ksize, sigma):
    """
    5 points
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be same as the nearst
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    """
    kernel = np.zeros((ksize, ksize))
    linear_data = np.linspace(-(ksize - 1) // 2., (ksize - 1) // 2., ksize)

    for i in range(ksize):
        for j in range(ksize):
            kernel[i, j] = np.exp(-0.5 * (np.square(linear_data[i]) + np.square(linear_data[j])) / np.square(sigma))

    return kernel / kernel.sum()


def cs4243_filter(image, kernel):
    """
    10 points
    Implement the convolution operation in a naive 4 nested for-loops,
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return:
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    padded_image = pad_zeros(image, Hk // 2, Wk // 2)
    rotated_kernel = cs4243_rotate180(kernel)

    for i in range(Hi):
        for j in range(Wi):
            for m in range(Hk):
                for n in range(Wk):
                    filtered_image[i, j] += padded_image[i + m, j + n] * rotated_kernel[m, n]

    return filtered_image


def pad_zeros(image, pad_height, pad_width):
    """
    Pad the image with zero pixels, e.g., given matrix [[1]] with pad_height=1 and pad_width=2, obtains:
    [[0 0 0 0 0]
    [0 0 1 0 0]
    [0 0 0 0 0]]
    :param image: numpy.ndarray
    :param pad_height: int
    :param pad_width: int
    :return padded_image: numpy.ndarray
    """
    height, width = image.shape
    new_height, new_width = height+pad_height*2, width+pad_width*2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height-pad_height, pad_width:new_width-pad_width] = image
    return padded_image


def cs4243_filter_fast(image, kernel):
    """
    10 points
    Implement a fast version of filtering algorithm.
    take advantage of matrix operation in python to replace the
    inner 2-nested for loops in filter function.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    padded_image = pad_zeros(image, Hk // 2, Wk // 2)
    rotated_kernel = cs4243_rotate180(kernel)

    for i in range(Hi):
        for j in range(Wi):
            filtered_image[i, j] += (padded_image[i:i + Hk, j:j + Wk] * rotated_kernel).sum()

    return filtered_image


def cs4243_filter_faster(image, kernel):
    """
    10 points
    Implement a faster version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk),also reshape the flipped
    kernel to be of shape (Hk*Hk, 1), then do matrix multiplication, and rehshape back
    to get the final output image.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

    padded_image = pad_zeros(image, Hk // 2, Wk // 2)
    rotated_kernel = cs4243_rotate180(kernel)

    im2col_raw_result = np.zeros((Hi * Wi, Hk * Wk))
    k = 0
    for i in range(Hi):
        for j in range(Wi):
            im2col_raw_result[k] = padded_image[i:i + Hk, j:j + Wk].reshape(1, -1)
            k += 1

    filtered_image = np.matmul(im2col_raw_result, rotated_kernel.reshape(-1, 1)).reshape(Hi, Wi)
    return filtered_image

def cs4243_downsample(image, ratio):
    """
    Downsample the image to its 1/(ratio^2),which means downsample the width to 1/ratio, and the height 1/ratio.
    for example:
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = downsample(A, 2)
    B=[[1, 3], [7, 9]]
    :param image:numpy.ndarray
    :param ratio:int
    :return:
    """
    width, height = image.shape[1], image.shape[0]
    return image[0:height:ratio, 0:width:ratio]


def cs4243_upsample(image, ratio):
    """
    upsample the image to its 2^ratio,
    :param image: image to be upsampled
    :param kernel: use same kernel to get approximate value for additional pixels
    :param ratio: which means upsample the width to ratio*width, and height to ratio*height
    :return res_image: upsampled image
    Tips: use cs4243_resize to help you
    """
    width, height = image.shape[1], image.shape[0]
    new_width, new_height = width*ratio, height*ratio
    res_image = np.zeros((new_height, new_width))
    res_image[0:new_height:ratio, 0:new_width:ratio] = image
    return res_image


def cs4243_gauss_pyramid(image, n=3):
    """
    10 points
    build a Gaussian Pyramid of level n
    :param image: original grey scaled image
    :param n: level of pyramid
    :return pyramid: list, with list[0] corresponding to blurred image at level 0
    Tips: you may need to call cs4243_gaussian_kernel() and cs4243_filter_faster()
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    pyramid = []

    current = image
    for i in range(n + 1):
        pyramid.append(current)
        current = cs4243_downsample(cs4243_filter_faster(current, kernel), 2)

    return pyramid


def cs4243_lap_pyramid(gauss_pyramid):
    """
    10 points
    build a Laplacian Pyramid from the corresponding Gaussian Pyramid
    :param gauss_pyramid: list, results of cs4243_gauss_pyramid
    :return lap_pyramid: list, with list[0] corresponding to image at level n-1
    """
    #use same Gaussian kernel

    kernel = cs4243_gaussian_kernel(7, 1)
    n = len(gauss_pyramid)
    lap_pyramid = [gauss_pyramid[n - 1]]  # the top layer is same as Gaussian Pyramid

    for i in range(n - 1):
        lap_pyramid.append(gauss_pyramid[n - 2 - i] - cs4243_filter_faster(cs4243_upsample(gauss_pyramid[n - 1 - i], 2), kernel) * 4)  # 4 is the scaling factor mentioned in forum

    return lap_pyramid


def cs4243_Lap_blend(A, B, mask):
    """
    10 points
    blend image with Laplacian pyramid
    :param A: image on left
    :param B: image on right
    :param mask: mask [0, 1]
    :return blended image: same size as input image
    Tips: use cs4243_gauss_pyramid() & cs4243_lap_pyramid() to help you
    """
    kernel = cs4243_gaussian_kernel(7, 1)

    n = 3
    gauss_pyramid_A = cs4243_gauss_pyramid(A, n)
    lap_pyramid_A = cs4243_lap_pyramid(gauss_pyramid_A)

    gauss_pyramid_B = cs4243_gauss_pyramid(B, n)
    lap_pyramid_B = cs4243_lap_pyramid(gauss_pyramid_B)

    gauss_pyramid_R = cs4243_gauss_pyramid(mask, n)

    lap_pyramid_combine = []
    for i in range(n + 1):
        lap_pyramid_combine.append(lap_pyramid_A[n - i] * gauss_pyramid_R[i] + lap_pyramid_B[n - i] * (1 - gauss_pyramid_R[i]))

    lap_pyramid_combine = list(reversed(lap_pyramid_combine))

    blended_image = lap_pyramid_combine[0]
    for i in range(n):
        blended_image = cs4243_filter_faster(cs4243_upsample(blended_image, 2), kernel) * 4 + lap_pyramid_combine[i + 1]

    return blended_image
