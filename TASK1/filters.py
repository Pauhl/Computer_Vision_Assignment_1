#####TUWIEN - CV: Task1 - Scale-Invariant Blob Detection
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import Tuple
import numpy as np
import cv2


def create_log_kernel(size: int, sig: float) -> float:
    """
    Returns a rotationally symmetric Laplacian of Gaussian kernel
    with given 'size' and standard deviation 'sig'

    Parameters
    ----------
    size : int
        size of kernel (must be odd) (int)
    sig : int
        standard deviation (float)
    
    Returns
    --------
    float
        kernel: filter kernel (size x size) (float)

    """

    kernel = np.zeros((size, size), np.float64)
    halfsize = int(np.floor(size / 2))
    r = range(-halfsize, halfsize + 1, 1)
    for x in r:
        for y in r:
            hg = (np.power(np.float64(x), 2) + np.power(np.float64(y), 2)) / (2 * np.power(np.float64(sig), 2))
            kernel[x + halfsize, y + halfsize] = -((1.0 - hg) * np.exp(-hg)) / (np.pi * np.power(sig, 4))

    return kernel - np.mean(kernel)


def get_log_pyramid(img: np.ndarray, sigma: float, k: float, levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a LoG scale space of given image 'img' with depth 'levels'
    The filter parameter 'sigma' increases by factor 'k' per level
    HINT: np.multiply(..), cv2.filter2D(..)

    Parameters
    ----------
    img : np.ndarray
        input image (n x m x 1) (float)
    sigma : float
        initial standard deviation for filter kernel
    levels : int
        number of layers of pyramid 

    Returns
    ---------
    np.ndarray
        scale_space : image pyramid (n x m x levels - float)
    np.ndarray
        all_sigmas : standard deviation used for every level (levels x 1 - float)
    """

    # student_code start
    sigma = float(sigma)
    scale_space = np.zeros((img.shape[0], img.shape[1], levels), dtype=float)
    all_sigmas = np.zeros((levels, 1), dtype=float)

    
    for i in range(levels):
        
        all_sigmas[i] = sigma

        filter_size = int(2 * np.floor(3 * sigma) + 1)
        kernel = create_log_kernel(filter_size, sigma)
        kernel = kernel * sigma**2

        scale_space[:, :, i] = np.abs(cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE))
        
        sigma *= k
    # student_code end

    return scale_space, all_sigmas
