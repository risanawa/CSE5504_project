# binary conversion
# noise reduction

import cv2
import numpy as np

def load_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Could not load image : {filepath}")
    return image

# BINARY CONVERSION ----------------------------

def convert_to_gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def binarize_image(gray_image, method="otsu"):
    """
    gray_image : input grayscale image 
    method : 'otsu' (easy) & 'adaptive' (hard/real)
    """

    # 'otsu' : good for high contrast, synthetic data 
    if method == "otsu":
        # global thresholding 
        thresh_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 'adaptive Gaussian thresholding' : good for uneven lightening, low contrast 
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            blockSize = 11,
            C = 2
        )

    else:
        raise ValueError("Define method as 'otsu' or 'adaptive'")
    
    return binary


# NOISE REDUCTION -----------------------------

def denoise(binary_image, kernal_size = 3):
    kernal = np.ones((kernal_size, kernal_size), np.uint8)

    # opening : remove noise from background
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernal)

    # closing : fill black holes
    cleaned = cv2.mophologyEx(cleaned, cv2.MORPH_CLOSE, kernal)

    return cleaned