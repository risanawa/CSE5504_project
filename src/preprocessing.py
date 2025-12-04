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

def binarize_image(gray_image, method="adaptive"):
    """
    gray_image : input grayscale image 
    method : 'otsu' (easy) & 'adaptive' (hard/real)
    """

    # 'otsu' : good for high contrast, synthetic data 
    if method == "otsu":
        # Check if background is bright (high mean value)
        if np.mean(gray_image) > 127:
            mode = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        else:
            mode = cv2.THRESH_BINARY + cv2.THRESH_OTSU
            
        thresh_val, binary = cv2.threshold(gray_image, 0, 255, mode)

    # 'adaptive Gaussian thresholding' : good for uneven lightening, low contrast 
    elif method == "adaptive":
        # Check if background is bright
        if np.mean(gray_image) > 127:
            thresh_type = cv2.THRESH_BINARY_INV
        else:
            thresh_type = cv2.THRESH_BINARY
            
        binary = cv2.adaptiveThreshold(
            gray_image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresh_type, # <--- Uses Inverse if background is bright
            blockSize=11, 
            C=2
        )

    else:
        raise ValueError("Define method as 'otsu' or 'adaptive'")
    
    return binary

# IF OTSU IS USED, NO DENOISING!

# NOISE REDUCTION -----------------------------

def denoise(binary_image, kernal_size = 3):
    kernal = np.ones((kernal_size, kernal_size), np.uint8)

    # opening : remove noise from background
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernal)

    # MAKE BIGGER?
    # closing : fill black holes 
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernal)

    return cleaned

def connect_segments(binary_image, kernel_size=5, iterations=2):
    """
    dilate to meger broken neurites 
    """
    # Safety Check: Ensure the background is actually black.
    # If the binary image came in wrong (White background), invert it here.
    if np.mean(binary_image) > 127:
        print("Warning: White background detected in bridging step. Inverting...")
        binary_image = cv2.bitwise_not(binary_image)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Dilation: Expands white regions.
    # This bridges gaps between the dots found in the thresholding step.
    bridged = cv2.dilate(binary_image, kernel, iterations=iterations)
    
    return bridged