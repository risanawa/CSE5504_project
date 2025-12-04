# connectivity analysis (CCA)
# object selection
# shape analysis
import numpy as np
import matplotlib.pyplot as plt
import cv2


# def connected_component_analysis(image, connectivity=2):
#     # Define connectivity structure
#     structure = ndimage.generate_binary_structure(2, connectivity)
    
#     # Label connected components
#     labeled_array, num_features = ndimage.label(image, structure=structure)
    
#     return labeled_array, num_features


def cca(binary_image):
    """ 
    returns : 
        num_labels - total number of objects found (including background)
        labels - image (each pixel will have the ID of the object it belongs to)
        stats - array that contains the info for each object (x, y, w, h, area)
        centroids - array containing the center (x, y) of each object 
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    return num_labels, labels, stats, centroids

def get_valid_obj(num_labels, labels, stats, min_area=50):
    # filter noise 
    # return list of binary mask for each object present 

    valid_masks = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # is it a neuron?
        if area > min_area:
            component_mask = (labels == i).astype("uint8") * 255
            valid_masks.append(component_mask)

    return valid_masks

def check_complexity(binary_mask):
    # is the object in need of untangling?
    
    return True

