# thinning/skeletonization
# structure analysis -- junctions
# cutting
# regrow -- dilation

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from scipy.ndimage import convolve

def get_skeleton(binary_mask):
    # reduce object to 1-pixel wide skeleton
    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)

    return skeleton.astype(np.uint8) * 255

def find_junctions(skeleton):
    # find crossing points (pixels with 3 or more neighbors)

    skeleton_bool = (skeleton > 0).astype(int)
    kernal = np.array([[1,1,1], [ 1,0,1], [1,1,1]])

    neighbor_count = convolve(skeleton_bool, kernal, mode="constant", cval=0)
    junctions = (skeleton_bool == 1) & (neighbor_count > 3)

    return np.argwhere(junctions)

def cut_skeleton(skeleton, junction_coords):
    # remove the junction pixels to separate the skeleton branches
    cut_skel = skeleton.copy()

    for (y, x) in junction_coords:
        cut_skel[y, x] = 0

    return cut_skel

def regrow_segments(cut_skeleton, original_mask):
    # watershed 
    num_labels, markers = cv2.connectedComponents(cut_skeleton)

    if num_labels <= 2:
        return markers
    
    # calculate the distance between each pixel and the background
    distance = cv2.distanceTransform(original_mask, cv2.DIST_L2, 5)

    # apply watershed 
    labels = watershed(-distance, markers, mask=original_mask)
    
    return labels 

def run_untangling(binary_mask, min_junctions=3):
    # 1. thinning
    skeleton = get_skeleton(binary_mask)

    # 2. structure analysis 
    junctions = find_junctions(skeleton)

    # untangle if junctions exist
    if len(junctions) == 0:
        return (binary_mask > 0).astype(int)
    
    # 3. cut
    cut_skel = cut_skeleton(skeleton, junctions)

    # 4. apply watershed
    final_labels = regrow_segments(cut_skel, binary_mask)

    return final_labels