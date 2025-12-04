# contouring 
# IOU calculation
import numpy as np
import cv2

def continuous_iou(mask1, mask2):
    # Element-wise minimum (soft intersection)
    intersection = np.minimum(mask1, mask2).sum()
    
    # Element-wise maximum (soft union)
    union = np.maximum(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0

def smooth_mask(mask):
    smoothed = cv2.GaussianBlur(mask.astype(float), (5, 5), 1.0)
    return smoothed

