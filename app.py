import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

from src.preprocessing import convert_to_gray, binarize_image, denoise, connect_segments
from src.untangling import run_untangling, get_skeleton, find_junctions
from src.segmentation import cca, get_valid_obj


def parse_yolo_boxes(bbox_text):
    boxes = []
    for line in bbox_text.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            
            boxes.append({
                'id': len(boxes) + 1,
                'x_min': x_min, 'y_min': y_min,
                'x_max': x_max, 'y_max': y_max
            })
    return boxes

# ===== MAIN APP =====

def main():
    st.title("Neuron Segmentation Pipeline - Simplified View")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload Image", type=['tif', 'tiff', 'png', 'jpg', 'bmp'])
    
    # Bbox input
    bbox_text = st.text_area(
        "Bounding Boxes (YOLO format):",
        "1 0.51918 0.62721 0.47163 0.45889\n1 0.26288 0.5 0.23229 1.0",
        height=100
    )
    
    if uploaded_file is None:
        st.info("Upload an image to start")
        return
    
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Parse boxes
    boxes = parse_yolo_boxes(bbox_text)
    
    if len(boxes) == 0:
        st.error("No valid bounding boxes found")
        return
    
    # ===== SIDEBAR CONTROLS =====
    st.sidebar.header("Processing Parameters")
    
    # Select box
    selected_id = st.sidebar.selectbox(
        "Choose box:",
        [b['id'] for b in boxes],
        format_func=lambda x: f"Box {x}"
    )
    
    st.sidebar.write("---")
    
    # Binary conversion
    method = st.sidebar.radio("Binary Method:", ["otsu", "adaptive"])
    
    # Denoising
    use_denoise = st.sidebar.checkbox("Apply denoising", value=True)
    denoise_kernel = st.sidebar.slider("Denoise Kernel:", 1, 15, 3, step=2) if use_denoise else 3
    
    # Connect segments
    use_connect = st.sidebar.checkbox("Connect broken segments", value=True)
    connect_kernel = st.sidebar.slider("Connect Kernel:", 3, 15, 5, step=2) if use_connect else 5
    connect_iterations = st.sidebar.slider("Connect Iterations:", 1, 5, 2) if use_connect else 2
    
    # Segmentation
    min_area = st.sidebar.slider("Min area (filter noise):", 50, 2000, 500, step=50)
    
    # Untangling
    use_untangle = st.sidebar.checkbox("Untangle touching neurons", value=True)
    
    # ===== GET SELECTED BOX =====
    box = [b for b in boxes if b['id'] == selected_id][0]
    
    # Convert to pixels
    x_min_px = int(box['x_min'] * width)
    y_min_px = int(box['y_min'] * height)
    x_max_px = int(box['x_max'] * width)
    y_max_px = int(box['y_max'] * height)
    
    # Crop to bounding box
    cropped = img_array[y_min_px:y_max_px, x_min_px:x_max_px]
    
    # ===== PROCESSING PIPELINE =====
    
    # Convert to grayscale
    gray = convert_to_gray(cropped)
    
    # Binarization
    binary = binarize_image(gray, method=method)
    
    # Denoising
    if use_denoise:
        cleaned = denoise(binary, kernal_size=denoise_kernel)
    else:
        cleaned = binary
    
    # SAVE STATE: Before connecting
    before_connect = cleaned.copy()
    
    # Connect segments (bridge gaps)
    if use_connect:
        cleaned = connect_segments(cleaned, kernel_size=connect_kernel, iterations=connect_iterations)
    
    # SAVE STATE: After connecting
    after_connect = cleaned.copy()
    
    # CCA - Find connected components
    num_labels, labels, stats, centroids = cca(cleaned)
    
    # Get valid objects
    neuron_masks = get_valid_obj(num_labels, labels, stats, min_area=min_area)
    
    # ===== COLLECT SKELETONIZATION DATA =====
    skeleton_data = []
    
    for i, mask in enumerate(neuron_masks):
        skeleton = get_skeleton(mask)
        junctions = find_junctions(skeleton)
        
        skeleton_data.append({
            'mask': mask,
            'skeleton': skeleton,
            'junctions': junctions,
            'num_junctions': len(junctions)
        })
    
    # ===== UNTANGLING =====
    final_masks = []
    
    if use_untangle:
        for i, mask in enumerate(neuron_masks):
            # Check if there are any junctions
            num_junctions = skeleton_data[i]['num_junctions']
            
            if num_junctions > 0:
                # Run untangling
                labels_untangled = run_untangling(mask, min_junctions=0)
                
                # Get unique labels
                unique_labels = np.unique(labels_untangled)
                unique_labels = unique_labels[unique_labels > 0]
                
                # Extract each segment
                for label_id in unique_labels:
                    segment_mask = (labels_untangled == label_id).astype(np.uint8) * 255
                    if segment_mask.sum() > 0:
                        final_masks.append(segment_mask)
            else:
                # Keep as is (no junctions found)
                final_masks.append(mask)
    else:
        final_masks = neuron_masks
    
    # Store masks in bbox coordinates for display
    final_masks_bbox = final_masks.copy()
    
    # Convert masks to full image coordinates for IoU evaluation
    final_masks_full = []
    for mask in final_masks:
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[y_min_px:y_max_px, x_min_px:x_max_px] = mask
        final_masks_full.append(full_mask)
    
    # ===== ORIGINAL IMAGE OVERVIEW =====
    st.subheader("Full Image with Bounding Boxes")
    
    fig_overview, ax_overview = plt.subplots(figsize=(12, 8))
    ax_overview.imshow(img_array)
    
    for b in boxes:
        bx_min = int(b['x_min'] * width)
        by_min = int(b['y_min'] * height)
        bx_max = int(b['x_max'] * width)
        by_max = int(b['y_max'] * height)
        
        # Highlight selected box
        color = 'red' if b['id'] == selected_id else 'red'
        linewidth = 3 if b['id'] == selected_id else 2
        
        rect = Rectangle(
            (bx_min, by_min),
            bx_max - bx_min,
            by_max - by_min,
            fill=False, edgecolor=color, linewidth=linewidth
        )
        ax_overview.add_patch(rect)
        ax_overview.text(bx_min, by_min - 5, f"Box {b['id']}", 
                        color=color, fontsize=12, weight='bold')
    
    ax_overview.axis('off')
    st.pyplot(fig_overview)
    plt.close()
    
    st.write("---")
    
    # ===== 4-STAGE VISUALIZATION =====
    
    st.subheader("Processing Pipeline Visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Stage 1: Original Image
    axes[0, 0].imshow(cropped)
    axes[0, 0].set_title("1. Original Image (Cropped)", fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    
    # Stage 2: After Connecting Broken Parts
    axes[0, 1].imshow(after_connect, cmap='gray')
    axes[0, 1].set_title(
        "2. Connected Segments" if use_connect else "2. No Connection Applied",
        fontsize=14, weight='bold'
    )
    axes[0, 1].axis('off')
    
    # Stage 3: Skeletonization with Junctions
    # Create composite image showing all skeletons
    skeleton_composite = np.zeros_like(after_connect, dtype=np.uint8)
    junction_points = []
    
    for data in skeleton_data:
        skeleton_composite = np.maximum(skeleton_composite, data['skeleton'])
        if len(data['junctions']) > 0:
            junction_points.extend(data['junctions'])
    
    # Show original mask as background, skeleton in red, junctions in blue
    axes[1, 0].imshow(after_connect, cmap='gray', alpha=0.3)
    axes[1, 0].imshow(skeleton_composite, cmap='Reds', alpha=0.7)
    
    if len(junction_points) > 0:
        junction_points = np.array(junction_points)
        axes[1, 0].scatter(
            junction_points[:, 1], 
            junction_points[:, 0],
            c='blue', s=100, marker='x', linewidths=3, label='Junctions'
        )
        axes[1, 0].legend(loc='upper right')
    
    total_junctions = sum(d['num_junctions'] for d in skeleton_data)
    axes[1, 0].set_title(
        f"3. Skeletonization",
        fontsize=14, weight='bold'
    )
    axes[1, 0].axis('off')
    
    # Stage 4: Final Output
    # Combine all final masks with different colors
    final_composite = np.zeros((*after_connect.shape, 3), dtype=np.uint8)
    
    # Generate distinct colors for each neuron
    np.random.seed(42)  # For consistent colors
    colors = []
    for i in range(len(final_masks)):
        color = tuple(np.random.randint(50, 255, size=3).tolist())
        colors.append(color)
    
    for i, mask in enumerate(final_masks_bbox):
        mask_bool = mask > 0
        final_composite[mask_bool] = colors[i]
    
    axes[1, 1].imshow(final_composite)
    axes[1, 1].set_title(
        f"4. Final Output",
        fontsize=14, weight='bold'
    )
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # ===== NEURON SELECTION =====
    if len(final_masks_bbox) > 0:
        st.write("---")
        st.subheader("Select a Neuron to Display")
        
        selected_neuron = st.selectbox(
            "Choose neuron:",
            range(1, len(final_masks_bbox) + 1),
            format_func=lambda x: f"Neuron #{x}"
        )
        
        # Display selected neuron
        st.write(f"**Displaying Neuron #{selected_neuron}:**")
        
        fig_selected, ax_selected = plt.subplots(figsize=(6, 6))
        ax_selected.imshow(final_masks_bbox[selected_neuron - 1], cmap='gray')
        ax_selected.set_title(f"Neuron #{selected_neuron}", fontsize=16, weight='bold')
        ax_selected.axis('off')
        st.pyplot(fig_selected)
        plt.close()
    
    # ===== IOU EVALUATION =====
    st.write("---")
    st.subheader("IoU Evaluation")
    
    gt_file = st.file_uploader("Upload Ground Truth", 
                               type=['txt'], key='gt_upload')
    
    if gt_file is not None:
        try:
            # Parse ground truth polygons
            polygons = []
            content = gt_file.read().decode('utf-8')
            for line in content.strip().split('\n'):
                coords = list(map(int, line.strip().split(',')))
                points = np.array(coords).reshape(-1, 2)
                polygons.append(points)
            
            # Convert to masks
            gt_masks = []
            for poly in polygons:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                gt_masks.append(mask)
            
            # Calculate IoU for each prediction
            from scipy.optimize import linear_sum_assignment
            
            n_pred = len(final_masks_full)
            n_gt = len(gt_masks)
            
            if n_pred > 0 and n_gt > 0:
                # Cost matrix
                cost_matrix = np.zeros((n_pred, n_gt))
                
                for i, pred_mask in enumerate(final_masks_full):
                    for j, gt_mask in enumerate(gt_masks):
                        pred_bool = pred_mask > 0
                        gt_bool = gt_mask > 0
                        intersection = np.logical_and(pred_bool, gt_bool).sum()
                        union = np.logical_or(pred_bool, gt_bool).sum()
                        iou = intersection / union if union > 0 else 0.0
                        cost_matrix[i, j] = -iou
                
                # Match
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                matches = [(i, j, -cost_matrix[i, j]) for i, j in zip(row_ind, col_ind) if -cost_matrix[i, j] > 0]
                matched_ious = [iou for _, _, iou in matches]
                
                if len(matched_ious) > 0:
                    mean_iou = np.mean(matched_ious)
                    st.success(f"**IoU: {mean_iou:.1%}**")
                    
                    # Visualization
                    fig_comp, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Ground truth
                    ax1.imshow(cropped)
                    for i, gt_mask in enumerate(gt_masks):
                        gt_crop = gt_mask[y_min_px:y_max_px, x_min_px:x_max_px]
                        contours, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            ax1.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, alpha=0.8)
                    ax1.set_title(f"Ground Truth", fontsize=14, weight='bold')
                    ax1.axis('off')
                    
                    # Predictions
                    ax2.imshow(cropped)
                    for i, pred_mask in enumerate(final_masks_bbox):
                        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            ax2.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2, alpha=0.8)
                    ax2.set_title(f"Predictions", fontsize=14, weight='bold')
                    ax2.axis('off')
                    
                    # Matched pairs with IoU scores
                    ax3.imshow(cropped)
                    for pred_idx, gt_idx, iou in matches:
                        # Draw prediction in red
                        pred_mask = final_masks_bbox[pred_idx]
                        contours_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours_pred:
                            ax3.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2, alpha=0.6)
                        
                        # Draw ground truth in green
                        gt_crop = gt_masks[gt_idx][y_min_px:y_max_px, x_min_px:x_max_px]
                        contours_gt, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours_gt:
                            ax3.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, alpha=0.6)
                    
                    ax3.set_title(f"Matched Pairs (Red=Pred, Green=GT)", fontsize=14, weight='bold')
                    ax3.axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig_comp)
                    plt.close()
                else:
                    st.warning("No matches found")
            else:
                st.warning(f"Cannot calculate IoU: {n_pred} predictions, {n_gt} ground truth")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # ===== COMPARE WITH OTHER MODEL =====
    st.write("---")
    st.subheader("Compare with Given Model")
    
    other_model_file = st.file_uploader("Upload Other Given Predictions", 
                                        type=['txt'], key='other_model_upload')
    
    if other_model_file is not None and gt_file is not None:
        try:
            # Parse other model polygons
            other_polygons = []
            content = other_model_file.read().decode('utf-8')
            for line in content.strip().split('\n'):
                coords = list(map(int, line.strip().split(',')))
                points = np.array(coords).reshape(-1, 2)
                other_polygons.append(points)
            
            # Convert to masks
            other_masks_full = []
            for poly in other_polygons:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                other_masks_full.append(mask)
            
            # Convert to bbox coordinates for display
            other_masks_bbox = []
            for mask in other_masks_full:
                bbox_mask = mask[y_min_px:y_max_px, x_min_px:x_max_px]
                other_masks_bbox.append(bbox_mask)
            
            # Calculate IoU for other model
            n_other = len(other_masks_full)
            
            if n_other > 0 and len(gt_masks) > 0:
                # Cost matrix
                cost_matrix_other = np.zeros((n_other, len(gt_masks)))
                
                for i, other_mask in enumerate(other_masks_full):
                    for j, gt_mask in enumerate(gt_masks):
                        other_bool = other_mask > 0
                        gt_bool = gt_mask > 0
                        intersection = np.logical_and(other_bool, gt_bool).sum()
                        union = np.logical_or(other_bool, gt_bool).sum()
                        iou = intersection / union if union > 0 else 0.0
                        cost_matrix_other[i, j] = -iou
                
                # Match
                row_ind_other, col_ind_other = linear_sum_assignment(cost_matrix_other)
                
                matches_other = [(i, j, -cost_matrix_other[i, j]) for i, j in zip(row_ind_other, col_ind_other) if -cost_matrix_other[i, j] > 0]
                matched_ious_other = [iou for _, _, iou in matches_other]
                
                if len(matched_ious_other) > 0:
                    mean_iou_other = np.mean(matched_ious_other)
                    st.info(f"**IoU: {mean_iou_other:.1%}**")
                    
                    # Comparison visualization
                    fig_comp2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Ground truth
                    ax1.imshow(cropped)
                    for i, gt_mask in enumerate(gt_masks):
                        gt_crop = gt_mask[y_min_px:y_max_px, x_min_px:x_max_px]
                        contours, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            ax1.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, alpha=0.8)
                    ax1.set_title(f"Ground Truth", fontsize=14, weight='bold')
                    ax1.axis('off')
                    
                    # Other model predictions
                    ax2.imshow(cropped)
                    for i, other_mask in enumerate(other_masks_bbox):
                        contours, _ = cv2.findContours(other_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            ax2.plot(contour[:, 0, 0], contour[:, 0, 1], 'b-', linewidth=2, alpha=0.8)
                    ax2.set_title(f"Given Model", fontsize=14, weight='bold')
                    ax2.axis('off')
                    
                    # Matched pairs
                    ax3.imshow(cropped)
                    for pred_idx, gt_idx, iou in matches_other:
                        # Draw prediction in blue
                        other_mask = other_masks_bbox[pred_idx]
                        contours_pred, _ = cv2.findContours(other_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours_pred:
                            ax3.plot(contour[:, 0, 0], contour[:, 0, 1], 'b-', linewidth=2, alpha=0.6)
                        
                        # Draw ground truth in green
                        gt_crop = gt_masks[gt_idx][y_min_px:y_max_px, x_min_px:x_max_px]
                        contours_gt, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours_gt:
                            ax3.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, alpha=0.6)
                    
                    ax3.set_title(f"Matched Pairs (Blue=Other, Green=GT)", fontsize=14, weight='bold')
                    ax3.axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig_comp2)
                    plt.close()
                    
               
                else:
                    st.warning("No matches found for other model")
            else:
                st.warning(f"Cannot calculate IoU: {n_other} predictions, {len(gt_masks)} ground truth")
        
        except Exception as e:
            st.error(f"Error processing other model file: {str(e)}")
    elif other_model_file is not None and gt_file is None:
        st.warning("Please upload ground truth first before comparing models")


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Neuron Segmentation")
    main()