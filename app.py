import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import from src folder
from src.preprocessing import convert_to_gray, binarize_image, denoise
from src.untangling import run_untangling
from src.segmentation import cca, get_valid_obj
from src.evaluation import continuous_iou, smooth_mask

# ===== BBOX PARSING =====

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
    st.title("Neuron Bounding Box Viewer")
    
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
    
    # ===== OVERVIEW =====
    st.subheader("Overview")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_array)
    
    for box in boxes:
        x_min_px = int(box['x_min'] * width)
        y_min_px = int(box['y_min'] * height)
        x_max_px = int(box['x_max'] * width)
        y_max_px = int(box['y_max'] * height)
        
        rect = Rectangle(
            (x_min_px, y_min_px),
            x_max_px - x_min_px,
            y_max_px - y_min_px,
            fill=False, edgecolor='red', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x_min_px, y_min_px - 5, f"Box {box['id']}", 
               color='red', fontsize=10, weight='bold')
    
    ax.axis('off')
    st.pyplot(fig)
    plt.close()
    
    # ===== SELECT BOX =====
    st.subheader("Select Box")
    selected_id = st.selectbox(
        "Choose box:",
        [b['id'] for b in boxes],
        format_func=lambda x: f"Box {x}"
    )
    
    # ===== PROCESSING CONTROLS =====
    st.write("---")
    st.subheader("Processing Controls")
    
    col_control1, col_control2 = st.columns(2)
    
    with col_control1:
        st.write("**Binary Conversion**")
        method = st.radio("Method:", ["otsu", "adaptive"])
        
        block_size = 11
        C = 2
        if method == "adaptive":
            block_size = st.slider("Block Size:", 3, 51, 11, step=2)
            C = st.slider("C (constant):", -10, 10, 2)
    
    with col_control2:
        st.write("**Morphological Operations**")
        use_denoise = st.checkbox("Apply denoising", value=True)
        kernel_size = 3
        if use_denoise:
            kernel_size = st.slider("Kernel Size:", 1, 15, 3, step=2)
        
        use_untangle = st.checkbox("Untangle touching neurons", value=False)
    
    st.write("---")
    
    # ===== GET SELECTED BOX =====
    box = [b for b in boxes if b['id'] == selected_id][0]
    
    # Convert to pixels
    x_min_px = int(box['x_min'] * width)
    y_min_px = int(box['y_min'] * height)
    x_max_px = int(box['x_max'] * width)
    y_max_px = int(box['y_max'] * height)
    
    # Crop to bounding box
    cropped = img_array[y_min_px:y_max_px, x_min_px:x_max_px]
    
    # ===== PROCESS IMAGE =====
    # Convert to grayscale
    gray = convert_to_gray(cropped)
    
    # Binarize
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
    else:
        binary = binarize_image(gray, method=method)
    
    # Apply denoising
    if use_denoise:
        processed = denoise(binary, kernal_size=kernel_size)
    else:
        processed = binary
    
    # Store for display
    processed_display = processed.copy()
    num_segments = 1
    
    # Apply untangling
    if use_untangle:
        labels = run_untangling(processed)
        num_segments = labels.max()
        
        # Create colored visualization of segments
        # Each segment gets a unique gray value
        processed_display = np.zeros_like(processed, dtype=np.uint8)
        
        for label_id in range(1, num_segments + 1):
            # Assign different intensities to each segment
            intensity = int((label_id / num_segments) * 200 + 55)  # Scale to 55-255 range
            processed_display[labels == label_id] = intensity
    
    # ===== DISPLAY =====
    st.subheader(f"Box {selected_id}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Original")
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.imshow(cropped)
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        st.write("Binary")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.imshow(binary, cmap='gray')
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close()
    
    with col3:
        st.write("Processed" + (" (Untangled)" if use_untangle else ""))
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        if use_untangle:
            # Show with color to see different segments
            ax3.imshow(processed_display, cmap='nipy_spectral')
        else:
            ax3.imshow(processed_display, cmap='gray')
        ax3.axis('off')
        st.pyplot(fig3)
        plt.close()
    
    # ===== STATS =====
    st.write(f"**Size:** {x_max_px - x_min_px} x {y_max_px - y_min_px} pixels")
    st.write(f"**Method:** {method}")
    if method == "adaptive":
        st.write(f"**Block Size:** {block_size}, **C:** {C}")
    if use_denoise:
        st.write(f"**Denoising Kernel:** {kernel_size}")
    if use_untangle:
        st.write(f"**Separated Segments:** {num_segments}")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()