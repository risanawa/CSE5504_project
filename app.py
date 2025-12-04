import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def convert_to_gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def binarize_image(gray_image, method="adaptive", block_size=11, C=2):
    if method == "otsu":
        thresh_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, C
        )
    else:
        raise ValueError("Method must be 'otsu' or 'adaptive'")
    return binary

def denoise(binary_image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

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
    
    # ===== PROCESSING CONTROLS (under box selector) =====
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
        st.write("**Noise Reduction**")
        use_denoise = st.checkbox("Apply denoising", value=True)
        kernel_size = 3
        if use_denoise:
            kernel_size = st.slider("Kernel Size:", 1, 15, 3, step=2)
    
    st.write("---")
    
    # ===== GET SELECTED BOX =====
    box = [b for b in boxes if b['id'] == selected_id][0]
    
    # Convert to pixels (exact bounding box, no padding)
    x_min_px = int(box['x_min'] * width)
    y_min_px = int(box['y_min'] * height)
    x_max_px = int(box['x_max'] * width)
    y_max_px = int(box['y_max'] * height)
    
    # Crop exactly to bounding box
    cropped = img_array[y_min_px:y_max_px, x_min_px:x_max_px]
    
    # ===== PROCESS IMAGE =====
    gray = convert_to_gray(cropped)
    binary = binarize_image(gray, method=method, block_size=block_size, C=C)
    
    if use_denoise:
        processed = denoise(binary, kernel_size=kernel_size)
    else:
        processed = binary
    
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
        st.write("Processed")
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        ax3.imshow(processed, cmap='gray')
        ax3.axis('off')
        st.pyplot(fig3)
        plt.close()
    
    # Stats
    st.write(f"**Size:** {x_max_px - x_min_px} x {y_max_px - y_min_px} pixels")
    st.write(f"**Method:** {method}")
    if method == "adaptive":
        st.write(f"**Block Size:** {block_size}, **C:** {C}")
    if use_denoise:
        st.write(f"**Kernel Size:** {kernel_size}")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()