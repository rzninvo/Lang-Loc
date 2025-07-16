import streamlit as st
import os
import json
import glob
from PIL import Image
import pandas as pd
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="ScanNet Image Annotation Tool",
    page_icon="🖼️",
    layout="wide"
)

# Dataset configuration - automatically loads from fixed path
DATASET_PATH = "/Users/abu/Desktop/master_project/data_sample"
ANNOTATIONS_FILE = "scannet_annotations.json"
UNINTERPRETABLE_FILE = "uninterpretable_images.json"

# Sample image configuration
SAMPLE_SCENE = "scene0000_00"
SAMPLE_VIEW = "view_1"
SAMPLE_DESCRIPTION = "A studio apartment On the left, there's a bed with blue bedding next to a worn rug and a wooden desk. A bicycle leans against a corrugated metal wall near the center, beside a dark L-shaped sofa with pillows and two ottomans in front. To the right, there's a small table or counter with stools. The bicycle is placed next to the TV cabinet, the TV is facing the shorter side of the L-shaped sofa."

def load_dataset_structure():
    """Load all scene directories from the dataset"""
    if not os.path.exists(DATASET_PATH):
        return []
    
    scene_dirs = []
    for item in os.listdir(DATASET_PATH):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path) and item.startswith('scene'):
            output_images_path = os.path.join(item_path, 'output_images')
            if os.path.exists(output_images_path):
                # Check if it has the expected view images
                views = [f"view_{i}.png" for i in range(1, 7)]
                if all(os.path.exists(os.path.join(output_images_path, view)) for view in views):
                    scene_dirs.append(item)
    
    return sorted(scene_dirs)

def load_existing_annotations():
    """Load existing annotations from file if it exists"""
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            with open(ANNOTATIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def load_uninterpretable_images():
    """Load existing uninterpretable images log from file if it exists"""
    if os.path.exists(UNINTERPRETABLE_FILE):
        try:
            with open(UNINTERPRETABLE_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'current_scene_index' not in st.session_state:
        st.session_state.current_scene_index = 0
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = load_existing_annotations()
    if 'uninterpretable_images' not in st.session_state:
        st.session_state.uninterpretable_images = load_uninterpretable_images()
    if 'scene_list' not in st.session_state:
        st.session_state.scene_list = load_dataset_structure()

# Initialize session state
initialize_session_state()

def save_annotations_to_file():
    """Save annotations to JSON file"""
    try:
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(st.session_state.annotations, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving annotations: {str(e)}")
        return False

def save_uninterpretable_to_file():
    """Save uninterpretable images log to JSON file"""
    try:
        with open(UNINTERPRETABLE_FILE, 'w') as f:
            json.dump(st.session_state.uninterpretable_images, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving uninterpretable images log: {str(e)}")
        return False

def get_scene_pose(image_index):
    """Calculate scene pose based on image index (60 degrees per image)"""
    return (image_index - 1) * 60

def get_current_image_path():
    """Get the path to the current image"""
    if not st.session_state.scene_list:
        return None
    
    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    current_view = f"view_{st.session_state.current_image_index + 1}.png"
    
    image_path = os.path.join(
        DATASET_PATH,
        current_scene,
        'output_images',
        current_view
    )
    
    return image_path if os.path.exists(image_path) else None

def get_sample_image_path():
    """Get the path to the sample image"""
    sample_path = os.path.join(
        DATASET_PATH,
        SAMPLE_SCENE,
        'output_images',
        f"{SAMPLE_VIEW}.png"
    )
    return sample_path if os.path.exists(sample_path) else None

def save_annotation(description):
    """Save the current annotation"""
    if not st.session_state.scene_list:
        return False
    
    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    current_view = f"view_{st.session_state.current_image_index + 1}"
    scene_pose = get_scene_pose(st.session_state.current_image_index + 1)
    
    annotation = {
        'scene_index': current_scene,
        'image_index': current_view,
        'scene_pose': scene_pose,
        'description': description,
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.annotations.append(annotation)
    return save_annotations_to_file()

def mark_uninterpretable():
    """Mark the current image as uninterpretable"""
    if not st.session_state.scene_list:
        return False
    
    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    current_view = f"view_{st.session_state.current_image_index + 1}"
    scene_pose = get_scene_pose(st.session_state.current_image_index + 1)
    
    # Check if this image is already marked as uninterpretable
    for item in st.session_state.uninterpretable_images:
        if item['scene_index'] == current_scene and item['image_index'] == current_view:
            return True  # Already marked, no need to add again
    
    uninterpretable_entry = {
        'scene_index': current_scene,
        'image_index': current_view,
        'scene_pose': scene_pose,
        'timestamp': datetime.now().isoformat(),
        'reason': 'marked_as_uninterpretable'
    }
    
    st.session_state.uninterpretable_images.append(uninterpretable_entry)
    return save_uninterpretable_to_file()

def is_current_image_uninterpretable():
    """Check if the current image is already marked as uninterpretable"""
    if not st.session_state.scene_list:
        return False
    
    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    current_view = f"view_{st.session_state.current_image_index + 1}"
    
    for item in st.session_state.uninterpretable_images:
        if item['scene_index'] == current_scene and item['image_index'] == current_view:
            return True
    return False

def next_image():
    """Navigate to the next image"""
    if not st.session_state.scene_list:
        return
    
    total_scenes = len(st.session_state.scene_list)
    images_per_scene = 6
    
    # Move to next image
    st.session_state.current_image_index += 1
    
    # If we've reached the end of current scene, move to next scene
    if st.session_state.current_image_index >= images_per_scene:
        st.session_state.current_image_index = 0
        st.session_state.current_scene_index += 1
        
        # If we've reached the end of all scenes, loop back to start
        if st.session_state.current_scene_index >= total_scenes:
            st.session_state.current_scene_index = 0

def previous_image():
    """Navigate to the previous image"""
    if not st.session_state.scene_list:
        return
    
    total_scenes = len(st.session_state.scene_list)
    images_per_scene = 6
    
    # Move to previous image
    st.session_state.current_image_index -= 1
    
    # If we've reached before the first image of current scene, move to previous scene
    if st.session_state.current_image_index < 0:
        st.session_state.current_scene_index -= 1
        st.session_state.current_image_index = images_per_scene - 1
        
        # If we've reached before the first scene, loop to last scene
        if st.session_state.current_scene_index < 0:
            st.session_state.current_scene_index = total_scenes - 1

# Main UI
st.title("🖼️ ScanNet Image Annotation Tool")

# Display dataset info in sidebar
st.sidebar.header("Dataset Information")
if st.session_state.scene_list:
    st.sidebar.success(f"Dataset loaded successfully!")
    st.sidebar.info(f"Total scenes: {len(st.session_state.scene_list)}")
    st.sidebar.info(f"Total images: {len(st.session_state.scene_list) * 6}")
    st.sidebar.info(f"Annotations completed: {len(st.session_state.annotations)}")
    st.sidebar.info(f"Uninterpretable images: {len(st.session_state.uninterpretable_images)}")
else:
    st.sidebar.error("No valid scenes found in the dataset")
    st.sidebar.info(f"Looking for data in: {DATASET_PATH}")

# Sample image and description
#st.header("📋 Sample Reference")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sample Image")
    sample_image_path = get_sample_image_path()
    if sample_image_path:
        try:
            sample_image = Image.open(sample_image_path)
            st.image(sample_image, caption=f"{SAMPLE_SCENE} - {SAMPLE_VIEW}", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading sample image: {str(e)}")
    else:
        st.error("Sample image not found")

with col2:
    st.subheader("Sample Description")
    st.info(SAMPLE_DESCRIPTION)
    
    st.subheader("Annotation Guidelines")
    guidelines = """
    A good description should include:
    - Objects visible in the scene
    - Spatial relationships between objects
    - Colors and materials
    - Overall scene context
    - Any notable features or details
    
    **Note:** If an image is too blurry, dark, corrupted, or otherwise impossible to interpret, use the "Uninterpretable" button instead of trying to describe it.
    """
    st.write(guidelines)

# Main annotation interface
if st.session_state.scene_list:
    st.header("🎯 Image Annotation")
    
    # Current image info
    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    current_view = f"view_{st.session_state.current_image_index + 1}"
    scene_pose = get_scene_pose(st.session_state.current_image_index + 1)
    
    # Progress indicator
    total_images = len(st.session_state.scene_list) * 6
    current_image_number = st.session_state.current_scene_index * 6 + st.session_state.current_image_index + 1
    
    st.progress(current_image_number / total_images)
    st.write(f"**Progress:** {current_image_number} / {total_images} images")
    
    # Image information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Scene", current_scene)
    with col2:
        st.metric("View", current_view)
    with col3:
        st.metric("Camera Pose", f"{scene_pose}°")
    with col4:
        if is_current_image_uninterpretable():
            st.error("⚠️ Uninterpretable")
        else:
            st.info("✅ Interpretable")
    
    # Display current image
    current_image_path = get_current_image_path()
    if current_image_path:
        try:
            image = Image.open(current_image_path)
            st.image(image, caption=f"{current_scene} - {current_view}", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        st.error("Image not found")
    
    # Annotation input
    st.subheader("📝 Describe this image:")
    description = st.text_area(
        "Enter your description here:",
        height=100,
        placeholder="Describe what you see in the image...",
        disabled=is_current_image_uninterpretable(),
        key=f"description_{current_scene}_{current_view}"
    )
    
    if is_current_image_uninterpretable():
        st.warning("This image has been marked as uninterpretable. Use navigation buttons to move to another image.")
    
    # Navigation buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            previous_image()
            st.rerun()
    
    with col2:
        if st.button("💾 Save & Next", use_container_width=True, type="primary", disabled=is_current_image_uninterpretable()):
            if description.strip():
                if save_annotation(description):
                    st.success("Annotation saved successfully!")
                    next_image()
                    st.rerun()
                else:
                    st.error("Failed to save annotation")
            else:
                st.error("Please enter a description before saving")
    
    with col3:
        if st.button("❌ Uninterpretable", use_container_width=True, help="Mark this image as uninterpretable (too blurry, dark, corrupted, etc.)"):
            if mark_uninterpretable():
                st.success("Image marked as uninterpretable!")
                next_image()
                st.rerun()
            else:
                st.error("Failed to mark image as uninterpretable")
    
    with col4:
        if st.button("⏭️ Skip", use_container_width=True):
            next_image()
            st.rerun()

else:
    st.error("Dataset not found. Please ensure the dataset is available at the specified path.")
    st.info(f"Expected path: {DATASET_PATH}")

# Show recent annotations and uninterpretable images for admin
col1, col2 = st.columns(2)

with col1:
    if st.session_state.annotations:
        with st.expander("📊 Recent Annotations (Admin View)"):
            df = pd.DataFrame(st.session_state.annotations)
            st.dataframe(df.tail(10))

with col2:
    if st.session_state.uninterpretable_images:
        with st.expander("⚠️ Uninterpretable Images (Admin View)"):
            df_uninterpretable = pd.DataFrame(st.session_state.uninterpretable_images)
            st.dataframe(df_uninterpretable.tail(10))

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    ### How to use this annotation tool:
    
    1. **View Sample**: Look at the sample image and description for reference
    2. **Annotate**: 
       - View the current image and its information
       - Write a detailed description in the text area
       - Click "Save & Next" to save your annotation and move to the next image
       - Use "Previous" to go back or "Skip" to move forward without saving
       - Click "Uninterpretable" if the image is too blurry, dark, corrupted, or otherwise impossible to interpret
    3. **Progress**: Track your progress with the progress bar
    4. **Auto-save**: All annotations and uninterpretable image logs are automatically saved to separate files
    
    ### File Output:
    - `scannet_annotations.json`: Contains all text annotations
    - `uninterpretable_images.json`: Contains log of images marked as uninterpretable
    
    ### Dataset Structure:
    ```
    /Users/abu/Desktop/master_project/data_sample/
    ├── scene0000_00/
    │   └── output_images/
    │       ├── view_1.png
    │       ├── view_2.png
    │       └── ...
    └── ...
    ```
    """)
