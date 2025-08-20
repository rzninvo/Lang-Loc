import streamlit as st

def render_instructions():
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
        ./data/scans/
        ├── scene0000_00
        │  ├── intrinsic
        │  |  ...
        │  ├──output
        │  │  ├── cache_scannetpp
        │  │  |   ├── ...
        │  │  ├── camera_pose.json
        │  │  ├── color
        │  │  │   ├── 002800.jpg
        │  │  │   └── ...
        │  │  ├── depth
        │  │  │   ├── 002800.png
        │  │  │   └── ...
        │  │  └── pose
        │  │  ├── 002800.txt
        │  │  └── ...
        │  ├── scene0000_00.aggregation.json
        │  ├── scene0000_00.txt
        │  ├── scene0000_00_vh_clean_2.0.010000.segs.json
        │  ├── scene0000_00_vh_clean_2.labels.ply
        │  └── scene0000_00_vh_clean_2.ply
        ├── ...

        """)
