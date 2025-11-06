import numpy as np
import streamlit as st
from PIL import Image


def load_image(file_like):
    img = Image.open(file_like).convert('RGB')
    return np.array(img)


st.set_page_config(page_title="R-CNN Car Detector", layout="wide")

TITLE_HTML = """
<h1 style='text-align: center; margin-bottom: 0.25rem;'>R-CNN Car Detector</h1>
<p style='text-align: center; color: gray; margin-top: 0;'>Upload, configure in sidebar, run, and view results in tabs</p>
"""


def create_ui():
    try:
        from car_detector import preds, get_ss
    except Exception as e:
        st.error(f"Failed to load detector dependencies: {e}")
        st.info("Please ensure required packages are installed as per requirements.txt, then restart the app.")
        return
    st.markdown(TITLE_HTML, unsafe_allow_html=True)

    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        threshold = st.slider("Confidence threshold", min_value=0.5, max_value=1.0, value=0.7)
        ss_mode = st.selectbox("Selective search", ("Single", "Fast", "Quality"), index=1)
        detect_button = st.button("Run Detection", type="primary", use_container_width=True)

    # Maintain image in session
    if image_file is not None:
        try:
            st.session_state["_img_np_"] = load_image(image_file)
        except Exception as e:
            st.warning(f"Could not load image: {e}")

    img = st.session_state.get("_img_np_")

    # Tabs
    tab_preview, tab_results, tab_about = st.tabs(["Preview", "Results", "About"]) 

    with tab_preview:
        st.subheader("Preview")
        if img is not None:
            st.image(img, use_column_width=True)
            st.caption(f"Threshold: {threshold} | Selective Search: {ss_mode}")
        else:
            st.info("Upload an image from the sidebar to begin.")

    # Run detection and persist results when the button is pressed
    if detect_button and img is not None:
        try:
            with st.spinner("Running selective search ..."):
                rects = get_ss(img, ss_mode)
                st.caption(f"Selective search generated {len(rects)} regions")

            progress_bar = st.progress(0)
            status = st.empty()
            status.info("Classifying regions ...")
            result_img = preds(img, threshold, rects, callback=lambda p: progress_bar.progress(int(min(1.0, max(0.0, p)) * 100)))
            status.empty()
            progress_bar.empty()

            # Save for Results tab
            st.session_state["_rects_count_"] = len(rects)
            st.session_state["_result_img_"] = result_img
            st.success("Detection complete. Check the Results tab.")
        except Exception as e:
            st.error(f"Detection failed: {e}")

    with tab_results:
        st.subheader("Detections")
        result_img = st.session_state.get("_result_img_")
        if result_img is not None:
            rects_count = st.session_state.get("_rects_count_", None)
            if rects_count is not None:
                st.caption(f"Regions evaluated: {rects_count}")
            st.image(result_img, use_column_width=True)
        else:
            st.info("No results yet. Use the sidebar to run detection.")

    with tab_about:
        st.markdown("""
        - This UI uses a sidebar for configuration and tabs for workflow separation.
        - Adjust the confidence threshold and selective search mode to influence detections.
        - Results show the annotated image after non-maximum suppression.
        """)


if __name__ == "__main__":
    create_ui()