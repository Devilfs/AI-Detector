"""Streamlit UI for AI Content Detection System."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import io
import json

# Import detection modules
try:
    from src.text_detection import TextEnsemble
    from src.image_detection import ImageEnsemble
    from src.utils import setup_logger
except ImportError as e:
    st.error(f"Failed to import detection modules: {e}")
    st.stop()

# Setup
st.set_page_config(
    page_title="AI Content Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'text_ensemble' not in st.session_state:
    st.session_state.text_ensemble = None
if 'image_ensemble' not in st.session_state:
    st.session_state.image_ensemble = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Setup logger
try:
    logger = setup_logger(__name__)
except Exception as e:
    st.error(f"Failed to setup logger: {e}")

# Utility functions
@st.cache_resource
def load_text_ensemble():
    """Load text detection ensemble."""
    try:
        return TextEnsemble()
    except Exception as e:
        st.error(f"Failed to load text detection models: {e}")
        return None

@st.cache_resource
def load_image_ensemble():
    """Load image detection ensemble."""
    try:
        return ImageEnsemble()
    except Exception as e:
        st.error(f"Failed to load image detection models: {e}")
        return None

def format_confidence(confidence):
    """Format confidence score with color."""
    if confidence > 0.8:
        return f"🔴 **{confidence:.1%}** (High)"
    elif confidence > 0.6:
        return f"🟡 **{confidence:.1%}** (Medium)"
    else:
        return f"🟢 **{confidence:.1%}** (Low)"

def create_confidence_chart(individual_predictions):
    """Create confidence chart for individual model predictions."""
    models = []
    confidences = []
    predictions = []
    
    for model, pred in individual_predictions.items():
        models.append(model.replace('_', ' ').title())
        confidences.append(pred['confidence'])
        predictions.append('AI' if pred['is_ai'] else 'Human')
    
    df = pd.DataFrame({
        'Model': models,
        'Confidence': confidences,
        'Prediction': predictions
    })
    
    fig = px.bar(
        df, 
        x='Model', 
        y='Confidence',
        color='Prediction',
        color_discrete_map={'AI': '#ff4b4b', 'Human': '#00d4aa'},
        title="Individual Model Predictions"
    )
    fig.update_layout(showlegend=True)
    return fig

def display_text_analysis(result, detailed=False):
    """Display text analysis results."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Prediction", 
            "🤖 AI Generated" if result['is_ai'] else "👤 Human Written"
        )
    
    with col2:
        st.metric("Confidence", format_confidence(result['confidence']))
    
    with col3:
        if 'processing_time_ms' in result:
            st.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")
    
    # Individual model results
    if 'individual_predictions' in result:
        st.subheader("Individual Model Results")
        
        # Create chart
        fig = create_confidence_chart(result['individual_predictions'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        if st.checkbox("Show detailed predictions"):
            for model, pred in result['individual_predictions'].items():
                with st.expander(f"{model.replace('_', ' ').title()} Details"):
                    st.json(pred)
    
    # Detailed features
    if detailed and 'detailed_features' in result:
        st.subheader("Detailed Analysis")
        
        for feature_type, features in result['detailed_features'].items():
            with st.expander(f"{feature_type.replace('_', ' ').title()} Features"):
                if isinstance(features, dict):
                    st.json(features)
                else:
                    st.write(features)

def display_image_analysis(result, image, detailed=False):
    """Display image analysis results."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, caption="Analyzed Image", use_column_width=True)
    
    with col2:
        st.metric(
            "Prediction", 
            "🤖 AI Generated" if result['is_ai'] else "📷 Real Photo"
        )
        st.metric("Confidence", format_confidence(result['confidence']))
        
        if 'processing_time_ms' in result:
            st.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")
    
    # Individual model results
    if 'individual_predictions' in result:
        st.subheader("Individual Model Results")
        
        # Create chart
        fig = create_confidence_chart(result['individual_predictions'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        if st.checkbox("Show detailed predictions"):
            for model, pred in result['individual_predictions'].items():
                with st.expander(f"{model.replace('_', ' ').title()} Details"):
                    st.json(pred)
    
    # Detailed features
    if detailed and 'detailed_features' in result:
        st.subheader("Detailed Analysis")
        
        for feature_type, features in result['detailed_features'].items():
            with st.expander(f"{feature_type.replace('_', ' ').title()} Features"):
                if isinstance(features, dict):
                    st.json(features)
                else:
                    st.write(features)

# Main UI
def main():
    st.title("🤖 AI Content Detection System")
    st.markdown("*A hybrid AI-generated text and image detector for high-accuracy detection*")
    
    # Sidebar
    st.sidebar.title("🛠️ Controls")
    
    # Model loading status
    st.sidebar.subheader("📊 Model Status")
    
    # Load models
    if st.session_state.text_ensemble is None:
        with st.spinner("Loading text detection models..."):
            st.session_state.text_ensemble = load_text_ensemble()
    
    if st.session_state.image_ensemble is None:
        with st.spinner("Loading image detection models..."):
            st.session_state.image_ensemble = load_image_ensemble()
    
    # Display model status
    text_status = "✅ Loaded" if st.session_state.text_ensemble else "❌ Failed"
    image_status = "✅ Loaded" if st.session_state.image_ensemble else "❌ Failed"
    
    st.sidebar.write(f"Text Models: {text_status}")
    st.sidebar.write(f"Image Models: {image_status}")
    
    # Detection mode selection
    st.sidebar.subheader("🔍 Detection Mode")
    detection_mode = st.sidebar.selectbox(
        "Choose what to analyze:",
        ["Text Detection", "Image Detection", "Batch Analysis", "Model Information"]
    )
    
    # Detailed analysis option
    detailed_analysis = st.sidebar.checkbox("Enable detailed analysis", value=False)
    
    # Main content area
    if detection_mode == "Text Detection":
        st.header("📝 Text Detection")
        
        if st.session_state.text_ensemble is None:
            st.error("Text detection models not available. Please check the logs.")
            return
        
        # Input methods
        input_method = st.radio(
            "Input method:",
            ["Type text", "Upload file", "Example texts"]
        )
        
        text_input = ""
        
        if input_method == "Type text":
            text_input = st.text_area(
                "Enter text to analyze:",
                height=200,
                placeholder="Paste or type the text you want to analyze here..."
            )
        
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt', 'md', 'pdf'],
                help="Supported formats: TXT, MD, PDF"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.type == "application/pdf":
                        st.warning("PDF support requires additional libraries. Please convert to TXT.")
                    else:
                        text_input = str(uploaded_file.read(), "utf-8")
                        st.success(f"File loaded: {len(text_input)} characters")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        elif input_method == "Example texts":
            examples = {
                "AI-Generated (GPT-like)": "The rapid advancement of artificial intelligence has fundamentally transformed the landscape of modern technology. Machine learning algorithms now process vast amounts of data with unprecedented efficiency, enabling sophisticated pattern recognition and predictive analytics across diverse applications.",
                "Human-Written (Academic)": "When I first started studying machine learning, I was completely overwhelmed by the math. Linear algebra, calculus, statistics - it felt like drinking from a fire hose. But my professor gave me some advice that changed everything: 'Don't try to understand everything at once. Pick one concept, work through examples, and build your intuition slowly.'",
                "AI-Generated (Creative)": "The ethereal moonlight cascaded through the ancient oak trees, creating a tapestry of silver shadows that danced across the forest floor. Each gentle breeze whispered secrets of forgotten times, while the nocturnal symphony of crickets provided a melodic backdrop to the mystical ambiance."
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            text_input = examples[selected_example]
            st.text_area("Selected text:", value=text_input, height=100, disabled=True)
        
        # Analysis button and results
        if st.button("🔍 Analyze Text", disabled=not text_input.strip()):
            if len(text_input.strip()) < 10:
                st.error("Text must be at least 10 characters long.")
            else:
                with st.spinner("Analyzing text..."):
                    try:
                        start_time = time.time()
                        
                        if detailed_analysis:
                            result = st.session_state.text_ensemble.analyze_detailed(text_input)
                            final_result = result['basic_prediction']
                            final_result['detailed_features'] = result.get('detailed_features', {})
                        else:
                            final_result = st.session_state.text_ensemble.predict(text_input)
                        
                        processing_time = (time.time() - start_time) * 1000
                        final_result['processing_time_ms'] = processing_time
                        
                        # Add to history
                        st.session_state.detection_history.append({
                            'type': 'text',
                            'timestamp': time.time(),
                            'result': final_result,
                            'input_preview': text_input[:100] + "..." if len(text_input) > 100 else text_input
                        })
                        
                        st.success("Analysis complete!")
                        display_text_analysis(final_result, detailed_analysis)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
    
    elif detection_mode == "Image Detection":
        st.header("🖼️ Image Detection")
        
        if st.session_state.image_ensemble is None:
            st.error("Image detection models not available. Please check the logs.")
            return
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload an image:",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Supported formats: JPG, PNG, WebP (max 10MB)"
        )
        
        if uploaded_image:
            try:
                # Load and display image
                image = Image.open(uploaded_image).convert('RGB')
                
                # Analysis
                if st.button("🔍 Analyze Image"):
                    with st.spinner("Analyzing image..."):
                        try:
                            start_time = time.time()
                            
                            if detailed_analysis:
                                result = st.session_state.image_ensemble.analyze_detailed(image)
                                final_result = result['basic_prediction']
                                final_result['detailed_features'] = result.get('detailed_features', {})
                            else:
                                final_result = st.session_state.image_ensemble.predict(image)
                            
                            processing_time = (time.time() - start_time) * 1000
                            final_result['processing_time_ms'] = processing_time
                            
                            # Add to history
                            st.session_state.detection_history.append({
                                'type': 'image',
                                'timestamp': time.time(),
                                'result': final_result,
                                'input_preview': f"Image ({image.width}x{image.height})"
                            })
                            
                            st.success("Analysis complete!")
                            display_image_analysis(final_result, image, detailed_analysis)
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    elif detection_mode == "Batch Analysis":
        st.header("📊 Batch Analysis")
        
        batch_type = st.selectbox("Batch type:", ["Text Files", "Images"])
        
        if batch_type == "Text Files":
            uploaded_files = st.file_uploader(
                "Upload multiple text files:",
                type=['txt', 'md'],
                accept_multiple_files=True
            )
            
            if uploaded_files and st.button("🔍 Analyze All Texts"):
                if st.session_state.text_ensemble is None:
                    st.error("Text detection models not available.")
                    return
                
                progress_bar = st.progress(0)
                results = []
                
                for i, file in enumerate(uploaded_files):
                    try:
                        text = str(file.read(), "utf-8")
                        result = st.session_state.text_ensemble.predict(text)
                        results.append({
                            'filename': file.name,
                            'prediction': 'AI' if result['is_ai'] else 'Human',
                            'confidence': result['confidence']
                        })
                    except Exception as e:
                        results.append({
                            'filename': file.name,
                            'prediction': 'Error',
                            'confidence': 0,
                            'error': str(e)
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Summary chart
                if len(df) > 0:
                    summary = df['prediction'].value_counts()
                    fig = px.pie(values=summary.values, names=summary.index, title="Batch Analysis Summary")
                    st.plotly_chart(fig)
        
        else:  # Images
            uploaded_images = st.file_uploader(
                "Upload multiple images:",
                type=['jpg', 'jpeg', 'png', 'webp'],
                accept_multiple_files=True
            )
            
            if uploaded_images and st.button("🔍 Analyze All Images"):
                if st.session_state.image_ensemble is None:
                    st.error("Image detection models not available.")
                    return
                
                progress_bar = st.progress(0)
                results = []
                
                for i, file in enumerate(uploaded_images):
                    try:
                        image = Image.open(file).convert('RGB')
                        result = st.session_state.image_ensemble.predict(image)
                        results.append({
                            'filename': file.name,
                            'prediction': 'AI' if result['is_ai'] else 'Real',
                            'confidence': result['confidence']
                        })
                    except Exception as e:
                        results.append({
                            'filename': file.name,
                            'prediction': 'Error',
                            'confidence': 0,
                            'error': str(e)
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_images))
                
                # Display results
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Summary chart
                if len(df) > 0:
                    summary = df['prediction'].value_counts()
                    fig = px.pie(values=summary.values, names=summary.index, title="Batch Analysis Summary")
                    st.plotly_chart(fig)
    
    elif detection_mode == "Model Information":
        st.header("🔧 Model Information")
        
        # Text models info
        if st.session_state.text_ensemble:
            st.subheader("📝 Text Detection Models")
            try:
                text_info = st.session_state.text_ensemble.get_model_info()
                st.json(text_info)
            except Exception as e:
                st.error(f"Failed to get text model info: {e}")
        
        # Image models info
        if st.session_state.image_ensemble:
            st.subheader("🖼️ Image Detection Models")
            try:
                image_info = st.session_state.image_ensemble.get_model_info()
                st.json(image_info)
            except Exception as e:
                st.error(f"Failed to get image model info: {e}")
        
        # System info
        st.subheader("💻 System Information")
        try:
            from src.utils import get_device_info
            device_info = get_device_info()
            st.json(device_info)
        except Exception as e:
            st.error(f"Failed to get device info: {e}")
    
    # Detection history in sidebar
    if st.session_state.detection_history:
        st.sidebar.subheader("📈 Recent Detections")
        
        for i, item in enumerate(reversed(st.session_state.detection_history[-5:])):
            pred_text = "🤖 AI" if item['result']['is_ai'] else "👤 Human"
            confidence = item['result']['confidence']
            
            st.sidebar.write(f"{pred_text} ({confidence:.1%})")
            st.sidebar.caption(f"{item['type'].title()}: {item['input_preview']}")
            st.sidebar.write("---")
        
        if st.sidebar.button("Clear History"):
            st.session_state.detection_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()