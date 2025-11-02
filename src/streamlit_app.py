"""
Explainable Retinal Disease Screening - Streamlit Web Interface
GPU-Accelerated with Enhanced User Experience
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
import os
from pathlib import Path
import time
import cv2

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

# Add models directory to path for explainability modules
models_dir = Path(__file__).parent.parent / 'models'
if models_dir.exists():
    sys.path.insert(0, str(models_dir))

# Import custom modules
from models.vignn import SceneGraphTransformer, create_scene_graph_model

# Check for explainability dependencies
GRADCAM_AVAILABLE = False
GRADCAM_LIBRARY = None
CAPTUM_AVAILABLE = False
SHAP_AVAILABLE = False
LIME_AVAILABLE = False
ELI5_AVAILABLE = False

try:
    import grad_cam
    GRADCAM_AVAILABLE = True
    GRADCAM_LIBRARY = 'grad-cam'
    print("✓ grad-cam available")
except ImportError:
    print("⚠ grad-cam not installed - GradCAM features will be disabled")
    GRADCAM_AVAILABLE = False
    GRADCAM_LIBRARY = None

try:
    import captum
    CAPTUM_AVAILABLE = True
    print("✓ captum available (Integrated Gradients, Saliency Maps)")
except ImportError:
    print("⚠ captum not installed - Integrated Gradients features will be disabled")

try:
    import shap
    SHAP_AVAILABLE = True
    print("✓ shap available (SHAP explanations)")
except ImportError:
    print("⚠ shap not installed - SHAP features will be disabled")

try:
    import lime
    LIME_AVAILABLE = True
    print("✓ lime available (Local Interpretable Model-agnostic Explanations)")
except ImportError:
    print("⚠ lime not installed - LIME features will be disabled")

try:
    import eli5
    ELI5_AVAILABLE = True
    print("✓ eli5 available (Explain Like I'm 5)")
except ImportError:
    print("⚠ eli5 not installed - ELI5 features will be disabled")

# Import ModelExplainer
try:
    from model_explainer import ModelExplainer
    EXPLAINABILITY_AVAILABLE = True
    print(f"✓ ModelExplainer loaded from: {models_dir}")
except ImportError as e:
    EXPLAINABILITY_AVAILABLE = False
    print(f"Warning: ModelExplainer not available: {e}")
    print(f"  Searched in: {models_dir}")

# Page configuration
st.set_page_config(
    page_title="Retinal AI Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #00897B;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00897B;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #00897B;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00695C;
    }
    .uploaded-image {
        border: 3px solid #00897B;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Disease name mapping (45 diseases from model metadata)
DISEASE_MAPPING = {
    "DR": "Diabetic Retinopathy",
    "ARMD": "Age-Related Macular Degeneration",
    "MH": "Macular Hole",
    "DN": "Diabetic Neuropathy",
    "MYA": "Myopic Retinopathy",
    "BRVO": "Branch Retinal Vein Occlusion",
    "TSLN": "Tessellation (Myopic Fundus Changes)",
    "ERM": "Epiretinal Membrane",
    "LS": "Laser Scars (Photocoagulation)",
    "MS": "Macular Scars",
    "CSR": "Central Serous Retinopathy",
    "ODC": "Optic Disc Cupping",
    "CRVO": "Central Retinal Vein Occlusion",
    "TV": "Tortuous Vessels",
    "AH": "Asteroid Hyalosis",
    "ODP": "Optic Disc Pallor",
    "ODE": "Optic Disc Edema",
    "ST": "Optociliary Shunt Vessels",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "PT": "Parafoveal Telangiectasia",
    "RT": "Retinal Traction Detachment",
    "RS": "Retinitis (Inflammatory Retinal Disease)",
    "CRS": "Chorioretinal Scars",
    "EDN": "Exudative Retinal Detachment",
    "RPEC": "Retinal Pigment Epithelial Changes",
    "MHL": "Lamellar Macular Hole",
    "RP": "Retinitis Pigmentosa",
    "CWS": "Cotton Wool Spots (Nerve Fiber Layer Infarcts)",
    "CB": "Coats Disease (Retinal Telangiectasia with Exudation)",
    "ODPM": "Optic Disc Pit Maculopathy",
    "PRH": "Preretinal Hemorrhage",
    "MNF": "Myelinated Nerve Fibers",
    "HR": "Hemorrhagic Retinopathy",
    "CRAO": "Central Retinal Artery Occlusion",
    "TD": "Tilted Disc (Congenital Disc Anomaly)",
    "CME": "Cystoid Macular Edema",
    "PTCR": "Post-Traumatic Chorioretinopathy",
    "CF": "Choroidal Folds",
    "VH": "Vitreous Hemorrhage",
    "MCA": "Retinal Macroaneurysm",
    "VS": "Vasculitis (Vessel Sheathing)",
    "BRAO": "Branch Retinal Artery Occlusion",
    "PLQ": "Optic Disc Drusen (Peripapillary Lesions)",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",
    "CL": "Choroidal Lesion"
}

DISEASE_CODES = list(DISEASE_MAPPING.keys())


@st.cache_data
def load_model_metadata():
    """Load model metadata from JSON file"""
    try:
        metadata_path = Path(__file__).parent.parent / 'models' / 'model_metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Model metadata loaded from: {metadata_path}")
            return metadata
        else:
            print(f"⚠ Model metadata not found at: {metadata_path}")
            return None
    except Exception as e:
        print(f"⚠ Error loading metadata: {e}")
        return None


@st.cache_resource
def load_model():
    """Load the retinal disease classification model with GPU support"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with st.spinner(f'🔄 Loading AI model on {device}...'):
            model_path = Path(__file__).parent.parent / 'models' / 'best_model_mobile.pth'
            
            if not model_path.exists():
                st.error(f"❌ Model not found at {model_path}")
                return None, None
            
            # Initialize model with correct SceneGraphTransformer parameters
            model = SceneGraphTransformer(
                num_classes=45,
                num_regions=12,
                hidden_dim=384,
                num_layers=2,
                num_heads=4,
                dropout=0.1,
                num_ensemble_branches=3
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            st.success(f'✅ Model loaded successfully on {device}!')
            return model, device
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


@st.cache_resource
def load_explainer(_model, _device):
    """Load model explainer for interpretability"""
    if not EXPLAINABILITY_AVAILABLE or _model is None:
        return None
    
    try:
        sys.path.append(str(Path(__file__).parent.parent / 'models'))
        from model_explainer import ModelExplainer
        
        explainer = ModelExplainer(
            model=_model,
            device=_device,
            disease_names=DISEASE_CODES
        )
        return explainer
    except Exception as e:
        st.warning(f"⚠️ Explainability features unavailable: {str(e)}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to tensor
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Transpose to CHW format and ensure float32
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return img_tensor


def predict(model, device, image_tensor):
    """Run inference on image"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        return probabilities
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None


def get_comprehensive_analysis(explainer, image_tensor, device, top_k=5):
    """
    Get comprehensive clinical analysis using ModelExplainer's lightweight explanation
    Returns detailed predictions with clinical insights, recommendations, and uncertainty metrics
    """
    try:
        image_tensor = image_tensor.to(device)
        
        # Use the explainer's comprehensive analysis method
        results = explainer.get_lightweight_explanation(image_tensor, top_k=top_k)
        
        return results
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        return None


def get_top_predictions(probabilities, top_k=5):
    """Get top K predictions with disease names"""
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        disease_code = DISEASE_CODES[idx]
        disease_name = DISEASE_MAPPING[disease_code]
        confidence = float(probabilities[idx])
        
        results.append({
            'code': disease_code,
            'name': disease_name,
            'confidence': confidence,
            'percentage': f"{confidence * 100:.2f}%"
        })
    
    return results


def plot_predictions(predictions):
    """Create interactive prediction chart"""
    df = pd.DataFrame(predictions)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['confidence'],
            y=df['name'],
            orientation='h',
            marker=dict(
                color=df['confidence'],
                colorscale='Tealgrn',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            text=df['percentage'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Top 5 Disease Predictions",
        xaxis_title="Confidence Score",
        yaxis_title="Disease",
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_confidence_gauge(confidence):
    """Create gauge chart for confidence level"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def get_clinical_recommendation(prediction):
    """Generate clinical recommendation based on prediction"""
    confidence = prediction['confidence']
    disease_name = prediction['name']
    
    if confidence >= 0.8:
        severity = "HIGH RISK"
        color = "error"
        recommendation = f"""
        **🚨 High Confidence Detection ({prediction['percentage']})**
        
        **Detected Condition:** {disease_name}
        
        **Recommendation:** 
        - Immediate consultation with an ophthalmologist is strongly recommended
        - Schedule appointment within 24-48 hours
        - Bring this screening result to your appointment
        - Avoid delay in seeking professional medical evaluation
        
        **Next Steps:**
        1. Contact your eye care provider immediately
        2. Prepare your medical history
        3. Schedule comprehensive eye examination
        """
    elif confidence >= 0.5:
        severity = "MODERATE RISK"
        color = "warning"
        recommendation = f"""
        **⚠️ Moderate Confidence Detection ({prediction['percentage']})**
        
        **Detected Condition:** {disease_name}
        
        **Recommendation:**
        - Schedule appointment with an eye care professional
        - Recommended within 1-2 weeks
        - Monitor for any vision changes
        - Consider retinal imaging
        
        **Next Steps:**
        1. Book eye examination appointment
        2. Monitor symptoms daily
        3. Seek immediate care if symptoms worsen
        """
    elif confidence >= 0.3:
        severity = "LOW RISK"
        color = "info"
        recommendation = f"""
        **ℹ️ Low Confidence Detection ({prediction['percentage']})**
        
        **Potential Condition:** {disease_name}
        
        **Recommendation:**
        - Routine eye examination recommended
        - Schedule within 1-3 months
        - Continue regular eye health monitoring
        - No immediate concern indicated
        
        **Next Steps:**
        1. Schedule routine eye check-up
        2. Maintain healthy eye care practices
        3. Return for follow-up screening
        """
    else:
        severity = "VERY LOW RISK"
        color = "success"
        recommendation = f"""
        **✅ Very Low Detection ({prediction['percentage']})**
        
        **Condition:** {disease_name}
        
        **Recommendation:**
        - Continue regular eye health monitoring
        - Annual eye examinations recommended
        - Maintain healthy lifestyle
        - No specific action required
        
        **Next Steps:**
        1. Schedule annual eye exam
        2. Practice good eye health habits
        3. Return if symptoms develop
        """
    
    return severity, recommendation, color


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">👁️ Explainable Retinal Disease Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Medical Image Analysis with GPU Acceleration</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/ophthalmology.png", width=100)
        st.title("🔬 Analysis Settings")
        
        # GPU Status
        device_status = "🟢 GPU Available" if torch.cuda.is_available() else "🔵 CPU Mode"
        st.info(f"**Device:** {device_status}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.success(f"**GPU:** {gpu_name}\n\n**Memory:** {gpu_memory:.2f} GB")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Configuration")
        top_k = st.slider("Top predictions to show", 3, 10, 5)
        show_explainability = st.checkbox("Show explainability features", value=EXPLAINABILITY_AVAILABLE)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)
        
        st.divider()
        
        # Model info
        st.subheader("📊 Model Information")
        
        # Load and display metadata
        metadata = load_model_metadata()
        if metadata:
            model_info = metadata.get('model_info', {})
            perf = metadata.get('performance', {})
            expl = metadata.get('explainability', {})
            
            info_text = f"""
        **Architecture:** {model_info.get('name', 'SceneGraphTransformer')}
        
        **Diseases:** {model_info.get('num_classes', 45)} retinal conditions
        
        **Input:** 224x224 RGB fundus images
        
        **Performance:**
        - F1: {perf.get('f1_score', 0):.4f}
        - AUC: {perf.get('auc_roc', 0):.4f}
        - Time: {perf.get('inference_time_ms', 0):.1f}ms
        
        **Explainability:** {'✓ Enabled' if expl.get('enabled') else '✗ Disabled'}
        """
        else:
            info_text = """
        **Architecture:** SceneGraphTransformer
        
        **Diseases:** 45 retinal conditions
        
        **Input:** 224x224 RGB fundus images
        
        **Optimization:** INT8 Quantized
        """
        
        st.info(info_text)
        
        st.divider()
        
        # Explainability Frameworks Status
        st.subheader("🔍 Explainability Tools")
        
        frameworks_status = []
        
        if GRADCAM_AVAILABLE:
            frameworks_status.append(f"✅ GradCAM ({GRADCAM_LIBRARY})")
        else:
            frameworks_status.append("❌ GradCAM")
        
        if CAPTUM_AVAILABLE:
            frameworks_status.append("✅ Captum (Integrated Gradients)")
        else:
            frameworks_status.append("❌ Captum")
        
        if SHAP_AVAILABLE:
            frameworks_status.append("✅ SHAP")
        else:
            frameworks_status.append("❌ SHAP")
        
        if LIME_AVAILABLE:
            frameworks_status.append("✅ LIME")
        else:
            frameworks_status.append("❌ LIME")
        
        if ELI5_AVAILABLE:
            frameworks_status.append("✅ ELI5")
        else:
            frameworks_status.append("❌ ELI5")
        
        frameworks_text = "\n".join(frameworks_status)
        
        total_available = sum([GRADCAM_AVAILABLE, CAPTUM_AVAILABLE, SHAP_AVAILABLE, LIME_AVAILABLE, ELI5_AVAILABLE])
        
        if total_available >= 4:
            st.success(f"**{total_available}/5 frameworks available**\n\n{frameworks_text}")
        elif total_available >= 2:
            st.info(f"**{total_available}/5 frameworks available**\n\n{frameworks_text}")
        else:
            st.warning(f"**{total_available}/5 frameworks available**\n\n{frameworks_text}\n\n💡 Install missing packages to enable all explainability features.")
        
        st.divider()
        
        # Medical disclaimer
        st.warning("""
        **⚠️ Medical Disclaimer**
        
        This is a screening tool for educational and research purposes. 
        
        **NOT A REPLACEMENT** for professional medical diagnosis.
        
        Always consult qualified healthcare professionals.
        """)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("❌ Cannot proceed without model. Please check configuration.")
        return
    
    # Load explainer if enabled
    explainer = None
    if show_explainability and EXPLAINABILITY_AVAILABLE:
        explainer = load_explainer(model, device)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Retinal Fundus Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Image source selection
            image_source = st.radio(
                "Select Image Source:",
                options=["📁 Upload from File", "📷 Capture from Camera"],
                horizontal=True
            )
            
            uploaded_file = None
            camera_image = None
            
            if image_source == "📁 Upload from File":
                uploaded_file = st.file_uploader(
                    "Choose a retinal image (JPG, PNG, JPEG)",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a clear retinal fundus photograph"
                )
                
                if uploaded_file is not None:
                    # Display uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Retinal Image", use_container_width=True)
                    
                    # Image info
                    st.info(f"""
                    **Image Details:**
                    - Format: {image.format}
                    - Size: {image.size}
                    - Mode: {image.mode}
                    """)
            
            else:  # Camera capture
                camera_image = st.camera_input(
                    "📷 Capture retinal image from camera",
                    help="Position the retinal fundus camera and capture the image"
                )
                
                if camera_image is not None:
                    # Display captured image
                    image = Image.open(camera_image)
                    st.image(image, caption="Captured Retinal Image", use_container_width=True)
                    
                    # Image info
                    st.info(f"""
                    **Image Details:**
                    - Source: Camera Capture
                    - Size: {image.size}
                    - Mode: {image.mode}
                    """)
                    uploaded_file = camera_image  # Treat camera image same as uploaded
        
        with col2:
            if uploaded_file is not None:
                # Analyze button
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("🧠 AI Analysis in progress..."):
                        # Preprocess
                        progress_bar = st.progress(0)
                        progress_bar.progress(25, text="Preprocessing image...")
                        
                        img_tensor = preprocess_image(image)
                        
                        # Check if comprehensive explainability is available
                        use_comprehensive = show_explainability and explainer is not None
                        
                        if use_comprehensive:
                            # Use comprehensive clinical analysis
                            progress_bar.progress(50, text="Running comprehensive analysis...")
                            start_time = time.time()
                            
                            comprehensive_results = get_comprehensive_analysis(
                                explainer, img_tensor, device, top_k
                            )
                            
                            inference_time = time.time() - start_time
                            
                            if comprehensive_results is not None:
                                progress_bar.progress(75, text="Generating clinical insights...")
                                
                                # Store comprehensive results
                                st.session_state['comprehensive_results'] = comprehensive_results
                                st.session_state['predictions'] = comprehensive_results['predictions']
                                st.session_state['clinical_insights'] = comprehensive_results['clinical_insights']
                                st.session_state['explainability_data'] = comprehensive_results['explainability']
                                st.session_state['image'] = image
                                st.session_state['inference_time'] = inference_time
                                st.session_state['use_comprehensive'] = True
                                
                                progress_bar.progress(100, text="Complete!")
                                time.sleep(0.5)
                                progress_bar.empty()
                                
                                st.success(f"✅ Comprehensive analysis complete in {inference_time:.3f} seconds!")
                                st.balloons()
                        else:
                            # Standard prediction
                            progress_bar.progress(50, text="Running inference...")
                            start_time = time.time()
                            probabilities = predict(model, device, img_tensor)
                            inference_time = time.time() - start_time
                            
                            if probabilities is not None:
                                progress_bar.progress(75, text="Generating results...")
                                
                                # Get top predictions
                                predictions = get_top_predictions(probabilities, top_k)
                                
                                # Store in session state
                                st.session_state['predictions'] = predictions
                                st.session_state['image'] = image
                                st.session_state['inference_time'] = inference_time
                                st.session_state['probabilities'] = probabilities
                                st.session_state['use_comprehensive'] = False
                                
                                progress_bar.progress(100, text="Complete!")
                                time.sleep(0.5)
                                progress_bar.empty()
                                
                                st.success(f"✅ Analysis complete in {inference_time:.3f} seconds!")
                                st.balloons()
    
    with tab2:
        st.header("Analysis Results Dashboard")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            inference_time = st.session_state['inference_time']
            use_comprehensive = st.session_state.get('use_comprehensive', False)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            # Handle both comprehensive and standard predictions
            if use_comprehensive:
                top_disease = predictions[0]['disease']
                top_confidence = predictions[0]['confidence_percentage']
            else:
                top_disease = predictions[0]['name']
                top_confidence = predictions[0]['percentage']
            
            with col1:
                st.metric("Top Prediction", top_disease, delta=top_confidence)
            
            with col2:
                st.metric("Confidence", top_confidence, delta=None)
            
            with col3:
                st.metric("Inference Time", f"{inference_time:.3f}s", delta=None)
            
            with col4:
                device_name = "GPU" if torch.cuda.is_available() else "CPU"
                st.metric("Device", device_name, delta=None)
            
            st.divider()
            
            # Show comprehensive clinical insights if available
            if use_comprehensive and 'clinical_insights' in st.session_state:
                clinical_insights = st.session_state['clinical_insights']
                
                # Overall assessment banner
                overall = clinical_insights['overall_assessment']
                st.subheader("🏥 Clinical Assessment Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Primary Diagnosis", overall['top_diagnosis'])
                with col2:
                    reliability = overall['reliability_score']
                    reliability_color = "🟢" if reliability >= 75 else "🟡" if reliability >= 50 else "🔴"
                    st.metric("Reliability Score", f"{reliability_color} {reliability:.1f}/100")
                with col3:
                    action = overall['clinical_action_required']
                    st.metric("Action Priority", action)
                
                st.divider()
                
                # Clinical Recommendations
                st.subheader("📋 Clinical Recommendations")
                recommendations = clinical_insights['recommendations']
                
                for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                    priority_emoji = {
                        'Urgent': '🔴',
                        'High': '🟠',
                        'Medium': '🟡',
                        'Low': '🟢',
                        'Review': '🔵'
                    }.get(rec['priority'], '⚪')
                    
                    with st.expander(f"{priority_emoji} {rec['type']} - {rec['priority']} Priority", expanded=(i==1)):
                        st.write(f"**Recommendation:** {rec['recommendation']}")
                        st.write(f"**Action Required:** {rec['action']}")
                
                st.divider()
                
                # Uncertainty Metrics
                st.subheader("📊 Reliability & Uncertainty Analysis")
                uncertainty = clinical_insights['uncertainty_metrics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reliability Score", f"{uncertainty['reliability_score']:.1f}/100")
                    st.caption(uncertainty['interpretation']['reliability'] + " reliability")
                
                with col2:
                    st.metric("Confidence Gap", f"{uncertainty['confidence_gap']:.2f}")
                    st.caption("Lower is better")
                
                with col3:
                    st.metric("Prediction Entropy", f"{uncertainty['entropy']:.3f}")
                    st.caption("Model uncertainty measure")
                
                # Multi-disease interactions
                if clinical_insights.get('multi_disease_interactions'):
                    st.divider()
                    st.subheader("🔗 Multi-Disease Interactions Detected")
                    interactions = clinical_insights['multi_disease_interactions']
                    
                    for interaction in interactions:
                        st.info(f"**{interaction['type']}**: {', '.join(interaction['diseases'])}\n\n"
                               f"*{interaction['note']}*")
                
                st.divider()
            
            # Main results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Prediction Confidence Scores")
                
                # Format predictions for plotting
                if use_comprehensive:
                    plot_data = [
                        {
                            'name': p['disease'],
                            'percentage': p['confidence_percentage'],
                            'confidence': p['confidence_score'],
                            'code': p['disease'][:10]  # Shortened code
                        }
                        for p in predictions
                    ]
                else:
                    plot_data = predictions
                
                fig = plot_predictions(plot_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Primary Detection")
                if use_comprehensive:
                    confidence_val = predictions[0]['confidence_score']
                else:
                    confidence_val = predictions[0]['confidence']
                fig_gauge = plot_confidence_gauge(confidence_val)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.divider()
            
            # Clinical recommendations (standard mode fallback)
            if not use_comprehensive:
                st.subheader("🏥 Clinical Assessment")
                
                top_pred = predictions[0]
                severity, recommendation, color = get_clinical_recommendation(top_pred)
                
                if color == "error":
                    st.error(f"**Severity Level:** {severity}")
                elif color == "warning":
                    st.warning(f"**Severity Level:** {severity}")
                elif color == "info":
                    st.info(f"**Severity Level:** {severity}")
                else:
                    st.success(f"**Severity Level:** {severity}")
                
                st.markdown(recommendation)
                
                st.divider()
            
            # Detailed predictions table
            st.subheader("📋 Detailed Predictions")
            
            if use_comprehensive:
                df_data = [
                    {
                        'Disease': p['disease'],
                        'Confidence': p['confidence_percentage'],
                        'Rank': p['rank'],
                        'Level': p['confidence_level'],
                        'Interpretation': p['clinical_interpretation']
                    }
                    for p in predictions
                ]
            else:
                df_data = [
                    {
                        'Disease': p['name'],
                        'Confidence': p['percentage'],
                        'Code': p['code']
                    }
                    for p in predictions
                ]
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Explainability section
            if show_explainability and explainer is not None and 'image' in st.session_state:
                st.divider()
                st.subheader("🔍 Explainability Analysis")
                
                # Show available frameworks status
                available_frameworks = []
                if GRADCAM_AVAILABLE:
                    available_frameworks.append(f"**GradCAM** ({GRADCAM_LIBRARY})")
                if CAPTUM_AVAILABLE:
                    available_frameworks.append("**Captum** (Integrated Gradients, Saliency Maps)")
                if SHAP_AVAILABLE:
                    available_frameworks.append("**SHAP** (SHapley Additive exPlanations)")
                if LIME_AVAILABLE:
                    available_frameworks.append("**LIME** (Local Interpretable Model-agnostic Explanations)")
                if ELI5_AVAILABLE:
                    available_frameworks.append("**ELI5** (Explain Like I'm 5)")
                
                if available_frameworks:
                    st.success(f"✅ **Available Explainability Frameworks:**\n\n" + "\n- ".join([""] + available_frameworks))
                else:
                    st.warning("⚠️ No explainability frameworks are currently available. Install the required packages to enable these features.")
                
                # Check if GradCAM is available
                if not GRADCAM_AVAILABLE:
                    st.error("""
                    ❌ **GradCAM Not Available**
                    
                    The `grad-cam` package is not installed in this environment.
                    
                    **To enable GradCAM features:**
                    
                    1. Install the package:
                       - `pip install grad-cam>=1.5.2`
                    2. Restart the application
                    
                    **Other available frameworks:**
                    - `pip install captum>=0.6.0` - Integrated Gradients, Saliency Maps
                    - `pip install shap>=0.42.0` - SHAP explanations
                    - `pip install lime>=0.2.0.1` - LIME explanations
                    - `pip install eli5>=0.13.0` - ELI5 explanations
                    """)
                else:
                    # GradCAM Visualization
                    with st.expander("🎯 View GradCAM Heatmap", expanded=True):
                        st.info(f"🔍 Using **{GRADCAM_LIBRARY}** for visualization")
                        with st.spinner("Generating attention heatmap..."):
                            try:
                                # Generate GradCAM
                                img_tensor = preprocess_image(st.session_state['image'])
                                
                                # Get target class - handle both comprehensive and standard modes
                                if use_comprehensive:
                                    # In comprehensive mode, find the disease name in DISEASE_CODES
                                    top_disease = predictions[0]['disease']
                                    # Try to find matching disease code
                                    target_class = None
                                    for idx, code in enumerate(DISEASE_CODES):
                                        if DISEASE_MAPPING[code] == top_disease or code in top_disease:
                                            target_class = idx
                                            break
                                    if target_class is None:
                                        # Fallback: use model prediction directly
                                        with torch.no_grad():
                                            output = model(img_tensor.to(device))
                                            probs = torch.sigmoid(output).cpu().numpy()[0]
                                            target_class = int(np.argmax(probs))
                                else:
                                    # Standard mode: get disease code
                                    target_class = DISEASE_CODES.index(predictions[0]['code'])
                                
                                heatmap = explainer.generate_gradcam(
                                    img_tensor.to(device),
                                    target_class=target_class
                                )
                                
                                # Display heatmap
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(st.session_state['image'], caption="Original Image", use_container_width=True)
                                with col2:
                                    st.image(heatmap, caption=f"GradCAM: {predictions[0].get('disease', predictions[0].get('name', 'Top Prediction'))}", use_container_width=True)
                                
                                st.info("""
                                **🔍 Heatmap Interpretation Guide:**
                                
                                - **🔴 Red/Hot Regions:** High importance areas where the AI focused for diagnosis
                                - **🟡 Yellow/Warm Regions:** Moderate importance areas contributing to the decision
                                - **🔵 Blue/Cool Regions:** Lower relevance areas with minimal impact
                                
                                The heatmap shows which parts of the retinal image influenced the AI's prediction most strongly.
                                Clinicians should verify that highlighted regions align with actual pathological features.
                                """)
                            
                            except Exception as e:
                                st.error(f"❌ Could not generate heatmap: {str(e)}")
                                st.info("💡 **Troubleshooting:** Ensure grad-cam is installed in the container.")
                    
                    # Captum - Integrated Gradients
                    if CAPTUM_AVAILABLE:
                        with st.expander("📊 Integrated Gradients (Captum)", expanded=False):
                            st.info("""
                            **About Integrated Gradients:**
                            
                            Integrated Gradients attributes prediction to input features by integrating gradients 
                            along a path from a baseline to the input. This method provides pixel-level importance scores.
                            
                            **Use Case:** Understanding which pixels contribute most to the model's prediction.
                            """)
                            
                            if st.button("Generate Integrated Gradients", key="ig_btn"):
                                with st.spinner("Computing Integrated Gradients..."):
                                    try:
                                        img_tensor = preprocess_image(st.session_state['image'])
                                        
                                        # Get results from explainer
                                        if use_comprehensive:
                                            target_class = None
                                            for idx, code in enumerate(DISEASE_CODES):
                                                if DISEASE_MAPPING[code] == predictions[0]['disease']:
                                                    target_class = idx
                                                    break
                                        else:
                                            target_class = DISEASE_CODES.index(predictions[0]['code'])
                                        
                                        ig_results = explainer.explain_integrated_gradients(
                                            img_tensor.to(device),
                                            target_classes=[target_class] if target_class is not None else None
                                        )
                                        
                                        if 'error' not in ig_results:
                                            for disease, attrs in ig_results.items():
                                                st.write(f"**{disease}:**")
                                                st.json(attrs['attribution_summary'])
                                        else:
                                            st.error(f"Error: {ig_results['error']}")
                                    except Exception as e:
                                        st.error(f"❌ Could not generate Integrated Gradients: {str(e)}")
                    
                    # SHAP Explanations
                    if SHAP_AVAILABLE:
                        with st.expander("🎲 SHAP Explanations", expanded=False):
                            st.info("""
                            **About SHAP (SHapley Additive exPlanations):**
                            
                            SHAP values explain model predictions by computing the contribution of each feature 
                            based on game theory. It provides consistent and locally accurate explanations.
                            
                            **Use Case:** Understanding feature importance and model behavior for individual predictions.
                            """)
                            
                            st.warning("""
                            ⚠️ **Note:** SHAP explanations for deep learning models can be computationally intensive.
                            Implementation requires model-specific adapters for retinal images.
                            """)
                    
                    # LIME Explanations
                    if LIME_AVAILABLE:
                        with st.expander("🔬 LIME Explanations", expanded=False):
                            st.info("""
                            **About LIME (Local Interpretable Model-agnostic Explanations):**
                            
                            LIME explains predictions by approximating the model locally with an interpretable model.
                            It perturbs the input and observes prediction changes to understand model behavior.
                            
                            **Use Case:** Model-agnostic explanations showing which image regions affect predictions.
                            """)
                            
                            st.warning("""
                            ⚠️ **Note:** LIME for medical images requires careful segmentation and perturbation strategies.
                            Implementation requires additional configuration for retinal fundus images.
                            """)
                    
                    # ELI5 Explanations
                    if ELI5_AVAILABLE:
                        with st.expander("📝 ELI5 Explanations", expanded=False):
                            st.info("""
                            **About ELI5 (Explain Like I'm 5):**
                            
                            ELI5 provides simple, human-readable explanations of machine learning models.
                            It supports various model types and can generate text and visual explanations.
                            
                            **Use Case:** Simplified explanations for non-technical stakeholders.
                            """)
                            
                            st.warning("""
                            ⚠️ **Note:** ELI5 is most effective with traditional ML models.
                            Deep learning support is limited and requires custom adapters.
                            """)
                    
                    # Framework Comparison Guide
                    with st.expander("📚 Explainability Framework Comparison", expanded=False):
                        comparison_df = pd.DataFrame({
                            'Framework': ['GradCAM', 'Captum (IG)', 'SHAP', 'LIME', 'ELI5'],
                            'Type': ['Visual', 'Visual + Numerical', 'Numerical', 'Visual + Numerical', 'Numerical'],
                            'Speed': ['Fast', 'Medium', 'Slow', 'Slow', 'Fast'],
                            'Medical Imaging': ['Excellent', 'Good', 'Good', 'Good', 'Limited'],
                            'Best For': [
                                'Quick visual insights',
                                'Detailed attribution analysis',
                                'Feature importance',
                                'Model-agnostic explanations',
                                'Simple text explanations'
                            ]
                        })
                        
                        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                        
                        st.info("""
                        **Recommendation for Retinal Screening:**
                        
                        1. **GradCAM** (Primary) - Best for visualizing which retinal regions influenced diagnosis
                        2. **Integrated Gradients** (Secondary) - Provides detailed pixel-level attribution
                        3. **SHAP/LIME** (Advanced) - For deeper model analysis and research purposes
                        4. **ELI5** (Documentation) - For generating simple reports
                        """)
        
        else:
            st.info("👆 Upload and analyze an image in the 'Upload & Analyze' tab to see results here.")
    
    with tab3:
        st.header("About This Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Purpose")
            st.write("""
            This application provides AI-powered screening for retinal diseases using 
            state-of-the-art deep learning technology with explainable AI features.
            """)
            
            st.subheader("🔬 Technology Stack")
            st.write("""
            - **Deep Learning:** PyTorch with CUDA GPU acceleration
            - **Architecture:** SceneGraphTransformer
            - **Optimization:** INT8 Quantization for efficient inference
            - **Explainability:** GradCAM, Captum, SHAP, LIME, ELI5
            - **Interface:** Streamlit with interactive visualizations
            
            **Explainability Frameworks:**
            - **GradCAM** - Visual attention heatmaps
            - **Captum** - Integrated Gradients, Saliency Maps
            - **SHAP** - SHapley Additive exPlanations
            - **LIME** - Local Interpretable Model-agnostic Explanations
            - **ELI5** - Simple, human-readable explanations
            """)
        
        with col2:
            st.subheader("📋 Detectable Conditions")
            st.write("""
            The model can screen for 45 different retinal diseases including:
            
            **Major Conditions:**
            - Diabetic Retinopathy
            - Age-Related Macular Degeneration
            - Macular Hole
            - Retinal Vein/Artery Occlusions
            - Glaucoma indicators
            
            **And 40 other retinal conditions...**
            """)
            
            st.subheader("⚡ Performance")
            st.write(f"""
            - **Inference Time:** ~200ms on GPU
            - **Model Size:** 119 MB (quantized)
            - **Input:** 224x224 RGB images
            - **Output:** 45 disease probabilities
            """)
        
        st.divider()
        
        st.subheader("📚 How to Use")
        st.write("""
        1. **Upload** a clear retinal fundus photograph
        2. **Analyze** the image using the AI model
        3. **Review** the top predictions and confidence scores
        4. **Read** clinical recommendations
        5. **Explore** explainability features (if enabled)
        6. **Consult** with healthcare professionals for proper diagnosis
        """)
        
        st.divider()
        
        st.error("""
        **⚠️ IMPORTANT MEDICAL DISCLAIMER**
        
        This application is for screening, educational, and research purposes only.
        It is NOT intended to replace professional medical diagnosis, treatment, or advice.
        
        **DO NOT** use this tool as the sole basis for medical decisions.
        
        **ALWAYS** consult with qualified healthcare professionals, including ophthalmologists
        and retinal specialists, for proper diagnosis, treatment recommendations, and medical care.
        
        The developers and operators of this tool assume no liability for medical decisions
        made based on its output.
        """)


if __name__ == "__main__":
    main()
