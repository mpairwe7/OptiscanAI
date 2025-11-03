"""
Mobile-Optimized Model Explainability Framework
Lightweight version for deployment with retinal disease classification models
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import warnings

# Suppress sklearn compatibility warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Fix numpy compatibility for sklearn (sklearn expects int32/int64 for labels, not int8)
# Note: For model quantization (INT8), this doesn't affect torch.qint8 used in the models
if not hasattr(np, 'int'):
    np.int = np.int32
if not hasattr(np, 'float'):
    np.float = np.float64

# ============================================================================
# DISEASE NAME MAPPING - Short Form to Full Name
# ============================================================================
DISEASE_NAME_MAPPING = {
    'N': 'Normal (No Disease)',
    'D': 'Diabetic Retinopathy',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-Related Macular Degeneration (AMD)',
    'H': 'Hypertensive Retinopathy',
    'M': 'Myopia',
    'O': 'Other Diseases/Abnormalities',
    'DN': 'Diabetic Neuropathy',
    'MH': 'Macular Hole',
    'ODC': 'Optic Disc Cupping',
    'TSLN': 'Tessellation',
    'ARMD': 'Age-Related Macular Degeneration',
    'MYA': 'Myopic Retinopathy',
    'BRVO': 'Branch Retinal Vein Occlusion',
    'HTN': 'Hypertensive Retinopathy',
    'CRS': 'Chorioretinal Scar',
    'ODC': 'Optic Disc Coloboma',
    'DN': 'Diabetic Neuropathy',
    'MH': 'Macular Hole',
    'MYA': 'Myopic Retinopathy',
    'ARMD': 'Age-Related Macular Degeneration',
    'ODC': 'Optic Disc Cupping/Coloboma',
    'BRVO': 'Branch Retinal Vein Occlusion',
    'HTN': 'Hypertensive Retinopathy',
    'TSLN': 'Tessellation (Myopic Changes)',
    'MH': 'Macular Hole',
    'CRS': 'Chorioretinal Scar',
    'RS': 'Retinitis',
    'EDN': 'Epiretinal Membrane',
    'RPEC': 'Retinal Pigment Epithelial Changes',
    'MHL': 'Macular Hole Lamellar',
    'LS': 'Laser Scars',
    'MS': 'Macular Scars',
    'CSR': 'Central Serous Retinopathy',
    'ODC': 'Optic Disc Cupping',
    'CRVO': 'Central Retinal Vein Occlusion',
    'TV': 'Tortuous Vessels',
    'AH': 'Asteroid Hyalosis',
    'ODP': 'Optic Disc Pallor',
    'ODE': 'Optic Disc Edema',
    'ST': 'Optociliary Shunt',
    'AION': 'Anterior Ischemic Optic Neuropathy',
    'PT': 'Parafoveal Telangiectasia',
    'RT': 'Retinal Traction',
    'RS': 'Retinitis/Retinal Scarring',
    'CWS': 'Cotton Wool Spots',
    'CB': 'Coats Disease/Retinal Exudates',
    'ODPM': 'Optic Disc Pit Maculopathy',
    'PRH': 'Preretinal Hemorrhage',
    'MNF': 'Myelinated Nerve Fibers',
    'HR': 'Hemorrhagic Retinopathy',
    'CRAO': 'Central Retinal Artery Occlusion',
    'TD': 'Tapetal Degeneration',
    'CME': 'Cystoid Macular Edema',
    'PTCR': 'Post-Traumatic Chorioretinopathy',
    'CF': 'Choroidal Folds',
    'VH': 'Vitreous Hemorrhage',
    'MCA': 'Macroaneurysm',
    'VS': 'Vessel Sheathing',
    'BRAO': 'Branch Retinal Artery Occlusion',
    'PLQ': 'Peripapillary Lesions/Drusen',
    'HPED': 'Hemorrhagic Pigment Epithelial Detachment',
    'CL': 'Choroidal Lesion'
}

def get_full_disease_name(short_name):
    """
    Convert short disease name to full descriptive name
    
    Args:
        short_name: Short disease abbreviation
    
    Returns:
        Full disease name or original if mapping not found
    """
    return DISEASE_NAME_MAPPING.get(short_name, short_name)

# Conditional imports with graceful fallbacks
try:
    from captum.attr import IntegratedGradients, Saliency, GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

# Import grad-cam library (pytorch-grad-cam package)
GRADCAM_AVAILABLE = False
GRADCAM_LIBRARY = None

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
    GRADCAM_LIBRARY = 'pytorch_grad_cam'
except ImportError:
    GRADCAM_AVAILABLE = False
    GRADCAM_LIBRARY = None


class ModelWrapper(torch.nn.Module):
    """
    Wrapper for models that return tuples to ensure single tensor output
    This is needed for compatibility with pytorch-grad-cam
    """
    def __init__(self, model):
        super().__init__()
        # Store reference to wrapped model without copying modules to avoid conflicts
        self._wrapped_model = model
    
    def forward(self, x):
        output = self._wrapped_model(x)
        # If model returns tuple, extract first element (main output)
        if isinstance(output, tuple):
            return output[0]
        return output
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._wrapped_model, name)


class ModelExplainer:
    """
    Lightweight model explainability for deployment
    
    Supported methods:
    - Grad-CAM (multiple variants)
    - Integrated Gradients
    - SHAP (GradientSHAP)
    - Saliency Maps
    """
    
    def __init__(self, model, device='cpu', disease_names=None, mobile_mode=False):
        """
        Args:
            model: PyTorch model
            device: 'cpu' or 'cuda'
            disease_names: List of disease class names (can be short forms)
            mobile_mode: If True, use only lightweight methods
        """
        self.original_model = model
        # Wrap model to handle tuple outputs for GradCAM compatibility
        self.model = ModelWrapper(model)
        self.device = device
        # Convert short names to full names
        if disease_names:
            self.disease_names_short = disease_names
            self.disease_names = [get_full_disease_name(name) for name in disease_names]
        else:
            self.disease_names_short = [f"Disease_{i}" for i in range(45)]
            self.disease_names = self.disease_names_short
        self.mobile_mode = mobile_mode
        self.model.eval()
        self.target_layer = self._get_target_layer()
        
    def _get_target_layer(self):
        """Find appropriate layer for CAM methods"""
        # Access the wrapped model
        model = self.model._wrapped_model if hasattr(self.model, '_wrapped_model') else self.model
        
        if hasattr(model, 'visual_encoder'):
            if hasattr(model.visual_encoder, 'blocks'):
                return model.visual_encoder.blocks[-1]
        
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.MultiheadAttention)):
                return module
        return None
    
    def explain_gradcam(self, image, target_classes=None, method='GradCAM'):
        """Generate Grad-CAM visualizations"""
        if not GRADCAM_AVAILABLE or self.target_layer is None:
            return {'error': 'GradCAM not available'}
        
        with torch.no_grad():
            output = self.model(image)
            # Handle models that return tuples (e.g., aux outputs)
            if isinstance(output, tuple):
                output = output[0]
            predictions = torch.sigmoid(output).cpu().numpy()[0]
        
        if target_classes is None:
            target_classes = np.argsort(predictions)[-3:][::-1]
        
        img_np = image.cpu().numpy()[0].transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        cam_method = {'GradCAM': GradCAM, 'GradCAMPlusPlus': GradCAMPlusPlus, 
                     'ScoreCAM': ScoreCAM}.get(method, GradCAM)
        
        try:
            # Note: use_cuda parameter removed in pytorch-grad-cam >= 1.4.0
            cam = cam_method(model=self.model, target_layers=[self.target_layer])
            
            results = {}
            for class_idx in target_classes:
                targets = [ClassifierOutputTarget(class_idx)]
                grayscale_cam = cam(input_tensor=image, targets=targets)[0]
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                
                results[self.disease_names[class_idx]] = {
                    'cam': grayscale_cam.tolist(),
                    'prediction': float(predictions[class_idx])
                }
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def explain_integrated_gradients(self, image, target_classes=None, n_steps=25):
        """Generate Integrated Gradients explanations"""
        if not CAPTUM_AVAILABLE:
            return {'error': 'Captum not available'}
        
        with torch.no_grad():
            output = self.model(image)
            # Handle models that return tuples (e.g., aux outputs)
            if isinstance(output, tuple):
                output = output[0]
            predictions = torch.sigmoid(output).cpu().numpy()[0]
        
        if target_classes is None:
            target_classes = np.argsort(predictions)[-2:][::-1]
        
        ig = IntegratedGradients(self.model)
        results = {}
        
        for class_idx in target_classes:
            attributions = ig.attribute(image, target=class_idx, n_steps=n_steps)
            attr_map = attributions.cpu().numpy()[0].transpose(1, 2, 0)
            attr_map = np.abs(attr_map).sum(axis=2)
            
            results[self.disease_names[class_idx]] = {
                'attribution_summary': {
                    'mean': float(attr_map.mean()),
                    'max': float(attr_map.max()),
                    'min': float(attr_map.min())
                },
                'prediction': float(predictions[class_idx])
            }
        
        return results
    
    def _assess_confidence_level(self, confidence_score):
        """Categorize confidence level with clinical interpretation"""
        if confidence_score >= 0.90:
            return {
                'level': 'Very High',
                'interpretation': 'Strong evidence for diagnosis',
                'reliability': 'High reliability - consider as primary diagnosis'
            }
        elif confidence_score >= 0.75:
            return {
                'level': 'High',
                'interpretation': 'Likely diagnosis with good evidence',
                'reliability': 'Good reliability - recommend clinical confirmation'
            }
        elif confidence_score >= 0.60:
            return {
                'level': 'Moderate',
                'interpretation': 'Possible diagnosis requiring review',
                'reliability': 'Moderate reliability - additional tests recommended'
            }
        elif confidence_score >= 0.45:
            return {
                'level': 'Low-Moderate',
                'interpretation': 'Weak evidence, consider differential diagnosis',
                'reliability': 'Lower confidence - clinical correlation essential'
            }
        else:
            return {
                'level': 'Low',
                'interpretation': 'Minimal evidence for this diagnosis',
                'reliability': 'Low confidence - likely not present'
            }
    
    def _generate_clinical_recommendations(self, predictions, top_indices):
        """Generate clinical recommendations based on predictions"""
        recommendations = []
        top_confidence = predictions[top_indices[0]]
        
        # High confidence - single disease
        if top_confidence >= 0.85 and (len(top_indices) < 2 or predictions[top_indices[1]] < 0.50):
            recommendations.append({
                'priority': 'High',
                'type': 'Primary Diagnosis',
                'recommendation': f'Strong evidence for {self.disease_names[top_indices[0]]}. Recommend confirmatory clinical examination and appropriate treatment protocol.',
                'action': 'Immediate clinical review and treatment planning'
            })
        
        # Multiple high confidence diseases
        elif len([i for i in top_indices if predictions[i] >= 0.60]) >= 2:
            high_conf_diseases = [self.disease_names[i] for i in top_indices if predictions[i] >= 0.60]
            recommendations.append({
                'priority': 'High',
                'type': 'Multiple Findings',
                'recommendation': f'Multiple retinal pathologies detected: {", ".join(high_conf_diseases[:3])}. Comprehensive ophthalmic evaluation recommended.',
                'action': 'Detailed examination for co-existing conditions'
            })
        
        # Moderate confidence
        elif 0.60 <= top_confidence < 0.85:
            recommendations.append({
                'priority': 'Moderate',
                'type': 'Probable Diagnosis',
                'recommendation': f'{self.disease_names[top_indices[0]]} likely present. Additional imaging or functional tests may improve diagnostic certainty.',
                'action': 'Consider OCT, fluorescein angiography, or visual field testing'
            })
        
        # Low confidence - normal or early stage
        else:
            recommendations.append({
                'priority': 'Low',
                'type': 'Monitoring',
                'recommendation': 'No significant pathology detected with high confidence. Consider routine follow-up or early-stage disease monitoring.',
                'action': 'Schedule regular screening, especially if risk factors present'
            })
        
        # Urgent findings detection
        urgent_conditions = ['Diabetic Retinopathy', 'Glaucoma', 'Retinal Detachment', 'Macular Degeneration']
        urgent_detected = [(i, predictions[i]) for i in top_indices[:3] 
                          if any(urgent in self.disease_names[i] for urgent in urgent_conditions) 
                          and predictions[i] >= 0.60]
        
        if urgent_detected:
            recommendations.insert(0, {
                'priority': 'Urgent',
                'type': 'Sight-Threatening Condition',
                'recommendation': f'Potential sight-threatening condition detected. Immediate ophthalmology referral recommended.',
                'action': 'Urgent specialist consultation within 24-48 hours'
            })
        
        return recommendations
    
    def _calculate_uncertainty_metrics(self, predictions):
        """Calculate uncertainty and reliability metrics"""
        # Entropy-based uncertainty
        epsilon = 1e-10
        entropy = -np.sum(predictions * np.log(predictions + epsilon) + 
                         (1 - predictions) * np.log(1 - predictions + epsilon))
        max_entropy = len(predictions) * np.log(2)
        normalized_entropy = entropy / max_entropy
        
        # Prediction variance
        prediction_variance = np.var(predictions)
        
        # Confidence gap (difference between top 2 predictions)
        sorted_preds = np.sort(predictions)[::-1]
        confidence_gap = sorted_preds[0] - sorted_preds[1] if len(sorted_preds) > 1 else sorted_preds[0]
        
        # Overall reliability score (0-100)
        reliability_score = (1 - normalized_entropy) * 50 + confidence_gap * 50
        
        return {
            'entropy': float(normalized_entropy),
            'variance': float(prediction_variance),
            'confidence_gap': float(confidence_gap),
            'reliability_score': float(reliability_score),
            'interpretation': {
                'entropy': 'Low' if normalized_entropy < 0.3 else ('Moderate' if normalized_entropy < 0.6 else 'High'),
                'reliability': 'High' if reliability_score >= 70 else ('Moderate' if reliability_score >= 50 else 'Low')
            }
        }
    
    def _detect_multi_disease_interactions(self, predictions, top_indices, threshold=0.50):
        """Detect and warn about co-existing conditions"""
        positive_diseases = [(i, predictions[i]) for i in range(len(predictions)) 
                            if predictions[i] >= threshold]
        
        interactions = []
        
        if len(positive_diseases) >= 2:
            disease_names = [self.disease_names[idx] for idx, _ in positive_diseases]
            
            # Common co-occurrences
            if any('Diabetic' in d for d in disease_names) and any('Macular' in d for d in disease_names):
                interactions.append({
                    'type': 'Common Co-occurrence',
                    'diseases': 'Diabetic Retinopathy + Macular Edema',
                    'note': 'Commonly co-exist. Macular edema is a frequent complication of diabetic retinopathy.',
                    'clinical_significance': 'Monitor both conditions closely'
                })
            
            if len(positive_diseases) >= 3:
                interactions.append({
                    'type': 'Multiple Pathologies',
                    'diseases': f'{len(positive_diseases)} conditions detected',
                    'note': f'Multiple retinal pathologies present: {", ".join([self.disease_names[i] for i, _ in positive_diseases[:3]])}',
                    'clinical_significance': 'Comprehensive evaluation needed for treatment prioritization'
                })
        
        return interactions
    
    def get_lightweight_explanation(self, image, top_k=3):
        """
        Get comprehensive lightweight explanations with clinical insights
        Returns JSON-serializable results including:
        - Predictions with confidence levels
        - Clinical recommendations
        - Uncertainty metrics
        - Multi-disease interactions
        - Visual explanations (GradCAM)
        """
        with torch.no_grad():
            output = self.model(image)
            # Handle models that return tuples (e.g., aux outputs)
            if isinstance(output, tuple):
                output = output[0]
            predictions = torch.sigmoid(output).cpu().numpy()[0]
        
        # Convert to Python list of native integers (not numpy int32/int64)
        # sklearn requires native Python int types, not numpy integer types
        top_indices = [int(i) for i in np.argsort(predictions)[-top_k:][::-1]]
        
        # Build detailed predictions with confidence assessments
        detailed_predictions = []
        for rank, idx in enumerate(top_indices):
            confidence_score = float(predictions[idx])
            confidence_assessment = self._assess_confidence_level(confidence_score)
            
            detailed_predictions.append({
                'disease': self.disease_names[idx],
                'confidence_score': confidence_score,
                'confidence_percentage': f'{confidence_score * 100:.1f}%',
                'rank': int(rank + 1),
                'confidence_level': confidence_assessment['level'],
                'clinical_interpretation': confidence_assessment['interpretation'],
                'reliability': confidence_assessment['reliability']
            })
        
        # Generate clinical recommendations
        recommendations = self._generate_clinical_recommendations(predictions, top_indices)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(predictions)
        
        # Detect multi-disease interactions
        interactions = self._detect_multi_disease_interactions(predictions, top_indices)
        
        # Build comprehensive results
        results = {
            'predictions': detailed_predictions,
            'clinical_insights': {
                'recommendations': recommendations,
                'uncertainty_metrics': uncertainty_metrics,
                'multi_disease_interactions': interactions if interactions else None,
                'overall_assessment': {
                    'top_diagnosis': self.disease_names[top_indices[0]],
                    'confidence': float(predictions[top_indices[0]]),
                    'reliability_score': uncertainty_metrics['reliability_score'],
                    'clinical_action_required': recommendations[0]['priority'] if recommendations else 'Review'
                }
            },
            'explainability': {},
            'metadata': {
                'total_diseases_evaluated': len(predictions),
                'diseases_above_threshold': int(np.sum(predictions >= 0.50)),
                'analysis_timestamp': 'runtime',
                'mobile_mode': self.mobile_mode
            }
        }
        
        # Add GradCAM if available
        if GRADCAM_AVAILABLE and self.target_layer is not None:
            results['explainability']['gradcam'] = self.explain_gradcam(
                image, target_classes=top_indices[:2], method='GradCAM'
            )
        
        # Add Integrated Gradients if available and not in mobile mode
        if CAPTUM_AVAILABLE and not self.mobile_mode:
            results['explainability']['integrated_gradients'] = self.explain_integrated_gradients(
                image, target_classes=top_indices[:2], n_steps=15
            )
        
        results['explainability']['methods_used'] = list(results['explainability'].keys())
        
        return results
    
    def generate_gradcam(self, image, target_class=None):
        """
        Generate GradCAM heatmap visualization for a specific target class
        
        Args:
            image: Input tensor [1, 3, H, W]
            target_class: Target class index (if None, uses top prediction)
        
        Returns:
            PIL Image of heatmap overlay
        """
        if not GRADCAM_AVAILABLE or self.target_layer is None:
            raise ValueError("GradCAM not available - ensure pytorch-grad-cam is installed")
        
        try:
            # Ensure image is on the correct device
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(image)}")
            
            image = image.to(self.device)
            
            # Get prediction if no target specified
            if target_class is None:
                with torch.no_grad():
                    output = self.original_model(image)
                    # Handle models that return tuples (e.g., aux outputs)
                    if isinstance(output, tuple):
                        output = output[0]
                    # Ensure output is a tensor
                    if not isinstance(output, torch.Tensor):
                        raise TypeError(f"Model output must be tensor, got {type(output)}")
                    predictions = torch.sigmoid(output).cpu().numpy()[0]
                    target_class = int(np.argmax(predictions))
            
            # Prepare image for visualization
            img_np = image.cpu().numpy()[0].transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_np = np.clip(img_np, 0, 1).astype(np.float32)
            
            # Create a wrapper specifically for GradCAM that ensures single tensor output
            class GradCAMModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    output = self.model(x)
                    if isinstance(output, tuple):
                        return output[0]
                    return output
            
            gradcam_model = GradCAMModelWrapper(self.original_model)
            gradcam_model.to(self.device)
            gradcam_model.eval()
            
            # Create GradCAM with the wrapped model
            # Note: use_cuda parameter removed in pytorch-grad-cam >= 1.4.0
            cam = GradCAM(
                model=gradcam_model,
                target_layers=[self.target_layer]
            )
            
            # Generate CAM
            targets = [ClassifierOutputTarget(target_class)]
            cam_output = cam(input_tensor=image, targets=targets)
            
            # Validate CAM output
            if not isinstance(cam_output, np.ndarray):
                raise TypeError(f"CAM output must be ndarray, got {type(cam_output)}")
            
            # Handle different output shapes
            if cam_output.ndim == 3:
                grayscale_cam = cam_output[0]
            elif cam_output.ndim == 2:
                grayscale_cam = cam_output
            else:
                raise ValueError(f"Unexpected CAM shape: {cam_output.shape}")
            
            # Ensure grayscale_cam is 2D
            if grayscale_cam.ndim != 2:
                raise ValueError(f"Grayscale CAM must be 2D, got shape {grayscale_cam.shape}")
            
            # Overlay on image
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # Validate visualization output
            if not isinstance(visualization, np.ndarray):
                raise TypeError(f"Visualization must be ndarray, got {type(visualization)}")
            
            # Convert to uint8 if needed
            if visualization.dtype != np.uint8:
                visualization = (visualization * 255).astype(np.uint8)
            
            # Convert to PIL Image
            return Image.fromarray(visualization)
            
        except Exception as e:
            raise RuntimeError(f"GradCAM generation failed: {str(e)}")
    
    def save_explanation_report(self, image, save_path='explanation.json'):
        """Generate and save lightweight explanation report"""
        results = self.get_lightweight_explanation(image)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        return results


def load_model_with_explainability(model_path, model_class, device='cpu'):
    """
    Load model with explainability support
    
    Args:
        model_path: Path to .pth model file
        model_class: Model class to instantiate
        device: 'cpu' or 'cuda'
    
    Returns:
        tuple: (model, explainer, metadata)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model
    model = model_class(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create explainer
    disease_names = checkpoint.get('disease_names', None)
    explainer = ModelExplainer(
        model=model,
        device=device,
        disease_names=disease_names,
        mobile_mode=True  # Default to lightweight mode
    )
    
    metadata = {
        'model_name': checkpoint.get('model_name', 'Unknown'),
        'num_classes': checkpoint['num_classes'],
        'f1_score': checkpoint.get('f1_score', None),
        'explainability_enabled': checkpoint.get('explainability', {}).get('enabled', False),
        'available_methods': checkpoint.get('explainability', {}).get('methods', [])
    }
    
    return model, explainer, metadata
