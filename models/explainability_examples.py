"""
Example usage of ModelExplainer with deployed models
"""

import torch
from PIL import Image
from torchvision import transforms
from model_explainer import load_model_with_explainability, ModelExplainer
import json

# ============================================================================
# EXAMPLE 1: Load model with explainability
# ============================================================================

def example_basic_usage():
    """Basic explainability usage"""
    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    image = Image.open('retinal_image.jpg')
    input_tensor = transform(image).unsqueeze(0)
    
    # Load model checkpoint manually
    checkpoint = torch.load('best_model_mobile.pth', map_location='cpu')
    
    # Create your model instance
    # model = YourModelClass(num_classes=checkpoint['num_classes'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    
    # Create explainer
    # explainer = ModelExplainer(
    #     model=model,
    #     device='cpu',
    #     disease_names=checkpoint['disease_names'],
    #     mobile_mode=True
    # )
    
    # Get comprehensive explanation with clinical insights
    # results = explainer.get_lightweight_explanation(input_tensor, top_k=5)
    
    # Print results structure:
    # results = {
    #     'predictions': [
    #         {
    #             'disease': 'Diabetic Retinopathy',
    #             'confidence_score': 0.87,
    #             'confidence_percentage': '87.0%',
    #             'rank': 1,
    #             'confidence_level': 'Very High',
    #             'clinical_interpretation': 'Strong evidence for diagnosis',
    #             'reliability': 'High reliability - consider as primary diagnosis'
    #         },
    #         ...
    #     ],
    #     'clinical_insights': {
    #         'recommendations': [
    #             {
    #                 'priority': 'High',
    #                 'type': 'Primary Diagnosis',
    #                 'recommendation': 'Strong evidence for Diabetic Retinopathy...',
    #                 'action': 'Immediate clinical review and treatment planning'
    #             }
    #         ],
    #         'uncertainty_metrics': {
    #             'entropy': 0.23,
    #             'confidence_gap': 0.35,
    #             'reliability_score': 82.5,
    #             'interpretation': {'reliability': 'High'}
    #         },
    #         'multi_disease_interactions': [...],
    #         'overall_assessment': {
    #             'top_diagnosis': 'Diabetic Retinopathy',
    #             'confidence': 0.87,
    #             'reliability_score': 82.5,
    #             'clinical_action_required': 'High'
    #         }
    #     },
    #     'explainability': {
    #         'gradcam': {...},
    #         'methods_used': ['gradcam']
    #     },
    #     'metadata': {
    #         'total_diseases_evaluated': 45,
    #         'diseases_above_threshold': 2
    #     }
    # }
    
    # Access specific insights:
    # print(f"Top Diagnosis: {results['clinical_insights']['overall_assessment']['top_diagnosis']}")
    # print(f"Confidence: {results['predictions'][0]['confidence_percentage']}")
    # print(f"Reliability Score: {results['clinical_insights']['uncertainty_metrics']['reliability_score']}")
    # for rec in results['clinical_insights']['recommendations']:
    #     print(f"[{rec['priority']}] {rec['recommendation']}")
    
    pass  # Uncomment above when you have the model class


# ============================================================================
# EXAMPLE 2: API endpoint integration
# ============================================================================

def example_api_endpoint():
    """Example Flask API with explainability"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Load model and explainer at startup
    # global model, explainer
    # checkpoint = torch.load('best_model_mobile.pth')
    # model = YourModelClass(num_classes=checkpoint['num_classes'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # explainer = ModelExplainer(model, device='cpu', mobile_mode=True)
    
    @app.route('/predict_with_explanation', methods=['POST'])
    def predict_with_explanation():
        # Get image from request
        # image_file = request.files['image']
        # image = Image.open(image_file)
        # input_tensor = transform(image).unsqueeze(0)
        
        # Get comprehensive predictions with clinical insights
        # results = explainer.get_lightweight_explanation(input_tensor, top_k=5)
        
        # Optional: Format for clinical display
        # response = {
        #     'diagnosis': {
        #         'primary': results['predictions'][0]['disease'],
        #         'confidence': results['predictions'][0]['confidence_percentage'],
        #         'confidence_level': results['predictions'][0]['confidence_level']
        #     },
        #     'clinical_recommendations': results['clinical_insights']['recommendations'],
        #     'reliability': {
        #         'score': results['clinical_insights']['uncertainty_metrics']['reliability_score'],
        #         'interpretation': results['clinical_insights']['uncertainty_metrics']['interpretation']
        #     },
        #     'all_findings': results['predictions'],
        #     'visual_explanation': results['explainability'].get('gradcam', {})
        # }
        
        # return jsonify(results)  # or jsonify(response) for formatted version
        pass
    
    # app.run(host='0.0.0.0', port=5000)


# ============================================================================
# EXAMPLE 3: Extract and use clinical recommendations
# ============================================================================

def example_clinical_recommendations():
    """Extract and format clinical recommendations"""
    # Load model and get predictions
    # checkpoint = torch.load('best_model_mobile.pth')
    # model = YourModelClass(...)
    # explainer = ModelExplainer(model, device='cpu', mobile_mode=True)
    
    # Get explanation
    # image = Image.open('retinal_image.jpg')
    # input_tensor = transform(image).unsqueeze(0)
    # results = explainer.get_lightweight_explanation(input_tensor, top_k=5)
    
    # Extract key clinical information
    # print("=" * 80)
    # print("CLINICAL REPORT")
    # print("=" * 80)
    
    # Primary diagnosis
    # primary = results['predictions'][0]
    # print(f"\nPrimary Diagnosis: {primary['disease']}")
    # print(f"Confidence: {primary['confidence_percentage']} ({primary['confidence_level']})")
    # print(f"Interpretation: {primary['clinical_interpretation']}")
    
    # Recommendations
    # print(f"\nClinical Recommendations:")
    # for rec in results['clinical_insights']['recommendations']:
    #     print(f"  [{rec['priority']}] {rec['type']}")
    #     print(f"  → {rec['recommendation']}")
    #     print(f"  Action: {rec['action']}\n")
    
    # Reliability metrics
    # metrics = results['clinical_insights']['uncertainty_metrics']
    # print(f"Reliability Assessment:")
    # print(f"  Overall Score: {metrics['reliability_score']:.1f}/100")
    # print(f"  Confidence Gap: {metrics['confidence_gap']:.2f}")
    # print(f"  Assessment: {metrics['interpretation']['reliability']} reliability")
    
    # Multi-disease detection
    # interactions = results['clinical_insights'].get('multi_disease_interactions')
    # if interactions:
    #     print(f"\nMulti-Disease Interactions:")
    #     for interaction in interactions:
    #         print(f"  {interaction['type']}: {interaction['diseases']}")
    #         print(f"  Note: {interaction['note']}")
    
    pass


# ============================================================================
# EXAMPLE 4: Batch processing with explainability and reporting
# ============================================================================

def example_batch_processing():
    """Process multiple images with explanations and generate reports"""
    import os
    
    # Load model and explainer
    # checkpoint = torch.load('best_model_mobile.pth')
    # model = YourModelClass(num_classes=checkpoint['num_classes'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # explainer = ModelExplainer(model, device='cpu', mobile_mode=True)
    
    image_dir = 'test_images/'
    output_dir = 'explanations/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    # for image_file in os.listdir(image_dir):
    #     if not image_file.endswith(('.jpg', '.png')):
    #         continue
    #     
    #     image_path = os.path.join(image_dir, image_file)
    #     image = Image.open(image_path)
    #     input_tensor = transform(image).unsqueeze(0)
    #     
    #     # Get comprehensive explanation
    #     results = explainer.get_lightweight_explanation(input_tensor, top_k=5)
    #     
    #     # Extract high-priority findings
    #     urgent_cases = [
    #         rec for rec in results['clinical_insights']['recommendations'] 
    #         if rec['priority'] in ['Urgent', 'High']
    #     ]
    #     
    #     # Save full results
    #     output_path = os.path.join(output_dir, f"{image_file}_explanation.json")
    #     with open(output_path, 'w') as f:
    #         json.dump(results, f, indent=2)
    #     
    #     # Generate summary report
    #     summary = {
    #         'image': image_file,
    #         'top_diagnosis': results['predictions'][0]['disease'],
    #         'confidence': results['predictions'][0]['confidence_percentage'],
    #         'reliability_score': results['clinical_insights']['uncertainty_metrics']['reliability_score'],
    #         'priority': results['clinical_insights']['recommendations'][0]['priority'],
    #         'requires_urgent_attention': len(urgent_cases) > 0
    #     }
    #     
    #     print(f"Processed: {image_file} | {summary['top_diagnosis']} ({summary['confidence']})")


if __name__ == '__main__':
    print("ModelExplainer Usage Examples - Enhanced with Clinical Insights")
    print("=" * 80)
    print("\n1. Basic Usage with Clinical Insights: example_basic_usage()")
    print("2. API Endpoint with Recommendations: example_api_endpoint()")
    print("3. Extract Clinical Recommendations: example_clinical_recommendations()")
    print("4. Batch Processing with Reporting: example_batch_processing()")
    print("\nNew Features:")
    print("  • Confidence levels with clinical interpretation")
    print("  • Automated clinical recommendations")
    print("  • Uncertainty and reliability metrics")
    print("  • Multi-disease interaction detection")
    print("  • Priority-based action recommendations")
    print("\nNote: Uncomment code sections and add your model class to run examples")
