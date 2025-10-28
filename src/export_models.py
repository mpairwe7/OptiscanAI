#!/usr/bin/env python3
"""
Automated Model Export from Kaggle
This script should be run at the end of training in Kaggle notebook
"""
import os
import json
import torch
from datetime import datetime
from pathlib import Path


def export_model_from_kaggle(
    model,
    model_name: str,
    metrics: dict,
    output_dir: str = "/kaggle/working/exports"
):
    """
    Export trained model with metadata for deployment
    
    Args:
        model: Trained PyTorch model
        model_name: Name of the model (e.g., 'GraphCLIP')
        metrics: Dictionary containing model metrics
        output_dir: Output directory for exports
    """
    
    # Create export directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_name = f"{model_name}_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"EXPORTING MODEL: {model_name}")
    print(f"{'='*80}")
    
    # 1. Export PyTorch checkpoint
    checkpoint_path = os.path.join(output_dir, f"{export_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'metrics': metrics,
        'timestamp': timestamp,
        'pytorch_version': torch.__version__
    }, checkpoint_path)
    print(f"✅ PyTorch checkpoint saved: {checkpoint_path}")
    
    # 2. Export metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'model_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
        'pytorch_version': torch.__version__,
        'input_shape': [1, 3, 224, 224],
        'num_classes': 45
    }
    
    metadata_path = os.path.join(output_dir, f"{export_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")
    
    # 3. Export to ONNX (optional, for production deployment)
    try:
        import torch.onnx
        onnx_path = os.path.join(output_dir, f"{export_name}.onnx")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ ONNX model saved: {onnx_path}")
        metadata['onnx_exported'] = True
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}")
        metadata['onnx_exported'] = False
    
    # 4. Create deployment manifest
    manifest = {
        'models': [{
            'name': model_name,
            'version': timestamp,
            'checkpoint': f"{export_name}.pth",
            'onnx': f"{export_name}.onnx" if metadata.get('onnx_exported') else None,
            'metadata': f"{export_name}_metadata.json",
            'metrics': metrics
        }]
    }
    
    manifest_path = os.path.join(output_dir, "deployment_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Deployment manifest saved: {manifest_path}")
    
    print(f"\n{'='*80}")
    print(f"EXPORT COMPLETE: {export_name}")
    print(f"{'='*80}")
    print(f"Total files exported: {len(os.listdir(output_dir))}")
    print(f"Export directory: {output_dir}")
    print(f"\n📦 Ready for deployment!")
    
    return {
        'checkpoint': checkpoint_path,
        'metadata': metadata_path,
        'manifest': manifest_path
    }


def export_all_trained_models(cv_results: dict, selected_models: dict):
    """
    Export all trained models from cross-validation results
    
    Args:
        cv_results: Cross-validation results dictionary
        selected_models: Dictionary of trained model instances
    """
    
    print("\n" + "="*80)
    print("EXPORTING ALL TRAINED MODELS")
    print("="*80)
    
    export_paths = {}
    
    for model_name, results in cv_results.items():
        if model_name in selected_models:
            model = selected_models[model_name]
            
            # Extract metrics
            metrics = {
                'mean_f1': float(results.get('mean_f1', 0)),
                'mean_auc': float(results.get('mean_auc', 0)),
                'mean_precision': float(results.get('mean_precision', 0)),
                'mean_recall': float(results.get('mean_recall', 0)),
                'std_f1': float(results.get('std_f1', 0))
            }
            
            # Export model
            paths = export_model_from_kaggle(model, model_name, metrics)
            export_paths[model_name] = paths
    
    print(f"\n✅ Exported {len(export_paths)} models successfully!")
    
    return export_paths


# Example usage in Kaggle notebook (add this at the end of Cell 46):
"""
# At the end of your training cell, add:

if cv_results:
    from export_models import export_all_trained_models
    export_paths = export_all_trained_models(cv_results, selected_models)
    
    print("\\n🚀 Models ready for deployment!")
    print("Download from: /kaggle/working/exports/")
"""
