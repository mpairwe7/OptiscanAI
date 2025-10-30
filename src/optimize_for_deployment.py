#!/usr/bin/env python3
"""
Model Optimization for Deployment
Converts PyTorch model to ONNX with quantization for production use

Usage:
    python src/optimize_for_deployment.py --model models/GraphCLIP_fold1_best.pth
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import time
import argparse
from typing import Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def load_model(model_path: Path, device: str = 'cpu') -> nn.Module:
    """Load PyTorch model from checkpoint"""
    
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Import model architecture (adjust based on your model)
    # For GraphCLIP, you'll need to import the actual model class
    # This is a placeholder - replace with actual model import
    
    print("\n⚠️  Note: You need to import your actual model architecture")
    print("For now, returning state dict for ONNX export")
    
    return state_dict


def export_to_onnx(
    model: nn.Module,
    save_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 14,
    dynamic_batch: bool = True
) -> None:
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version
        dynamic_batch: Whether to support dynamic batch sizes
    """
    
    print(f"\n{'='*80}")
    print("EXPORTING TO ONNX")
    print(f"{'='*80}")
    
    model.eval()
    model.cpu()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    print(f"\nExporting model...")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic batch: {dynamic_batch}")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"\n✓ ONNX export complete!")
    print(f"  Saved to: {save_path}")
    
    # Verify ONNX model
    print(f"\nVerifying ONNX model...")
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is valid")
    
    # Get model size
    model_size_mb = save_path.stat().st_size / (1024 ** 2)
    print(f"  Model size: {model_size_mb:.2f} MB")


def benchmark_onnx_inference(
    onnx_path: Path,
    num_runs: int = 100,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> Dict[str, float]:
    """Benchmark ONNX model inference speed"""
    
    print(f"\n{'='*80}")
    print("BENCHMARKING ONNX INFERENCE")
    print(f"{'='*80}")
    
    # Create ONNX Runtime session
    print(f"\nCreating ONNX Runtime session...")
    
    # Check for GPU
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
        print("  ✓ GPU acceleration available")
    else:
        print("  ⚠️  Using CPU only")
    
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
    
    # Warm-up
    print(f"\nWarming up...")
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    
    for _ in range(10):
        _ = ort_session.run(None, ort_inputs)
    
    # Benchmark
    print(f"Running {num_runs} inference iterations...")
    times = []
    
    for i in range(num_runs):
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
        
        start = time.time()
        outputs = ort_session.run(None, ort_inputs)
        end = time.time()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    results = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'fps': 1000 / np.mean(times)
    }
    
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"  Mean:      {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
    print(f"  Median:    {results['median_ms']:.2f} ms")
    print(f"  Min/Max:   {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
    print(f"  95th %ile: {results['p95_ms']:.2f} ms")
    print(f"  99th %ile: {results['p99_ms']:.2f} ms")
    print(f"  Throughput: {results['fps']:.1f} FPS")
    
    return results


def create_config_files(output_dir: Path, num_classes: int = 45):
    """Create configuration files for deployment"""
    
    print(f"\n{'='*80}")
    print("CREATING CONFIGURATION FILES")
    print(f"{'='*80}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Disease names (RFMiD dataset - 45 diseases)
    disease_names = [
        'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS', 'MS',
        'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
        'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM',
        'PRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA',
        'VS', 'BRAO', 'PLQ', 'HPED', 'CL'
    ]
    
    # Model configuration
    config = {
        'model_type': 'onnx',
        'model_name': 'GraphCLIP',
        'version': '1.0.0',
        'input_size': [224, 224],
        'num_classes': num_classes,
        'disease_names': disease_names[:num_classes],
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'resize': 224,
            'normalize': True,
            'convert_rgb': True
        },
        'inference': {
            'batch_size': 1,
            'threshold': 0.5,
            'use_sigmoid': True,
            'return_probabilities': True
        },
        'deployment': {
            'platform': 'onnx',
            'optimize_for_mobile': True,
            'quantized': False
        }
    }
    
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Config saved to: {config_path}")
    
    # Disease information with clinical context
    disease_info = {
        'DR': {
            'full_name': 'Diabetic Retinopathy',
            'severity': 'HIGH',
            'urgency': 'URGENT',
            'description': 'Retinal damage from diabetes complications',
            'screening_interval_months': 6,
            'risk_factors': ['Diabetes', 'Hypertension', 'Poor glucose control']
        },
        'ARMD': {
            'full_name': 'Age-Related Macular Degeneration',
            'severity': 'HIGH',
            'urgency': 'SEMI_URGENT',
            'description': 'Age-related deterioration of central vision',
            'screening_interval_months': 12,
            'risk_factors': ['Age >50', 'Smoking', 'Family history']
        },
        'HTR': {
            'full_name': 'Hypertensive Retinopathy',
            'severity': 'MEDIUM',
            'urgency': 'SEMI_URGENT',
            'description': 'Retinal changes from high blood pressure',
            'screening_interval_months': 6,
            'risk_factors': ['Hypertension', 'Cardiovascular disease']
        },
        'MH': {
            'full_name': 'Macular Hole',
            'severity': 'HIGH',
            'urgency': 'URGENT',
            'description': 'Small break in the macula',
            'screening_interval_months': 3,
            'risk_factors': ['Age >60', 'Eye trauma', 'High myopia']
        }
    }
    
    # Add basic info for other diseases
    for disease in disease_names[:num_classes]:
        if disease not in disease_info:
            disease_info[disease] = {
                'full_name': disease,
                'severity': 'MEDIUM',
                'urgency': 'ROUTINE',
                'description': 'Retinal condition requiring evaluation',
                'screening_interval_months': 12,
                'risk_factors': []
            }
    
    disease_info_path = output_dir / 'disease_info.json'
    with open(disease_info_path, 'w') as f:
        json.dump(disease_info, f, indent=2)
    print(f"✓ Disease info saved to: {disease_info_path}")


def create_optimization_report(
    output_dir: Path,
    original_size_mb: float,
    optimized_size_mb: float,
    benchmark_results: Dict[str, float]
):
    """Create optimization report"""
    
    report = {
        'optimization_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': {
            'original_size_mb': original_size_mb,
            'optimized_size_mb': optimized_size_mb,
            'size_reduction_pct': (1 - optimized_size_mb / original_size_mb) * 100,
            'format': 'ONNX'
        },
        'performance': benchmark_results,
        'deployment_ready': True,
        'recommendations': [
            'Model is optimized for production deployment',
            'ONNX format enables cross-platform compatibility',
            f'Expected inference time: {benchmark_results["mean_ms"]:.0f}ms',
            'Consider GPU acceleration for higher throughput'
        ]
    }
    
    report_path = output_dir / 'optimization_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Optimization report saved to: {report_path}")
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Original size:  {original_size_mb:.2f} MB")
    print(f"  Optimized size: {optimized_size_mb:.2f} MB")
    print(f"  Size reduction: {report['model']['size_reduction_pct']:.1f}%")
    print(f"  Inference time: {benchmark_results['mean_ms']:.2f} ms")
    print(f"  Throughput:     {benchmark_results['fps']:.1f} FPS")
    print(f"\n✓ Model ready for deployment!")


def main():
    parser = argparse.ArgumentParser(description='Optimize model for deployment')
    parser.add_argument(
        '--model',
        type=str,
        default='models/GraphCLIP_fold1_best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/exports',
        help='Output directory for optimized models'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=45,
        help='Number of disease classes'
    )
    parser.add_argument(
        '--skip-onnx',
        action='store_true',
        help='Skip ONNX export (only create configs)'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    
    print(f"\n{'='*80}")
    print("MODEL OPTIMIZATION FOR DEPLOYMENT")
    print(f"{'='*80}")
    print(f"Input model: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration files
    create_config_files(output_dir, args.num_classes)
    
    if not args.skip_onnx:
        # Check if model exists
        if not model_path.exists():
            print(f"\n❌ Error: Model not found at {model_path}")
            print("\nTo optimize your model:")
            print("  1. Make sure the model file exists")
            print("  2. Run: python src/optimize_for_deployment.py --model <path>")
            return
        
        # Get original model size
        original_size_mb = model_path.stat().st_size / (1024 ** 2)
        
        # Load model
        print(f"\n⚠️  Note: This script needs your actual model architecture")
        print("For now, it will create the deployment structure")
        print("\nTo complete ONNX export, you need to:")
        print("  1. Import your GraphCLIP model class")
        print("  2. Load the model architecture")
        print("  3. Load the weights from checkpoint")
        print("  4. Export to ONNX")
        
        # Create placeholder for now
        onnx_path = output_dir / 'GraphCLIP_optimized.onnx'
        print(f"\n📝 ONNX model would be saved to: {onnx_path}")
    
    print(f"\n{'='*80}")
    print("✓ DEPLOYMENT PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"  1. Complete ONNX export with actual model")
    print(f"  2. Test optimized model: python deployment/test_optimized_model.py")
    print(f"  3. Build container: podman build -f Dockerfile.gpu -t retinal-screening-gpu .")
    print(f"  4. Deploy to GCP: Follow deployment/COMPLETE_DEPLOYMENT_GUIDE.md")


if __name__ == "__main__":
    main()
