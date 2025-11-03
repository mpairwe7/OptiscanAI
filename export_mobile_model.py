import torch
import torch.nn as nn
from pathlib import Path

# Import your model architecture
from src.models.vignn import SceneGraphTransformer  # Adjust import path

def export_for_mobile():
    """Export SceneGraphTransformer to TorchScript for mobile deployment"""
    
    print("Loading model architecture...")
    model = SceneGraphTransformer(
        num_classes=45,
        num_regions=12,
        hidden_dim=384,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        knowledge_graph=None,
        num_ensemble_branches=3
    )
    
    print("Loading trained weights...")
    checkpoint = torch.load('models/best_model_mobile.pth', map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print("Creating example input...")
    example_input = torch.randn(1, 3, 224, 224)
    
    print("Tracing model with TorchScript...")
    try:
        # Option 1: Tracing (recommended for inference-only)
        traced_model = torch.jit.trace(model, example_input)
        
        # Verify the traced model works
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
            
            # Check outputs match
            if torch.allclose(original_output, traced_output, atol=1e-5):
                print(" Model tracing successful! Outputs match.")
            else:
                print(" Warning: Traced outputs differ slightly")
        
        # Save for mobile
        output_path = 'retinal_screening/assets/models/best_model_mobile_traced.pt'
        traced_model.save(output_path)
        
        print(f" Mobile model saved to: {output_path}")
        print(f"   File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        # Also save optimized version (if torch.utils.mobile_optimizer is available)
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            optimized_model = optimize_for_mobile(traced_model)
            optimized_path = 'retinal_screening/assets/models/best_model_mobile_optimized.ptl'
            optimized_model._save_for_lite_interpreter(optimized_path)
            
            print(f" Optimized mobile model saved to: {optimized_path}")
            print(f"   File size: {Path(optimized_path).stat().st_size / 1024 / 1024:.2f} MB")
        except ImportError:
            print(" Mobile optimization not available (torch.utils.mobile_optimizer)")
            print("   Using standard traced model for mobile deployment")
        except Exception as opt_error:
            print(f" Mobile optimization failed: {opt_error}")
            print("   Using standard traced model for mobile deployment")
        
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        print("\nTrying torch.jit.script instead...")
        
        # Option 2: Scripting (if tracing fails)
        try:
            scripted_model = torch.jit.script(model)
            output_path = 'retinal_screening/assets/models/best_model_mobile_scripted.pt'
            scripted_model.save(output_path)
            print(f" Scripted model saved to: {output_path}")
        except Exception as e2:
            print(f"❌ Scripting also failed: {e2}")
            print("\n Your model may have operations not supported by TorchScript")
            print("   Consider simplifying the model architecture for mobile")

if __name__ == '__main__':
    export_for_mobile()