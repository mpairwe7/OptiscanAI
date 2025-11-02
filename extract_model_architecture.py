#!/usr/bin/env python3
"""
Extract the exact model architecture from the checkpoint file
to understand what the trained model expects
"""

import torch
from pathlib import Path

# Load the checkpoint
checkpoint_path = Path("/home/darkhorse/Downloads/MLOPS_V1/models/best_model_mobile.pth")

if checkpoint_path.exists():
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\nCheckpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n✓ Found 'model_state_dict' with {len(state_dict)} parameters")
            
            # Group parameters by module
            modules = {}
            for key in state_dict.keys():
                parts = key.split('.')
                if len(parts) > 0:
                    module_name = parts[0]
                    if module_name not in modules:
                        modules[module_name] = []
                    modules[module_name].append(key)
            
            print(f"\n📦 Model Structure ({len(modules)} top-level modules):")
            print("="*80)
            
            for module_name, params in sorted(modules.items()):
                print(f"\n{module_name}/ ({len(params)} parameters)")
                # Show first few and unique patterns
                unique_patterns = set()
                for p in params[:10]:
                    pattern = '.'.join(p.split('.')[:-1])  # Remove weight/bias
                    unique_patterns.add(pattern)
                
                for pattern in sorted(unique_patterns)[:5]:
                    print(f"  - {pattern}")
                
                if len(params) > 10:
                    print(f"  ... and {len(params) - 10} more")
            
            print("\n" + "="*80)
            print("\n🔍 Key Findings:")
            print("="*80)
            
            # Check region_extractor structure
            region_ext_params = [k for k in state_dict.keys() if k.startswith('region_extractor.')]
            if region_ext_params:
                print(f"\n region_extractor: {len(region_ext_params)} parameters")
                resolution_proj = [k for k in region_ext_params if 'resolution_projections' in k]
                fusion = [k for k in region_ext_params if 'fusion' in k]
                print(f"   - Has resolution_projections: {len(resolution_proj) > 0}")
                print(f"   - Has fusion layers: {len(fusion) > 0}")
                if resolution_proj:
                    print(f"     Example: {resolution_proj[0]}")
                if fusion:
                    print(f"     Example: {fusion[0]}")
            
            # Check relation_attn structure
            relation_attn_params = [k for k in state_dict.keys() if k.startswith('relation_attn.')]
            if relation_attn_params:
                print(f"\n✅ relation_attn: {len(relation_attn_params)} parameters")
                has_qkv = any('qkv' in k for k in relation_attn_params)
                has_separate = any('q_proj' in k or 'k_proj' in k or 'v_proj' in k for k in relation_attn_params)
                print(f"   - Has combined qkv: {has_qkv}")
                print(f"   - Has separate q/k/v projections: {has_separate}")
                print("   Parameters:")
                for param in relation_attn_params:
                    print(f"     - {param}")
            
            # Check ensemble branches
            ensemble_params = [k for k in state_dict.keys() if 'ensemble' in k.lower()]
            if ensemble_params:
                print(f"\n🔀 Ensemble components: {len(ensemble_params)} parameters")
                print(f"   First few: {ensemble_params[:5]}")
            
            # Check for other key components
            print(f"\n📊 Other Components:")
            for component in ['region_proj', 'spatial_encoder', 'region_type_embed', 'uncertainty_estimator', 'classifier']:
                comp_params = [k for k in state_dict.keys() if component in k]
                if comp_params:
                    print(f"   ✓ {component}: {len(comp_params)} parameters")
            
            # Print shapes of key parameters
            print(f"\n📏 Key Parameter Shapes:")
            print("="*80)
            key_params = [
                'region_extractor.encoder.patch_embed.proj.weight',
                'region_type_embed',
                'ensemble_fusion.0.weight',
                'classifier.0.weight',
                'classifier.4.weight'
            ]
            for param_name in key_params:
                if param_name in state_dict:
                    shape = state_dict[param_name].shape
                    print(f"   {param_name}: {shape}")
            
            # Metadata
            if 'best_f1' in checkpoint:
                print(f"\n📈 Model Performance:")
                print(f"   Best F1: {checkpoint['best_f1']:.4f}")
            if 'best_auc' in checkpoint:
                print(f"   Best AUC: {checkpoint['best_auc']:.4f}")
            
        else:
            print(f"\n⚠️  'model_state_dict' key not found")
            print(f"Available keys: {list(checkpoint.keys())}")
    else:
        print(f"\n⚠️  Checkpoint is not a dictionary")
        
else:
    print(f"❌ Checkpoint not found at: {checkpoint_path}")
