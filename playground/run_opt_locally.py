#!/usr/bin/env python3
"""
OPT-125M Local Analysis Script
==============================

This script loads the OPT-125M model from Hugging Face and performs detailed
architectural analysis including layer dimensions, parameter counts, and
basic inference testing.

Usage:
    python playground/run_opt_locally.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import pandas as pd
from collections import OrderedDict


class OPTAnalyzer:
    """Comprehensive analyzer for OPT-125M model architecture and parameters."""
    
    def __init__(self, model_name: str = "facebook/opt-125m"):
        """Initialize the analyzer with model loading."""
        print(f"Loading OPT model: {model_name}")
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token to eos token to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,  # Use float32 for analysis
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully!")
        print(f"Model type: {type(self.model)}")
        
    def get_model_summary(self) -> Dict[str, Any]:
        """Get high-level model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        config = self.model.config
        
        summary = {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": getattr(config, 'ffn_dim', 'N/A'),
            "max_position_embeddings": config.max_position_embeddings,
            "layer_norm_eps": getattr(config, 'layer_norm_eps', 'N/A'),
            "activation_function": getattr(config, 'activation_function', 'N/A'),
        }
        
        return summary
    
    def analyze_layer_dimensions(self) -> pd.DataFrame:
        """Analyze dimensions and parameter counts for each layer."""
        layer_info = []
        
        # Get the main model (unwrap if needed)
        if hasattr(self.model, 'model'):
            core_model = self.model.model
        else:
            core_model = self.model
            
        print(f"Core model type: {type(core_model)}")
        print(f"Core model attributes: {list(core_model.__dict__.keys())}")
        
        # Analyze embeddings
        if hasattr(core_model, 'decoder') and hasattr(core_model.decoder, 'embed_tokens'):
            embed_tokens = core_model.decoder.embed_tokens
            layer_info.append({
                'layer_name': 'embed_tokens',
                'layer_type': 'Embedding',
                'shape': list(embed_tokens.weight.shape),
                'num_parameters': embed_tokens.weight.numel(),
                'dtype': str(embed_tokens.weight.dtype)
            })
        
        if hasattr(core_model, 'decoder') and hasattr(core_model.decoder, 'embed_positions'):
            embed_pos = core_model.decoder.embed_positions
            layer_info.append({
                'layer_name': 'embed_positions',
                'layer_type': 'Embedding',
                'shape': list(embed_pos.weight.shape),
                'num_parameters': embed_pos.weight.numel(),
                'dtype': str(embed_pos.weight.dtype)
            })
        
        # Analyze transformer layers
        if hasattr(core_model, 'decoder') and hasattr(core_model.decoder, 'layers'):
            layers = core_model.decoder.layers
            print(f"Found {len(layers)} transformer layers")
            
            for i, layer in enumerate(layers):
                # Self attention
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    
                    # Q, K, V projections
                    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                        if hasattr(attn, proj_name):
                            proj = getattr(attn, proj_name)
                            layer_info.append({
                                'layer_name': f'layer_{i}.self_attn.{proj_name}',
                                'layer_type': 'Linear',
                                'shape': list(proj.weight.shape),
                                'num_parameters': proj.weight.numel() + (proj.bias.numel() if proj.bias is not None else 0),
                                'dtype': str(proj.weight.dtype)
                            })
                
                # Feed forward network
                if hasattr(layer, 'fc1'):
                    fc1 = layer.fc1
                    layer_info.append({
                        'layer_name': f'layer_{i}.fc1',
                        'layer_type': 'Linear',
                        'shape': list(fc1.weight.shape),
                        'num_parameters': fc1.weight.numel() + (fc1.bias.numel() if fc1.bias is not None else 0),
                        'dtype': str(fc1.weight.dtype)
                    })
                
                if hasattr(layer, 'fc2'):
                    fc2 = layer.fc2
                    layer_info.append({
                        'layer_name': f'layer_{i}.fc2',
                        'layer_type': 'Linear',
                        'shape': list(fc2.weight.shape),
                        'num_parameters': fc2.weight.numel() + (fc2.bias.numel() if fc2.bias is not None else 0),
                        'dtype': str(fc2.weight.dtype)
                    })
                
                # Layer norms
                for norm_name in ['self_attn_layer_norm', 'final_layer_norm']:
                    if hasattr(layer, norm_name):
                        norm = getattr(layer, norm_name)
                        layer_info.append({
                            'layer_name': f'layer_{i}.{norm_name}',
                            'layer_type': 'LayerNorm',
                            'shape': list(norm.weight.shape),
                            'num_parameters': norm.weight.numel() + (norm.bias.numel() if norm.bias is not None else 0),
                            'dtype': str(norm.weight.dtype)
                        })
        
        # Final layer norm and LM head
        if hasattr(core_model, 'decoder') and hasattr(core_model.decoder, 'final_layer_norm'):
            final_norm = core_model.decoder.final_layer_norm
            layer_info.append({
                'layer_name': 'final_layer_norm',
                'layer_type': 'LayerNorm',
                'shape': list(final_norm.weight.shape),
                'num_parameters': final_norm.weight.numel() + (final_norm.bias.numel() if final_norm.bias is not None else 0),
                'dtype': str(final_norm.weight.dtype)
            })
        
        if hasattr(self.model, 'lm_head'):
            lm_head = self.model.lm_head
            layer_info.append({
                'layer_name': 'lm_head',
                'layer_type': 'Linear',
                'shape': list(lm_head.weight.shape),
                'num_parameters': lm_head.weight.numel() + (lm_head.bias.numel() if lm_head.bias is not None else 0),
                'dtype': str(lm_head.weight.dtype)
            })
        
        return pd.DataFrame(layer_info)
    
    def verify_parameter_count(self) -> Dict[str, int]:
        """Verify parameter count using different methods."""
        # Method 1: Direct count
        total_params_direct = sum(p.numel() for p in self.model.parameters())
        
        # Method 2: Named parameters
        named_params = dict(self.model.named_parameters())
        total_params_named = sum(p.numel() for p in named_params.values())
        
        # Method 3: Layer analysis
        layer_df = self.analyze_layer_dimensions()
        total_params_analysis = layer_df['num_parameters'].sum()
        
        # Check for weight tying
        weight_tying_detected = False
        if hasattr(self.model, 'lm_head') and hasattr(self.model.model.decoder, 'embed_tokens'):
            lm_head_weight = self.model.lm_head.weight
            embed_weight = self.model.model.decoder.embed_tokens.weight
            weight_tying_detected = torch.equal(lm_head_weight, embed_weight)
        
        return {
            "direct_count": total_params_direct,
            "named_parameters": total_params_named,
            "layer_analysis": total_params_analysis,
            "discrepancy": total_params_direct - total_params_analysis,
            "weight_tying_detected": weight_tying_detected,
            "explanation": "LM head shares weights with token embeddings" if weight_tying_detected else "No weight sharing detected"
        }
    
    def get_detailed_architecture_info(self) -> Dict[str, Any]:
        """Get detailed architectural information."""
        config = self.model.config
        
        # Calculate theoretical parameter counts
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        ffn_dim = config.ffn_dim
        
        # Embedding parameters
        token_embeddings = vocab_size * hidden_size
        position_embeddings = config.max_position_embeddings * hidden_size
        
        # Per-layer parameters
        # Self-attention: 4 linear layers (q, k, v, out) each hidden_size x hidden_size
        attention_params_per_layer = 4 * (hidden_size * hidden_size + hidden_size)  # +bias
        
        # Feed-forward: 2 linear layers
        ffn_params_per_layer = (hidden_size * ffn_dim + ffn_dim) + (ffn_dim * hidden_size + hidden_size)
        
        # Layer norms: 2 per layer
        layernorm_params_per_layer = 2 * (hidden_size + hidden_size)  # weight + bias
        
        total_per_layer = attention_params_per_layer + ffn_params_per_layer + layernorm_params_per_layer
        
        # Final layer norm
        final_layernorm = hidden_size + hidden_size  # weight + bias
        
        # LM head (usually tied with token embeddings)
        lm_head_params = vocab_size * hidden_size
        
        theoretical_total = (token_embeddings + position_embeddings + 
                           num_layers * total_per_layer + 
                           final_layernorm + lm_head_params)
        
        return {
            "config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "ffn_dim": ffn_dim,
                "num_attention_heads": config.num_attention_heads,
                "max_position_embeddings": config.max_position_embeddings
            },
            "theoretical_params": {
                "token_embeddings": token_embeddings,
                "position_embeddings": position_embeddings,
                "attention_per_layer": attention_params_per_layer,
                "ffn_per_layer": ffn_params_per_layer,
                "layernorm_per_layer": layernorm_params_per_layer,
                "total_per_layer": total_per_layer,
                "all_layers": num_layers * total_per_layer,
                "final_layernorm": final_layernorm,
                "lm_head": lm_head_params,
                "theoretical_total": theoretical_total
            }
        }
    
    def analyze_attention_structure(self) -> Dict[str, Any]:
        """Analyze the attention mechanism structure for surgical modifications."""
        config = self.model.config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        attention_info = {
            "num_attention_heads": num_heads,
            "head_dimension": head_dim,
            "total_attention_dim": hidden_size,
            "attention_layers": []
        }
        
        # Analyze each attention layer
        if hasattr(self.model.model.decoder, 'layers'):
            for i, layer in enumerate(self.model.model.decoder.layers):
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    layer_info = {
                        "layer_idx": i,
                        "q_proj_shape": list(attn.q_proj.weight.shape),
                        "k_proj_shape": list(attn.k_proj.weight.shape),
                        "v_proj_shape": list(attn.v_proj.weight.shape),
                        "out_proj_shape": list(attn.out_proj.weight.shape),
                        "has_bias": {
                            "q_proj": attn.q_proj.bias is not None,
                            "k_proj": attn.k_proj.bias is not None,
                            "v_proj": attn.v_proj.bias is not None,
                            "out_proj": attn.out_proj.bias is not None,
                        }
                    }
                    attention_info["attention_layers"].append(layer_info)
        
        return attention_info
    
    def get_surgical_modification_targets(self) -> Dict[str, Any]:
        """Identify key components for potential surgical modifications."""
        targets = {
            "embedding_layers": {
                "token_embeddings": "model.decoder.embed_tokens.weight",
                "position_embeddings": "model.decoder.embed_positions.weight",
                "shape_token": f"[{self.model.config.vocab_size}, {self.model.config.hidden_size}]",
                "shape_position": f"[{self.model.config.max_position_embeddings}, {self.model.config.hidden_size}]"
            },
            "attention_components": [],
            "feedforward_components": [],
            "normalization_components": []
        }
        
        # Attention and FFN components for each layer
        for i in range(self.model.config.num_hidden_layers):
            # Attention components
            attn_components = {
                "layer_idx": i,
                "q_proj": f"model.decoder.layers.{i}.self_attn.q_proj.weight",
                "k_proj": f"model.decoder.layers.{i}.self_attn.k_proj.weight", 
                "v_proj": f"model.decoder.layers.{i}.self_attn.v_proj.weight",
                "out_proj": f"model.decoder.layers.{i}.self_attn.out_proj.weight",
                "shape": f"[{self.model.config.hidden_size}, {self.model.config.hidden_size}]"
            }
            targets["attention_components"].append(attn_components)
            
            # FFN components
            ffn_components = {
                "layer_idx": i,
                "fc1": f"model.decoder.layers.{i}.fc1.weight",
                "fc2": f"model.decoder.layers.{i}.fc2.weight",
                "fc1_shape": f"[{self.model.config.ffn_dim}, {self.model.config.hidden_size}]",
                "fc2_shape": f"[{self.model.config.hidden_size}, {self.model.config.ffn_dim}]"
            }
            targets["feedforward_components"].append(ffn_components)
            
            # Normalization components
            norm_components = {
                "layer_idx": i,
                "self_attn_layer_norm": f"model.decoder.layers.{i}.self_attn_layer_norm.weight",
                "final_layer_norm": f"model.decoder.layers.{i}.final_layer_norm.weight",
                "shape": f"[{self.model.config.hidden_size}]"
            }
            targets["normalization_components"].append(norm_components)
        
        # Final components
        targets["output_components"] = {
            "final_layer_norm": "model.decoder.final_layer_norm.weight",
            "lm_head": "lm_head.weight",
            "weight_tied_with_embeddings": True,  # Based on our analysis
            "shape": f"[{self.model.config.vocab_size}, {self.model.config.hidden_size}]"
        }
        
        return targets
    
    def test_inference(self, prompt: str = "The future of artificial intelligence is") -> Dict[str, Any]:
        """Test basic inference with the model."""
        print(f"\nTesting inference with prompt: '{prompt}'")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "input_ids_shape": inputs.input_ids.shape,
            "output_ids_shape": outputs.shape
        }
    
    def visualize_parameter_distribution(self, layer_df: pd.DataFrame) -> None:
        """Create visualizations of parameter distribution across layers."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Parameter count by layer type
        plt.subplot(2, 2, 1)
        layer_type_params = layer_df.groupby('layer_type')['num_parameters'].sum()
        plt.pie(layer_type_params.values, labels=layer_type_params.index, autopct='%1.1f%%')
        plt.title('Parameter Distribution by Layer Type')
        
        # Plot 2: Parameter count by layer (top 20)
        plt.subplot(2, 2, 2)
        top_layers = layer_df.nlargest(20, 'num_parameters')
        plt.barh(range(len(top_layers)), top_layers['num_parameters'])
        plt.yticks(range(len(top_layers)), top_layers['layer_name'], fontsize=8)
        plt.xlabel('Number of Parameters')
        plt.title('Top 20 Layers by Parameter Count')
        
        # Plot 3: Layer dimensions heatmap (for Linear layers)
        plt.subplot(2, 2, 3)
        linear_layers = layer_df[layer_df['layer_type'] == 'Linear'].copy()
        if not linear_layers.empty:
            # Extract dimensions
            linear_layers['dim1'] = linear_layers['shape'].apply(lambda x: x[0] if len(x) > 0 else 0)
            linear_layers['dim2'] = linear_layers['shape'].apply(lambda x: x[1] if len(x) > 1 else 0)
            
            # Create scatter plot
            plt.scatter(linear_layers['dim1'], linear_layers['dim2'], 
                       s=linear_layers['num_parameters']/1000, alpha=0.6)
            plt.xlabel('Input Dimension')
            plt.ylabel('Output Dimension')
            plt.title('Linear Layer Dimensions (size = param count)')
        
        # Plot 4: Cumulative parameter count
        plt.subplot(2, 2, 4)
        layer_df_sorted = layer_df.sort_values('num_parameters', ascending=False)
        cumsum = layer_df_sorted['num_parameters'].cumsum()
        plt.plot(range(len(cumsum)), cumsum / cumsum.iloc[-1] * 100)
        plt.xlabel('Layer Index (sorted by param count)')
        plt.ylabel('Cumulative Parameters (%)')
        plt.title('Cumulative Parameter Distribution')
        
        plt.tight_layout()
        plt.savefig('opt_125m_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    def run_full_analysis(self) -> None:
        """Run complete analysis of the OPT-125M model."""
        print("=" * 60)
        print("OPT-125M COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        # Model summary
        print("\n1. MODEL SUMMARY")
        print("-" * 30)
        summary = self.get_model_summary()
        for key, value in summary.items():
            print(f"{key:25}: {value:,}" if isinstance(value, int) else f"{key:25}: {value}")
        
        # Parameter verification
        print("\n2. PARAMETER COUNT VERIFICATION")
        print("-" * 30)
        param_counts = self.verify_parameter_count()
        for key, value in param_counts.items():
            if isinstance(value, int):
                print(f"{key:20}: {value:,}")
            else:
                print(f"{key:20}: {value}")
        
        # Detailed architecture
        print("\n3. DETAILED ARCHITECTURE ANALYSIS")
        print("-" * 30)
        arch_info = self.get_detailed_architecture_info()
        print("Configuration:")
        for key, value in arch_info["config"].items():
            print(f"  {key:25}: {value:,}")
        
        print("\nTheoretical Parameter Breakdown:")
        for key, value in arch_info["theoretical_params"].items():
            print(f"  {key:25}: {value:,}")
        
        # Layer analysis
        print("\n4. LAYER-BY-LAYER ANALYSIS")
        print("-" * 30)
        layer_df = self.analyze_layer_dimensions()
        
        # Group by layer type
        print("\nParameter count by layer type:")
        type_summary = layer_df.groupby('layer_type').agg({
            'num_parameters': ['count', 'sum', 'mean']
        }).round(2)
        print(type_summary)
        
        print(f"\nDetailed layer information (showing first 10 rows):")
        print(layer_df.head(10).to_string(index=False))
        
        print(f"\nTotal layers analyzed: {len(layer_df)}")
        print(f"Total parameters from analysis: {layer_df['num_parameters'].sum():,}")
        
        # Attention structure analysis
        print("\n5. ATTENTION STRUCTURE ANALYSIS")
        print("-" * 30)
        attention_info = self.analyze_attention_structure()
        print(f"Number of attention heads: {attention_info['num_attention_heads']}")
        print(f"Head dimension: {attention_info['head_dimension']}")
        print(f"Total attention dimension: {attention_info['total_attention_dim']}")
        print(f"Attention layers analyzed: {len(attention_info['attention_layers'])}")
        
        # Show first few attention layers
        if attention_info['attention_layers']:
            print("\nFirst 3 attention layers:")
            for layer_info in attention_info['attention_layers'][:3]:
                print(f"  Layer {layer_info['layer_idx']}:")
                print(f"    Q/K/V/Out shapes: {layer_info['q_proj_shape']}")
                print(f"    Has bias: {layer_info['has_bias']}")
        
        # Surgical modification targets
        print("\n6. SURGICAL MODIFICATION TARGETS")
        print("-" * 30)
        surgical_targets = self.get_surgical_modification_targets()
        
        print("Key modification targets for surgery:")
        print(f"  Token embeddings: {surgical_targets['embedding_layers']['token_embeddings']}")
        print(f"  Position embeddings: {surgical_targets['embedding_layers']['position_embeddings']}")
        print(f"  Attention layers: {len(surgical_targets['attention_components'])} layers")
        print(f"  FFN layers: {len(surgical_targets['feedforward_components'])} layers")
        print(f"  Layer norms: {len(surgical_targets['normalization_components'])} layers")
        
        print("\nExample attention modification targets (Layer 0):")
        if surgical_targets['attention_components']:
            layer_0 = surgical_targets['attention_components'][0]
            for key, value in layer_0.items():
                if key != 'layer_idx':
                    print(f"  {key}: {value}")
        
        # Inference test
        print("\n7. INFERENCE TEST")
        print("-" * 30)
        inference_result = self.test_inference()
        print(f"Prompt: {inference_result['prompt']}")
        print(f"Generated: {inference_result['generated_text']}")
        print(f"Input shape: {inference_result['input_ids_shape']}")
        print(f"Output shape: {inference_result['output_ids_shape']}")
        
        # Visualizations
        print("\n8. GENERATING VISUALIZATIONS")
        print("-" * 30)
        self.visualize_parameter_distribution(layer_df)
        
        # Save detailed analysis
        layer_df.to_csv('opt_125m_layer_analysis.csv', index=False)
        print("Detailed layer analysis saved to 'opt_125m_layer_analysis.csv'")
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)


def main():
    """Main function to run the OPT analysis."""
    try:
        analyzer = OPTAnalyzer()
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
