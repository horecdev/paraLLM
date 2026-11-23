import torch
from safetensors import safe_open

layers = []
tied_embeds_classifier = None
final_norm = None

with safe_open("models/qwen2.5-0.5b-instruct/model.safetensors", framework="pt", device="cpu") as f:
    tied_embeds_classifier = f.get_tensor("model.embed_tokens.weight")
    final_norm = f.get_tensor("model.norm.weight")

    for layer_idx in range(0, 23 + 1):
        layer_dict = {
            "input_layernorm.weight" : f.get_tensor(f"model.layers.{layer_idx}.input_layernorm.weight"),
            "mlp.down_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.mlp.down_proj.weight"),
            "mlp.gate_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
            "mlp.up_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight"),
            "post_attention_layernorm.weight" : f.get_tensor(f"model.layers.{layer_idx}.post_attention_layernorm.weight"),
            "self_attn.k_proj.bias" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.bias"),
            "self_attn.k_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
            "self_attn.q_proj.bias" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.bias"),
            "self_attn.q_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
            "self_attn.v_proj.bias" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.bias"),
            "self_attn.v_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
            "self_attn.o_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
        }
        
        layers.append(layer_dict)

    print(f"Key amount: {len(list(f.keys()))}")
    amount_parsed_keys = 2 # tied_embeds_classifiers, final_norm
    for layer_dict in layers:
        amount_parsed_keys += len(list(layer_dict.keys()))

    print(f"Parsed keys: {amount_parsed_keys}")



        

    