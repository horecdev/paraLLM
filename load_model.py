import torch
import json
from safetensors import safe_open

all_weights = {}

with open("models/qwen2.5-0.5b-instruct/config.json", "r", encoding="utf-8") as file:
    cfg = json.load(file)
    
with safe_open("models/qwen2.5-0.5b-instruct/model.safetensors", framework="pt", device="cpu") as f:
    all_weights["model.embed_tokens.weight"] = f.get_tensor("model.embed_tokens.weight")
    all_weights["model.norm.weight"] = f.get_tensor("model.norm.weight")

    for layer_idx in range(0, 23 + 1):
        all_weights.update({
            f"model.layers.{layer_idx}.input_layernorm.weight" : f.get_tensor(f"model.layers.{layer_idx}.input_layernorm.weight"),
            f"model.layers.{layer_idx}.mlp.down_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.mlp.down_proj.weight"),
            f"model.layers.{layer_idx}.mlp.gate_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
            f"model.layers.{layer_idx}.mlp.up_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight"),
            f"model.layers.{layer_idx}.post_attention_layernorm.weight" : f.get_tensor(f"model.layers.{layer_idx}.post_attention_layernorm.weight"),
            f"model.layers.{layer_idx}.self_attn.k_proj.bias" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.bias"),
            f"model.layers.{layer_idx}.self_attn.k_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
            f"model.layers.{layer_idx}.self_attn.q_proj.bias" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.bias"),
            f"model.layers.{layer_idx}.self_attn.q_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
            f"model.layers.{layer_idx}.self_attn.v_proj.bias" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.bias"),
            f"model.layers.{layer_idx}.self_attn.v_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
            f"model.layers.{layer_idx}.self_attn.o_proj.weight" : f.get_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
        })
        

    print(set(all_weights) ^ set(f.keys()))

tied_embeds_classifier = all_weights["model.embed_tokens.weight"]
final_norm = all_weights["model.norm.weight"]

ids = torch.tensor([[1, 2, 3, 4]]) # [1, 4]
x = tied_embeds_classifier[ids]

for layer_idx in range(0, 23 + 1):
    h = torch.rms_norm(x, all_weights[f"model.layers.{layer_idx}.input_layernorm.weight"], eps=cfg["rms_norm_eps"])
    
    q = 