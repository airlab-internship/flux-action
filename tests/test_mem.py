import json
from pathlib import Path

import torch
from flux_action.inference.offline import load_observation
from flux_action.policy import FluxActionPolicy
import torch
import flux_action.models.transformer_inf_fp8r as fp8r

def emulated_scaled_mm(activation_q, weight_q_nk, activation_scale, weight_scale):
    a = activation_q.to(torch.bfloat16) * activation_scale[:, None].to(torch.bfloat16)
    b = weight_q_nk.T.to(torch.bfloat16) * weight_scale[None, :].to(torch.bfloat16)
    return a @ b

fp8r._scaled_mm = emulated_scaled_mm
fp8r.FP8RInferenceDiT._check_runtime_device = staticmethod(lambda device: None)

policy = FluxActionPolicy.from_pretrained(
    "outputs/droid",
    subfolder="variants/fp8r",  # prequantized FP8r weights
    device="cuda",
)
print("quantization:", policy.config.quantization)  # should print fp8r
policy.set_text_encoder_offload(True)  # keep the text encoder on CPU between caption-cache misses
policy.prepare_inference(compile=False)

task = json.loads(Path("outputs/public-droid/observation.json").read_text())["task"]
observation = load_observation(
    Path("outputs/public-droid/observation.npz"), policy.config, task, "cuda"
)

with torch.inference_mode():
    plan = policy.predict_action_chunk(observation)  # first call encodes the caption
print(plan.shape)  # torch.Size([1, 32, 8])
print(f"First call peak (with text encoder): {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB")

torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    plan = policy.predict_action_chunk(observation)  # caption is cached now
print(f"Inference peak: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB")