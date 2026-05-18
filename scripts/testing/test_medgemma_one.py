import os
import sys

import torch
from huggingface_hub import get_token
from transformers import pipeline

model_id = "google/medgemma-4b-it"
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or get_token()

if not token:
    sys.exit(
        "Missing Hugging Face token. Run hf_login.py or set HF_TOKEN to a "
        f"token that has access to {model_id}."
    )

print(f"Loading {model_id}...", flush=True)
if not torch.cuda.is_available():
    print(
        "CUDA GPU is not available. This may take a long time on CPU.",
        flush=True,
    )

try:
    pipeline_kwargs = {
        "task": "text-generation",
        "model": model_id,
        "token": token,
        "dtype": torch.bfloat16,
    }
    if torch.cuda.is_available():
        pipeline_kwargs["device_map"] = "auto"
    else:
        pipeline_kwargs["device"] = -1

    pipe = pipeline(
        **pipeline_kwargs,
    )
except OSError as exc:
    if "gated repo" in str(exc).lower() or "403" in str(exc):
        sys.exit(
            f"Cannot access {model_id}. Visit "
            f"https://huggingface.co/{model_id} and request/accept access "
            "for the Hugging Face account used by your saved login or HF_TOKEN."
        )
    raise

print("Model loaded. Generating one example...", flush=True)

prompt = """
You are generating educational clinical explanations for laboratory results.

Given the laboratory result below, write a short explanation in 2-3 sentences.

Rules:
- Use cautious medical language.
- Do not make a final diagnosis.
- Do not recommend treatment.
- Do not invent missing information.
- Mention that the result should be interpreted in clinical context.

Input:
Patient sex: F
Admission type: EMERGENCY
Admission diagnosis: PNEUMONIA

Laboratory test: Hemoglobin
Fluid: Blood
Category: Hematology
Measured value: 9.8 g/dL
Abnormal flag: abnormal

Output:
"""

result = pipe(
    prompt,
    max_new_tokens=30,
    do_sample=False
)

print(result[0]["generated_text"])
"""
Output:
The hemoglobin level is 9.8 g/dL. This is slightly below the normal range for females, which is typically 12-1
"""