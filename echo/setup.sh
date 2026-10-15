#!/usr/bin/env bash
# One-time pod setup: python deps, Qwen3-14B, the J-lens, WikiText. VLLM=1 also builds the /root/vllm venv.
set -euo pipefail
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1
pip install -q -U "transformers>=4.56" "peft>=0.15" datasets safetensors huggingface_hub pytest
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())"
hf download Qwen/Qwen3-14B > /dev/null
hf download neuronpedia/jacobian-lens qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt --local-dir /root/lens > /dev/null
python -c "from datasets import load_dataset; load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')"
if [ -n "${VLLM:-}" ]; then
  pip install -q uv
  uv venv -q /root/vllm --python 3.12
  uv pip install -q --python /root/vllm/bin/python vllm
  /root/vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, torch.__version__, torch.cuda.is_available())"
fi
echo SETUP_DONE
