@echo off
call D:\d-awww\LlamaFactory\llama_env\Scripts\activate.bat
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import qwen_vl_utils; print('qwen-vl-utils:', qwen_vl_utils.__version__)"
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Qwen2_5_VL import OK')"
