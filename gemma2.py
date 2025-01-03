import sys 
sys.path.append("gemma_pytorch") 
from gemma.config import GemmaConfig, get_model_config
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
import contextlib
import os
import torch

# Load the model
VARIANT = "9b" 
MACHINE_TYPE = "cuda" 
weights_dir = '/home/yangrumei/.cache/kagglehub/models/google/gemma-2/pyTorch/gemma-2-9b-it/1' 

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)

model_config = get_model_config(VARIANT)
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")

device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
  model = GemmaForCausalLM(model_config)
  model.load_weights(weights_dir)
  model: GemmaForCausalLM = model.to(device).eval()

# Use the model

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn><eos>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn><eos>\n"

prompt = (
    USER_CHAT_TEMPLATE.format(
        prompt="鹿儿岛的景点推荐"
    )
    + MODEL_CHAT_TEMPLATE.format(prompt="北京大学")
    + USER_CHAT_TEMPLATE.format(prompt="他的qs是多少")
    + "<start_of_turn>model\n"
)

model.generate(
    USER_CHAT_TEMPLATE.format(prompt=prompt),
    device=device,
    output_len=1000,
)