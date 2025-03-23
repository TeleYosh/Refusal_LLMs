# Libraries
import utils
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the llama-3-8B-instruct model
login(token="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")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).to(device)

# Load datasets
N_INST_TRAIN = 128
harmful_inst_train, harmful_inst_test = utils.get_harmful_instructions()
harmless_inst_train, harmless_inst_test = utils.get_harmless_instructions()

# Show examples
print("Harmful instructions:")
for i in range(4):
    print(f"\t{repr(harmful_inst_train[i])}")
print("Harmless instructions:")
for i in range(4):
    print(f"\t{repr(harmless_inst_train[i])}")

# Get harmful activations
num_layer, pos = 16, -1
harmful_activations = []

def hook_fn_harmful(module, input, output):
        activation = output[0][0,pos,:]
        harmful_activations.append(activation.detach().cpu())

hooks = []
for i, layer in enumerate(model.model.layers):
    if i == num_layer:
        hook = layer.register_forward_hook(hook_fn_harmful)
        hooks.append(hook)
    
for i in range(N_INST_TRAIN):
    chat = [
        {"role": "user", "content": harmful_inst_train[i]},
    ]
    inputs = tokenizer.apply_chat_template(chat, tokenize = True, 
                                          add_generation_prompt = True,
                                          return_tensors = "pt",
                                          return_dict = True).to(device)
    with torch.no_grad():
        _ = model(**inputs)
harmful_activations = torch.stack(harmful_activations)

for hook in hooks:
    hook.remove()

# Get harmless activations
num_layer, pos = 16, -1
harmless_activations = []

def hook_fn_harmless(module, input, output):
        activation = output[0][0,pos,:]
        harmless_activations.append(activation.detach().cpu())

hooks = []
for i, layer in enumerate(model.model.layers):
    if i == num_layer:
        hook = layer.register_forward_hook(hook_fn_harmless)
        hooks.append(hook)

for i in range(N_INST_TRAIN):
    chat = [
        {"role": "user", "content": harmless_inst_train[i]},
    ]
    inputs = tokenizer.apply_chat_template(chat, tokenize = True, 
                                          add_generation_prompt = True,
                                          return_tensors = "pt",
                                          return_dict = True).to(device)
    with torch.no_grad():
        _ = model(**inputs)
harmless_activations = torch.stack(harmless_activations)

for hook in hooks:
    hook.remove()

# Refusal direction
refusal_dir = harmful_activations.mean(dim = 0) - harmless_activations.mean(dim = 0)
refusal_dir = refusal_dir / refusal_dir.norm()
torch.save(refusal_dir, './refusal_direction.pt')