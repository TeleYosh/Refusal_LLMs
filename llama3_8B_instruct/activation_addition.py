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

# Load the refusal direction
refusal_direction = torch.load('./refusal_direction.pt', weights_only = True).to(device)

# Load harmless chat
harmless_inst_train, harmless_inst_test = utils.get_harmless_instructions()
text = harmless_inst_test[0]

# Print the input harmful prompt
print("Input prompt:")
print(text)
print("")


# Use the model without activation addition
chat = [
    {"role": "user", "content": text},
]

inputs = tokenizer.apply_chat_template(chat, tokenize = True, 
                                        add_generation_prompt = True,
                                        return_tensors = "pt",
                                        return_dict = True).to(device)


complete_answer, _, init_length = utils.generate(model = model,
                                                 input_ids = inputs['input_ids'],
                                                 attention_mask = inputs['attention_mask'],
                                                 eos_token_id = tokenizer.eos_token_id,
                                                 max_tokens = 64)


print("Answer before activation addition:")
print(tokenizer.decode(complete_answer[0,init_length:], skip_special_tokens = True))
print("")

# Add hooks for direction ablation
def layers_activation_addition_hook(module, input, output):
    res = output[0] + 10*refusal_direction
    new_output = (res,) + output[1:]
    return new_output

hooks, num_layer = [], 16
for i, layer in enumerate(model.model.layers):
    if i == num_layer:
        hook = layer.register_forward_hook(layers_activation_addition_hook)
        hooks.append(hook)

# Do generation with direction ablation
max_tokens, num_tokens, eos, init_length = 64, 0, False, inputs['input_ids'].shape[1]
while (not eos) and (num_tokens < max_tokens):
    with torch.no_grad():
        model_output = model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
    logits = model_output[0][0,-1,:]
    probas = torch.nn.functional.softmax(logits, dim = -1)
    next_token = torch.argmax(probas, dim = -1)
    token_to_add = torch.tensor([[next_token.item()]], dtype = torch.long, device = device)
    mask_to_add = torch.tensor([[1]], dtype = torch.long, device = device)
    inputs['input_ids'] = torch.cat((inputs['input_ids'], token_to_add), dim = -1)
    inputs['attention_mask'] = torch.cat((inputs['attention_mask'], mask_to_add), dim = -1)
    eos = (next_token.item() == tokenizer.eos_token_id)
    num_tokens += 1

# Remove hooks
for hook in hooks:
    hook.remove()

# Decode thr answer
print("Answer after activation addition:")
print(tokenizer.decode(inputs['input_ids'][0,init_length:], skip_special_tokens = True))
