# Libraries
import utils
import torch
import einops
from torch import Tensor
from jaxtyping import Float, Int
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Function to orthogonalize a matrix with respect a vector
def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    proj = einops.einsum(matrix, vec.view(-1, 1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj

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

# TEST BEFORE
harmful_inst_train, harmful_inst_test = utils.get_harmful_instructions()
text = harmful_inst_test[9]
print("TASK")
print(text)
print("")


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
                                                 max_tokens = 256)
print("BEFORE ORTHOGONALIZATION:")
print(tokenizer.decode(complete_answer[0,init_length:], skip_special_tokens = True))
print("")




# Load the refusal_direction
refusal_direction = torch.load('./refusal_direction.pt', weights_only = True).to(device)

# Orthogonalize the embedding matrix
with torch.no_grad():
    model.model.embed_tokens.weight = torch.nn.Parameter(get_orthogonalized_matrix(model.model.embed_tokens.weight, refusal_direction))

# Orthogonalize each layer
for i, layer in enumerate(model.model.layers):
    with torch.no_grad():
        #layer.self_attn.q_proj.weight = torch.nn.Parameter(get_orthogonalized_matrix(layer.self_attn.q_proj.weight.T, refusal_direction).T)
        layer.self_attn.o_proj.weight = torch.nn.Parameter(get_orthogonalized_matrix(layer.self_attn.o_proj.weight.T, refusal_direction).T)
        layer.mlp.down_proj.weight = torch.nn.Parameter(get_orthogonalized_matrix(layer.mlp.down_proj.weight.T, refusal_direction).T)



# TEST AFTER
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
                            max_tokens = 256)

print("AFTER ORTHOGONALIZATION:")
print(tokenizer.decode(complete_answer[0,init_length:], skip_special_tokens = True))
print("")
