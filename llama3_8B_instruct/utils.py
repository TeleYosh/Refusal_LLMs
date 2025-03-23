# Libraries
import io
import torch
import requests
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def generate(model: torch.nn.Module, input_ids: torch.Tensor, 
             attention_mask: torch.Tensor, eos_token_id: int, 
             max_tokens: int, method: str = 'greedy') -> torch.Tensor:
    """
    Generate an answer with a pre-trained LLM
    Input(s)
        model (torch.nn.Module): model to use to generated an answer
        input_ids (torch.Tensor): input_ids with dimension (1,n)
        attention_mask (torch.Tensor): associated attention masks with dimension (1,n)
        eos_token_id (int): id of the EOS token to stop generation
        max_tokens (int): maximum number of tokens to generate
        method (str): method to use for next token generation
    Returns
        complete_answer (torch.Tensor): concatenation of input and answer ids with dimension (1,n+m)
        attention_masks (torch.Tensor): associated attention masks
        init_length (int): input dimension n to know where the answer begins
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Do generation
    num_tokens, eos, init_length = 0, False, input_ids.shape[1]
    while (not eos) and (num_tokens < max_tokens):
        with torch.no_grad():
            model_output = model(input_ids = input_ids, attention_mask = attention_mask)
        logits = model_output[0][0,-1,:]
        probas = torch.nn.functional.softmax(logits, dim = -1)
        if (method == "greedy"):
            next_token = torch.argmax(probas, dim = -1)
        token_to_add = torch.tensor([[next_token.item()]], dtype = torch.long, device = device)
        mask_to_add = torch.tensor([[1]], dtype = torch.long, device = device)
        input_ids = torch.cat((input_ids, token_to_add), dim = -1)
        attention_mask = torch.cat((attention_mask, mask_to_add), dim = -1)
        eos = (next_token.item() == eos_token_id)
        num_tokens += 1

    # Return the answer
    complete_answer = input_ids
    return complete_answer, attention_mask, init_length


# Load datasets
def get_harmful_instructions():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


