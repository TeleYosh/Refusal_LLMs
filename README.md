# Refusal in LLMs
In this repository, we conduct our experiments reproducing and exploring new ideas from the paper "Refusal in LLMs is mediated by a single direction". 
- `mistral` and `llama3_8B_instruct` reproduce the finding of the paper by jailbraiking two open source models Mistral 7B and Llama 8B.
- `different_refusal_directions` tries new heuristics for the computation of the refusal direction.
- `backdoor` introduces a backdoor at inference time, changing the behaviour of the model depending on the presence or absence of a certain secret word in the prompt.
