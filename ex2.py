from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Mistral initialization
mistral_name = "mistralai/Mistral-7B-Instruct-v0.3"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_name)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_name, torch_dtype=torch.float16, device_map="auto").eval()

# Llama initialization
llama_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_name, torch_dtype=torch.float16, device_map="auto").eval()

# Create prompts
prompts = [
    "Tom lent his bike to Peter because his was broken. Whose bike was broken?",
    "Jenny handed the binoculars to Rachel because she had the clearer view. Who had the clearer view?",
    "Laura lent her favorite umbrella to Sophia because hers was torn. Whose umbrella was torn?",
    "Carla passed the microphone to Olivia because she had the louder voice. Who had the louder voice?",
    "The engineers called the contractors because they had missed the deadline. Who had missed the deadline?"
]


def ask_model(model, tokenizer, prompt, max_new_tokens: int = 128):
    """
    Return the raw text that the model generates for a single prompt.
    :param model: model to generate text with.
    :param tokenizer: tokenizer of the given model.
    :param prompt: prompt to generate text for.
    :param max_new_tokens: maximum number of tokens to generate.
    :return: generated text with the model given the prompt.
    """
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**tokens, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# Generate output of Mistral
for prompt in prompts:
    print(ask_model(mistral_model, mistral_tokenizer, prompt))
    print()

# Test hypothesis
test_prompts = [
    "Because his was brand-new, Peter borrowed Tom's bike. Whose bike was brand-new?",
    "Peter was lent Tom's bike because his was broken. Whose bike was broken?",
    "Olivia was handed the microphone by Carla because she had a louder voice. Who had a louder voice?"
]

for prompt in test_prompts:
    print(ask_model(mistral_model, mistral_tokenizer, prompt))
    print()

# Generate output of Llama
for prompt in prompts:
    print(ask_model(llama_model, llama_tokenizer, prompt))
    print()
