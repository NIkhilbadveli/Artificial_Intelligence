"""
We will use DialoGPT for this task.
DialoGPT is a large-scale tunable neural conversational response generation model trained on 147M conversations
extracted from Reddit.
The good thing is that you can fine-tune it with your dataset to achieve better performance than training from scratch.
https://www.thepythoncode.com/article/conversational-ai-chatbot-with-huggingface-transformers-in-python - for ref.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/DialoGPT-medium"  # small and large are also available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def greedy_search_bot():
    """
    This function will return the response from DialoGPT and it uses greedy search.
    """
    # Chat for 5 times
    chat_history = []
    for i in range(5):
        text = input('>> You: ')
        encoded_text = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        model_input = torch.cat([encoded_text, chat_history], dim=-1) if i > 0 else encoded_text
        chat_history = model.generate(model_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(chat_history[:, model_input.shape[-1]:][0], skip_special_tokens=True)
        print(f'>> Bot: {output}')


def beam_search_bot():
    """
    This function will return the response from DialoGPT, and it uses beam search.
    """
    # Chat for 5 times
    chat_history = []
    for i in range(5):
        text = input('>> You: ')
        encoded_text = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        model_input = torch.cat([encoded_text, chat_history], dim=-1) if i > 0 else encoded_text
        chat_history = model.generate(model_input, max_length=1000, pad_token_id=tokenizer.eos_token_id, num_beams=3,
                                      early_stopping=True)
        output = tokenizer.decode(chat_history[:, model_input.shape[-1]:][0], skip_special_tokens=True)
        print(f'>> Bot: {output}')


def topk_sample_bot():
    """
    This function will return the response from DialoGPT, and it uses topK sampling.
    """
    # Chat for 5 times
    chat_history = []
    for i in range(5):
        text = input('>> You: ')
        encoded_text = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        model_input = torch.cat([chat_history, encoded_text], dim=-1) if i > 0 else encoded_text
        chat_history = model.generate(model_input, max_length=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True,
                                      temperature=0.75, top_k=100)
        output = tokenizer.decode(chat_history[:, model_input.shape[-1]:][0], skip_special_tokens=True)
        print(f'>> Bot: {output}')


topk_sample_bot()
