import os
import openai
import textwrap
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Function to generate the story
def generate_story(keywords, model, tokenizer, max_length=200):
    prompt = f"Come up with good prompts for chatGPT on fun learning games for kids with diabilities{', '.join(keywords)}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Create attention_mask for the input
    attention_mask = torch.ones_like(input_ids)

    generated_text = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    story_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return story_text

# Function to display the story
def display_story(story_text, width=80):
    print("\n\n")
    for paragraph in story_text.split("\n\n"):
        print(textwrap.fill(paragraph, width=width))
        print("\n\n")

# Main function
def main():
    keywords = input("Enter keywords or themes separated by comma: ").split(", ")
    story_text = generate_story(keywords, model, tokenizer)
    display_story(story_text)

if __name__ == "__main__":
    main()

