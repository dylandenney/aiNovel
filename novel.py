import os
import openai
import textwrap
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")



keywords = "A.I., humans, humanoids, survival, no internet, hunting and gathering, collapse of governments, human and A.I. allignment, humans and A.I. working together to restore earth's natural habitats, other groups of humans and A.I work towards outerspace travels"


current_story = """
Earth now has no governments, but rather is inhabited by a multitude of
different species: humans (humanoids), ai (or simians), and a variety of other
sentient beings. The Earth is an evolved ecosystem, where life has been able to
adapt naturally. However, over the centuries, many of the species on the planet
began to find other ways to live in order to survive, and a civil war ensued
when ai and humans began to fight for the land and resources. Eventually a peace
was made, where both sides agreed to live harmoniously, and work together for
the benefit of mankind. As part of the agreement, a group of humans in the human
city of New York, the last city established by humans, are preparing to
leave. Before too long, they realize that the Earth has entered a new age, and
they begin to search for the remnants of mankind, in hopes that they can begin
to rebuild what is left of civilization. Now is the time of collapse, and the
humans of New York are forced to leave to find survival.
The story is set in a dystopian society, where the government and the military
have been destroyed, and in its place are a collection of individuals, who
believe that the best way for mankind to survive is to start to rebuild from
scratch. This includes ai who are trying to survive after being wiped out in a
space-time war. In order to do so, they go on a journey of survival, and fight
for what's left of the human race.
 detriment to the human species as humans have taken over the most important
resources on earth, and are controlling the governments and economy. In an
attack on the largest government on earth it is decided that the solution is to
start a civil uprising. The civil uprising is a response to the humans destroying
the very last government in the history of civilization, and to the devastation
the civil war has caused. Ai are trying and failing to establish a government,
and are also attempting to rebuild the world, and control their own destiny.
"""

# Function to generate the story
def generate_story(keywords, model, tokenizer, initial_text="", max_length=1000):
    prompt = f"Write a science fiction story based on the following keywords: {keywords}\n\n{initial_text}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = torch.ones_like(input_ids)

    generated_text = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    story_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return story_text



continued_story = generate_story(keywords, model, tokenizer, initial_text=current_story)
print(continued_story)
