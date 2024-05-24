# %%

from transformers import AutoTokenizer, CLIPTextModelWithProjection

# %%

# with visual information (TextModelWithProjection)
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

outputs = model(**inputs)
text_embeds = outputs.text_embeds
# %%
text_embeds
# %%
import torch

# Normalize the embeddings
text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)

# Calculate cosine similarity
similarity = torch.mm(text_embeds, text_embeds.T)
print(similarity)

# %%

from transformers import AutoTokenizer, CLIPTextModelWithProjection

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize the input text
text = "cat"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Pass the tokenized inputs to the model
outputs = model(**inputs)
token_embeds = outputs.last_hidden_state

# Map token IDs back to words and print token embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, embed in zip(tokens, token_embeds[0]):
    print(f"Token: {token}, Embedding: {embed}")

# %%
# %%
