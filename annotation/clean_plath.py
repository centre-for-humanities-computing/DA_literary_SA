# %%

import pandas as pd
# %%
# open txt
with open('data/colossus_plath.txt', 'r') as f:
    all_data = f.readlines()

#all_data.split('\n\n')
poems_raw = [x.split('\\n\\n') for x in all_data]
# split the lists into sublists
poems_r = []
for poem in poems_raw:
    poem_r = []
    for line in poem:
        poem_r.append(line.split('\\n'))
    poems_r.append(poem_r)
#poems_r = [x.split('\\n') for x in poems_raw]

poems = poems_r[0]
len(poems)
# %%
poems
# %%
titles = [x[0] for x in poems]
all_lines = [x[0:] for x in poems]
all_lines = [item for sublist in all_lines for item in sublist]
all_lines = [x for x in all_lines if x != '']
# %%
len(all_lines)
# %%
# dump to json
import json
with open('data/plath_lines.json', 'w') as f:
    json.dump(all_lines, f)
# %%

# Clean ariel too

with open('annotation/ariel_clean.txt', 'r') as f:
    all_data = f.readlines()
all_listed_raw = [x for x in all_data if x != '\n']
all_listed_clean = [x.replace('\n', '') for x in all_listed_raw]
all_listed_clean

# %%
# split all data into poems, so whenever there is '', a new poem begins
unsplit = [x.replace('\n', '') for x in all_data]

# how to split the list on the uppercase lists?
split = []
for i in range(len(unsplit)):
    if unsplit[i].isupper():
        split.append(i)

poems = []
for i in range(len(split)):
    if i == 0:
        poems.append(unsplit[:split[i]-1])
    else:
        poems.append(unsplit[split[i-1]:split[i]])

poems
len(poems)

# %%
poems[39] # missing 1

# %%
all_listed_clean[-30:]
# %%
from nltk.tokenize import word_tokenize

all_words = []
for line in all_listed_clean:
    words = word_tokenize(line)
    all_words += words

len(all_words)

# %%
# dump to json
# with open('data/ariel_lines.json', 'w') as f:
#     json.dump(all_listed_clean, f)

# %%

# load hemingway and check number of words
with open('annotation/old_man.txt', 'r') as f:
    all_data = f.readlines()
# %%
joined = ''.join(all_data)
all_words = word_tokenize(joined)
len(all_words)
# %%
len(all_words)/1923

# %%
