# %%
from utils import *

# %%
with open('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/resources/mrc_dict_imageability.json', 'r') as f:
    dict_mrc = json.load(f)
# %%

# we want to lemmatize the keys

# Define a function to lemmatize a single key
def lemmatize_key(key):
    return lmtzr.lemmatize(key)

# Define a function to lemmatize all keys of a dictionary
def lemmatize_dict_keys(dictionary):
    lemmatized_dict = {}
    for key, value in dictionary.items():
        lemmatized_key = lemmatize_key(key).lower()
        lemmatized_dict[lemmatized_key] = value
    return lemmatized_dict

# Lemmatize the keys of the dictionary
lemmatized_dict = lemmatize_dict_keys(dict_mrc)
lemmatized_dict['was']['imag']

# imageability goes from 100-?
# Extract 'imag' values from each dictionary
imag_values = [entry['imag'] for entry in lemmatized_dict.values()]

# Calculate maximum and min
max_imag = max(imag_values)
min_imag = min(imag_values)

print("Maximum of 'imag' column:", max_imag)
print("Min of 'imag' column:", min_imag)
# %%

# set input path for data
input_path = 'data/EmoTales/emoTales.json' #'data/emobank_w_features_and_cats.json' #'data/FB_data_w_features.json' 
title = 'EmoTales'
print(title)
# texts should contain sentences and SA scores

# %%
with open(input_path, 'r') as f:
    all_data = json.load(f)

df = pd.DataFrame.from_dict(all_data)
#df.columns = ['ANNOTATOR_1', 'SENTENCE']
print(len(df))
df.head()

# %%
# loop through df

imageabilities_avg = []

for i, row in df.iterrows():
    words = []
    sent = row['SENTENCE']
    toks = nltk.wordpunct_tokenize(sent.lower())
    lems = [lmtzr.lemmatize(word) if word != 'was' else word for word in toks] # there's an error wit lemmatizing was
    words += lems

    imageabilities = []

    for word in words:
        if word in lemmatized_dict.keys():
            imageabilities.append(lemmatized_dict[word]['imag'])
            #print(lemmatized_dict[word]['imag'])
        else:
            imageabilities.append(np.nan)
    
    imageabilities_avg.append(np.nanmean(imageabilities))
# %%
df['imageability_avg'] = imageabilities_avg
df.head()

# %%
# dump to json

# %%
