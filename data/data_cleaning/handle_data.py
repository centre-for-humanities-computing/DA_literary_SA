# %%

from utils import *
from functions import *
import json
import pandas as pd

# %%
with open('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/data/all_DANISH_w_sensorimotor.json', 'r') as f:
    all_data = json.load(f)

all_data
# %%
data = pd.DataFrame.from_dict(all_data)
data.head()
# %%
data['CATEGORY'].value_counts()
# %%
# # make category column where category == 'fairytale' if id is string
# data['CATEGORY'] = data['id'].apply(lambda x: 'fairytales' if x in ['havfrue', 'aelling', 'skyggen'] else 'hymns')
# data.tail()
# %%
# update the json with the new file
# with open('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/data/all_texts_w_sensorimotor.json', 'w') as f:
#     json.dump(data.to_dict(), f)
# %%
data.columns

len(data)

# %% 
# add vader to danish data

sid =  SentimentIntensityAnalyzer()

def sentimarc_vader(text, untokd=True):
    if untokd:
        sents = nltk.sent_tokenize(text)
        print(len(sents))
    else: sents = text
    arc=[]
    for sentence in sents:
        compound_pol = sid.polarity_scores(sentence)['compound']
        arc.append(compound_pol)
    return arc

vader_scores = sentimarc_vader(data['SENTENCE_ENGLISH'].values, untokd=False)
data['vader'] = vader_scores
data.head()



# %% merge hemingway and hymns/hca data
with open('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/data/HEMINGWAY_w_features.json', 'r') as f:
    hemingway = json.load(f)

hemingway_df = pd.DataFrame.from_dict(hemingway)
hemingway_df.columns
len(hemingway_df)
# %%
hemingway_df.columns = ['ANNOTATOR_1', 'ANNOTATOR_2', 'HUMAN', 'SENTENCE', 'tr_base', 'tr_stars',
       'tr_twt_roberta', 'tr_xlm_roberta', 'vader', 'avg_concreteness',
       'concreteness', 'avg_valence', 'avg_arousal', 'avg_dominance',
       'Auditory.mean', 'Gustatory.mean', 'Haptic.mean', 'Interoceptive.mean',
       'Olfactory.mean', 'Visual.mean']


# %%
hemingway_df['id'] = 'hemingway'
hemingway_df['CATEGORY'] = 'prose'
hemingway_df.head()
# %%
# merge
#merged = pd.concat([data, hemingway_df], axis=0).reset_index(drop=True)
merged = pd.concat([data, hemingway_df], axis=0).reset_index(drop=True)
merged.tail()
# %%
len(merged)
# %%
# make it a dict
merged_dict = merged.to_dict()
len(merged['id'])

# %%
merged.columns
# %%
# to json
with open('data/all_texts_w_sensorimotor.json', 'w') as f:
    json.dump(merged_dict, f)
# %%
