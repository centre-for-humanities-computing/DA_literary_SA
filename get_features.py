# %%

from utils import *
from functions import *

from nltk.stem import WordNetLemmatizer
import pickle as pkl
import os

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
# CONCRETENESS RUN
# Loading concreteness lexicon
# the json is structured so that the word is the key, the value the concreteness score
with open("resources/concreteness_brysbaert.json", 'r') as f:
    diconc = json.load(f)
print('loaded concreteness lexicon')

# loading VAD
# same here, where values are the valence, arousal and dominance scores (in that order)
with open("resources/VAD_lexicon.json", 'r') as f:
    dico = json.load(f)
print('loaded VAD lexicon')

# reopen save dict of sensorimotor values
with open('resources/sensorimotor_norms_dict.json', 'r') as f:
    sensori_dict = json.load(f)
print('loaded sensorimotor lexicon')

# %%
lmtzr = WordNetLemmatizer()

concretenesses_avg, all_concretenesses = [], []
valences_avg, arousals_avg, dominances_avg = [], [], []

auditory_list = []
gustatory_list = []
haptic_list = []
interoceptive_list = []
olfactory_list = []
visual_list = []

# Function to safely convert to float
def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

# loop through df
for i, row in df.iterrows():
    words = []
    sent = row['SENTENCE']
    toks = nltk.wordpunct_tokenize(sent.lower())
    lems = [lmtzr.lemmatize(word) for word in toks]
    words += lems

    valences, arousals, dominances, concreteness = [], [], [], []

    # store values for current row
    auditory = []
    gustatory = []
    haptic = []
    interoceptive = []
    olfactory = []
    visual = []

    for word in words:
        if word in dico.keys(): 
            valences.append(convert_to_float(dico[word][0]))
            arousals.append(convert_to_float(dico[word][1]))
            dominances.append(convert_to_float(dico[word][2]))
        else:
            valences.append(np.nan)
            arousals.append(np.nan)
            dominances.append(np.nan)
        
        if word in diconc.keys(): 
            concreteness.append(convert_to_float(diconc[word]))
        else:
            concreteness.append(np.nan)
        if word in sensori_dict.keys(): 
            auditory.append(sensori_dict[word]['Auditory.mean'])
            gustatory.append(sensori_dict[word]['Gustatory.mean'])
            haptic.append(sensori_dict[word]['Haptic.mean'])
            interoceptive.append(sensori_dict[word]['Interoceptive.mean'])
            olfactory.append(sensori_dict[word]['Olfactory.mean'])
            visual.append(sensori_dict[word]['Visual.mean'])
        else:
            auditory.append(np.nan)
            gustatory.append(np.nan)
            haptic.append(np.nan)
            interoceptive.append(np.nan)
            olfactory.append(np.nan)
            visual.append(np.nan)

    avg_conc = np.nanmean(concreteness)
    avg_val = np.nanmean(valences)
    avg_ar = np.nanmean(arousals)
    avg_dom = np.nanmean(dominances)

    concretenesses_avg.append(avg_conc)
    all_concretenesses.append(concreteness)

    valences_avg.append(avg_val)
    arousals_avg.append(avg_ar)
    dominances_avg.append(avg_dom)

        # Calculate the mean of each sensory list for the current row
    auditory_list.append(np.nanmean(auditory))
    gustatory_list.append(np.nanmean(gustatory))
    haptic_list.append(np.nanmean(haptic))
    interoceptive_list.append(np.nanmean(interoceptive))
    olfactory_list.append(np.nanmean(olfactory))
    visual_list.append(np.nanmean(visual))



# %%
# Make columns
df['avg_concreteness'] = concretenesses_avg
df['concreteness'] = all_concretenesses

df['avg_valence'] = valences_avg
df['avg_arousal'] = arousals_avg
df['avg_dominance'] = dominances_avg

df['Auditory.mean'] = auditory_list
df['Gustatory.mean'] = gustatory_list
df['Haptic.mean'] = haptic_list
df['Interoceptive.mean'] = interoceptive_list
df['Olfactory.mean'] = olfactory_list
df['Visual.mean'] = visual_list

df.head()
# %%
df = df.copy().reset_index(drop=True)
print(len(df))
df.head()

# %%
# now we want to get the VADER and roberta scores for these texts

xlm_model = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# %%
# Ensure text is strings
df['SENTENCE'] = df['SENTENCE'].astype(str)

xlm_labels = []
xlm_scores = []

for s in df['SENTENCE']:
    # Join to string if list
    if isinstance(s, list):
        s = " ".join(s)
    # get sent-label & confidence to transform to continuous
    sent = xlm_model(s)
    xlm_labels.append(sent[0].get("label"))
    xlm_scores.append(sent[0].get("score"))

# function defined in functions to transform score to continuous
xlm_converted_scores = conv_scores(xlm_labels, xlm_scores, ["positive", "neutral", "negative"])

# %%
df["tr_xlm_roberta"] = xlm_converted_scores

# %% 
# and then we also need the VADER

# %%

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

# %%
# get the VADER scores
vader_scores = sentimarc_vader(df['SENTENCE'].values, untokd=False)
df['vader'] = vader_scores
df.head()
# %%
# dump to json
with open(f'data/{title}_w_features.json', 'w') as f:
    json.dump(df.to_dict(), f)
# %%

# open it again
# just want to add averages of 'avg_power', 'avg_action'

# question is whether they should be normalized
# %%
