# %%

from utils import *
from functions import *

# %%

fb_dat = pd.read_csv("data/dataset-fb-valence-arousal-anon.csv", index_col=0)
fb_dat.head()
# %%
# get rid of mulitlevel index
fb_dat = fb_dat.reset_index(drop=False)
fb_dat.head()
# %%

listed_text = fb_dat['Anonymized Message'].values
len(listed_text)
# %%
fb_dat['text'] = fb_dat['Anonymized Message']

# %%
# check len of messages
lens = [len(str(x)) for x in fb_dat['text']]
print('max:', max(lens), 'min:', min(lens), 'mean:', np.mean(lens))
fb_dat['textlen'] = lens
# let's set a max of 514 (which is max for roberta)
fb_dat = fb_dat.loc[fb_dat['textlen'] < 514]
len(fb_dat)


# %%
# the json is structured so that the word is the key, the value the concreteness score
with open("resources/concreteness_brysbaert.json", 'r') as f:
    diconc = json.load(f)
print('loaded concreteness lexicon')

# loading VAD
# same here, where values are the valence, arousal and dominance scores (in that order)
with open("resources/VAD_lexicon.json", 'r') as f:
    dico = json.load(f)
print('loaded VAD lexicon')
# %%
# now get values of words in each sentence and the means
concretenesses_avg, all_concretenesses = [], []
valences_avg, arousals_avg, dominances_avg = [], [], []

for i, row in fb_dat.iterrows():
    words = []
    sent = row['text']
    # tokenize and lemmatize
    toks = nltk.wordpunct_tokenize(str(sent).lower())
    lems = [lmtzr.lemmatize(word) for word in toks]
    words += lems

    # get text features
    valences, arousals, dominances, concreteness = [], [], [], []
    # match with dictionaries
    for word in words:
        if word in dico.keys(): 
            valences.append(dico[word][0])
            arousals.append(dico[word][1])
            dominances.append(dico[word][2])
        else:
            valences.append(np.nan)
            arousals.append(np.nan)
            dominances.append(np.nan)
        
        if word in diconc.keys(): 
            concreteness.append(diconc[word])
        else:
            concreteness.append(np.nan)

    # get mean values per sentence
    avg_conc = mean_when_floated(concreteness)
    avg_val = mean_when_floated(valences)
    avg_ar = mean_when_floated(arousals)
    avg_dom = mean_when_floated(dominances)

    concretenesses_avg.append(avg_conc)
    all_concretenesses.append(concreteness)
    valences_avg.append(avg_val)
    arousals_avg.append(avg_ar)
    dominances_avg.append(avg_dom)

# %%
# Make columns
fb_dat['avg_concreteness'] = concretenesses_avg
fb_dat['concreteness'] = all_concretenesses
fb_dat['avg_valence'] = valences_avg
fb_dat['avg_arousal'] = arousals_avg
fb_dat['avg_dominance'] = dominances_avg
fb_dat.head()

# %%
# just checking
fb_dat['HUMAN'] = (fb_dat['Valence1'] + fb_dat['Valence2']) / 2
sns.scatterplot(data=fb_dat, x='HUMAN', y='avg_valence', size=10, alpha=0.4)

#%%

fb_dat['harousal'] = (fb_dat['Arousal1'] + fb_dat['Arousal2']) / 2
sns.scatterplot(data=fb_dat, x='harousal', y='avg_arousal', size=10, alpha=0.4)



# %%
# now we want to get the VADER and roberta scores for these texts

xlm_model = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# %%
# Ensure text is strings
fb_dat['text'] = fb_dat['text'].astype(str)

xlm_labels = []
xlm_scores = []

for s in fb_dat['text']:
    # Join to string if list
    if isinstance(s, (float, int)):
        s = str(s)
    elif isinstance(s, list):
        s = " ".join(s)
    # get sent-label & confidence to transform to continuous
    sent = xlm_model(s)
    # Get the sentiment analysis result, truncating the input if it's too long
    #sent = xlm_model(s, truncation=True)
    xlm_labels.append(sent[0].get("label"))
    xlm_scores.append(sent[0].get("score"))
    
# function defined in functions to transform score to continuous
xlm_converted_scores = conv_scores(xlm_labels, xlm_scores, ["positive", "neutral", "negative"])

# %%
fb_dat["tr_xlm_roberta"] = xlm_converted_scores

# %%
# get the VADER scores
vader_scores = sentimarc_vader(fb_dat['text'].values, untokd=False)
fb_dat['vader'] = vader_scores

# %%
fb_dat.columns
# %%
fb_dat.head()

# %%
fb_dat_df = fb_dat[['HUMAN', 'harousal', 'Valence1', 'Valence1', 'Arousal1', 'Arousal2', 'text', 'avg_concreteness', 'concreteness',
       'avg_valence', 'avg_arousal', 'avg_dominance', 'tr_xlm_roberta',
       'vader']]
# %%
# rename columns to match the other datasets
fb_dat_df.columns = ['HUMAN', 'harousal', 'VALENCE_HUMAN_1_FB', 'VALENCE_HUMAN_2_FB', 'AROUSAL_HUMAN_1_FB', 'AROUSAL_HUMAN_2_FB', 'SENTENCE', 
                     'avg_concreteness', 'concreteness', 'avg_valence', 'avg_arousal', 'avg_dominance', 'tr_xlm_roberta', 'vader']

# %%

fb_dat_df.head()

# %%
# Now we can save this to a dict and load it in the analysis script
# dump to json
# fb_dat_dict = fb_dat_df.to_dict(orient='records')
# with open('data/FB_data_w_features.json', 'w') as f:
#     json.dump(fb_dat_dict, f)

# %%