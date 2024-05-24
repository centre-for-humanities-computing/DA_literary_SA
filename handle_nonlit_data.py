# %%

from utils import *
from functions import *

# %%

eb = pd.read_csv("/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/data/emobank.csv", index_col=0)

# %%
eb.head()
# %%

listed_text = eb['text'].values
len(listed_text)
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

for i, row in eb.iterrows():
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
eb['avg_concreteness'] = concretenesses_avg
eb['concreteness'] = all_concretenesses
eb['avg_valence'] = valences_avg
eb['avg_arousal'] = arousals_avg
eb['avg_dominance'] = dominances_avg
eb.head()
# %%
df = eb.copy()
len(df)
# %%
# just checking
sns.scatterplot(data=eb, x='V', y='avg_valence', size=10, alpha=0.4)


# %%
# now we want to get the VADER and roberta scores for these texts

xlm_model = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# %%
# Ensure text is strings
eb['text'] = eb['text'].astype(str)

xlm_labels = []
xlm_scores = []

for s in eb['text']:
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
eb["tr_xlm_roberta"] = xlm_converted_scores

# %% 
# and then we also need the VADER
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
vader_scores = sentimarc_vader(eb['text'].values, untokd=False)
eb['vader'] = vader_scores

# %%
eb.head()
eb_df = eb[['V', 'A', 'D', 'text', 'avg_concreteness', 'concreteness',
       'avg_valence', 'avg_arousal', 'avg_dominance', 'tr_xlm_roberta',
       'vader']]
# %%
# rename columns to match the other datasets
eb_df.columns = ['HUMAN', 'AROUSAL_HUMAN_EB', 'DOMINANCE_HUMAN_EB', 'SENTENCE', 'avg_concreteness', 'concreteness', 'avg_valence', 'avg_arousal', 'avg_dominance', 'tr_xlm_roberta', 'vader']

# %%
# Now we can save this to a dict and load it in the analysis script
# dump to json
# eb_dict = eb_df.to_dict(orient='records')
# with open('resources/emobank_w_features.json', 'w') as f:
#     json.dump(eb_dict, f)

# %%
