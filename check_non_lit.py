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
    toks = nltk.wordpunct_tokenize(str(sent).lower())
    lems = [lmtzr.lemmatize(word) for word in toks]
    words += lems

    valences, arousals, dominances, concreteness = [], [], [], []

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
valences_avg
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
from transformers import pipeline


# %%
# twitter-xlm-roberta-base-sentiemnt
def conv_scores(lab, sco, spec_lab): #insert exact labelnames in order positive, negative og as positive, neutral, negative
    
    converted_scores = []
    
    if len(spec_lab) == 2:
        spec_lab[0] = "positive"
        spec_lab[1] = "negative"

        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            else:
                converted_scores.append(-sco[i])
            
    if len(spec_lab) == 3:
        spec_lab[0] = "positive"
        spec_lab[1] = "neutral"
        spec_lab[2] = "negative"
        
        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            elif lab[i] == "neutral":
                converted_scores.append(0)
            else:
                converted_scores.append(-sco[i])
    
    return converted_scores

# %%
xlm_model = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# %%
xlm_labels = []
xlm_scores = []

for s in eb['text']:
    sent = xlm_model(s)
    xlm_labels.append(sent[0].get("label"))
    xlm_scores.append(sent[0].get("score"))

xlm_converted_scores = conv_scores(xlm_labels, xlm_scores, ["positive", "neutral", "negative"])

# %%
eb["cardiffnlp/twitter-xlm-roberta-base-sentiment"] = xlm_converted_scores

# %% 
# and then we also need the VADER and stuff but work in progress