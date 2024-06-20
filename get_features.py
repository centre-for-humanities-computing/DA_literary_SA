
# %%
# 
from utils import *
from functions import *


# %%
# set input path for data
input_path = 'data/fiction4_data.json'
title = input_path.split('/')[1].split('_')[0]
print('data treated:', title.upper())
# texts should contain sentences and SA annotated scores

# %%
with open(input_path, 'r') as f:
    all_data = json.load(f)

df = pd.DataFrame.from_dict(all_data)
print('len data:', len(df))
df.head()


# %%
# PART 1: loading dicts, getting feature values
print('# PART 1: loading dicts, getting feature values')

# Loading concreteness lexicon
# the json is structured so that the word is the key, the value the concreteness score
with open("resources/concreteness_brysbaert.json", 'r') as f:
    diconc = json.load(f)
print('loaded concreteness lexicon, len:', len(diconc))

# loading VAD
# same here, where values are the valence, arousal and dominance scores (in that order)
with open("resources/VAD_lexicon.json", 'r') as f:
    dico = json.load(f)
print('loaded VAD lexicon, len:', len(dico))

# reopen save dict of sensorimotor values
with open('resources/sensorimotor_norms_dict.json', 'r') as f:
    sensori_dict = json.load(f)
print('loaded sensorimotor lexicon, len:', len(sensori_dict))

# and get the imageability dict from MRC psycholinguistics database
with open('Resources/mrc_psychol_dict.json', 'r') as f:
    dict_mrc = json.load(f)
print('loaded imageability lexicon, len:', len(dict_mrc))

# this one needs some extra cleaning, since keys are not lemmatized

# %%
words = ['dog', 'feeling', 'stomach', 'outside', 'tree', 'heart', 'stone']
print('interoceptive values test')
for word in words:
    print(word, sensori_dict[word]['Interoceptive.mean'])

# %%
[x for x in df['SENTENCE_ENGLISH'] if len(x) < 1]
# %%

concretenesses_avg, all_concretenesses = [], []
valences_avg, arousals_avg, dominances_avg = [], [], []

auditory_list = []
gustatory_list = []
haptic_list = []
interoceptive_list = []
olfactory_list = []
visual_list = []

imageability_avg = []

datasets_english = ['emobank', 'emotales', 'FB', 'plath']

if title in datasets_english:
    use_col = 'SENTENCE'
else:
    print("Using col 'SENTENCE_ENGLISH' for calculating the dictionary values")
    use_col = 'SENTENCE_ENGLISH'

# loop through df
for i, row in df.iterrows():
    words = []
     # make sure we're using the english sentence (also for Danish texts)
    sent = row[use_col]

    toks = nltk.wordpunct_tokenize(sent.lower())
    lems = [lmtzr.lemmatize(word) for word in toks]
    words += lems

    # lists to store values for current row
    valences, arousals, dominances, concreteness = [], [], [], []

    auditory = []
    gustatory = []
    haptic = []
    interoceptive = []
    olfactory = []
    visual = []

    imageabilities = []
    
    # get the VAD values
    for word in words:
        if word in dico.keys(): 
            valences.append(convert_to_float(dico[word][0]))
            arousals.append(convert_to_float(dico[word][1]))
            dominances.append(convert_to_float(dico[word][2]))
        else:
            valences.append(np.nan)
            arousals.append(np.nan)
            dominances.append(np.nan)
        
        # get concreteness
        if word in diconc.keys(): 
            concreteness.append(np.nanmean(diconc[word]))
        else:
            concreteness.append(np.nan)

        # get the sensorimotor values
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
        
        # get imageability
        if word in dict_mrc.keys():
            imageabilities.append(dict_mrc[word]['imag'])
        else:
            imageabilities.append(np.nan)
    

    # save everything and get the means per sentence
    valences_avg.append(np.nanmean(valences))
    arousals_avg.append(np.nanmean(arousals))
    dominances_avg.append(np.nanmean(dominances))

    concretenesses_avg.append(np.nanmean(concreteness))
    all_concretenesses.append(concreteness)

    auditory_list.append(np.nanmean(auditory))
    gustatory_list.append(np.nanmean(gustatory))
    haptic_list.append(np.nanmean(haptic))
    interoceptive_list.append(np.nanmean(interoceptive))
    olfactory_list.append(np.nanmean(olfactory))
    visual_list.append(np.nanmean(visual))

    imageability_avg.append(np.nanmean(imageabilities))



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

df['avg_imageability'] = imageability_avg


df.head()
# %%
# checkup
df = df.copy().reset_index(drop=True)
print(len(df))
df.head()

# %%
# PART 2: sentiment analysis
print('# PART 2: sentiment analysis')
# now we want to get the VADER and roberta scores for these texts

xlm_model = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")


# %%
#
if title in datasets_english: # make sure we're using the english sentence (also for Danish texts)
        use_col = 'SENTENCE'
        # Ensure text is strings
        df['SENTENCE'] = df['SENTENCE'].astype(str)
else:
    print('check that you use the right col for the mixed language dataset, set it manually')
    use_col = 'SENTENCE_ENGLISH'
    print(f'using col {use_col}')
    # # Ensure text is strings
    # df['SENTENCE_ENGLISH'] = df['SENTENCE_ENGLISH'].astype(str)

xlm_labels = []
xlm_scores = []

for s in df[use_col]:
    # Join to string if list
    if isinstance(s, list):
        s = " ".join(s)
    # get sent-label & confidence to transform to continuous
    sent = xlm_model(s)
    xlm_labels.append(sent[0].get("label"))
    xlm_scores.append(sent[0].get("score"))

# function defined in functions to transform score to continuous
xlm_converted_scores = conv_scores(xlm_labels, xlm_scores, ["positive", "neutral", "negative"])
df["tr_xlm_roberta"] = xlm_converted_scores


# %%
# get the VADER scores
vader_scores = sentimarc_vader(df[use_col].values, untokd=False)
df['vader'] = vader_scores
df.head()

# %%
# Check for nan values

nan_counts = df.isna().sum()
print("NaN counts per column:")
print(nan_counts)

nan_rows_annotators = df[df[['HUMAN', 'tr_xlm_roberta']].isna().any(axis=1)]
print("Rows with NaN values in SA columns:")
print(nan_rows_annotators)

# %%
# dump to json
#with open(f'{input_path}', 'w') as f:
with open('fiction4_20june.json', 'w') as f:
    json.dump(df.to_dict(), f)
# %%
print(f'treated {title.upper()}: \n VAD, concreteness, sensorimotor and imageability calculated \n -- json updated!')



# %% 
# We want to get the correlation between RoBERTa and Humans for each of our datasets
# %%
# get the overall correlation between human/vader
correlation_results = stats.spearmanr(df['tr_xlm_roberta'], df['HUMAN'])
corr_value = round(correlation_results[0], 3)
p_value = round(correlation_results[1], 5)
print('correlation, human vs roberta (all on ENGLISH):', corr_value, p_value)


# %%
# for the datasets that have different categories, we want to check the correlation for each category with the human mean
map_categories = {'emobank': 'category', 'fiction4': 'CATEGORY'}

if title in map_categories:
    for cat in df[map_categories[title]].unique():
        dat = df.loc[df[map_categories[title]] == cat]

        correlation_results = stats.spearmanr(dat['tr_xlm_roberta'], dat['HUMAN'])
        corr_value = round(correlation_results[0], 3)
        p_value = round(correlation_results[1], 5)
        print(f'correlation, human vs roberta for cat: {cat}', corr_value, p_value)



# %%
# for the bilingual dataset, check the difference between applying roberta in each language
if title == 'fiction4':

    # make two groups based on original language
    dk_data = df.loc[(df['CATEGORY'] == 'hymns') | (df['CATEGORY'] == 'fairy tales')]
    print(len(dk_data))
    en_data = df.loc[(df['CATEGORY'] == 'prose') | (df['CATEGORY'] == 'poetry')]
    print(len(en_data))

    # first print the correlation of the SA on English translations (already computed)
    # just isolating the originally danish texts
    correlation_results = stats.spearmanr(dk_data['tr_xlm_roberta'], dk_data['HUMAN'])
    corr_value = round(correlation_results[0], 3)
    p_value = round(correlation_results[1], 5)
    print('correlation, human vs roberta DANISH texts but SA on ENGLISH translations:', corr_value, p_value)

    # and we print the correlation between human/roberta for the originally english texts
    correlation_results = stats.spearmanr(en_data['tr_xlm_roberta'], en_data['HUMAN'])
    corr_value = round(correlation_results[0], 3)
    p_value = round(correlation_results[1], 5)
    print('correlation, human vs roberta (SA on original ENGLISH fiction4):', corr_value, p_value)

    # then we want to redo the roberta values directly on the danish texts
    xlm_labels = []
    xlm_scores = []

    for s in dk_data['SENTENCE']:
        # Join to string if list
        if isinstance(s, list):
            s = " ".join(s)
        # get sent-label & confidence to transform to continuous
        sent = xlm_model(s)
        xlm_labels.append(sent[0].get("label"))
        xlm_scores.append(sent[0].get("score"))

    # function defined in functions to transform score to continuous
    xlm_converted_scores = conv_scores(xlm_labels, xlm_scores, ["positive", "neutral", "negative"])
    dk_data["tr_xlm_roberta"] = xlm_converted_scores

    correlation_results = stats.spearmanr(dk_data['tr_xlm_roberta'], dk_data['HUMAN'])
    corr_value = round(correlation_results[0], 3)
    p_value = round(correlation_results[1], 5)
    print('correlation, human vs roberta (SA on DANISH fiction4):', corr_value, p_value)




# %%
