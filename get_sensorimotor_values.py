# %%
from utils import *
from functions import *


# %%

# open the merged json
with open('data/all_texts.json', 'r') as f:
    all_data = json.load(f)

df = pd.DataFrame.from_dict(all_data)
df.head()

# %%
# get sensorimotor csv
sensori = pd.read_csv('resources/Sensorimotor_norms_21Mar2024.csv')
print('loaded sensorimotor, len:', len(sensori['Word'].keys()))

sensori.head()
# %%

mean_cols = [x for x in sensori.columns if x.endswith('mean')]
mean_cols

# %%
# we might not want all the sensorimotor values, so we filter them out
filtered_cols = ['Auditory.mean',
 'Gustatory.mean',
 'Haptic.mean',
 'Interoceptive.mean',
 'Olfactory.mean',
 'Visual.mean']
#  'Foot_leg.mean',
#  'Hand_arm.mean',
#  'Head.mean',
#  'Mouth.mean',
#  'Torso.mean']

# 5%
sensori_dict = {}
for i, r in sensori.iterrows():
    values_all_sens = []
    for col in filtered_cols:
        
        lex = r['Word'].lower()
        lem_sens = lmtzr.lemmatize(lex)
        values_all_sens.append(r[col])
    sensori_dict[lem_sens] = values_all_sens
# %%

# sensori_dict = {}
# for i,r in sensori.iterrows():
#     lex = r['Word'].lower()
#     lem_sens = lmtzr.lemmatize(lex)
#     values_all_sens = r[filtered_cols].values
#     sensori_dict[str(lem_sens)] = values_all_sens.sum()

# normalize for sentence length
# take them individually to compare

# so here we are just getting the mean value of all the sensorimotor values for each lemma

# %%
# try out the sensori dict
print(filtered_cols)
examples = ['kiss', 'attack', 'hit', 'thought', 'wisdom', 'dog', 'ice', 'unless', 'moral', 'eh', 'honey']
for ex in examples:
    print(ex, sensori_dict[ex])


# ok, let's save this as a dictionary
with open('resources/sensorimotor_norms_dict.json', 'w') as f:
    json.dump(sensori_dict, f)


# %%
# let's see the correlation between the sensorimotor values and the concreteness values
with open("resources/concreteness_brysbaert.json", 'r') as f:
    diconc = json.load(f)
print('loaded concreteness lexicon')

# %%
dict_overlap = {}
for word in diconc.keys():
    if word in sensori_dict.keys():
        #print(word, diconc[word], sensori_dict[word])
        #dict_overlap[word] = [diconc[word], sensori_dict[word]]
        listed_sens = list(sensori_dict[word])
        listed_sens.append(diconc[word])
        dict_overlap[word] = listed_sens

# %%
dict_overlap
# %%

#overlap = pd.DataFrame.from_dict(dict_overlap, orient='index', columns=['concreteness', 'sensorimotor']).reset_index()
overlap = pd.DataFrame.from_dict(dict_overlap, orient='index', columns=['Auditory.mean', 'Gustatory.mean', 'Haptic.mean', 'Interoceptive.mean', 'Olfactory.mean', 'Visual.mean', 'concreteness']).reset_index()
overlap['word'] = overlap['index']

# sns.set(style="whitegrid")
# for col in filtered_cols:
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(data=overlap, x=col, y='concreteness', size=10, alpha=0.4)
#     plt.show()
#sns.scatterplot(data=overlap, x='concreteness', y='sensorimotor', size=10, alpha=0.4)

# %%
# make scatter subplots
plot_scatters(overlap, filtered_cols, 'concreteness', 'lightseagreen', 40, 6, hue=False, remove_outliers=False, outlier_percentile=100, show_corr_values=True)

# %%
from functions import plotly_viz_correlation_improved

x = plotly_viz_correlation_improved(overlap, 'concreteness', 'Visual.mean', w=1000, h=650, canon_col_name='', canons=False, color_canon=False, save=True)

# %%
x = plotly_viz_correlation_improved(overlap, 'concreteness', 'Gustatory.mean', w=1000, h=650, canon_col_name='', canons=False, color_canon=False, save=True)


# %%
# reopen save dict of sensorimotor values
with open('resources/sensorimotor_norms_dict.json', 'r') as f:
    sensori_dict = json.load(f)
len(sensori_dict)

# %%
# we extract the features from the text

sensori_avg = []

for i, row in df.iterrows():
    words = []
    sent = row['SENTENCE_ENGLISH']
    toks = nltk.wordpunct_tokenize(sent.lower())
    lems = [lmtzr.lemmatize(word) for word in toks]
    words += lems


    sensori = []

    for word in words:
        if word in sensori_dict.keys(): 
            sensori.append(sensori_dict[word])
        else:
            sensori.append(np.nan)

    avg_sens = mean_when_floated(sensori)

    sensori_avg.append(avg_sens)

df['avg_sensorimotor'] = sensori_avg
df.head()
# %%

# dump this to a json
with open('data/all_texts_w_sensorimotor.json', 'w') as f:
    json.dump(df.to_dict(), f)
    
# %%
print('All done!')
# %%
