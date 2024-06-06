# %%
from utils import *
from functions import *


# %%

# open the merged json
with open('data/FB_data_w_features.json', 'r') as f:
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
        auditory, gustatory, haptic, interoceptive, olfactory, visual = r['Auditory.mean'], r['Gustatory.mean'], r['Haptic.mean'], r['Interoceptive.mean'], r['Olfactory.mean'], r['Visual.mean']

        lex = r['Word'].lower()
        lem_sens = lmtzr.lemmatize(lex)
        #values_all_sens.append(r[col]) # for taking all

    sensori_dict[lem_sens] = {'Auditory.mean': auditory, 'Gustatory.mean': gustatory, 'Haptic.mean': haptic, 'Interoceptive.mean': interoceptive, 'Olfactory.mean': olfactory, 'Visual.mean': visual}
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
        #listed_sens.append(diconc[word])
        dict_overlap[word] = {'Auditory.mean': sensori_dict[word]['Auditory.mean'], 'Gustatory.mean': sensori_dict[word]['Gustatory.mean'], 
                              'Haptic.mean': sensori_dict[word]['Haptic.mean'], 'Interoceptive.mean': sensori_dict[word]['Interoceptive.mean'], 
                              'Olfactory.mean': sensori_dict[word]['Olfactory.mean'], 'Visual.mean': sensori_dict[word]['Visual.mean'], 
                              'concreteness': diconc[word]}

# %%
dict_overlap
# %%

#overlap = pd.DataFrame.from_dict(dict_overlap, orient='index', columns=['concreteness', 'sensorimotor']).reset_index()
overlap = pd.DataFrame.from_dict(dict_overlap, orient='index', columns=['Auditory.mean', 'Gustatory.mean', 'Haptic.mean', 'Interoceptive.mean', 'Olfactory.mean', 'Visual.mean', 'concreteness']).reset_index()
overlap['word'] = overlap['index']

overlap.head()
# %%
# make scatter subplots
plot_scatters(overlap, filtered_cols, 'concreteness', 'lightseagreen', 40, 6, hue=False, remove_outliers=False, outlier_percentile=100, show_corr_values=True)

# %%
from functions import plotly_viz_correlation_improved

x = plotly_viz_correlation_improved(overlap, 'concreteness', 'Visual.mean', w=1000, h=650, hoverdata_column='word', canon_col_name='', canons=False, color_canon=False, save=True)

# %%
x = plotly_viz_correlation_improved(overlap, 'concreteness', 'Gustatory.mean', w=1000, h=650, hoverdata_column='word', canon_col_name='', canons=False, color_canon=False, save=True)


# %%
# reopen save dict of sensorimotor values
with open('resources/sensorimotor_norms_dict.json', 'r') as f:
    sensori_dict = json.load(f)
len(sensori_dict)

# %%
sensori_dict

# %%
sensori_dict['far']['Auditory.mean']

# %%
import nltk
import numpy as np

auditory_list = []
gustatory_list = []
haptic_list = []
interoceptive_list = []
olfactory_list = []
visual_list = []

# Iterate over each row in the DataFrame
for i, row in df.iterrows():
    words = []
    sent = row['SENTENCE']
    toks = nltk.wordpunct_tokenize(sent.lower())
    lems = [lmtzr.lemmatize(word) for word in toks]
    words += lems

    # store values for current row
    auditory = []
    gustatory = []
    haptic = []
    interoceptive = []
    olfactory = []
    visual = []

    for word in words:
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

    # Calculate the mean of each sensory list for the current row
    auditory_list.append(np.nanmean(auditory))
    gustatory_list.append(np.nanmean(gustatory))
    haptic_list.append(np.nanmean(haptic))
    interoceptive_list.append(np.nanmean(interoceptive))
    olfactory_list.append(np.nanmean(olfactory))
    visual_list.append(np.nanmean(visual))

# Assign the lists to the DataFrame columns
df['Auditory.mean'] = auditory_list
df['Gustatory.mean'] = gustatory_list
df['Haptic.mean'] = haptic_list
df['Interoceptive.mean'] = interoceptive_list
df['Olfactory.mean'] = olfactory_list
df['Visual.mean'] = visual_list


df.head()
# %%

# dump this to a json
#with open('data/FB_data_w_features.json', 'w') as f:
    #json.dump(df.to_dict(), f)
    
# %%
print('All done!')
# %%
