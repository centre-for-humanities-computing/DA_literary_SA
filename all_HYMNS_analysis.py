# %%
import os
from utils import *
from functions import *

# %%
# set out path for visualizations
output_path = 'figures/'


#%%
with open(f"/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/concreteness/data/HYMNS_all_values_w_years.json", 'r') as f:
    all_data = json.load(f)

merged = pd.DataFrame.from_dict(all_data)
merged


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

# %%
# now get values of words in each sentence and the means
concretenesses_avg, all_concretenesses = [], []
valences_avg, arousals_avg, dominances_avg = [], [], []

for i, row in merged.iterrows():
    words = []
    sent = row['SENTENCE_ENGLISH']
    toks = nltk.wordpunct_tokenize(sent.lower())
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
# Make columns
merged['avg_concreteness'] = concretenesses_avg
merged['concreteness'] = all_concretenesses

merged['avg_valence'] = valences_avg
merged['avg_arousal'] = arousals_avg
merged['avg_dominance'] = dominances_avg
merged.head()
# %%
df = merged.copy()
len(df)
# %%
# let's make some groups
# filtering out the neutral human scores
filtered = df.loc[(df['HUMAN'] <= 5) | (df['HUMAN'] >= 6)].reset_index(drop=False)
print(len(filtered))

# threshold for transformer scores
threshold = 0.1

# implicit group
implicit_df = filtered.loc[(abs(filtered['sentida_MODERN']) <= threshold) & (abs(filtered['tr_xlm_roberta']) <= threshold)]
print('len_IMplicit_group:', len(implicit_df))

# explicit group
explicit_df = filtered.loc[(abs(filtered['sentida_MODERN']) > threshold) & (abs(filtered['tr_xlm_roberta']) > threshold)] # & (abs(filtered['arc_sentida']) > threshold5)]#  # difference in concreteness is bigger if i do | instead of & between the last pair, but explicit group is then also bigger
print('len_EXplicit_group:', len(explicit_df))


# %%
# statistics
measure_list = ['avg_concreteness', 'avg_arousal', 'avg_valence', 'avg_dominance']

ustats = []
pvals = []

for measure in measure_list:
    df1 = explicit_df.loc[explicit_df[measure].notnull()]
    df2 = implicit_df.loc[implicit_df[measure].notnull()]
    print()
    print('groupsizes', len(df2), len(df1))
    values1, values2 = df1[measure], df2[measure]

    u_stat, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
    ustats.append(u_stat)
    pvals.append(p_value)

    print(measure, "U statistic:", u_stat, "P-value:", np.round(p_value, 5))
    print()


# %%
# Boxplots
implicit_df['GROUP'] = 1
explicit_df['GROUP'] = 0

df1 = explicit_df.loc[explicit_df['avg_valence'].notnull()]
df2 = implicit_df.loc[implicit_df['avg_valence'].notnull()]

both_groups = pd.concat([df1, df2])


measure_list = ['avg_valence', 'avg_arousal', 'avg_dominance', 'avg_concreteness']
sns.set_style("whitegrid")
x = pairwise_boxplots_canon(both_groups, measure_list, category='GROUP', category_labels=['implicit', 'explicit'], 
                            plottitle='HYMNS', outlier_percentile=100, remove_outliers=False, h=11, w=13, save=True)

# %% 
# Let's also get mean and std for each measure
for measure in measure_list:
    print(measure)
    print('implicit mean:', implicit_df[measure].mean(), 'std:', implicit_df[measure].std())
    print('explicit mean:', explicit_df[measure].mean(), 'std:', explicit_df[measure].std())


# %%
measure_list = ['avg_valence', 'avg_arousal', 'avg_dominance', 'avg_concreteness']
labels = measure_list

ced_plot(implicit_df, explicit_df, measure_list, labels, save=True, save_title='HYMNS')


# %%
# Inspect sentences
implicit_df_happy = implicit_df.loc[implicit_df['HUMAN'] > 6]
implicit_df_sad = implicit_df.loc[implicit_df['HUMAN'] < 4]

implicit_df_happy['sentida_MODERN_rounded'] = implicit_df_happy['sentida_MODERN'].round(2)
implicit_df_sad['sentida_MODERN_rounded'] = implicit_df_sad['sentida_MODERN'].round(2)


implicit_df_happy[['SENTENCE', 'avg_concreteness', 'HUMAN', 'sentida_MODERN_rounded']].nlargest(30, 'avg_concreteness')

# %%

x = list(implicit_df['DIFF_HUMAN_SENTIDA'])
y = list(implicit_df['avg_concreteness'])

# do linear reg
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print('sentida vs concreteness in implicit group')
print('slope:', slope, 'intercept:', intercept, 'r:', r_value, 'p:', p_value, 'std_err:', std_err)

# %%

# %%
# see correlation between difference of human vs system and concreteness
sns.regplot(data=implicit_df, x='DIFF_HUMAN_SENTIDA', y='avg_concreteness', scatter_kws={'alpha':0.5, 's': 50})

# check the correlation in the implicit group
correlation, p_value = spearmanr(implicit_df['DIFF_HUMAN_SENTIDA'], implicit_df['avg_concreteness'])
print('implicit group; corr:', correlation, 'p:', p_value)

# check the correlation in the explicit group
correlation, p_value = spearmanr(explicit_df['DIFF_HUMAN_SENTIDA'], explicit_df['avg_concreteness'])
print('explicit group; corr:', correlation, 'p:', p_value)









# %%
# try adjusting with conceteness
for i, row in df.iterrows():
    sa_score = row['sentida_MODERN']

    concreteness_score = row['avg_concreteness']

    #adjustment_ar = arousal_score # * threshold
    adjustment_conc = concreteness_score * 0.05

    if abs(sa_score) < 0.5:
        if concreteness_score > 0:
            if sa_score >= 0:
                score_conc = sa_score + adjustment_conc
            if sa_score < 0:
                score_conc = sa_score - adjustment_conc
        else:
            score_conc = sa_score
    else:
        score_conc = sa_score

    df.at[i, 'adjusted_sa_conc'] = score_conc
    #df.at[i, 'adjusted_roberta_arousal'] = roberta_arousal

sns.heatmap(df[['HUMAN', 'sentida_MODERN', 'adjusted_sa_conc']].corr(method='spearman'), annot=True)

# %%

# %%