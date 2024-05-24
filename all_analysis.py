# %%
import os
from utils import *
from functions import *

# set out path for visualizations
output_path = 'figures/'
# set input path for data
input_path = 'data/FB_data_w_features.json' #'data/emobank_w_features.json'
# set save-title
save_title = input_path.split('/')[-1].split('.')[0]
print('data treated:', save_title)

# %%
# # open and merge the different datasets and get only some of the columns

# with open(f"/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/concreteness/data/HCA_all_values_w_years.json", 'r') as f:
#     all_data = json.load(f)

# df = pd.DataFrame.from_dict(all_data)
# hca = df[['avg_concreteness', 'avg_arousal', 'avg_valence', 'avg_dominance', 'tr_xlm_roberta', 'sentida', 'HUMAN', 'SENTENCE', 'SENTENCE_ENGLISH', 'id']]
# hca.columns = hymns.columns = ['avg_concreteness', 'avg_arousal', 'avg_valence', 'avg_dominance', 'tr_xlm_roberta', 'sentida_MODERN', 'HUMAN', 'SENTENCE', 'SENTENCE_ENGLISH', 'id'] # rename sentida to match the other
# print(len(hca))

# with open(f"/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/concreteness/data/HYMNS_all_values_w_years.json", 'r') as f:
#     all_dat = json.load(f)

# merged = pd.DataFrame.from_dict(all_dat)
# hymns = merged[['avg_concreteness', 'avg_arousal', 'avg_valence', 'avg_dominance', 'tr_xlm_roberta', 'sentida_MODERN', 'HUMAN', 'SENTENCE', 'SENTENCE_ENGLISH', 'YEAR']]
# hymns.columns = ['avg_concreteness', 'avg_arousal', 'avg_valence', 'avg_dominance', 'tr_xlm_roberta', 'sentida_MODERN', 'HUMAN', 'SENTENCE', 'SENTENCE_ENGLISH', 'id'] # make year 'id'
# print(len(hymns))

# df = pd.concat([hca, hymns])
# print(len(df))
# df.head()

# %%
# open the merged json
with open(input_path, 'r') as f:
    all_data = json.load(f)

df = pd.DataFrame.from_dict(all_data)
df.head()

# %%
# we want to normalize the dictionary scores before using it to filter out the groups, but check that its needed
#filtered[dictionary_used] = normalize(filtered[dictionary_used])
# and the human values if needed
df['HUMAN'] = normalize(df['HUMAN'], scale_zero_to_ten=True)
df.head()
# I'm not too happy about this normalization of human scores business

# %%
# GROUPING
filtered = df.loc[(df['HUMAN'] <= 5) | (df['HUMAN'] >= 6)].reset_index(drop=False)
print('filtered:', len(filtered))

threshold = 0.1
dictionary_used = 'vader' #sentida_MODERN

# implicit group
implicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) <= threshold) & (abs(filtered[dictionary_used]) <= threshold)]
print('len_IMplicit_group:', len(implicit_df))

# explicit group
explicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) > threshold) & (abs(filtered[dictionary_used]) > threshold)] # & (abs(filtered['arc_sentida']) > threshold)]#  # difference in concreteness is bigger if i do | instead of & between the last pair, but explicit group is then also bigger
print('len_EXplicit_group:', len(explicit_df))

# %%
# statistics
measure_list = ['avg_valence', 'avg_arousal', 'avg_dominance', 'avg_concreteness']#, 'avg_sensorimotor']

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


measure_list = ['avg_valence', 'avg_arousal', 'avg_dominance', 'avg_concreteness']#, 'avg_sensorimotor']
sns.set_style("whitegrid")
x = pairwise_boxplots_canon(both_groups, measure_list, category='GROUP', category_labels=['implicit', 'explicit'], 
                            plottitle=save_title, outlier_percentile=100, remove_outliers=False, h=8, w=11, save=True)


# %%

# plot the CEDs and do the kolmogorov-smirnov test

labels = ['avg_valence', 'avg_arousal', 'avg_dominance', 'avg_concreteness']#, 'avg_sensorimotor']

ced_plot(implicit_df, explicit_df, measure_list, labels, save=True, save_title=save_title)

# 
# %%

histplot_two_groups(implicit_df, explicit_df, measure_list, labels, l=28, h=5, density=True, save=True, save_title=save_title)

# %%
# we want to plot the distribution of the concreteness values for the two groups
plt.figure(figsize=(10, 4))
sns.set_theme(style="whitegrid", font_scale=1.5)
sns.histplot(data=implicit_df, x='avg_concreteness', color='blue', kde=True, stat='density', label='Implicit')
sns.histplot(data=explicit_df, x='avg_concreteness', color='red', kde=True, stat='density', label='Explicit')

# %%
