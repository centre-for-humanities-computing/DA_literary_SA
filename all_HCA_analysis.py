# %%
import os
from utils import *
from functions import *

# %%
# open
with open(f"/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/concreteness/data/HCA_all_values_w_years.json", 'r') as f:
    all_data = json.load(f)

df = pd.DataFrame.from_dict(all_data)
df


# %%
# let's check some
filtered = df.loc[(df['HUMAN'] <= 5) | (df['HUMAN'] >= 6)].reset_index(drop=False)
print('filtered:', len(filtered))

# implicit group
implicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) <= 0.1) & (abs(filtered['tr_alexandrainst']) <= 0.1)]
print('len_IMplicit_group:', len(implicit_df))

# explicit group
explicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) > 0.1) & (abs(filtered['tr_alexandrainst']) > 0.1)] # & (abs(filtered['arc_sentida']) > 0.15)]#  # difference in concreteness is bigger if i do | instead of & between the last pair, but explicit group is then also bigger
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
                            plottitle='H.C. Andersen', outlier_percentile=100, remove_outliers=False, h=8, w=11, save=True)


# %%

# plot the CEDs and do the kolmogorov-smirnov test

labels = ['Valence', 'Arousal','Dominance', 'Concreteness']

ced_plot(implicit_df, explicit_df, measure_list, labels, save=True, save_title='HCA')

# %%
