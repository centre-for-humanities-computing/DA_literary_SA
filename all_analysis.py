# %%
import os
from utils import *
from functions import *

# set out path for visualizations
output_path = 'figures/'
# set input path for data
input_path = 'data/emobank_w_features_and_cats.json' #'data/FB_data_w_features.json' #
# set save-title
save_title = input_path.split('/')[-1].split('.')[0]

filter = True

if filter == True:
    print('data treated:', save_title, '-- filtered for length == True')
    save_title += '_filtered'
else:
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

data = pd.DataFrame.from_dict(all_data)
data.head()

# %%
data.columns
# %%
# try filtering out where sentences are too short
# we want to tokenize first
if filter == True:
    data['SENTENCE_TOKENNIZED'] = data['SENTENCE'].apply(lambda x: nltk.wordpunct_tokenize(x.lower()))
    lens = data['SENTENCE_TOKENNIZED'].apply(lambda x: len(x))
    data['SENTENCE_LENGTH'] = lens
    df = data.loc[data['SENTENCE_LENGTH'] > 5].reset_index(drop=True)
    print('len filtered data', len(df))
else:
    print('len:', len(df))
    df = data

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
# we want to see if there is a correlation between the human/roberta absolute diff and the concreteness
# in both groups
df['HUMAN_NORM'] = normalize(df['HUMAN'], scale_zero_to_ten=False)
df['ROBERTA_HUMAN_DIFF'] = abs(abs(df['HUMAN_NORM']) - abs(df['tr_xlm_roberta']))

x = plotly_viz_correlation_improved(df, 'avg_concreteness', 'ROBERTA_HUMAN_DIFF', w=800, h=650, hoverdata_column='category', canon_col_name='', canons=False, color_canon=False, save=False)

# %%
# we want to try and see if the correlation improves at differente thresholds of sentence length
thresholds = [0, 5, 10, 15, 20, 25, 30, 35]
scores_list = ['avg_arousal', 'avg_dominance', 'avg_concreteness']

# going back to the original unfiltered dataframe, data
data['HUMAN_NORM'] = normalize(data['HUMAN'], scale_zero_to_ten=False)
data['ROBERTA_HUMAN_DIFF'] = abs(abs(data['HUMAN_NORM']) - abs(data['tr_xlm_roberta']))

for threshold in thresholds:
    print('no. words/sentence threshold:', threshold)
    data_filtered = data.loc[(data['SENTENCE_LENGTH'] > threshold)]
    print('len of df:', len(data_filtered), ' texts')
    plot_scatters(data_filtered, scores_list, 'ROBERTA_HUMAN_DIFF', 'pink', 25, 8, hue=False, remove_outliers=False, outlier_percentile=100, show_corr_values=True)

# %%
# this is only for the emobank data with categories
if save_title == 'emobank_w_features_and_cats':

    # let's try filtering for the different categories and correlating diff score to the features
    categories = data['category'].unique()
    thresholds = [0, 5, 10, 15, 20, 25, 30, 35]

    category_data_all = {}

    for category in categories:
        category_data_per_threshold = {}

        category_df = data.loc[data['category'] == category]

        for threshold in thresholds:
            print('Category:', category, '- No. words/sentence threshold:', threshold)

            # Filter data based on category and threshold
            data_filtered_for_s_len = category_df.loc[category_df['SENTENCE_LENGTH'] > threshold]

            # Drop NaNs before correlation
            data_filtered_for_s_len_dropna = data_filtered_for_s_len.dropna(subset=['ROBERTA_HUMAN_DIFF', 'avg_concreteness', 'avg_arousal'])

            # Calculate correlation for concreteness
            correlation_conc = stats.spearmanr(data_filtered_for_s_len_dropna['ROBERTA_HUMAN_DIFF'], data_filtered_for_s_len_dropna['avg_concreteness'])
            corr_value_conc = round(correlation_conc[0], 3)
            p_value_conc = round(correlation_conc[1], 5)

            # Calculate correlation for arousal
            correlation_arousal = stats.spearmanr(data_filtered_for_s_len_dropna['ROBERTA_HUMAN_DIFF'], data_filtered_for_s_len_dropna['avg_arousal'])
            corr_value_arousal = round(correlation_arousal[0], 3)
            p_value_arousal = round(correlation_arousal[1], 5)

            # Store results
            category_data_per_threshold[threshold] = {'no_texts': len(data_filtered_for_s_len_dropna), 
                                                    'corr_conc': corr_value_conc, 
                                                    'p_conc': p_value_conc, 
                                                    'corr_arousal': corr_value_arousal, 
                                                    'p_arousal': p_value_arousal}

        category_data_all[category] = category_data_per_threshold

    category_data_all

# %%
