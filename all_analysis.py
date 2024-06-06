# %%
import os
from utils import *
from functions import *

# set out path for visualizations
output_path = 'figures/'
# set input path for data
input_path =  'data/all_texts_w_sensorimotor.json'#'data/emobank_w_features_and_cats.json'#'data/FB_data_w_features.json'#'data/all_texts_w_sensorimotor.json' #'data/EmoTales_w_features.json' #'data/all_texts_w_sensorimotor.json'''data/all_texts_w_sensorimotor.json'#

# set save-title
save_title = input_path.split('/')[-1].split('.')[0]

filter = False

if filter == True:
    print('data treated:', save_title, '-- filtered for length == True')
    save_title += '_filtered'
else:
    print('data treated:', save_title)


# %%
# open the merged json
with open(input_path, 'r') as f:
    all_data = json.load(f)

data = pd.DataFrame.from_dict(all_data)

# Tokenize and get len of sentence
data['SENTENCE_TOKENNIZED'] = data['SENTENCE'].apply(lambda x: nltk.wordpunct_tokenize(x.lower()))
lens = data['SENTENCE_TOKENNIZED'].apply(lambda x: len(x))
data['SENTENCE_LENGTH'] = lens

data.tail()

# %%
data.columns

# %%
# try filtering out where sentences are too short
# we want to tokenize first
if filter == True:
    df = data.loc[data['SENTENCE_LENGTH'] > 5].reset_index(drop=True)
    print('len filtered data', len(df))
else:
    df = data
    print('len df, unfiltered:', len(df))

df['avg_visual'] = df['Visual.mean']
df['avg_haptic'] = df['Haptic.mean']
df['avg_interoceptive'] = df['Interoceptive.mean']

# %%
# we want to normalize the dictionary scores before using it to filter out the groups, but check that its needed
#filtered[dictionary_used] = normalize(filtered[dictionary_used])
# and the human values if needed

# adjust huamn range if using emobank
data_to_normalize = ['emobank_w_features_and_cats','EmoTales_w_features', 'FB_data_w_features']

if save_title in data_to_normalize:
    df['HUMAN'] = normalize(df['HUMAN'], scale_zero_to_ten=True) # we scale it 0-10 to get it comparable to human
    df.head()

    print(f'{save_title} avg valence scores was normalized 0-10')
# I'm not thrilled about this normalization of human scores business

# %%
sns.histplot(df['HUMAN'])
# %%
# GROUPING
filtered = df.loc[(df['HUMAN'] <= 4.5) | (df['HUMAN'] >= 5.5)].reset_index(drop=False)
print('filtered:', len(filtered))

threshold = 0.4
# we want to use VADER for english, sentida for danish texts?
# we need to decide whether we want to filter on both dict and roberta...
#dictionary_used = 'sentida_MODERN'#'vader'

# implicit group
implicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) <= threshold)]# & (abs(filtered[dictionary_used]) <= threshold)]
print('len_IMplicit_group:', len(implicit_df))

# explicit group
explicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) > threshold)]# & (abs(filtered[dictionary_used]) > threshold)] # & (abs(filtered['arc_sentida']) > threshold)]#  # difference in concreteness is bigger if i do | instead of & between the last pair, but explicit group is then also bigger
print('len_EXplicit_group:', len(explicit_df))

# %%
# statistics
measure_list = ['avg_arousal', 'avg_concreteness', 'avg_imageability', 'avg_visual', 'avg_haptic']#, 'avg_interoceptive']#, 'avg_sensorimotor'] # avg_dominance # 'avg_valence', 

# if it is EmoTales, we also have annotations for valence, so use it
if save_title == 'EmoTales_w_features':
    measure_list = measure_list + ['avg_action', 'avg_power']
    print('EmoTales avg POW & ACT is also used')
    width_plot = 15
else:
    width_plot = 11

# and use V, D if it is EmoBank
if save_title == 'emobank_w_features_and_cats':
    measure_list = measure_list + ['avg_harousal']
    print('EmoBank avg human dominance & arousal is also used')


print('measures considered:', measure_list)

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


sns.set_style("whitegrid")
x = pairwise_boxplots_canon(both_groups, measure_list, category='GROUP', category_labels=['implicit', 'explicit'], 
                            plottitle=save_title, outlier_percentile=100, remove_outliers=False, h=8, w=width_plot, save=True)


# %%

# plot the CEDs and do the kolmogorov-smirnov test
sensorimotor = ['Auditory.mean', 'Gustatory.mean', 'Haptic.mean', 'Interoceptive.mean', 'Olfactory.mean', 'Visual.mean']

ced_plot(implicit_df, explicit_df, measure_list, measure_list, save=True, save_title=save_title)
ced_plot(implicit_df, explicit_df, sensorimotor, sensorimotor, save=True, save_title=save_title + '_sensorimotor')
# 

# %%
print('whole corpus')
histplot_two_groups(implicit_df, explicit_df, measure_list, measure_list, l=28, h=5, title_plot=f"{save_title.split('_')[0]} All texts", density=True, save=True, save_title=save_title)

# if there are categories in the data, we want to show each category seperately.
# categories are in differently named columns, so we use a map
column_map = {'emobank_w_features_and_cats': 'category','all_texts_w_sensorimotor':'CATEGORY'} #'EmoTales_w_features':'ID', 

if save_title in column_map.keys():
    categories = df[column_map[save_title]].unique()
    for cat in categories:
        implicit_df_cat = implicit_df.loc[implicit_df[column_map[save_title]] == cat]
        explicit_df_cat = explicit_df.loc[explicit_df[column_map[save_title]] == cat]
        print(f'{cat}: GROUPS: len implicit:', len(implicit_df_cat), 'len explicit:', len(explicit_df_cat))
        histplot_two_groups(implicit_df_cat, explicit_df_cat, measure_list, measure_list, l=28, h=5, title_plot=cat, density=True, save=True, save_title=save_title + '_' + cat)

# elif 'ID' in df.columns:
#     df['CATEGORY'] = df['ID']
#     categories = df['ID'].unique()

#     for cat in categories:
#         implicit_df_cat = implicit_df.loc[implicit_df['ID'] == cat]
#         explicit_df_cat = explicit_df.loc[explicit_df['ID'] == cat]
#         print(f'GROUPS: len implicit in {cat}:', len(implicit_df_cat), 'len explicit:', len(explicit_df_cat))
#         histplot_two_groups(implicit_df_cat, explicit_df_cat, measure_list, labels, l=28, h=5, title_plot=cat, density=True, save=True, save_title=save_title + '_' + cat)

# %%
# and for the sensorimotor values
if save_title in column_map.keys():
    categories = df[column_map[save_title]].unique()

    for cat in categories:
        implicit_df_cat = implicit_df.loc[implicit_df[column_map[save_title]] == cat]
        explicit_df_cat = explicit_df.loc[explicit_df[column_map[save_title]] == cat]
        print(f'GROUPS: len implicit in {cat}:', len(implicit_df_cat), 'len explicit:', len(explicit_df_cat))
        histplot_two_groups(implicit_df_cat, explicit_df_cat, sensorimotor, sensorimotor, l=35, h=5, title_plot=cat, density=True, save=False, save_title=save_title + '_' + cat + '_sensorimotor')
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

for measure in measure_list:
    print(f'All genres -- {measure} corr w. disagreement')
    x = plotly_viz_correlation_improved(df, measure, 'ROBERTA_HUMAN_DIFF', w=800, h=350, hoverdata_column='SENTENCE', canon_col_name='', color_canon=False, save=False)

# and if there are categories, do it for every category
if save_title in column_map.keys():
    categories = df[column_map[save_title]].unique()

    for cat in categories:
    # we want to check the corr between disagreement and sentence length
        cat_df = df.loc[df[column_map[save_title]] == cat]
        for measure in measure_list:
            print(cat, 'arousal')
            x = plotly_viz_correlation_improved(cat_df, measure, 'ROBERTA_HUMAN_DIFF', w=800, h=350, hoverdata_column='SENTENCE', canon_col_name='', color_canon=False, save=False)
     

# %%
# Some experiments of filtering for various thresholds of sentence length
#
# %%
# we want to try and see if the correlation improves at differente thresholds of sentence length
# the whole data (not divided into categories)
# and visualizing it
thresholds = [0, 5, 10, 15, 20, 25, 30]
scores_list = ['avg_arousal', 'avg_concreteness', 'avg_imageability']

# going back to the original (certainly) unfiltered dataframe -- 'data'
data['HUMAN_NORM'] = normalize(data['HUMAN'], scale_zero_to_ten=False)
data['ROBERTA_HUMAN_DIFF'] = abs(abs(data['HUMAN_NORM']) - abs(data['tr_xlm_roberta']))

# for threshold in thresholds:
#     print('no. words/sentence threshold:', threshold)
#     data_filtered = data.loc[(data['SENTENCE_LENGTH'] > threshold)]
#     print('len of df:', len(data_filtered), ' texts')
#     plot_scatters(data_filtered, scores_list, 'ROBERTA_HUMAN_DIFF', 'pink', 20, 6, hue=False, remove_outliers=False, outlier_percentile=100, show_corr_values=True)

# %%
# correlation at different thresholds per each category
# this is only for the emobank data with categories


if save_title in column_map:

    # Let's try filtering for the different categories and correlating diff score to the features
    categories = data[column_map[save_title]].unique()

    category_data_all = {}

    for category in categories:
        category_data_per_threshold = {}

        category_df = data.loc[data[column_map[save_title]] == category]

        for threshold in thresholds:

            category_data_threshold = {}

            # Filter data based on category and threshold
            data_filtered_for_s_len = category_df.loc[category_df['SENTENCE_LENGTH'] > threshold]
            # Drop NaNs before correlation
            data_filtered_for_s_len_dropna = data_filtered_for_s_len.dropna(subset=['ROBERTA_HUMAN_DIFF'] + measure_list)

            measure_results = {}

            for measure in measure_list:
                # Calculate correlation for each measure
                corr, pval = stats.spearmanr(data_filtered_for_s_len_dropna['ROBERTA_HUMAN_DIFF'], data_filtered_for_s_len_dropna[measure])
                measure_results[measure] = {'correlation': round(corr, 3), 'p-value': round(pval, 5)}

            # Store results
            category_data_threshold['no_texts'] = len(data_filtered_for_s_len_dropna)
            category_data_threshold['scores'] = measure_results

            category_data_per_threshold[threshold] = category_data_threshold

        category_data_all[category] = category_data_per_threshold

category_data_all

# %%
# We just want the correlation of the whole data with the 5 word sentence threshold

data_filtered_for_s_len = data.loc[data['SENTENCE_LENGTH'] > 5]
data_filtered_for_s_len_dropna = data_filtered_for_s_len.dropna(subset=['ROBERTA_HUMAN_DIFF', 'avg_concreteness', 'avg_arousal', 'avg_imageability'])
print('len data', len(data_filtered_for_s_len_dropna))
for feature in measure_list:
    correlation_conc = stats.spearmanr(data_filtered_for_s_len_dropna['ROBERTA_HUMAN_DIFF'], data_filtered_for_s_len_dropna[feature])
    corr_value_conc = round(correlation_conc[0], 3)
    p_value_conc = round(correlation_conc[1], 5)
    print('correlation::', feature, corr_value_conc, p_value_conc)

# %%
# and get the mean, std, median for the features across the categories
# get the mean, std, median for the features across the categories
features = measure_list + ['ROBERTA_HUMAN_DIFF'] # 'avg_valence', 
data_unfiltered = {}

print('All data')
for feature in features:
            # get mean, std, median
        mean = round(data[feature].mean(), 3)
        std = round(data[feature].std(), 3)
        data_unfiltered[feature] = {'mean': mean, 'std': std}#, 'median': median}

print(pd.DataFrame.from_dict(data_unfiltered))


category_data_unfiltered = {}

if save_title in column_map:
    categories = data[column_map[save_title]].unique()

    for category in categories:
        category_data_feature = {}
        category_df = data.loc[data[column_map[save_title]] == category]
        
        # We loop through each category
        for feature in features:
            # Get mean, std, median
            mean = round(category_df[feature].mean(), 3)
            std = round(category_df[feature].std(), 3)
            median = round(category_df[feature].median(), 3)
        
            category_data_feature[feature] = {'mean': mean, 'std': std, 'median': median}
        
        category_data_unfiltered[category] = category_data_feature
    print('Data per category')
    print(pd.DataFrame.from_dict(category_data_unfiltered))


# %%

print('All done!')

# %%
