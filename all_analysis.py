# %%
import os
from utils import *
from functions import *

# set out path for visualizations
output_path = 'figures/'
# set input path for data
input_path =  'data/fiction4_data.json'
# make a save-title
save_title = input_path.split('/')[-1].split('_')[0]

filter = False
len_threshold = 5

if filter == True:
    print('data treated:', save_title, '-- filtered for length == True')
    save_title += '_filtered'
else:
    print('dataset treated:', save_title.upper())

# %%
# open the merged json and make df
with open(input_path, 'r') as f:
    all_data = json.load(f)
data = pd.DataFrame.from_dict(all_data)
print(len(data))
data.head()

# %%
# Tokenize and get len of sentences/posts
data['SENTENCE_TOKENNIZED'] = data['SENTENCE'].apply(lambda x: nltk.wordpunct_tokenize(x.lower()))
lens = data['SENTENCE_TOKENNIZED'].apply(lambda x: len(x))
data['SENTENCE_LENGTH'] = lens

# filtering where sentences are too short
if filter == True:
    df = data.loc[data['SENTENCE_LENGTH'] > len_threshold].reset_index(drop=True)
    print(f'data is FILTERED, sentences < len {len_threshold} are removed')
    print('len original/len filtered', len(data), len(df))
else:
    df = data
    print('data is UNFILTERED')
    print('len data:', len(df))

# %%
# maybe clean
if 'emobank' in save_title:
    # get rid of the SemEval "genre"
    df = df.loc[df['category'] != 'SemEval']
    print('SemEval "genre" removed from EmoBank')
print('avg words per sentence/post:', round(df['SENTENCE_LENGTH'].mean(),1), '--std:', round(df['SENTENCE_LENGTH'].std(),1))
print('number of sentences/posts:', len(df))
print('number of words:', df['SENTENCE_LENGTH'].sum())

# if we are treating our own assembled corpus, get some numbers on each category
if save_title == 'fiction4':
    data['CATEGORY'].value_counts()

    for cat in data['CATEGORY'].unique():
        print(cat, '--')
        counts_df = df.loc[data['CATEGORY'] == cat]
        s_len = counts_df['SENTENCE_LENGTH'].mean()
        no_words = counts_df['SENTENCE_LENGTH'].sum()
        lens = len(counts_df)
        print(cat, 'len data:', lens, 'avg s_len:', s_len, 'no words:', no_words)
        

# rename columns
df['avg_visual'] = df['Visual.mean']
df['avg_haptic'] = df['Haptic.mean']
df['avg_interoceptive'] = df['Interoceptive.mean']

if save_title == 'FB':
    df['avg_harousal'] = df['harousal']

# %%
# we want to normalize the dictionary scores before using it to filter out the groups for certain datasets
# as well as the human values if needed

# adjust huamn range if using these datasets
data_to_normalize = ['emobank','emotales', 'FB']

if save_title in data_to_normalize:
    df['HUMAN'] = normalize(df['HUMAN'], scale_zero_to_ten=True) # we scale it 0-10 to get it comparable to the Fiction4 corpus
    df.head()

    print(f'{save_title} avg human valence scores were normalized to range 0-10')
# I'm not thrilled about this normalization of human scores business

# %%
sns.histplot(df['HUMAN'])
[x for x in df['HUMAN'] if x > 10]
# %%
# EXPERIMENT 1
print('EXPERIMENT 1')
# Here we make two groups, one of "implicit" and one of "explicit" sentiment, signifying sentences where human/model diverge/converge

# First, we filter out the lukewarm human ratings (neutral scores)
filtered = df.loc[(df['HUMAN'] <= 4.5) | (df['HUMAN'] >= 5.5)].reset_index(drop=False)
print('len filtered:', len(filtered))

# then we set the threshold for when the model scores something neutral
threshold = 0.1

# make implicit group
implicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) <= threshold)]# & (abs(filtered[dictionary_used]) <= threshold)]
print('len IMplicit group:', len(implicit_df))

# make explicit group
explicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) > threshold)]# & (abs(filtered[dictionary_used]) > threshold)] # & (abs(filtered['arc_sentida']) > threshold)]#  # difference in concreteness is bigger if i do | instead of & between the last pair, but explicit group is then also bigger
print('len EXplicit group:', len(explicit_df))

# %%
# statistics
measure_list = ['avg_arousal', 'avg_concreteness', 'avg_imageability', 'avg_visual', 'avg_haptic', 'avg_interoceptive']#, 'avg_interoceptive']#, 'avg_sensorimotor'] # avg_dominance # 'avg_valence', 

# if it is EmoTales, we also have annotations for valence, so use it
if save_title == 'emotales':
    measure_list = measure_list + ['avg_action', 'avg_power']
    print('EmoTales avg POW & ACT is also used')
    width_plot = 25 # make plots a bit bigger since using more features
else:
    width_plot = 20

# and use V, D if it is EmoBank
if save_title == 'emobank':
    measure_list = ['avg_harousal'] + measure_list
    print('EmoBank avg human dominance & arousal is also used')

if save_title == 'FB':
    measure_list = ['avg_harousal'] + measure_list 
    print('FB avg human dominance & arousal is also used')


print('measures considered:', measure_list)

# %%

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

    print(measure.upper(), "U statistic:", u_stat, "P-value:", np.round(p_value, 5))
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
                            plottitle=save_title.upper(), outlier_percentile=100, remove_outliers=False, h=9, w=width_plot, save=True)


# %%

# plot the CEDs and do the kolmogorov-smirnov test
sensorimotor = ['Auditory.mean', 'Gustatory.mean', 'Olfactory.mean']

ced_plot(implicit_df, explicit_df, measure_list, measure_list, save=True, save_title=save_title)
ced_plot(implicit_df, explicit_df, sensorimotor, sensorimotor, save=True, save_title=save_title + '_sensorimotor')
# 


# %%
print(f'whole {save_title.upper()}')
histplot_two_groups(implicit_df, explicit_df, measure_list, measure_list, l=28, h=5, title_plot=f"{save_title.split('_')[0]} All texts", density=True, save=True, save_title=save_title)

# if there are categories in the data, we want to show each category seperately.
# categories are in differently named columns, so we use a map
column_map = {'emobank': 'category','fiction3':'CATEGORY', 'fiction4': 'CATEGORY', 'fiction3': 'CATEGORY'} #'EmoTales_w_features':'ID', 

if save_title in column_map.keys():
    categories = df[column_map[save_title]].unique()
    for cat in categories:
        implicit_df_cat = implicit_df.loc[implicit_df[column_map[save_title]] == cat]
        explicit_df_cat = explicit_df.loc[explicit_df[column_map[save_title]] == cat]
        print(f'{cat.upper()}: Groups: len implicit/explicit:', len(implicit_df_cat), '/', len(explicit_df_cat))
        histplot_two_groups(implicit_df_cat, explicit_df_cat, measure_list, measure_list, l=35, h=5, title_plot=cat, density=True, save=False, save_title=save_title + '_' + cat)

# %%
# and for the sensorimotor values
if save_title in column_map.keys():
    categories = df[column_map[save_title]].unique()

    for cat in categories:
        implicit_df_cat = implicit_df.loc[implicit_df[column_map[save_title]] == cat]
        explicit_df_cat = explicit_df.loc[explicit_df[column_map[save_title]] == cat]
        print(f'SENSORIMOTOR NORMS: GROUPS: len implicit/explicit in {cat.upper()}:', len(implicit_df_cat), '/', len(explicit_df_cat))
        histplot_two_groups(implicit_df_cat, explicit_df_cat, sensorimotor, sensorimotor, l=35, h=5, title_plot=cat, density=True, save=False, save_title=save_title + '_' + cat + '_sensorimotor')

# %%
# EXPERIMENT 2
print('EXPERIMENT 2')
# we want to see if there is a correlation between the human/roberta absolute diff and the concreteness
# in both groups
df['HUMAN_NORM'] = normalize(df['HUMAN'], scale_zero_to_ten=False)
df['ROBERTA_HUMAN_DIFF'] = abs(abs(df['HUMAN_NORM']) - abs(df['tr_xlm_roberta']))

for measure in measure_list:
    print(f'All genres -- {measure} corr w. disagreement')
    x = plotly_viz_correlation_improved(df, measure, 'ROBERTA_HUMAN_DIFF', w=800, h=350, hoverdata_column='SENTENCE', canon_col_name='', color_canon=False, save=False)


# %%
# we want to try and see if the correlation improves at differente thresholds of sentence length
# the whole data (not divided into categories)
# and visualizing it
thresholds = [0, 5, 10, 15, 20, 25, 30]
scores_list = ['avg_arousal', 'avg_concreteness', 'avg_imageability', 'avg_interoceptive']

# going back to the original (certainly) unfiltered dataframe -- 'data'
df['HUMAN_NORM'] = normalize(df['HUMAN'], scale_zero_to_ten=False)
df['ROBERTA_HUMAN_DIFF'] = abs(abs(df['HUMAN_NORM']) - abs(df['tr_xlm_roberta']))

# uncomment to plot all scatterplots
# for threshold in thresholds:
#     print('no. words/sentence threshold:', threshold)
#     data_filtered = df.loc[(df['SENTENCE_LENGTH'] > threshold)]
#     print('len of df:', len(data_filtered), ' texts')
#     plot_scatters(data_filtered, scores_list, 'ROBERTA_HUMAN_DIFF', 'pink', 20, 6, hue=False, remove_outliers=False, outlier_percentile=100, show_corr_values=True)

# correlation at different thresholds per each category
# this is only for the data with categories
if save_title in column_map:

    # Let's try filtering for the different categories and correlating diff score to the features
    categories = df[column_map[save_title]].unique()
    category_data_all = {}

    for category in categories:
        print(category)
        category_data_per_threshold = {}

        category_df = df.loc[df[column_map[save_title]] == category]

        for threshold in thresholds:
            category_data_threshold = {}

            # Filter data based on category and threshold
            data_filtered_for_s_len = category_df.loc[category_df['SENTENCE_LENGTH'] > threshold]

            # Drop NaNs before correlation
            data_filtered_for_s_len_dropna = data_filtered_for_s_len.dropna(subset=['ROBERTA_HUMAN_DIFF', 'avg_concreteness', 'avg_arousal', 'avg_imageability'])

            measure_results = {}

            for measure in measure_list:
                # Calculate correlation for each measure
                corr = stats.spearmanr(data_filtered_for_s_len_dropna['ROBERTA_HUMAN_DIFF'], data_filtered_for_s_len_dropna[measure])
                measure_results[measure] = {'correlation': round(corr[0], 3), 'p-value': round(corr[1], 5)}
                print(category, 'T:', threshold, measure, 'correlation::', round(corr[0], 3), 'p', round(corr[1], 5))

# %%
# We just want the correlation of the whole data with the 5 word sentence threshold
threshold = 5
data_filtered_for_s_len = df.loc[df['SENTENCE_LENGTH'] > threshold]

# drop NaNs before correlation
data_filtered_for_s_len_dropna = data_filtered_for_s_len.dropna(subset=['ROBERTA_HUMAN_DIFF', 'avg_concreteness', 'avg_arousal', 'avg_imageability'])
print('len data', len(data_filtered_for_s_len_dropna))

print(f'correlation of whole data with word/sentence threshold of: {threshold}')

for measure in measure_list:
    correlation_results = stats.spearmanr(data_filtered_for_s_len_dropna['ROBERTA_HUMAN_DIFF'], data_filtered_for_s_len_dropna[measure])
    corr_value = round(correlation_results[0], 3)
    p_value = round(correlation_results[1], 5)
    print('correlation::', measure, corr_value, p_value)

# %%
# and get the mean, std, median for the features across the categories
features = measure_list + ['ROBERTA_HUMAN_DIFF']
data_unfiltered = {}

print('All data')
for feature in features:
            # get mean, std, median
        mean = round(df[feature].mean(), 3)
        std = round(df[feature].std(), 3)
        data_unfiltered[feature] = {'mean': mean, 'std': std}#, 'median': median}

print(pd.DataFrame.from_dict(data_unfiltered))


category_data_unfiltered = {}

if save_title in column_map:
    categories = df[column_map[save_title]].unique()

    for category in categories:
        category_data_feature = {}
        category_df = df.loc[df[column_map[save_title]] == category]
        
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

df.head()

# %%
