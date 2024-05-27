# %%
import pandas as pd
import json

# %%
filename = '/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/data/EmoTales/View #1/View#1_dimensions.xlsx'
# v1.head()
sheet_to_df_map = pd.read_excel(filename, sheet_name=None, header=1)
sheet_to_df_map.keys()

# get an id tag in there so we can merge
lenghts = []
for sheet in sheet_to_df_map.keys():
       sheet_to_df_map[sheet].head()
       sheet_to_df_map[sheet]['id'] = sheet
      # print and rename columns
       print(len(sheet_to_df_map[sheet].columns))
       lenghts.append(len(sheet_to_df_map[sheet]))
       # sum the lengths to check against final length
       
       if len(sheet_to_df_map[sheet].columns) == 20:
              sheet_to_df_map[sheet].columns = ['Sentence', 'Annotator_1', 'Annotator_1_ACT', 'Annotator_1_POW', 'Annotator_2',
                                             'Annotator_2_ACT', 'Annotator_2_POW', 'Annotator_3', 'Annotator_3_ACT', 'Annotator_3_POW',
                                             'Annotator_4', 'Annotator_4_ACT', 'Annotator_4_POW', 'Annotator_5',
                                             'Annotator_5_ACT', 'Annotator_5_POW', 'Annotator_6', 'Annotator_6_ACT', 'Annotator_6_POW', 'id']
       elif len(sheet_to_df_map[sheet].columns) == 32:
              sheet_to_df_map[sheet].columns = ['Sentence', 'Annotator_1', 'Annotator_1_ACT', 'Annotator_1_POW', 'Annotator_2',
                                             'Annotator_2_ACT', 'Annotator_2_POW', 'Annotator_3', 'Annotator_3_ACT', 'Annotator_3_POW',
                                             'Annotator_4', 'Annotator_4_ACT', 'Annotator_4_POW', 'Annotator_5',
                                             'Annotator_5_ACT', 'Annotator_5_POW', 'Annotator_6', 'Annotator_6_ACT', 'Annotator_6_POW', 
                                             'Annotator_7', 'Annotator_7_ACT', 'Annotator_7_POW',
                                             'Annotator_8', 'Annotator_8_ACT', 'Annotator_8_POW',
                                             'Annotator_9', 'Annotator_9_ACT', 'Annotator_9_POW',
                                             'Annotator_10', 'Annotator_10_ACT', 'Annotator_10_POW','id']
       elif len(sheet_to_df_map[sheet].columns) == 39:
              sheet_to_df_map[sheet].columns = ['Sentence', 'Annotator_1', 'Annotator_1_ACT', 'Annotator_1_POW', 'Annotator_2',
                                                 'Annotator_2_ACT', 'Annotator_2_POW', 'Annotator_3', 'Annotator_3_ACT', 'Annotator_3_POW',
                                                 'Annotator_4', 'Annotator_4_ACT', 'Annotator_4_POW', 'Annotator_5',
                                                 'Annotator_5_ACT', 'Annotator_5_POW', 'Annotator_6', 'Annotator_6_ACT', 'Annotator_6_POW', 
                                                 'Annotator_7', 'Annotator_7_ACT', 'Annotator_7_POW',
                                                 'Annotator_8', 'Annotator_8_ACT', 'Annotator_8_POW',
                                                 'Annotator_9', 'Annotator_9_ACT', 'Annotator_9_POW',
                                                 'Annotator_10', 'Annotator_10_ACT', 'Annotator_10_POW',
                                                 'Annotator_11', 'Annotator_11_ACT', 'Annotator_11_POW',
                                                 'Annotator_12', 'Annotator_12_ACT', 'Annotator_12_POW','id']
       elif len(sheet_to_df_map[sheet].columns) == 44:
              sheet_to_df_map[sheet].columns = ['Sentence', 'Annotator_1', 'Annotator_1_ACT', 'Annotator_1_POW', 'Annotator_2',
                                                 'Annotator_2_ACT', 'Annotator_2_POW', 'Annotator_3', 'Annotator_3_ACT', 'Annotator_3_POW',
                                                 'Annotator_4', 'Annotator_4_ACT', 'Annotator_4_POW', 'Annotator_5',
                                                 'Annotator_5_ACT', 'Annotator_5_POW', 'Annotator_6', 'Annotator_6_ACT', 'Annotator_6_POW', 
                                                 'Annotator_7', 'Annotator_7_ACT', 'Annotator_7_POW',
                                                 'Annotator_8', 'Annotator_8_ACT', 'Annotator_8_POW',
                                                 'Annotator_9', 'Annotator_9_ACT', 'Annotator_9_POW',
                                                 'Annotator_10', 'Annotator_10_ACT', 'Annotator_10_POW',
                                                 'Annotator_11', 'Annotator_11_ACT', 'Annotator_11_POW',
                                                 'Annotator_12', 'Annotator_12_ACT', 'Annotator_12_POW',
                                                 'Annotator_13', 'Annotator_13_ACT', 'Annotator_13_POW',
                                                 'Annotator_14', 'Annotator_14_ACT', 'Annotator_14_POW','id']
       elif len(sheet_to_df_map[sheet].columns) == 23:
              sheet_to_df_map[sheet].columns = ['Sentence', 'Annotator_1', 'Annotator_1_ACT', 'Annotator_1_POW', 'Annotator_2',
                                                 'Annotator_2_ACT', 'Annotator_2_POW', 'Annotator_3', 'Annotator_3_ACT', 'Annotator_3_POW',
                                                 'Annotator_4', 'Annotator_4_ACT', 'Annotator_4_POW', 'Annotator_5',
                                                 'Annotator_5_ACT', 'Annotator_5_POW', 'Annotator_6', 'Annotator_6_ACT', 'Annotator_6_POW', 
                                                 'Annotator_7', 'Annotator_7_ACT', 'Annotator_7_POW','id']
       elif len(sheet_to_df_map[sheet].columns) == 35:
              sheet_to_df_map[sheet].columns = ['Sentence', 'Annotator_1', 'Annotator_1_ACT', 'Annotator_1_POW', 'Annotator_2',
                                                 'Annotator_2_ACT', 'Annotator_2_POW', 'Annotator_3', 'Annotator_3_ACT', 'Annotator_3_POW',
                                                 'Annotator_4', 'Annotator_4_ACT', 'Annotator_4_POW', 'Annotator_5',
                                                 'Annotator_5_ACT', 'Annotator_5_POW', 'Annotator_6', 'Annotator_6_ACT', 'Annotator_6_POW', 
                                                 'Annotator_7', 'Annotator_7_ACT', 'Annotator_7_POW',
                                                 'Annotator_8', 'Annotator_8_ACT', 'Annotator_8_POW',
                                                 'Annotator_9', 'Annotator_9_ACT', 'Annotator_9_POW',
                                                 'Annotator_10', 'Annotator_10_ACT', 'Annotator_10_POW',
                                                 'Annotator_11', 'Annotator_11_ACT', 'Annotator_11_POW','id']
       elif len(sheet_to_df_map[sheet].columns) == 38:
              sheet_to_df_map[sheet].columns = ['Sentence', 'Annotator_1', 'Annotator_1_ACT', 'Annotator_1_POW', 'Annotator_2',
                                                 'Annotator_2_ACT', 'Annotator_2_POW', 'Annotator_3', 'Annotator_3_ACT', 'Annotator_3_POW',
                                                 'Annotator_4', 'Annotator_4_ACT', 'Annotator_4_POW', 'Annotator_5',
                                                 'Annotator_5_ACT', 'Annotator_5_POW', 'Annotator_6', 'Annotator_6_ACT', 'Annotator_6_POW', 
                                                 'Annotator_7', 'Annotator_7_ACT', 'Annotator_7_POW',
                                                 'Annotator_8', 'Annotator_8_ACT', 'Annotator_8_POW',
                                                 'Annotator_9', 'Annotator_9_ACT', 'Annotator_9_POW',
                                                 'Annotator_10', 'Annotator_10_ACT', 'Annotator_10_POW',
                                                 'Annotator_11', 'Annotator_11_ACT', 'Annotator_11_POW',
                                                 'Annotator_12', 'Annotator_12_ACT', 'Annotator_12_POW','id']
       else:
            print('unknown number of columns', sheet)

print(sum(lenghts))

sheet_to_df_map['Rapunzel'].head()

# %%
# merge all sheets
df = pd.concat(sheet_to_df_map.values())
df.columns
len(df)

# %%
# replace '<br>' with '' and &#39; with ' in sentence column (encoding errors, hope that was all)
df['Sentence'] = df['Sentence'].str.replace('<br>', ' ')
df['Sentence'] = df['Sentence'].str.replace('&#39;', "'")

df.head()

# %%

# now we want to get a mean of the ACT and POW columns for each sentence as well as the annotator_1 etc columns
act_cols = [col for col in df.columns if 'ACT' in col]
pow_cols = [col for col in df.columns if 'POW' in col]
# and if colname ends with number
annotator_cols = [col for col in df.columns if col[-1].isdigit()]

for i, r in df.iterrows():
       # mean of ACT and POW
       df.loc[i, 'ACT_mean'] = r[act_cols].mean()
       df.loc[i, 'POW_mean'] = r[pow_cols].mean()
       # mean of annotators
       df.loc[i, 'HUMAN'] = r[annotator_cols].mean()

df.head()

# %%
# uppercase column names
df.columns = df.columns.str.upper()
df.head()

# %%
# drop to json
# with open('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/data/EmoTales/emoTales.json', 'w') as f:
#     f.write(df.to_json(orient='records'))
# %%
# try opening it with encoding to see if it fixes the ecnoding errors in sentences
with open('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/data/EmoTales/emoTales.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dict_data = pd.DataFrame(data)
dict_data.head()
# %%

# we want to get the spearman correlation and the krippendorff for the annotators
# inter annotator reliability
# Spearman correlation between annotators
# for inter rater reliability
from statsmodels.stats import inter_rater as irr
import krippendorff as kd
from scipy.stats import spearmanr

# %%
from itertools import combinations

coefficient_means_VAL_all = []
kds = []

stories = dict_data['ID'].unique()
stories_w_no_corr = []

for id in stories:
    
       coefficients_VAL = [] # update this for every story

       dt = dict_data.loc[dict_data['ID'] == id]

       #dt = dt.dropna(axis=1, how='all')

       cols_ANN = [col for col in annotator_cols if col in dt.columns]

       krippendorff = kd.alpha([x for x in dt[cols_ANN].values], level_of_measurement='interval') # steven's level of measurement must be one of 'nominal', 'ordinal', 'interval', 'ratio'
       kds.append(krippendorff)

       for col1, col2 in combinations(cols_ANN, 2):
              if col1 in dt.columns and col2 in dt.columns:
                     correlation, p_value = spearmanr(dt[col1], dt[col2])

              else:
                     print('columns not in df')
              if correlation >= 0:
                     coefficients_VAL.append(correlation)
                     
              #else:
                     #print('correlation is negative')
       #print(len(coefficients_VAL))

       if len(coefficients_VAL) > 0:
              mean_corr_VAL = sum(coefficients_VAL) / len(coefficients_VAL)
              coefficient_means_VAL_all.append(mean_corr_VAL)
       else:
              print('no coefficients for story', id)
              coefficient_means_VAL_all.append(0)
              stories_w_no_corr.append(id)

print(coefficient_means_VAL_all)

print(len(stories))
print(len(coefficient_means_VAL_all), len(kds))

print(len(stories_w_corr))
# I CANNOT GET THIS TO WORK AND DONT KNOW WHY
# I have to filter out the three stories

print('average corr', sum(coefficient_means_VAL_all) / len(coefficient_means_VAL_all))
# let's make this a df
df_coefficients = pd.DataFrame({'ID': stories, 'mean_corr': coefficient_means_VAL_all, 'krippendorff': kds})
df_coefficients.head(10)
# %%
# %%
# IRR 2, fleiss & krippendorff
# Need to transpose here cause fleiss expects certain format
transposed = np.array([x for x in human_scores.values()]).transpose()
print('fleiss_kappa: ', irr.fleiss_kappa(irr.aggregate_raters(transposed)[0], method='fleiss'))
# from web: Note that Fleiss is not perfectly applicable to a rating situation with a relative metric: it assumes that this is a classification task, not a ranking. 
# Fleiss is not sensitive to how far apart the ratings are; it knows only that the ratings differed: a (0,1) paring is just as damaging as a (0,3) pairing.

# No need to transpose for krippendorff
print('krippendorf: ', kd.alpha([x for x in human_scores.values()], level_of_measurement='interval')) # steven's level of measurement must be one of 'nominal', 'ordinal', 'interval', 'ratio'
# I'm assuming here that our ratings are on interval scale (absolute 0 and absolute 10) and not ordinal

# Ok, it's perhaps nice to do, but we can also just go with the correlation

# %%
