# %%
from utils import *

from functions import *
# %%
# set input path for data
input_path = 'data/plath_data.json'
title = input_path.split('/')[1].split('_')[0]
print('data treated:', title.upper())
# texts should contain sentences and SA scores

# %%
with open(input_path, 'r') as f:
    all_data = json.load(f)

df = pd.DataFrame.from_dict(all_data)
#df.columns = ['ANNOTATOR_1', 'SENTENCE']
print('len data:', len(df))
df.head()

# %%

nan_counts = df.isna().sum()
print("NaN counts per column:")
print(nan_counts)

nan_rows_annotators = df[df[['ANNOTATOR_1', 'ANNOTATOR_2', 'ANNOTATOR_3']].isna().any(axis=1)]
print("Rows with NaN values in annotator columns:")
print(nan_rows_annotators)


# %%
yuri_listed = df['ANNOTATOR_3']
[x for x in yuri_listed if x > 10]
#df['ANNOTATOR_3'] = df['ANNOTATOR_3'].replace(56, 5)
[x for x in df['ANNOTATOR_3'] if x > 10]


# %%
# inter annotator reliability
# Spearman correlation between annotators
df['ANNOTATOR_1'] = df['ANNOTATOR_1'].astype(int)
df['ANNOTATOR_2'] = df['ANNOTATOR_2'].astype(int)
df['ANNOTATOR_3'] = df['ANNOTATOR_3'].astype(int)


correlation1, p_value = spearmanr(df['ANNOTATOR_1'], df['ANNOTATOR_2'])
print("P-E: IRR: Spearman:", round(correlation1, 3), "p-value:", round(p_value,5))

correlation2, p_value = spearmanr(df['ANNOTATOR_2'], df['ANNOTATOR_3'])
print("E-Y: IRR: Spearman:", round(correlation2, 3), "p-value:", round(p_value,5))

correlation3, p_value = spearmanr(df['ANNOTATOR_1'], df['ANNOTATOR_3'])
print("P-Y: IRR: Spearman:", round(correlation3, 3), "p-value:", round(p_value,5))

mean_corr = (correlation1 + correlation2 + correlation3) / 3
print('mean spearman: ', round(mean_corr,3))

# and krippendorff
from krippendorff import alpha as krippendorff_alpha

# Convert annotation data to float
annotator_1_float = df['ANNOTATOR_1'].astype(float)
annotator_2_float = df['ANNOTATOR_2'].astype(float)
annotator_3_float = df['ANNOTATOR_3'].astype(float)

# Calculate Krippendorff's alpha
krip = krippendorff_alpha([annotator_1_float, annotator_2_float, annotator_3_float])
print("IRR: Krippendorff:", round(krip, 3))

# %%
print("Additional task, model-human agreement in steps")
# So ideally we want to check the agreement w e.g. roBERTa at each point of adding an annotator?
# Remember, we see me & Yuri agreeing more with the model together than separately...

# set input path for data
input_path = 'data/fiction4_data.json'
title = input_path.split('/')[1].split('_')[0]
print('data treated:', title.upper())
# texts should contain sentences and SA scores

# %%
with open(input_path, 'r') as f:
    all_data = json.load(f)

df = pd.DataFrame.from_dict(all_data)
#df.columns = ['ANNOTATOR_1', 'SENTENCE']
print('len data:', len(df))
df.head()

# and so on

# %%
