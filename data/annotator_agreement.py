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
# inter annotator reliability
# Spearman correlation between annotators
df['ANNOTATOR_1'] = df['ANNOTATOR_1'].astype(int)
df['ANNOTATOR_2'] = df['ANNOTATOR_2'].astype(int)

correlation, p_value = spearmanr(df['ANNOTATOR_1'], df['ANNOTATOR_2'])
print("IRR: Spearman:", round(correlation, 3), "p-value:", round(p_value,5))

# and krippendorff
from krippendorff import alpha as krippendorff_alpha

# Convert annotation data to float
annotator_1_float = df['ANNOTATOR_1'].astype(float)
annotator_2_float = df['ANNOTATOR_2'].astype(float)

# Calculate Krippendorff's alpha
krip = krippendorff_alpha([annotator_1_float, annotator_2_float])
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
