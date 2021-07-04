import pandas as pd
import numpy as np

# create target csn once from xslx
label_names = ['trunk-flex', 'scapular-e', 'scapular-r', 'shoulder-flex', 'elbow-flex', 'distal-dys-syn']
label_indices = ['1', '2', '3', '4', '5', '9']
df = pd.read_excel(r'../Data/Soroka/Patients/movement_classification.xlsx')
df = df.replace(to_replace='[A-Z]+ ', value=',', regex=True)
df = df.replace(to_replace='[A-Z]', value='', regex=True)

new_df = pd.DataFrame() # if also want severity, add here
for column in df.columns[4:-2]:
    df['experiment'] = column
    df['compensation'] = df[column].str.replace(' ', '')

    new_df = new_df.append(df[['patient', 'experiment', 'compensation']])
new_df['compensation'] = new_df['compensation'].str.split(',')
new_df['compensation'] = new_df['compensation'] .apply(lambda x: list(
    set(x) & set(label_indices)) if x is not np.nan else [''])

for name, index in zip(label_names, label_indices):
    new_df[name] = new_df['compensation'].apply(lambda x: 1 if index in x else 0)
new_df = new_df.reset_index(drop=True)

new_df.to_csv(r'../Data/Soroka/Patients/patients_target.csv', index=False)
