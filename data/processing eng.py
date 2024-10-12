import pandas as pd
import re
# Preprocessing english data

def clean_output(text):
    return re.sub(r'[\(\)\[\]]', '', text)

def clean_output_special(text):
    if pd.isna(text):  
        return text
    return re.sub(r'[\[\]]', '', str(text)) 

df_literal = pd.read_csv('C:/Users/nas/Desktop/STUD/NN Project/english/data_annotation_mihan_literal.csv', delimiter=';', encoding='ISO-8859-1')


df_literal['Output'] = df_literal['Output'].apply(clean_output_special)

df_literal['Input'] = df_literal['Input'].str.strip()
df_literal['Output'] = df_literal['Output'].str.strip()
df_literal = df_literal.dropna(subset=['Input', 'Output'])
# size 350


# Output (Subject, Verb, Object For Active voices) and (Object, Verb, Subjet For Passive Voices)  
df_met = pd.read_csv('C:/Users/nas/Desktop/STUD/NN Project/english/data_annotation_mihan_metaphorical.csv', delimiter=';', encoding='ISO-8859-1')

df_met['Input'] = df_met['Input'].str.strip()
df_met['Output'] = df_met['Output'].str.strip()

df_met = df_met.dropna(subset=['Input', 'Output'])

df_met['Output'] = df_met['Output'].apply(clean_output)
# size 350

# Now Viet's 

df_Viet = pd.read_csv('C:/Users/nas/Desktop/STUD/NN Project/english/data-annotation_Viet_refined.csv')
df_Viet = df_Viet.rename(columns={'Tuple SVO': 'Output'})
df_Viet = df_Viet.rename(columns={'Sentence': 'Input'})

# swap the columns to match the other dataset
cols = list(df_Viet.columns)

cols[0], cols[1] = cols[1], cols[0]

# why 399 ? 235 literal and 164 metaphores ? sure 
df_Viet = df_Viet[cols]
df_Viet['Output'] = df_Viet['Output'].apply(clean_output)
          
empty_count = 0
met_count = 0

for index, row in df_Viet.iterrows():
    if row['Output'] == '':
        empty_count += 1
    else:
        met_count += 1

print(met_count,empty_count )

eng_df_all = pd.concat([df_Viet, df_met, df_literal], ignore_index=True)
