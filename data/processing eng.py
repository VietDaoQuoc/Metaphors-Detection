import pandas as pd
import re
# Preprocessing english data
 
def clean_output(text):
    # Remove commas, parentheses, and square brackets
    return re.sub(r'[\(\)\[\]\']', '', text)
 
def clean_output_special(text):
    if pd.isna(text):
        return text
    return re.sub(r'[\[\]\']' , '', text)
df_literal = pd.read_csv('/home/lyoko/Documents/Personal_Project/Metaphors-Detection/data/en/data_annotation_mihan_literal.csv', delimiter=';', encoding='ISO-8859-1')
 
 
df_literal['Output'] = df_literal['Output'].apply(clean_output_special)
df_literal['Input'] = df_literal['Input'].str.strip()
df_literal['Output'] = df_literal['Output'].str.strip()
df_literal = df_literal.dropna(subset=['Input', 'Output'])
# size 350
# for index, row in df_cleaned.iterrows():
#     print(index)
#     print(row['Input'])
#     print(row['Output'])
 
 
# Output (Subject, Verb, Object For Active voices) and (Object, Verb, Subjet For Passive Voices)
df_met = pd.read_csv('/home/lyoko/Documents/Personal_Project/Metaphors-Detection/data/en/data_annotation_mihan_metaphorical.csv', delimiter=';', encoding='ISO-8859-1')
 
df_met['Output'] = df_met['Output'].apply(clean_output)
df_met['Input'] = df_met['Input'].str.strip()
df_met['Output'] = df_met['Output'].str.strip()
 
df_met = df_met.dropna(subset=['Input', 'Output'])
 
 
# size 350
 
# Now Viet's
 
df_Viet = pd.read_csv('/home/lyoko/Documents/Personal_Project/Metaphors-Detection/data/en/data-annotation_Viet_refined_updated.csv')
df_Viet = df_Viet.rename(columns={'Tuple SVO': 'Output'})
df_Viet = df_Viet.rename(columns={'Sentence': 'Input'})
 
# swap the columns to match the other dataset
cols = list(df_Viet.columns)
 
# Swap the first two elements
cols[0], cols[1] = cols[1], cols[0]
 
# Reorder the DataFrame using the new column order
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
eng_df_all.to_csv('combined.csv', index=False)

 
eng_df_all = eng_df_all.dropna()
 
# Put a placeholder for all the instances where there is no metaphor, as NAN cannot be procesed later
eng_df_all['Output'] = eng_df_all['Output'].replace("", "#,#,#")
 
# pad the ouptuts and ensure there is always a triple 
def ensure_triple(data):
    result = []
    for item in data:
        item_list = [x.strip() for x in item.split(",")]
        # If the item is a tuple or list, convert it to a list and check its length
        if len(item_list) < 3:
            item_list.append('#')
            # If it has less than 3 elements, add 'nothing' to fill the missing slots
            while len(item_list) < 3:
                item_list.append("#")
            print(item_list)
        item =", ".join(item_list)
    return data
 
eng_df_all['Output'] = ensure_triple(eng_df_all['Output'])
eng_df_all.to_csv('combined_1.csv', index=False)

eng_df_all