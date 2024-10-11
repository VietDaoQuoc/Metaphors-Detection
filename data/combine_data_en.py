import pandas as pd

df1 = pd.read_csv('/home/lyoko/Documents/Personal_Project/Metaphors-Detection/en/data_annotation_mihan_literal.csv')

df2 = pd.read_csv('/home/lyoko/Documents/Personal_Project/Metaphors-Detection/en/data_annotation_mihan_metaphorical.csv')

combined_df = pd.concat([df1, df2], ignore_index=True)

combined_df.to_csv('combined_data_mihan.csv', index=False) 
