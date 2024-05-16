import pandas as pd

path = "/hcds_vol/private/skes/ares_test/CCP_example_files/human_validation_set__nq_reformatted.tsv"
dataframe = pd.read_csv(path, sep='\t')

dataset_len = len(dataframe)
print(f'Context_Relevance_Label ratio = {round(dataframe["Context_Relevance_Label"].tolist().count(1)/dataset_len, 2)}')
print(f'Answer_Faithfulness_Label ratio = {round(dataframe["Answer_Faithfulness_Label"].tolist().count(1)/dataset_len, 2)}')
print(f'Answer_Relevance_Label ratio = {round(dataframe["Answer_Relevance_Label"].tolist().count(1)/dataset_len, 2)}')
print(f'total length of CCP_finetune_dataset is {dataset_len}')
