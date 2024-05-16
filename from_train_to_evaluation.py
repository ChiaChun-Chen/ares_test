from ares import ARES
from classifier import classifier

############################################
# Parameters for all
model_choice = "google-bert/bert-base-multilingual-cased"
label_column = ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label"]
checkpoints = ["", "", ""]
# Parameters for training the classifier
training_dataset = ["CCP_example_files/human_validation_set__nq_reformatted.tsv"]
training_dataset_path = "CCP_example_files/total_training_dataset.tsv"
num_epochs = 10
assigned_batch_size = 16
# Parameters for classifying
max_token_length = 512
unlabelled_data = "CCP_example_files/test_record.tsv"
PPI_file_name = "CCP_example_files/classified_for_ppi_google_bert_bert_base_multilingual_cased_with_no_train.tsv"
# Parameters for evaluating
evaluation_datasets = [PPI_file_name]
evaluation_num = len(evaluation_datasets)
evaluation_results = "CCP_example_files/evaluation_results_with_no_train.txt"
############################################

# # train_classifier
# classifier_config = {
#     "training_dataset": training_dataset,
#     "training_dataset_path":  training_dataset_path,
#     "validation_set": ["CCP_example_files/human_validation_set__nq_reformatted.tsv"], 
#     "label_column": label_column, 
#     "num_epochs": num_epochs, 
#     "patience_value": 3, 
#     "learning_rate": 5e-6,
#     "assigned_batch_size": assigned_batch_size,  
#     "gradient_accumulation_multiplier": 32,
#     "model_choice": model_choice,
# }

# ares = ARES(classifier_model=classifier_config)
# results = ares.train_classifier()
# print(f'Finish training classifier')

# classify
classifier(
    max_token_length=max_token_length,
    model_choice=model_choice,
    checkpoints=checkpoints,
    unlabelled_data=unlabelled_data,
    label_column=label_column,
    PPI_file_name=PPI_file_name
)

# evaluate
ppi_config = { 
    "evaluation_datasets": evaluation_datasets, 
    "few_shot_examples_filepath": "CCP_example_files/few_shot_prompt_filename.tsv",
    "checkpoints": checkpoints,
    "labels": label_column, 
    "GPT_scoring": False, 
    "gold_label_path": "CCP_example_files/human_validation_set__nq_reformatted.tsv",
    "model_choice": model_choice,
}

ares_module = ARES(ppi=ppi_config)
result_dataframe = ares_module.evaluate_RAG()

# Save evaluation results
with open(evaluation_results, 'a+') as f:
    f.write('model choice: ' + model_choice + '\n')
    f.write('evaluation datasets: ' + str(evaluation_datasets) + '\n')
    f.write('labels: ' + str(label_column) + '\n')
    f.write('checkpoints: ' + str(checkpoints) + '\n')
    for j, label in enumerate(label_column):
        f.write(label + '\n')
        for i in range(evaluation_num*j, evaluation_num*(j+1)):
            f.write(result_dataframe.columns[0] + ':' + str(result_dataframe.iloc[i][0]) + '\n')
            f.write(result_dataframe.columns[1] + ':' + str(result_dataframe.iloc[i][1]) + '\n')
            f.write(result_dataframe.columns[2] + ':' + str(result_dataframe.iloc[i][2]) + '\n')
            f.write(result_dataframe.columns[3] + ':' + str(result_dataframe.iloc[i][3]) + '\n')
            f.write(result_dataframe.columns[4] + ':' + str(result_dataframe.iloc[i][4]) + '\n')

print(f'Save evaluation results to {evaluation_results}')