from ares import ARES
from classifier import classifier

############################################
# Parameters for all
model_choice = "google-bert/bert-base-multilingual-cased"
label_column = ["Answer_Relevance_Label"]
checkpoints = ["checkpoints/google-bert-bert-base-multilingual-cased/Answer_Relevance_Label_human_validation_set_2024-05-06_21:21:30.pt"]
# Parameters for training the classifier
training_dataset = ["/hcds_vol/private/skes/ARES/synthetic_datasets/datasets/nq_synthetic_queries_v4.1.tsv"]
num_epochs = 10
assigned_batch_size = 16
# Parameters for classifying
max_token_length = 512
unlabelled_data = "CCP_example_files/test_record.tsv"
PPI_file_name = "CCP_example_files/classified_for_ppi_google_bert_bert_base_multilingual_cased_2024-05-06_19:42:03.tsv"
# Parameters for evaluating
evaluation_datasets = [PPI_file_name]

############################################

# # train_classifier
# classifier_config = {
#     "training_dataset": training_dataset, 
#     "validation_set": ["CCP_example_files/human_validation_set.tsv"], 
#     "label_column": label_column, 
#     "num_epochs": num_epochs, 
#     "patience_value": 3, 
#     "learning_rate": 5e-5,
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
    "gold_label_path": "CCP_example_files/human_validation_set.tsv",
    "model_choice": model_choice,
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)