from transformers import AutoTokenizer, AutoModel
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import CustomBERTModel
import torch
import pandas as pd
import pyarrow as pa
import datasets
from torch.utils.data import DataLoader
import re
from tqdm.auto import tqdm

############################################
# Parameters for the classification script #

model_choice = "google-bert/bert-base-multilingual-cased"
checkpoint = "checkpoints/google-bert-bert-base-multilingual-cased/Context_Relevance_Label_human_validation_set_2024-04-27_11:28:24.pt"
unlabelled_data = "CCP_example_files/test_record.tsv"
label_column = "Context_Relevance_Label"
PPI_file_name = "CCP_example_files/classified_for_ppi.tsv"

############################################

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

def prepare_dataset_for_evaluation(dataframe, label_column: str, text_column: str, assigned_batch_size, tokenizer):
    from datasets.utils.logging import disable_progress_bar
    disable_progress_bar()

    eval_set_text = [dataframe.iloc[i][text_column] for i in range(len(dataframe))]
    eval_set_label = [dataframe.iloc[i][label_column] for i in range(len(dataframe))]
    
    test_dataset_pandas = pd.DataFrame({'label': eval_set_label, 'text': eval_set_text})
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

    classification_dataset = datasets.DatasetDict({'test' : test_dataset_arrow})
    tokenized_datasets = classification_dataset.map(lambda examples: tokenize_function(tokenizer, examples), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)
    return eval_dataloader

def combine_query_document(query: str, document: str, answer=None):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" ")[:512])

    if len(query.split(" ")) > 100:
        query = (" ").join(query.split(" ")[:30])

    if answer is None:
        return query + " | " + cleaned_document
    else:
        try:
            return query + " | " + cleaned_document + " | " + answer
        except:
            breakpoint()
            print("Error with combine_query_document")
            print("Query: " + str(query))
            print("Cleaned Document: " + str(cleaned_document))
            print("Answer: " + str(answer))
            return str(query) + " | " + str(cleaned_document) + " | " + str(answer)
        
def transform_data(evaluation_set, label_column):
    eval_set = pd.read_csv(evaluation_set, sep="\t")
    eval_set['Question'] = eval_set['Query']
    eval_set['Document'] = eval_set['Document'].str.strip()
    eval_set = eval_set[eval_set["Document"].str.len() > 10]

    if "Context" in label_column:
        eval_set['concat_text'] = [combine_query_document(eval_set.iloc[i]['Question'], eval_set.iloc[i]['Document']) for i in range(len(eval_set))]
    else:
        eval_set['concat_text'] = [combine_query_document(eval_set.iloc[i]['Question'], eval_set.iloc[i]['Document'], eval_set.iloc[i]['Answer']) for i in range(len(eval_set))]

    # print(eval_set.duplicated()) # Check for duplicates
    eval_set = eval_set.drop_duplicates(["concat_text"])

    return eval_set

def postprocess_data(dataframe, label_column, numpy_array):
    dataframe[label_column] = numpy_array
    dataframe = dataframe.drop(columns=['Question', 'concat_text']) # Drop the columns that are not needed

    return dataframe

def classify_data(eval_dataloader, model, device):
    progress_bar = tqdm(range(len(eval_dataloader)))
    total_prediction = torch.tensor([], dtype=torch.long).to(device)
    for batch in eval_dataloader:
        with torch.no_grad():
            new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}
            outputs = model(**new_batch)
            logits = outputs
            predictions = torch.argmax(logits, dim=-1)

            total_prediction = torch.cat((total_prediction, predictions), 0)
            progress_bar.update(1)

    return total_prediction.cpu().numpy()

############################################
# Main script #
# Load the model
tokenizer = AutoTokenizer.from_pretrained(model_choice)
model = CustomBERTModel(2, model_choice)
checkpoint_dict = torch.load(checkpoint)
model.load_state_dict(checkpoint_dict)

# Load the data
eval_set = transform_data(unlabelled_data, label_column)
eval_dataloader = prepare_dataset_for_evaluation(eval_set, label_column, "concat_text", 1, tokenizer)

# Prepare device setup
torch.cuda.empty_cache()
device = torch.device("cuda:0")
model.to(device)

# Classify the data
model.eval()
total_prediction = classify_data(eval_dataloader, model, device)
print("total_prediction: ", total_prediction)

# Post-process the data
eval_set = postprocess_data(eval_set, label_column, total_prediction)
print(eval_set.head())

# Save the data
eval_set.to_csv(PPI_file_name, index=False, sep="\t")
print("Data saved to: " + PPI_file_name)