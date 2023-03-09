import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

def get_train_dataframe(filename):
    df = pd.read_csv(filename)
    # df['sentence'] = df['title'] + '.' + df['title'] + '.' + df['title'] + '.' + df['abstract']
    # df['sentence'] = df['title'] + '.' + df['abstract']
    df['sentence'] = df['text']

    df.fillna(value='None', inplace=True)

    return df


def get_sentences_labels_list(filename):
    df = get_train_dataframe(filename)
    # df = pd.read_csv(filename, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    sentences = df.sentence.values
    labels = df.label.values
    return sentences, labels


# get input_ids and attention_masks
def get_ii_am(tokenizer, sentences):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,
                            max_length = 512,
                            padding = 'max_length',
                            truncation = True, 
                            return_attention_mask = True,
                            return_tensors = 'pt'
                        )
        
        # Add the encoded abstract to the list. 
        # print(encoded_dict['input_ids'])
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        # print(encoded_dict['attention_mask'])
        attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks
    

def get_tensor_dataset(tokenizer, sentences, labels):
    input_ids, attention_masks = get_ii_am(tokenizer, sentences)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset

# input tensor dataset
def get_train_val_dataloader(train_dataset, val_dataset):
    # prepare dataloader
    # The DataLoader needs to know our batch size for training, so we specify it 
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
    # size of 16 or 32.
    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    return train_dataloader, validation_dataloader

