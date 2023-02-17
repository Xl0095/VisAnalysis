from transformers import BertTokenizer
from utils import my_dataset, my_model
import train


if __name__ == '__main__':
    seed_val = 2023
    
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # load dataset
    sentences, labels = my_dataset.get_sentences_labels_list('data/train_test.csv')
    # get tensor dataset
    dataset = my_dataset.get_tensor_dataset(tokenizer, sentences, labels)

    res_file = 'res/result.csv'
    log_file = 'log/log.txt'
    # train
    train.train(dataset, seed_val, log_file, res_file)
