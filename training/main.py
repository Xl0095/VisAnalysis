from transformers import BertTokenizer
from utils import my_dataset, my_model
import train


if __name__ == '__main__':
    seed_val = 2023
    # model_name = 'bert-base-uncased'
    model_name = 'allenai/scibert_scivocab_uncased'

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # load dataset
    # sentences, labels = my_dataset.get_sentences_labels_list('data/train_test.csv')
    # sentences, labels = my_dataset.get_sentences_labels_list('data/in_domain_train.tsv')
    sentences, labels = my_dataset.get_sentences_labels_list('data/imdb_train_1k_imbalanced_0.csv')

    # get tensor dataset
    dataset = my_dataset.get_tensor_dataset(tokenizer, sentences, labels)

    res_file = 'res/result_imdb-imb0_scibert.csv'
    log_file = 'log/log_imdb-imb0_scibert.txt'
    # train
    train.train(model_name, dataset, seed_val, log_file, res_file)
