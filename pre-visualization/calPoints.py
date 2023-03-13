import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import en_core_web_sm
import re
import json
import time

def write_log(file, msg, trigger):
    if trigger:
        with open(file, 'a+', encoding='utf-8') as f:
            f.write('[' + time.strftime("%Y-%m-%d %H:%M:%S") + ']' + str(msg) + '\n')


def main():
    model_name = 'bert-base-uncased'
    # load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = BertModel.from_pretrained(model_name)

    # tak sentences vectors
    sentences_vectors =[]
    for sent in sentences_list:
        outputs = model(**tokenizer(str(sent), return_tensors='pt', max_length=512, truncation=True))
        sentences_vectors.append(outputs[1].tolist()[0])
    
    print('sentences finished')

    domain_vectors = []
    domain_and_all_vectors = []
    other_vectors = []
    other_and_all_vectors = [] 

    single_vectors = []
    concat_vectors = []

    for domain, i in domain_and_all:
        outputs = model(**tokenizer(str(domain), return_tensors='pt', max_length=16, truncation=True))
        vect = outputs[1].tolist()[0]
        single_vectors.append(vect)
        concat_vectors.append(vect + sentences_vectors[i])
    
    print('domain words finished')
    for ow, i in other_and_all:
        outputs = model(**tokenizer(str(ow), return_tensors='pt', max_length=16, truncation=True))
        vect = outputs[1].tolist()[0]
        single_vectors.append(vect)
        concat_vectors.append(vect + sentences_vectors[i])
    
    print('other words finished')

    # scalar = MinMaxScaler(feature_range=(0, 1))
    # data = scalar.fit_transform(np.array(sent_vectors))
    data1 = np.array(single_vectors)
    data2 = np.array(concat_vectors)
    
    labels = pd.Series([1] * len(domain_and_all) + [-1] * len(other_and_all), name='label')
    single_words = pd.Series([domain for domain, _ in domain_and_all] + [other for other, _ in other_and_all], name='words')
    concat_words = pd.Series([domain + sentences_list[i] for domain, i in domain_and_all] + [other + sentences_list[i] for other, i in other_and_all], name='words')
    dimensions = [2]
    columns = [[], [], ['x1', 'x2'], ['x1', 'x2', 'x3']]

    for dim in dimensions:
        """
          , NMF(n_components=dim), LatentDirichletAllocation(n_components=dim)
          , 'NMF', 'LDA'
        """
        rd_models = [PCA(n_components=dim), TSNE(n_components=dim), MDS(n_components=dim), Isomap(n_components=dim), LocallyLinearEmbedding(n_components=dim), SpectralEmbedding(n_components=dim)]
        rd_model_names = ['PCA', 'TSNE', 'MDS', 'Isomap', 'LLE', 'SE']

        for i in range(len(rd_model_names)):
            rd_model = rd_models[i]
            res = rd_model.fit_transform(data1)
            res_df = pd.DataFrame(res, columns=columns[dim])
            save_df = pd.concat([single_words, res_df, labels], axis=1)
            # save_df = pd.concat([df['doi'], res_df, df['label']], axis=1)
            # save_df.fillna(-1, inplace=True)
            save_file = 'middle/single-words' + '_' + rd_model_names[i] + '_2d.csv'
            save_df.to_csv(save_file, encoding='utf-8', index=False)
            
            res = rd_model.fit_transform(data2)
            res_df = pd.DataFrame(res, columns=columns[dim])
            save_df = pd.concat([concat_words, res_df, labels], axis=1)
            # save_df = pd.concat([df['doi'], res_df, df['label']], axis=1)
            # save_df.fillna(-1, inplace=True)
            save_file = 'middle/concat-words' + '_' + rd_model_names[i] + '_2d.csv'
            save_df.to_csv(save_file, encoding='utf-8', index=False)

            print(fr'-------------bertmodel {model_name}, dim {dim}, rd model {rd_model_names[i]} finished---------------')
    # 

if __name__ == '__main__':
    nlp = en_core_web_sm.load()
    
    # read file
    df = pd.read_csv('data/all_domain_ta.csv')

    with open('data/doi_paper_keywords.json') as f:
        paper_keywords = json.load(f)
    
    log_file = 'log/cal_points.txt'
    log_trigger = True
    # df.fillna(-1, inplace=True)
    
    # str list
    # all_domain = []
    # other_words = []

    # tuple list
    domain_and_all = []
    other_and_all = []
    
    # all 65 papers tak
    sentences_list = []

    domain_numbers_in_tak = 0
    paper_numbers_have_domain = 0
    app_number = 0
    paper_exist_do_number = 0
    
    # can't use nltk 
    # so just manually save word and stem
    stem_dt = {}

    for i, row in df.iterrows():
        # get basic infos
        doi, title, abstract, label, domain_str, problem_str = row
        write_log(log_file, f'{doi} start; domain_str:{domain_str}', log_trigger)
        # only application papers so have  domain
        # split app domain from domain_str
        if len(domain_str) > 2:
            domain = [' '.join(re.split('\s+', s[1:-1])) for s in re.split(', ', domain_str[1:-1])]
            paper_exist_do_number += 1
        else:
            domain = []
        app_number += len(domain)

        write_log(log_file, f'domain:{domain}', log_trigger)
        stem_domain = []
        for do in domain:
            doc = nlp(do)
            # tmp list for saving domain stem words
            tmp = []
            for token in doc:
                stem_dt[token.text] = token.lemma_
                tmp.append(token.lemma_)
            stem_domain.append(' '.join(tmp))

        write_log(log_file, f'stem_domain:{stem_domain}', log_trigger)

        # if len(domain_str) > 2:
        #     domain = [s[1:-1] for s in re.split(', ', domain_str[1:-1])]
        #     app_number += len(domain)

        # get keywords
        keywords = paper_keywords[doi]

        # concat title, abstract, keywords with seperator '.'
        sentences = (title + '.' + abstract + '.' + '.'.join(keywords)).lower()
        write_log(log_file, f'sentences:{sentences}', log_trigger)

        # save sentences at pos i
        sentences_list.append(sentences) 
        
        do_exist = []
        in_tak = False
        # directly delete domain words
        ######### #########################################################################
        # WRONG: If a domain is sub of another domain, something wrong when delete domain #
        ###################################################################################
        for do in domain:
            sent_list = sentences.split(do)
            if len(sent_list) > 1:
                do_exist.append(True)
                in_tak = True
            else:
                do_exist.append(False)

        # delete domain words if the stemming words match
        doc = nlp(sentences)
        for token in doc:
            stem_dt[token.text] = token.lemma_

        # stem sentences
        ###
        ### Attention!!!! stop words probobaly in domain, so remove stop words after match domain words
        ###
        # remove stop words
        stem_sent = ' '.join(token.lemma_ for token in doc)
        
        for j, stem_do in enumerate(stem_domain):
            # domain_and_all.append((do, i))
            domain_and_all.append((stem_do, i))

            # without stem domain; remove stop wordsd
            sent_list_wosd = stem_sent.split(stem_do)
            if len(sent_list_wosd) > 1: 
                # if have domain, app numbers ++
                do_exist[j] = True
                in_tak = True
                # copy sent_list without domain
                # a little hard because there has no direct association from stem and origin words
                # just ignore this first

                # update stem_sent
                # stem_sent = '.'.join([sent.strip() for sent in sent_list_wosd])
        # one of the domain appears in tak
        if in_tak:
            paper_numbers_have_domain += 1
        elif len(domain) > 0:
            write_log(log_file, '####################THIS PAPER NO DOMAIN IN TAK####################', log_trigger)
        
        for j, exist in enumerate(do_exist):
            if exist:
                domain_numbers_in_tak += 1
            else:
                write_log(log_file, f'domain words: {domain[j]} IS NOT IN TAK!!', log_trigger)
        
        # only delete stem domain words
        # for do in domain:
        #     sentences = '.'.join([s.strip() for s in sentences.split(do)])
        # doc = nlp(sentences)
        # stem_sent = ' '.join(token.lemma_ for token in doc)
        for stem_do in stem_domain:
            stem_sent = '.'.join([sent.strip() for sent in stem_sent.split(stem_do)])

        doc = nlp(stem_sent) 
        write_log(log_file, f'stem_sent:{stem_sent}', log_trigger)
        stem_sent_rm_sw = ' '.join(['.' if token.is_stop else token.lemma_ for token in doc])
        write_log(log_file, f'stem_sent_rm_sw:{stem_sent_rm_sw}', log_trigger)

        doc = nlp(stem_sent_rm_sw)
        # split stem_sent_rm_sw into sentences
        sent_list = [str(sent) for sent in doc.sents]

        # split and get word and bigrams from sent_list
        # no domain words or stop words left
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
        X = vectorizer.fit_transform(sent_list)
        res = list(vectorizer.get_feature_names_out())

        # get_res
        # other_words.extend(res)
        # save res and sentences
        other_and_all.extend([(str(s), i) for s in res])

    write_log(log_file, f'number of app annotation: {app_number}', True)
    write_log(log_file, f'number of paper exists domain : {paper_exist_do_number}', True)    
    write_log(log_file, f'number of app annotation in tak: {domain_numbers_in_tak} / {app_number}', True)
    write_log(log_file, f'number of paper at least 1 domain in tak: {paper_numbers_have_domain} / {paper_exist_do_number}', True)

    # constrcut sentence
    # df['sentence'] = df['title'] + '.' + df['title'] + '.' + df['title'] + '.' + df['abstract']
    # key_mls = [('section_headers', 512), ('captions', 512)]
    
    main()

    # for key, ml in key_mls:
    #     df['sentence'] = df[key]
        
    #     # get sentences and labels
    #     sentences = df.sentence.values
    #     labels = df.label.values
    #     # model_prefix = [('bert-base-uncased', 'bert'), ('allenai/scibert_scivocab_uncased', 'scibert')]
    #     model_name, file_prefix = ('bert-base-uncased', 'bert')
    #     # for model_name, file_prefix in model_prefix:
    #     main()
