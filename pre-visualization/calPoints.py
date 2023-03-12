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

def write_log(file, msg):
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
            save_df = pd.concat([res_df, labels], axis=1)
            # save_df = pd.concat([df['doi'], res_df, df['label']], axis=1)
            # save_df.fillna(-1, inplace=True)
            save_file = 'middle/single-words' + '_' + rd_model_names[i] + '_2d.csv'
            save_df.to_csv(save_file, encoding='utf-8', index=False)
            
            res = rd_model.fit_transform(data2)
            res_df = pd.DataFrame(res, columns=columns[dim])
            save_df = pd.concat([res_df, labels], axis=1)
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
    
    # can't use nltk 
    # so just manually save word and stem
    stem_dt = {}

    for i, row in df.iterrows():
        # get basic infos
        doi, title, abstract, label, domain_str, problem_str = row
        # only application papers so have  domain
        # split app domain from domain_str
        if len(domain_str) > 2:
            domain = [s[1:-1] for s in re.split(', ', domain_str[1:-1])]
        else:
            domain = []
        app_number += len(domain)

        stem_domain = []
        for do in domain:
            doc = nlp(do)
            # tmp list for saving domain stem words
            tmp = []
            for token in doc:
                stem_dt[token.text] = token.lemma_
                tmp.append(token.lemma_)
            stem_domain.append(' '.join(tmp))

        # if len(domain_str) > 2:
        #     domain = [s[1:-1] for s in re.split(', ', domain_str[1:-1])]
        #     app_number += len(domain)

        # get keywords
        keywords = paper_keywords[doi]

        # concat title, abstract, keywords with seperator '.'
        sentences = (title + '.' + abstract + '.' + '.'.join(keywords)).lower()

        # save sentences at pos i
        sentences_list.append(sentences) 
        
        doc = nlp(sentences)
        for token in doc:
            stem_dt[token.text] = token.lemma_

        # stem sentences
        # remove stop words
        stem_sent_rm_sw = ' '.join(['.' if token.is_stop else token.lemma_ for token in doc])
       
        in_tak = False
        
        for j in range(len(domain)):
            stem_do = stem_domain[j]
            do = domain[j]
            
            # domain_and_all.append((do, i))
            domain_and_all.append((stem_do, i))
            # without stem domain; remove stop wordsd
            sent_list_wosd = stem_sent_rm_sw.split(stem_do)
            if len(sent_list_wosd) > 1: 
                # if have domain, app numbers ++
                domain_numbers_in_tak += 1
                in_tak = True

                # copy sent_list without domain
                # a little hard because there has no direct association from stem and origin words
                # just ignore this first

                # update stem_sent_rm_sw
                stem_sent_rm_sw = '.'.join([sent.strip() for sent in sent_list_wosd])
        # one of the domain appears in tak
        if in_tak:
            paper_numbers_have_domain += 1
            
        # split stem_sent_rm_sw into sentences
        doc2 = nlp(stem_sent_rm_sw) 
        sent_list = [str(sent) for sent in doc2.sents]

        # split and get word and bigrams from sent_list
        # no domain words or stop words left
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
        X = vectorizer.fit_transform(sent_list)
        res = list(vectorizer.get_feature_names_out())

        # get_res
        # other_words.extend(res)
        # save res and sentences
        other_and_all.extend([(str(s), i) for s in res])

    write_log('log/cal_Points.txt', f'number of app annotation: {app_number}')
    write_log('log/cal_Points.txt', f'number of paper exists domain : 31')    
    write_log('log/cal_Points.txt', f'number of app annotation in tak: {domain_numbers_in_tak} / {app_number}')
    write_log('log/cal_Points.txt', f'number of paper at least 1 domain in tak: {paper_numbers_have_domain} / 31')

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
