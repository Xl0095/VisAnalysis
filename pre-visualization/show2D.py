import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import os

def save_img(_df, _col, _figTitle, _filename):
    _df.plot.scatter('x1', 'x2', c=_col, s=1)
    plt.gcf().set_size_inches(8, 6)
    plt.title(_figTitle)              
    plt.xlim((x1lim_min, x1lim_max))
    plt.ylim((x2lim_min, x2lim_max))
    plt.savefig(_filename, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    col_all = []
    col_domain = []
    col_others = []
    # blue, red, gray
    colors = ['#0000FF', '#FF0000', '#D3D3D3']
    df = pd.read_csv('middle/single-words_PCA_2d.csv')
    labels = df.label.values

    for label in labels:
        col_all.append(colors[int(label)])
        if label != -1:
            col_domain.append(colors[int(label)])
        else:
            col_others.append(colors[int(label)])

    # berts = ['bert']
    rds = ['PCA', 'NMF', 'LDA', 'TSNE', 'MDS', 'Isomap', 'LLE', 'SE']
    stands = ['2d']
    # keys = ['section_headers', 'captions']
    pres = ['single-words', 'concat-words']
    # stands = ['2d', '2d_standard']

    for pre in pres:
        for rd in rds:
            for stand in stands:
                filename = 'middle/' + pre + '_' + rd + '_2d.csv'
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    df.fillna(-1, inplace=True)
                    
                    # get min x1 limit and x2 limit
                    x1s = df.x1.values
                    x2s = df.x2.values
                    x1lim_min = min(x1s)
                    x1lim_min = x1lim_min - abs(x1lim_min) / 10
                    x1lim_max = max(x1s)
                    x1lim_max = x1lim_max + abs(x1lim_max) / 10
                    x2lim_min = min(x2s)
                    x2lim_min = x2lim_min - abs(x2lim_min) / 10
                    x2lim_max = max(x2s)
                    x2lim_max = x2lim_max + abs(x2lim_max) / 10
                    
                    save_img(df, col_all, pre + '_' + rd + 'points all', 'img/' + pre + '_' + rd + '_all.jpg')
                    df_domain = df.loc[df['label'] != -1]
                    save_img(df_domain, col_domain, pre + '_' + rd + 'points domain', 'img/' + pre + '_' + rd + '_domain.jpg')
                    df_others = df.loc[df['label'] == -1]
                    save_img(df_others, col_others, pre + '_' + rd + 'points others', 'img/' + pre + '_' + rd + '_others.jpg')

                    if 'concat-words_LLE' in filename:
                        x2_mean = np.mean(x2s)
                        x2_std = np.std(x2s)
                        _3_lower = x2_mean - 3 * x2_std
                        _3_upper = x2_mean + 3 * x2_std
                        
                        # remove outlier
                        df_domain_ = df_domain.loc[(_3_lower < df_domain['x2']) & (df_domain['x2'] < _3_upper)]
                        df_others_ = df_others.loc[(_3_lower < df_others['x2']) & (df_others['x2'] < _3_upper)]
                        df_ = pd.concat([df_domain_, df_others_])

                        col_domain_ = col_domain[len(df_domain) - len(df_domain_):]
                        col_others_ = col_others[len(df_others) - len(df_others_):]
                        col_all_ = col_domain_ + col_others_

                        x1s = df_.x1.values
                        x2s = df_.x2.values
                        x1lim_min = min(x1s)
                        x1lim_min = x1lim_min - abs(x1lim_min) / 10
                        x1lim_max = max(x1s)
                        x1lim_max = x1lim_max + abs(x1lim_max) / 10
                        x2lim_min = min(x2s)
                        x2lim_min = x2lim_min - abs(x2lim_min) / 10
                        x2lim_max = max(x2s)
                        x2lim_max = x2lim_max + abs(x2lim_max) / 10
                        
                        save_img(df_, col_all_, pre + '_' + rd + ' points all processed', 'img/' + pre + '_' + rd + '_all_processed.jpg')
                        save_img(df_domain_, col_domain_, pre + '_' + rd + ' points domain processed', 'img/' + pre + '_' + rd + '_domain_processed.jpg')
                        save_img(df_others_, col_others_, pre + '_' + rd + ' points others processed', 'img/' + pre + '_' + rd + '_others_processed.jpg')
