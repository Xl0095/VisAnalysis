import matplotlib.pyplot as plt 
import pandas as pd 
import os

if __name__ == '__main__':
    col_all = []
    col_domain = []
    col_others = []
    # blue, red, gray
    colors = ['#0000FF', '#FF0000', '#696969']
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
                    
                    plt.xlim((x1lim_min, x1lim_max))
                    plt.xyim((x2lim_min, x2lim_max))
                    df.plot.scatter('x1', 'x2', c=col_all, s=10)
                    plt.gcf().set_size_inches(1, 1)
                    plt.title(pre + '_' + rd + 'points all')              
                    plt.savefig('img/' + pre + '_' + rd + '_all.jpg', bbox_inches='tight', dpi=300)
                    plt.close()

                    df_domain = df.loc[df['label'] != -1]
                    plt.xlim((x1lim_min, x1lim_max))
                    plt.xyim((x2lim_min, x2lim_max))
                    df_domain.plot.scatter('x1', 'x2', c=col_domain, s=10)
                    plt.gcf().set_size_inches(1, 1)
                    plt.title(pre + '_' + rd + 'points domain')
                    plt.savefig('img/' + pre + '_' + rd + '_domain.jpg', bbox_inches='tight', dpi=300)
                    plt.close()

                    df_others = df.loc[df['label'] == -1]
                    plt.xlim((x1lim_min, x1lim_max))
                    plt.xyim((x2lim_min, x2lim_max))
                    df_others.plot.scatter('x1', 'x2', c=col_others, s=10)
                    plt.gcf().set_size_inches(1, 1)s
                    plt.title(pre + '_' + rd + 'points others')
                    plt.savefig('img/' + pre + '_' + rd + '_others.jpg', bbox_inches='tight', dpi=300)
                    plt.close()

