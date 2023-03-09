import matplotlib.pyplot as plt 
import pandas as pd 
import os

if __name__ == '__main__':
    col_all = []
    col_labeled = []
    # blue, red, gray
    colors = ['#0000FF', '#FF0000', '#696969']
    df = pd.read_csv('middle/single-words_PCA_2d.csv')
    labels = df.label.values

    for label in labels:
        col_all.append(colors[int(label)])
        if label != -1:
            col_labeled.append(colors[int(label)])

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
                    df.plot.scatter('x1', 'x2', c=col_all, s=10)
                    plt.gcf().set_size_inches(8, 6)
                    plt.title(pre + '_' + rd + 'points')
                    plt.savefig('img/' + pre + '_' + rd + '.jpg', bbox_inches='tight', dpi=300)
                    plt.close()

                    df_label = df.loc[df['label'] != -1]
                    df_label.plot.scatter('x1', 'x2', c=col_labeled, s=10)
                    plt.gcf().set_size_inches(8, 6)
                    plt.title(pre + '_' + rd + '_labeled points')
                    plt.savefig('img/' + pre + '_' + rd + '_labeled.jpg', bbox_inches='tight', dpi=300)
                    plt.close()
