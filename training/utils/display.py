import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# res_cnt = 0

def print_table(training_stats, out_file):
    # display output
    # Display floats with two decimal places.
    pd.options.display.precision = 4
    # pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    # df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    # df_stats = df_stats.set_index('epoch')
    # df_stats.to_csv(fr'/home/xl/VisAnalysis/training/log/text/result.csv', encoding='utf-8', index=False)
    # print(df_stats)
    # return df_stats
    training_stats.to_csv(out_file, encoding='utf-8', index=False)
    print(training_stats)
    return training_stats

"""
def print_fig(df_stats):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title(fr"Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.savefig(fr'/home/xl/VisAnalysis/training/log/img/result_{res_cnt}.jpg')
    # plt.show()

def print_res(training_stats):
    # global res_cnt

    df_stats = print_table(training_stats)
    # print_fig(df_stats)

    # res_cnt += 1
"""