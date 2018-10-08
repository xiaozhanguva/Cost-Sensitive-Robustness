import examples.problems as pblm
from examples.heatmap import *
import setproctitle
import pandas as pd
import os

from matplotlib.ticker import FuncFormatter
plt.switch_backend('agg')
plt.rcdefaults()

if __name__ == '__main__':
    # robust heatmap for MNIST
    num_classes = 10        # in total, 10 digit classes for MNIST
    args = pblm.argparser(prefix='mnist', opt='adam', starting_epsilon=0.05, epsilon=0.2)
    labels = ['digit ' + x for x in list(map(str, range(num_classes)))]
    filepath = ('results/'+args.proctitle+'_robustProbs.csv')

    # load the pairwise robust error matrix
    df = pd.read_csv(filepath, sep='\t', skiprows=[0], nrows=num_classes, 
                    usecols=np.arange(1,num_classes+1), header=None)
    robust_prob_mat = df.applymap(lambda x: float(x.strip('%'))).values
    for i in range(num_classes):
        robust_prob_mat[i,i] = np.nan
    robust_prob_mat = np.ma.masked_invalid(robust_prob_mat)

    fig, ax = plt.subplots()
    ticks = np.arange(0, np.nanmax(robust_prob_mat), step=2.0, dtype=float)
    im = heatmap(robust_prob_mat, labels, labels, ax=ax, cmap="OrRd", 
                cbarlabel="Robust error rate", cbar_kw={'ticks':ticks, 'format':'%.0f%%'})

    texts = annotate_heatmap(im, data=robust_prob_mat, valfmt="{x:.1f}%", fontsize=8)
    fig.tight_layout()

    save_filepath = 'results/'+os.path.dirname(args.proctitle)+'/robust_heatmap.pdf'
    fig.savefig(save_filepath)
    plt.savefig('test.png')

