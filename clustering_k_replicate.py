import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

def convert_to_dict(fpath):
    with open(fpath, 'r') as f:
        results = json.load(f)
    results_list = []
    for k in results:
        vio_iner = results[k][0]
        vio_inter = results[k][1]
        mal_iner = results[k][2]
        mal_inter = results[k][3]
        
        for it in range(len(vio_iner)):
            results_list.append((int(k), it, vio_iner[it], 'iner','violence'))
            results_list.append((int(k), it, vio_inter[it], 'inter', 'violence'))
            results_list.append((int(k), it, mal_iner[it], 'iner', 'malpractice'))
            results_list.append((int(k), it, mal_inter[it], 'inter', 'malpractice'))
    
    df = pd.DataFrame(results_list, columns=['k', 'iteration', 'distance', 'dist_type', 'event_type'])
    return df


gh_cnn_df = convert_to_dict('downloads/clustering-k/gh_cnn_results.json')
g = sns.factorplot(x="k", y="distance", hue="dist_type", col="event_type", data=gh_cnn_df, kind="box")
g.fig.suptitle('Ghana', fontsize=12)
g.fig.subplots_adjust(top=0.9)
plt.tight_layout()

ph_cnn_df = convert_to_dict('downloads/clustering-k/ph_cnn_results.json')
g = sns.factorplot(x="k", y="distance", hue="dist_type", col="event_type", data=ph_cnn_df, kind="box")
g.fig.suptitle('Philippines', fontsize=12)
g.fig.subplots_adjust(top=0.9)
plt.tight_layout()

vz_cnn_df = convert_to_dict('downloads/clustering-k/vz_cnn_results.json')
g = sns.factorplot(x="k", y="distance", hue="dist_type", col="event_type", data=vz_cnn_df, kind="box")
g.fig.suptitle('Venezuela', fontsize=12)
g.fig.subplots_adjust(top=0.9)
plt.tight_layout()

plt.show()
