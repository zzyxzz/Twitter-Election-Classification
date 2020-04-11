import matplotlib.pyplot as plt
import json


def plot_2d(name, twod, keywords):
    x, y = list(zip(*twod))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y)

    for i, txt in enumerate(keywords):
        ax.annotate(txt, (x[i], y[i]))
    plt.scatter(x=x, y=y)
    plt.title(name)
    plt.tight_layout()


def plot_embedding(fpath):
    with open(fpath, 'r') as f:
        plot_pairs = json.load(f)
    for name in plot_pairs:
        plot_2d(name, plot_pairs[name]['2d'], plot_pairs[name]['keywords'])
    plt.show()


if __name__ == '__main__':
    plot_embedding('downloads/embed_2d/embedding_plot.json')
