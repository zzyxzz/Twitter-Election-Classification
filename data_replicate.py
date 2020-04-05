from lib.preprocess_twitter_dataset import process_violence
from download_data import download_data
import click
import os


@click.command()
@click.option("--data_path", type=click.Path(exists=True), help="path to file has Tweet IDs")
@click.option("--name", type=click.Choice(['ph', 'gh', 'vz']), help="name of the dataset")
def download_and_process_tweets(data_path, name):
    if name == 'vz':
        lang = 'Spanish'
        stopword_path = 'downloads/stopwords/spanish_stopwords.txt'
    else:
        lang = 'English'
        stopword_path = 'downloads/stopwords/english_stopwords.txt'

    if not os.path.exists('downloads/raw/'):
        os.mkdir('downloads/raw/')
    if not os.path.exists('downloads/processed/'):
        os.mkdir('downloads/processed/')

    save_path_raw = 'downloads/raw/{}-tweets.csv'.format(name)
    save_path_processed = 'downloads/processed/{}-tweets.csv'.format(name)
    download_data(data_path=data_path, save_path=save_path_raw)

    process_violence(lang=lang, data_path=save_path_raw, stopword_path=stopword_path, save_path=save_path_processed)
    print "processed tweets are saved to {}".format(save_path_processed)


if __name__ == '__main__':
    download_and_process_tweets()
