import tweepy
import json
import csv
import click
import logging

logging.basicConfig()


def set_api():
    with open('lib/twitterAPI.json', 'r') as f:
        twitter_auth = json.load(f)

    print twitter_auth
    print 'setting oath'
    auth = tweepy.OAuthHandler(twitter_auth['consumer_token'], twitter_auth['consumer_secret'])
    auth.set_access_token(twitter_auth['access_token'], twitter_auth['access_secret'])

    print 'setting api'
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api


def download_data(data_path, save_path):
    api = set_api()
    with open(data_path, 'r') as rf, open('{}'.format(save_path), 'w') as wf:
        dict_reader = csv.DictReader(rf)
        dict_writer = csv.DictWriter(wf, fieldnames=dict_reader.fieldnames + ['text'])
        dict_writer.writeheader()
        count = 0
        suc_count = 0
        for line in dict_reader:
            count += 1
            tid = line['tid']
            try:
                tweet = api.get_status(tid)
                print tweet.text
                line.update({'text': tweet.text.encode('utf-8')})
                dict_writer.writerow(line)
                suc_count += 1
            except tweepy.TweepError as e:
                print "tweet cannot be downloaded due to: {}".format(e)
        print "expect {} tweets, {} downloaded".format(count, suc_count)


@click.command()
@click.option("--data_path", type=click.Path(exists=True), help="path to file has Tweet IDs")
@click.option("--save_path", type=click.Path(), help="path to file to save tweets")
def run_download_data(data_path, save_path):
    download_data(data_path, save_path)


if __name__ == '__main__':
    run_download_data()
