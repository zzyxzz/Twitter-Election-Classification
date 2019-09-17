import tweepy
import json
import csv
import sys

with open('lib/twitterAPI.json', 'r') as f:
    for line in f:
        twitter_auth = json.loads(line)

print twitter_auth
print 'setting oath'
auth = tweepy.OAuthHandler(twitter_auth['consumer_token'], twitter_auth['consumer_secret'])
auth.set_access_token(twitter_auth['access_token'], twitter_auth['access_secret'])

print 'setting api'
api = tweepy.API(auth)


def download_data(data_path, save_path):
    with open(data_path, 'r') as rf, open('{}'.format(save_path), 'w') as wf:
        dict_reader = csv.DictReader(rf)
        dict_writer = csv.DictWriter(wf, fieldnames=dict_reader.fieldnames + ['text'])
        dict_writer.writeheader()
        count = 0
        for line in dict_reader:
            count += 1
            if count == 11:
                break
            tid = line['tid']
            try:
                tweet = api.get_status(tid)
                print tweet.text
                line.update({'text': tweet.text.encode('utf-8')})
                dict_writer.writerow(line)
            except Exception as e:
                print "tweet cannot be downloaded due to: {}".format(e)


if __name__ == '__main__':
    data_fp, save_fp = "/path/to/twitter/data", "/path/to/save/data"
    download_data(data_path=data_fp, save_path=save_fp)
