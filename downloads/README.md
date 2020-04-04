# Dataset
Tweet IDs of the dataset can be accessed from [EV dataset](http://researchdata.gla.ac.uk/564/).
The following instructions show how to download and pre-processed tweets if any customised pre-processing is required.
 
## Download tweets
Tweets can be downloaded using Twitter API using the provided script ``download_data.py``. Before running the script, you need to configure the Twitter API ``consumer_token``, ``consumer_secret``, ``access_token``, ``access_secret`` variables in ```lib/twitterAPI.json``` for accessing the Twitter API.
For more information about accessing Twitter API, please check info about [Twitter API access](https://developer.twitter.com/en/apply-for-access.html). In addtion to Twitter API access, you need tweet IDs of the [EV dataset](http://researchdata.gla.ac.uk/564/), which can be accessed separately. 

The EV dataset is in csv format (shown in the example below):
```
tid,uid,handler,election,violence,date
796303280896,1048772612,CDEEEE,yes,violence,2016-11-01
798360141124,2283746327,ABCCCC,yes,no,unknown
796269422126,1427364627,CBAAAA,no,no,unknown
813424462817,1293837483,EDCCCC,no,no,unknown
...
```

Once the EV dataset is downloaded, run
```
python download_data.py --data_path "path/to/ev/data/file" --save_path "file/path/to/save/downloaded/tweets"
```
The downloaded csv file will be in csv format (shown in the example below):
```
tid,uid,handler,election,violence,date,text
796303280896,1048772612,CDEEEE,yes,violence,2016-11-01,This is a tweet about electoral violence
798360141124,2283746327,ABCCCC,yes,no,unknown,This is a random tweet about election
796269422126,1427364627,CBAAAA,no,no,unknown,This is a random tweet
813424462817,1293837483,EDCCCC,no,no,unknown,This is a random tweet
...
```
Then tweets will be saved to the file path provided.

## Preprocess tweets
To preprocess the tweets, e.g. tokenization, remove stopwords, stemming, just use the ```lib/preprocess_twitter_dataset.py```.
Available options:
```
Options:
  --lang [English|Spanish|None]  language of the dataset
  --data_path PATH               path to the downloaded tweets dataset
  --stopword_path PATH           path to stopword file
  --save_path PATH               path to save processed data
  --help                         Show this message and exit.
```
The ``` --lang ``` option will let the codes choose the right stemmer for different languages. Note: If ```None``` is provided, then tweets will be processed without stemming.
For Venezuela tweets data, set ```--lang=Spanish```. For Ghana and Philippines tweet data, set ```--lang=English```.

e.g. for Venezuela tweets dataset:
```
python lib/preprocess_twitter_dataset.py --lang Spanish --data_path "file/path/to/venezuela/tweets/data" --save_path "path/to/save/processed/tweets/data --stopword_path "file/path/to/stopwords"
```
The processed tweets are tokenized and in csv format (shown in example below assuming violence vs non-violence, with stopword removal and stemming)
```
1,tweet,elector,violenc
0,random,tweet,elect
0,random,tweet,
0,random,tweet,
...
```
