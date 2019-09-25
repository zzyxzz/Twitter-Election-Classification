# Twitter-Election-Classification

This repository maintains the required codes for training CNN models used in Twitter Election Classification.

# Creating an environment from an environment.yml file

Use the terminal or an Anaconda Prompt for the following steps:

- Create the environment from the ``environment.yml`` file:
```
      conda env create -f environment.yml
```
- Activate the new environment: ``conda activate myenv``
- Verify that the new environment was installed correctly:

```
      conda env list
```
You can also check [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for creating environment from yaml file.

# Download tweets
Tweets can be downloaded using Twitter API using the provided script ``download_data.py``. Before running the script, you need to add the Twitter API's ``consumer_token``, ``consumer_secret``, ``access_token``, ``access_secret`` to ```lib/twitterAPI.json```. For more information about accessing Twitter API, please check info about [Twitter API access](https://developer.twitter.com/en/apply-for-access.html).
In addtion to Twitter API access, you need tweet ID of the [EV dataset](http://researchdata.gla.ac.uk/564/), which can be accessed separately.

Once the dataset is downloaded, run
```
python download_data.py --data_path "path/to/ev/data/file" --save_path "file/path/to/save/downloaded/tweets"
```
Then tweets will be save to the file path you provided.

# Preprocess tweets
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

# Train models
Train new models are straightforward, just run ``` cnn_train.py ``` and provide necessary parameters. 
If relevant parameters related to pre-trained word embeddings
 e.g.
 ```
 --vocab_path             File path to the vocabulary file that has all the words in the pre-trained embedding, one word per line
 --vector_path            File path to the embedding vector in .txt format, one vector per line
 ```
are not provided, run
```
python cnn_train --dataset_path "file/path/to/processed/venezuela/data" --lang "es"
```
the CNN model will not use pre-trained embeddings.

To use pre-trained embeddings, run
```
python cnn_train --dataset_path "file/path/to/processed/venezuela/data" --lang "es" --vocab_path "path/to/vocab/file --vector_path "path/to/embedding/vector/file/in/text/format"
```
 
# Reproduce
In order to reproduce the results of CNN models, run
```
python cnn_pred_reprod.py
``` 
The parameters (saved in ```lib/rep_settings.py```) will be automatically loaded.

For SVM models, run
```
python svm_pred_reprod.py
```
Due to the randomness of weight initialization and data availability (e.g. tweets deleted by Twitter user result in fewer data), the results may vary slightly.

 
