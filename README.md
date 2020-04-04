# Twitter-Election-Classification

This repository maintains the required codes for training CNN models used in Twitter Election Classification.

# Creating an environment from an environment.yml file

Our code has a number of Python dependencies with particular versions. We recommend use of the Anaconda distribution of Python, and use of the ``environment.yml`` to initialise the dependencies. 

Use the terminal or an Anaconda Prompt for the following steps:

- Create the environment from the ``environment.yml`` file:
```
      conda env create -f environment.yml
```

- Verify that the new environment was installed correctly:
```
      conda env list
```

- Activate the new environment: 
```
      conda activate myenv
```

You can also check [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for creating environment from yaml file.

NOTE: Before running any command in this instruction, please make sure you are in the ```Twitter-Election-Classification``` folder.

# Replication
This section shows how to replicate the CNN and SVM results. 

## Download and process tweets
Tweets can be downloaded using Twitter API using the provided script ``data_replicate.py``. Before running the script, you need to configure the Twitter API ``consumer_token``, ``consumer_secret``, ``access_token``, ``access_secret`` variables in ```lib/twitterAPI.json``` for accessing the Twitter API.
For more information about accessing Twitter API, please check info about [Twitter API access](https://developer.twitter.com/en/apply-for-access.html). 

#### Apply for Twitter API access

```
https://developer.twitter.com/en/apply-for-access.html
```
Open ```lib/twitterAPI.json``` and copy paste your ```consumer_token```, ```consumer_secret```, ```access_token``` and ```access_secret``` into
```
{"consumer_token": YourConsumerToken, "consumer_secret": YourConsumerSecret, "access_token": YourAccessToken, "access_secret": YourAccessSecret}
```

#### Start downloading and pre-processing

In addtion to Twitter API access, you need tweet IDs of the [EV dataset](http://researchdata.gla.ac.uk/564/), which can be accessed separately. 

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

for downloading and pre-processing tweets of Ghana dataset:
```
python data_replicate.py --data_path /path/to/ghana/dataset/csv/file --name gh
```
for downloading and pre-processing tweets of Philippines dataset:
```
python data_replicate.py --data_path /path/to/philippines/dataset/csv/file --name ph
```
for downloading and pre-processing tweets of Venezuela dataset:
```
python data_replicate.py --data_path /path/to/venezuela/dataset/csv/file --name vz
```
Tweets will be automatically downloaded from Twitter and processed. 
Tweets before pre-processing are saved in folder ```download/raw```.
Pre-processed tweets are saved in folder ```download/processed```.

## Run the models to get results
To obtain the results of CNN models, run:
```
python cnn_replicate.py
``` 

To obtain the results of SVM models, run
```
python svm_replicate.py
```
Results will be printed on your screen.

Due to the randomness of weight initialization and data availability (e.g. tweets deleted by Twitter user result in fewer data), the results may vary slightly.

# Train models for new Twitter dataset
The new Twitter dataset should have the same csv format as the processed tweets described in ``downloads/README.md``. Train new models are straightforward, just run ``` cnn_train.py ``` and provide necessary parameters. 

If relevant parameters related to pre-trained word embeddings
 e.g.
 ```
 --vocab_path             File path to the vocabulary file that has all the words in the pre-trained embedding, one word per line
 --vector_path            File path to the embedding vector in .txt format, one vector per line
 ```
are not provided, run
```
python cnn_train.py --dataset_path "file/path/to/processed/twitter/dataset/csv/file" --lang "lang"
```
the CNN model will not use pre-trained embeddings.

To use pre-trained embeddings, run
```
python cnn_train.py --dataset_path "file/path/to/processed/twitter/dataset/csv/file" --lang "lang" --vocab_path "path/to/vocab/file --vector_path "path/to/embedding/vector/file/in/text/format"
```

#### Example for training models from processed Twitter dataset
train model from the processed Venezuela election dataset
```
python cnn_train.py --dataset_path downloads/processed/ph-tweets.csv --lang "es"
```

 
