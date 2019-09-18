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
Tweets can be downloaded using Twitter API using the provided script ``download_data.py``. Before running the script, you need the Twitter API ``consumer_token``, ``consumer_secret``, ``access_token``, ``access_secret``. For more information about accessing Twitter API, please check info about [Twitter API access](https://developer.twitter.com/en/apply-for-access.html).
In addtion to Twitter API access, you need tweet ID of the [EV dataset](http://researchdata.gla.ac.uk/564/), which can be accessed separately. 

# Preprocess tweets
To preprocess the tweets, e.g. tokenization, remove stopwords, stemming, just use the ```lib/preprocess_twitter_dataset.py```.

# Train models
Train new models are straightforward, just run ``` cnn_train.py ``` and provide necessary parameters. If relevant parameters related to pre-trained word embeddings are not provided,
the CNN model will not use pre-trained embeddings.

# Reproduce
In order to reproduce the results of CNN models:
```
python cnn_pred_reprod.py
``` 
For SVM models:
```
python svm_pred_reprod.py
```
Due to the randomness of weight initialization and data availability (e.g. tweets deleted by Twitter user), the results may vary slightly.

 
