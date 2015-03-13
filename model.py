from os.path import abspath, dirname, join, exists
from itertools import islice, chain

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from indicoio import batch_image_features

label_names = [
    'airplane', 
    'automobile', 
    'bird',
    'cat', 
    'deer',
    'dog', 
    'frog',
    'horse', 
    'ship', 
    'truck'
]

directory  = abspath(dirname(__file__))
file_path  = lambda suffix : join(directory, 'data', "cifar10-%s.pkl" % suffix)
url_path   = lambda suffix : 'https://s3.amazonaws.com/cifar10/cifar10-%s.pkl' % suffix


def download_data():
    """
    Downloads and saves cifar10 pickle files if not present
    """
    for suffix in ('train', 'test'):
        data_file, data_url = file_path(suffix), url_path(suffix)
        if not exists(data_file):
            print "Downloading %s to %s..." % (data_url, data_file)
            response = requests.get(data_url)
            with open(data_file, 'w') as fd:
                fd.write(response.content)


def batch(iterable, size):
    """
    Iterate over iterable in batches
    """
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([batchiter.next()], batchiter)


def transform_data():
    """
    Passes the downloaded data through the indico imagefeatures API
    """
    train = pd.read_pickle(file_path('train'))
    test  = pd.read_pickle(file_path('test'))

    # limit to the first 10000 training examples 
    train = train[:10000]
    
    train.name, test.name = 'train', 'test'

    for df in (train, test):
        imagefeatures = []

        i = 0
        batch_size = 50
        n = len(df.data)/batch_size

        print "Fetching %s imagefeatures..." % (df.name)
        for df_batch in batch(df.data, batch_size):
            print "\t%d/%d" % (i, n)
            imagefeatures.extend(batch_image_features(df_batch)) 
            i += 1

        df['features'] = imagefeatures

        df.to_pickle(file_path("cifar10-%s-features.pkl" % df.name))


def train_model():
    """
    Train a scikit-learn model on top of the imagefeatures
    """
    print "Loading data into memory..."
    train = pd.read_pickle(file_path('train-features'))
    test  = pd.read_pickle(file_path('test-features'))


def train_imagefeatures_model():
    """
    Train a logistic regression model using indico's imagefeatures as inputs
    """
    model = LogisticRegression()
    print "Fitting image features model..."
    model.fit(np.vstack(train['features'].values), train['labels'].values)

    print "Scoring image features model..."
    score = model.score(np.vstack(test['features'].values), test['labels'].values)
    print "Accuracy: %f\n" % score


def train_traditional_model():
    """
    Train a logistic regression model using raw pixels as inputs
    """
    model = LogisticRegression()
    print "Fitting pixel based model..."
    train_pixels = np.vstack([row.flatten() for row in train['data'].values])
    model.fit(train_pixels, train['labels'].values)

    print "Scoring pixel based model..."
    test_pixels = np.vstack([row.flatten() for row in test['data'].values])
    score = model.score(test_pixels, test['labels'].values)
    print "Accuracy: %f\n" % score


if __name__ == '__main__':
    download_data()
    transform_data()
    train_imagefeatures_model()
