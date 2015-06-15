# ImageFeaturesClassifier

Using indico's imagefeatures API and scikit-learn to produce a solve a novel image classification task with small amounts of training data.

CIFAR-10 Performance (Logistic regression on raw pixels):
```
100   examples: 19%
500   examples: 24%
1000  examples: 25%
```

CIFAR-10 Performance (Logistic regression on indico imagefeatures):
```
100   examples: 45%     
500   examples: 64% 
1000  examples: 67%
```

Usage
-----

```
python model.py
```

In order to use indico's batch imagefeatures API, you'll also have to set your username and password in your `~/.indicorc` file.

```
[auth]
api_key = "YOUR_API_KEY"
```

