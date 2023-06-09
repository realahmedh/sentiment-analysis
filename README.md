# Sentiment Analysis

In this project, I implemented sentiment analysis using the Naive Bayes classifier from the Natural Language Toolkit (NLTK) library. I used the movie reviews corpus from NLTK to train and test our classifier.

## Requirements

To run this project, you need the following:

- Python 3.x
- NLTK library

You can install the NLTK library by running the following command:

```python
!pip install nltk
```

## Dataset

I used the movie reviews corpus from NLTK, which contains 2000 movie reviews classified as positive or negative. I randomly shuffled the reviews and split them into a training set and a test set.

```python
import nltk
from nltk.corpus import movie_reviews
import random

nltk.download('movie_reviews')

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
```

## Feature Extraction

I defined a feature extractor function that takes a document (a list of words) as input and returns a dictionary of features. The feature dictionary contains 2000 boolean features, where each feature indicates whether a particular word from a pre-defined list of 2000 most common words in the corpus is present in the document or not.

```python
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

```

## Training

I use the feature extractor to convert each document in the training set into a feature set, which is a dictionary of boolean features. I then used these feature sets along with the corresponding labels (positive or negative) to train a Naive Bayes classifier.

```python
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)
```

## Testing

I tested the accuracy of the classifier on the test set.

```python
print(nltk.classify.accuracy(classifier, test_set))
```

The output of the above code should be:

```
0.78
```

This means that our classifier correctly predicted the sentiment of 78% of the reviews in the test set.

## Most Informative Features

I used the `show_most_informative_features()` method of the classifier to display the 5 most informative features, i.e., the 5 words that are most strongly associated with positive or negative sentiment.

```python
classifier.show_most_informative_features(5)
```

The output of the above code should be:

```
Most Informative Features
        contains(ludicrous) = True              neg : pos    =      9.0 : 1.0
           contains(turkey) = True              neg : pos    =      8.3 : 1.0
        contains(atrocious) = True              neg : pos    =      7.7 : 1.0
          contains(suvari) = True              neg : pos    =      7.0 : 1.0
          contains(shoddy) = True              neg : pos    =      6.4 : 1.0
```

These are the 5 most informative features, along with their ratios of occurrence in negative to positive reviews. For example, the word "ludicrous" is 9 times more likely to appear in a negative review than in a positive review. 

## Conclusion

In this project, I implemented sentiment analysis using the Naive Bayes classifier from the NLTK library. I used the movie reviews corpus from NLTK to train and test our classifier, and achieved an accuracy of 78% on the test set. I also identified the 5 most informative features, which are the words that are most strongly associated with positive or negative sentiment.
