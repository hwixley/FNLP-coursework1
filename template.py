"""
Foundations of Natural Language Processing

Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

```
[hostname]s1234567 python3 s1234567.py
```

Before submission executed your code with ``--answers`` flag
```
[hostname]s1234567 python3 s1234567.py --answers
```
include generated answers.py file.

Best of Luck!
"""
from ast import operator
from cgi import test
from collections import defaultdict, Counter
import enum
from lib2to3.pgen2 import token
from ntpath import join
from operator import itemgetter
from string import punctuation
from bleach import clean

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach
from sklearn.multiclass import OutputCodeClassifier  # import corpora

# Import the Twitter corpus and LgramModel
from nltk_model import *  # See the README inside the nltk_model folder for more information

# Import the Twitter corpus and LgramModel
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

def ppEandT(eAndTs):
    '''
    Pretty print a list of entropy-tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None
    '''

    for entropy, tweet in eAndTs:
        print("{:.3f} [{}]".format(entropy, ", ".join(tweet)))


def compute_accuracy(classifier, data):
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :type data: list(tuple(list(any), str))
    :param data: A list with tuples of the form (list with features, label)
    :rtype float
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct/len(data)


def apply_extractor(extractor_f, data):
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :type extractor_f: (str, str, str, str) -> list(any)
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :type data: list(tuple(str))
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :rtype list(tuple(list(any), str))
    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """
    def __init__(self, classifier_class, train_features, **kwargs):
        """

        :type classifier_class: a class object of nltk.classify.api.ClassifierI
        :param classifier_class: the kind of classifier we want to create an instance of.
        :type train_features: list(tuple(list(any), str))
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype dict(any, int)
        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype str
        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n = 10):
        self.classifier_obj.show_most_informative_features(n)

# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1 [7 marks]
def train_LM(corpus):
    '''
    Build a bigram letter language model using LgramModel
    based on the all-alpha subset the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel
    '''
    # subset the corpus to only include all-alpha tokens,
    # converted to lower-case (_after_ the all-alpha check)
    corpus_tokens = [w.lower() for w in corpus.words(corpus.fileids()) if w.isalpha()]
    
    # Return a smoothed (using the default estimator) padded bigram
    # letter language model
    return LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)


# Question 2 [7 marks]
def tweet_ent(file_name, bigram_model):
    '''
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets'''

    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 (remaining) tokens, converted
    # to lowercase
    list_of_tweets = xtwc.sents(file_name)
    alpha_tweets = [[token.lower() for token in tweet if token.isalpha()] for tweet in list_of_tweets]
    cleaned_list_of_tweets = [alpha_tweet for alpha_tweet in alpha_tweets if len(alpha_tweet) >= 5]

    # Construct a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average word for the tweet, and return the list of
    #  (entropy,tweet) tuples sorted by entropy
    ents = {idx: np.mean([bigram_model.entropy(word, pad_left=True, pad_right=True, perItem=True) for word in tweet]) for idx, tweet in enumerate(cleaned_list_of_tweets)}
    sorted_ents = sorted(ents.items(), key=lambda item: item[1])
    list_of_tuples = [(item[1], cleaned_list_of_tweets[item[0]]) for item in sorted_ents]

    return list_of_tuples



# Question 3 [8 marks]
def open_question_3():
    '''
    Question: What differentiates the beginning and end of the list
    of tweets and their entropies?

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""
    The entropy values represent the average uncertainty the model
    has with classifying all the words in the given tweet.
    The first tweets are all English words, the most common being
    conjunctions ("and"), noun articles ("the"), and nouns
    ("weather", "love").
    The last tweets mainly consisted of non-ASCII logograms from
    other languages. This was to be expected given these languages
    are evidently not likely to be used in an English tweet.""")[0:500]


# Question 4 [8 marks]
def open_question_4() -> str:
    '''
    Problem: noise in Twitter data

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""
    We should remove all non-English tweets (non-ASCII) from the corpus
    as these characters/words are obviously not relevant for
    developing an English NL model.
    We can identify non-English tweets by checking if they contain 
    non-ASCII characters as ASCII is only used for the English language.""")[0:500]


# Question 5 [15 marks]
def tweet_filter(list_of_tweets_and_entropies):
    '''
    Compute entropy mean, standard deviation and using them,
    likely non-English tweets in the all-ascii subset of list 
    of tweets and their letter bigram entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average letter bigram entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             non-English tweets and entropies
    '''
    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    idx = int(0.9*len(list_of_tweets_and_entropies))
    list_of_ascii_tweets_and_entropies = list_of_tweets_and_entropies[0:idx]

    # Extract a list of just the entropy values
    list_of_entropies = [tweet[0] for tweet in list_of_ascii_tweets_and_entropies]

    # Compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # Compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # Get a list of "probably not English" tweets, that is
    #  "ascii" tweets with an entropy greater than (mean + std_dev))
    threshold = mean + standard_deviation
    list_of_not_English_tweets_and_entropies = [tweet for tweet in list_of_ascii_tweets_and_entropies if tweet[0] > threshold]

    # Return mean, standard_deviation,
    #  list_of_ascii_tweets_and_entropies,
    #  list_of_not_English_tweets_and_entropies
    return mean, standard_deviation, list_of_ascii_tweets_and_entropies, list_of_not_English_tweets_and_entropies


# Question 6 [15 marks]
def open_question_6():
    """
    Suppose you are asked to find out what the average per word entropy of English is.
    - Name 3 problems with this question, and make a simplifying assumption for each of them.
    - What kind of experiment would you perform to estimate the entropy after you have these simplifying assumptions?
       Justify the main design decisions you make in your experiment.
    :rtype: str
    :return: your answer [1000 chars max]
    """
    return inspect.cleandoc("""
    Problems:
    This question is rather vague because...
    1. It does not detail the era of English.
    English has been spoken for centuries and has evolved massively using all the data across
    these centuries together would not be useful due to completely different dialects so I will
    assume this question refers to the 21st century.
    2. It does not detail the dialect(s) of English.
    Due to the fact that English is so widely spoken (many different countries) there are many
    different dialects with various differences in the spelling of words (ie. British English vs.
    American English). This is problematic as it would result in not representing equivalent words
    ("colour" vs. "color") as the same thing. So I will assume the question refers to British
    English.
    3. It does not detail the genre of English.
    Corpora data typically have a genre based on where the data was scraped from (ie. Web News)
    and thus these are not representative of typical English usage (ie. conversational English).
    So I will assume that this question refers to balanced genre corpora extracted from the Web.

    Experiment:
    1. Find a British English corpora from the 21st-century with a balanced genre that has been
    extracted from the web.
    2. Perform necessary preprocessing: tokenise the corpus (split sentences into words).
    3. Compute a dictionary of word frequencies for all the words in the corpus.
    4. Compute a dictionary of word priors by dividing the word frequency dictionary by the
    sum of all frequencies.
    5. Create a function that calculates entropy of a word using its prior (-prior*log(prior)).
    4. Iterate this function over all unique words in the corpus and take the mean to get the
    average word entropy for this corpus.


    1. Given Zipf's law we know that the frequency of any word is inversely proportional
    to its rank in the frequency table indicating the majority of words in English
    are rarely used. Thus almost making this metric insignificant as the average will likely
    be quite large.
    """)[:1000]


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 7 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data, alpha):
        """
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data):
        """
        Compute the set of all possible features from the (training) data.
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :rtype: set(any)
        :return: The set of all features used in the training data for all classes.
        """
        return {ftr for el in data for ftr in el[0]}

    @staticmethod
    def train(data, alpha, vocab):
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :type data: list(tuple(list(any), str))
        :param data: A list of tuples ([f1, f2, ... ], c) with the first element
                     being a list of features and the second element being its class.

        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing

        :type vocab: set(any)
        :param vocab: The set of all features used in the training data for all classes.


        :rtype: tuple(dict(str, float), dict(str, dict(any, float)))
        :return: Two dictionaries: the prior and the likelihood (in that order).
        We expect the returned values to relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """
        assert alpha >= 0.0

        likelihood, prior = {}, {}
        dclasses, dftrs = [], []

        # Compute raw frequency distributions
        cfdist = {}
        for el in data:
            dclasses.append(el[1])

            if not el[1] in cfdist.keys():
                cfdist[el[1]] = {}

            for ftr in el[0]:
                dftrs.append(ftr)

                if ftr in cfdist[el[1]].keys():
                    cfdist[el[1]][ftr] = cfdist[el[1]][ftr] + 1
                else:
                    cfdist[el[1]][ftr] = 1

        classes = set(dclasses)
        ftrs = set(dftrs)

        class_counts = {c: dclasses.count(c) for c in classes}
        ftr_counts = {f: dftrs.count(f) for f in ftrs}

        # Compute prior (MLE). Compute likelihood with smoothing.
        #num_samples = np.sum(list(class_counts.values()))
        num_ftrs = np.sum(list(ftr_counts.values()))

        for c in classes:
            prior[c] = class_counts[c]/len(data)
            likelihood[c] = {}
            for v in vocab:
                if not v in cfdist[c].keys():
                    cfdist[c][v] = 0

                prob_cv = cfdist[c][v]/ftr_counts[v]
                prob_v = ftr_counts[v]/num_ftrs
                likelihood[c][v] = ((prob_cv) + alpha)/(prior[c] + alpha*len(vocab))
                assert likelihood[c][v] >= 0

            tot_prob = np.sum(list(likelihood[c].values()))
            for v in vocab:
                likelihood[c][v] = likelihood[c][v]/tot_prob

            assert abs(np.sum(list(likelihood[c].values())) - 1) <= 1e-12
            assert prior[c] >= 0
        assert abs(np.sum(list(prior.values())) - 1) <= 1e-12

        return prior, likelihood



    def prob_classify(self, d):
        """
        Compute the probability P(c|d) for all classes.
        :type d: list(any)
        :param d: A list of features.
        :rtype: dict(str, float)
        :return: The probability p(c|d) for all classes as a dictionary.
        """
        classes = set(self.likelihood.keys())
        c_probs = {}

        cftr_lh_count = {}
        for ftr in d:
            if ftr in self.vocab:
                cftr_lh_count[ftr] = 0
                for c in classes:
                    cftr_lh_count[ftr] += self.likelihood[c][ftr]

        for c in classes:
            ftr_likelihoods = [self.likelihood[c][ftr]/cftr_lh_count[ftr] for ftr in d if ftr in self.vocab]
            c_probs[c] = np.prod(ftr_likelihoods)
            assert c_probs[c] >= 0

        tot_prob = np.sum(list(c_probs.values()))
        #print(tot_prob)
        for c in classes:
            c_probs[c] = c_probs[c]/tot_prob

        assert abs(np.sum(list(c_probs.values())) - 1) <= 1e-12

        return c_probs


    def classify(self, d):
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :type d: list(any)
        :param d: A list of features.
        :rtype: str
        :return: The most likely class.
        """
        probs = self.prob_classify(d)
        return max(probs, key=probs.get)

        



# Question 8 [10 marks]
def open_question_8() -> str:
    """
    How do you interpret the differences in accuracy between the different ways to extract features?
    :rtype: str
    :return: Your answer of 500 characters maximum.
    """
    return inspect.cleandoc("""
    The best accuracy was achieved using a sequence of words. Indicating that
    this model is most useful when passed a sequence of words.

    My NB accuracy is worse than all LR scores in Table 1. I believe this
    difference can mainly be attributed to the NB independence assumption
    as this infers probability distributions about features that are likely
    not true. Thus a model that does not assume any distribution would be
    more useful.
    """)[:500]


# Feature extractors used in the table:
# see your_feature_extractor for documentation on arguments and types.
def feature_extractor_1(v, n1, p, n2):
    return [v]


def feature_extractor_2(v, n1, p, n2):
    return [n1]


def feature_extractor_3(v, n1, p, n2):
    return [p]


def feature_extractor_4(v, n1, p, n2):
    return [n2]


def feature_extractor_5(v, n1, p, n2):
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]

# Q9.1: Supplementary Function
# ----------------------------
# Get features for the given word
# -------------------------------
def word_ftrs(word, str):
    features = []
    #features.append((f"{str}_steps", len(word)*)
    features.append((f"{str}_count",len(word)))
    features.append((f"{str}_0upper",word[0].isupper()))
    features.append((f"{str}_0vowel", word[0].lower() in "aeiou"))
    features.append((f"{str}_0", word[0]))
    #features.append((f"{str}_-1plural",  word[-2:].lower() == "es" if len(word) > 1 else False))
    #features.append((f"{str}_-1s",  word[-1].lower() == "s"))
    features.append((f"{str}_-1", word[-1]))

    return features

# Q9.1: Supplementary Function
# ----------------------------
# Parse a word into 2 numerical values
# ------------------------------------
# Uses a step as a means to give more significance
# to adjacent letter combinations (which also helps
# prevent classifying 2 words with the same letters
# as the same thing)
def parse_word(word, step):
    # primes: used to hold the identifiers for all our characters
    #primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,59,61,67,71,73,79,83,89,97,101,103,107,109,113, 127, 131, 139, 149, 151, 157, 163, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, -269]
    letters = "eariotnslcudpmhgbfywkvxzjq" # ordered by English letter frequency
    #numbers = "0123456789" # ascending order
    #punctuation = ".?!,:;'\"-$%&*()+#/" # arbitrary order

    sum = 0
    prod = 1
    temp_sum = 0
    temp_prod = 1
    for i, c in enumerate(word):
        #idx = -1
        #if c.lower() in letters:
        #    idx = letters.index(c.lower())
        #elif c in numbers:
        #    idx = 26 + numbers.index(c)
        #elif c in punctuation:
        #    idx = 36 + punctuation.index(c)
        char = c
        if c.lower() in letters:
            char = c.lower()

        if i % step == 0:
            if step == 1:
                sum += ord(char)
            else:
                sum += temp_sum*ord(char) #primes[idx]
            prod = prod*(temp_prod + ord(char)) #primes[idx])
            temp_sum = 0
            temp_prod = 1
        else:
            temp_sum += ord(char) #primes[idx]
            temp_prod = temp_prod*ord(char) #primes[idx]

    return sum, prod

# Q9.1: Supplementary Function
# ----------------------------
# Extracts features from the numerical values associated
# with these words (calculated in parse_word)
# ------------------------------------------------------
def parse_words(sums, prods):
    N = len(sums)
    S = len(sums[0])
    assert(N == len(prods))
    assert(S == len(prods[0]))
    features = []

    suml, prodl, spl, s_pl = [], [], [], []

    sum, prod = 0, 1
    max_sum, max_prod, max_sp, max_s_p = -9e+10, -9e+10, -9e+10, -9e+10
    for j in range(S):
        #tsum = 0
        #tprod = 1

        for i in range(N):
            #tsum += sums[i][j]
            #tprod = tprod*prods[i][j]
            sum += sums[i][j]
            prod = prod*prods[i][j]
        features.append(sum) #, prod, sum*prod, sum + prod))
        features.append(prod)
        features.append(sum*prod)
        features.append(sum + prod)
        """
        suml.append(sum) #, prod, sum*prod, sum + prod))
        prodl.append(prod)
        spl.append(sum*prod)
        s_pl.append(sum + prod)

        if sum > max_sum:
            max_sum = sum
        if prod > max_prod:
            max_prod = prod
        if spl[-1] > max_sp:
            max_sp = sum*prod
        if s_pl[-1] > max_s_p:
            max_s_p = sum+prod
    features = [el/max_sum for el in suml] + [el/max_prod for el in prod] +  [el/max_sp for el in spl] + [el/max_s_p for el in s_pl]
    """
    return features

# Q9.1: Supplementary Function
# ----------------------------
# Extracts features for the given word combinations
# -------------------------------------------------
def dfunc2(data, strs):
    #vowels = "aeiou"
    #abc = "abcdefghijklmnopqrstuvwxyz"
    features = []

    sums = []
    prods = []
    for i, d in enumerate(data):
        steps = [1,2,3]
        sum_list = []
        prod_list = []

        for s in steps:
            sum, prod = parse_word(d, s)
            features.append((f"{strs[i]}_{s}_sum",sum))
            features.append((f"{strs[i]}_{s}_prod",prod))
            sum_list.append(sum)
            prod_list.append(prod)

        sums.append(sum_list)
        prods.append(prod_list)

    ftrs = parse_words(sums, prods)

    return features + ftrs


def dfunc(data, strs):
    vowels = "aeiou"
    abc = "abcdefghijklmnopqrstuvwxyz"
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,59,61,67,71,73,79,83,89,97,101,103]
    features = []

    t_vprod, t_tprod, t_sprod, t_uprod = 1, 1, 1, 1
    t_vsum, t_tsum, t_ssum, t_usum = 0, 0, 0, 0
    for i,d in enumerate(data):
        #print(train_d.similar(d))
        #features.append((f"{strs[i]}_similar",train_d.similar(d)))

        #features.append((f"{strs[i]}_count",len(d)))
        #features.append((f"{strs[i]}_0upper",d[0].isupper()))
        #features.append((f"{strs[i]}_0vowel",d[0] in vowels))
        #features.append((f"{strs[i]}_0",d[0]))
        #features.append((f"{strs[i]}_-1",d[-1]))
        #features.append((f"{strs[i]}_0alpha", d[0].lower() in abc))
        #features.append((f"{strs[i]}_-1alpha", d[-1].lower() in abc))

        #alpha_count = 0
        vowel_count = 0
        vprod, tprod, sprod, uprod = 1, 1, 1, 1
        vsum, tsum, ssum, usum = 0, 0, 0, 0
        for j,c in enumerate(d):
            if c.lower() in vowels:
                vowel_count += 1

            val = primes[abc.find(c.lower())]
            vprod = vprod*val
            vsum += val
            if j>0 and j % 2 == 0:
                tsum += (val - primes[abc.find(d[j-1].lower())])**2
                tprod = tprod*(val - primes[abc.find(d[j-1].lower())])**2
            if j>0 and j % 3 == 0:
                ssum += (val - primes[abc.find(d[j-1].lower())] - primes[abc.find(d[j-2].lower())]**2)#/primes[abc.find(d[j-2].lower())]
                sprod = sprod*((val - primes[abc.find(d[j-1].lower())] - primes[abc.find(d[j-2].lower())]))**2#/primes[abc.find(d[j-2].lower())])**2
            if j>0 and j % 4 == 0:
                usum += (val - primes[abc.find(d[j-1].lower())] - primes[abc.find(d[j-2].lower())]**2 - primes[abc.find(d[j-3].lower())]**3)#/primes[abc.find(d[j-2].lower())]
                uprod = uprod*((val - primes[abc.find(d[j-1].lower())] - primes[abc.find(d[j-2].lower())]**2 - primes[abc.find(d[j-3].lower())]**3))**2#/primes[abc.find(d[j-2].lower())])**2

        #features.append(alpha_count) #(f"{strs[i]}_alpha_count",alpha_count))
        #features.append(alpha_count/len(d)) #(f"{strs[i]}_alpha_mean",alpha_count/len(d)))
        features.append(vowel_count) #(f"{strs[i]}_vowel_count",vowel_count))
        features.append(vowel_count/len(d)) #(f"{strs[i]}_vowel_mean",vowel_count/len(d)))
        features.append(vprod) #(f"{strs[i]}_vprod",vprod))
        features.append(vsum) #(f"{strs[i]}_vsum",vsum))
        features.append(tsum) #(f"{strs[i]}_tsum",tsum))
        features.append(tprod) #(f"{strs[i]}_tprod",tprod))
        features.append(ssum) #(f"{strs[i]}_ssum",ssum))
        features.append(sprod) #(f"{strs[i]}_sprod",sprod))
        features.append(usum) #(f"{strs[i]}_usum",usum))
        features.append(uprod) #(f"{strs[i]}_uprod",uprod))
            #features.append(idx)
            #features.append(abc.find(c.lower())**2*d.lower().count(c.lower()))

        #for v in vowels:
            #features.append(d.lower().count(v))
        for letter in abc:
            features.append((f"{strs[i]}-{letter}_count",d.lower().count(letter)))
        features.append((f"{strs[i]}_upper_count",np.sum([1 for c in d if c.isupper()])))

        t_sprod = t_sprod*sprod
        t_vprod = t_vprod*vprod
        t_tprod = t_tprod*tprod
        t_uprod = t_uprod*uprod
        t_ssum += ssum
        t_vsum += vsum
        t_tsum += tsum
        t_usum += usum

        #features.append((f"{strs[i]}_alpha_count",t_alpha_count))
        #features.append((f"{strs[i]}_vowel_count",t_vowel_count))
        features.append((f"{strs[i]}_vprod",t_vprod))
        features.append((f"{strs[i]}_vsum",t_vsum))
        features.append((f"{strs[i]}_tsum",t_tsum))
        features.append((f"{strs[i]}_tprod",t_tprod))
        features.append((f"{strs[i]}_ssum",t_ssum))
        features.append((f"{strs[i]}_sprod",t_sprod))
        features.append((f"{strs[i]}_usum",t_usum))
        features.append((f"{strs[i]}_uprod",t_uprod))
    return features

# Question 9.1 [5 marks]
def your_feature_extractor(v, n1, p, n2):
    """vsumvsum
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.
    :type v: str
    :param v: The verb.
    :type n1: str
    :param n1: Head of the object NP (Noun Phrase).
    :type p: str
    :param p: The preposition.
    :type n2: str
    :param n2: Head of the NP embedded in the PP (Prepositional Phrase).
    :rtype: list(any)
    :return: A list of features produced by you.
    """
    data = [v, n1, p, n2]
    #lfreqs = {"e": 57, "a": 43, "r": 39, "i": 38, "o": 37, "t": 36}
    features = [("v", v), ("n1", n1), ("p", p), ("n2", n2), ("n1-p", (n1, p)), ("n2-p", (n2, p)), ("v-n2", (v, n2)), ("v-p", (v, p))]
    features = features + word_ftrs(v, "v") + word_ftrs(n1, "n1") + word_ftrs(n2, "n2") + word_ftrs(p, "p")
    # Verb features

    #features = features + [("v-p", v+p), ("n1-p", n1+p), ("n2-p", n2+p), ("n1-n2-p", n1+n2+p)]
    ptags = [ptag[1] for ptag in nltk.pos_tag(data)]
    #print(ptags)
    #joint_ptags = []
    #for i in range(len(ptags)):
    #    for j in range(len(ptags)):
    #        if i != j:
    #            joint_ptags.append(ptags[i] + ptags[j])
    features = features + ptags #+ joint_ptags
    #features = features + dfunc2(ptags[0:2], ["V", "N1"]) + dfunc2(ptags[2:4], ["P", "N2"])
    #features = features + [nltk.pos_tag(data)]
    #features = features + [nltk.pos_tag(v), nltk.pos_tag(n1), nltk.pos_tag(n2)]
    #features.append("s" == n1[-1])
    #features = features + [nltk.pos_tag_sents(v), nltk.pos_tag(n1), nltk.pos_tag(n2)]
    
    
    if "ing" == v[-3:]:
        features.append(True)
        features.append(False)
        features.append(False)
        features.append(v[:-3])
    elif "ed" == v[-2:]:
        features.append(False)
        features.append(True)
        features.append(False)
        features.append(v[:-2])
    elif "s" == v[-1]:
        features.append(False)
        features.append(False)
        features.append(True)
        features.append(v[:-1])
    else:
        features.append(False)
        features.append(False)
        features.append(False)
        features.append(v)
    
    #features.append(v == "is")
    #features.append(v == "be")
    #features.append(v == "am")
    #features.append(v == "have")
    #features.append(v == "do")
    #features.append(v == "go")
    #Noun1 features
    features.append(n1[-1] == "s")
    features.append("?" in v)
    features.append("?" in p)
    features.append("?" in n1)
    features.append("?" in n2)

    #step_sizes = [2,3]
    #for d in data:
    #    for s in step_sizes:
    #        steps = int(len(d)/s)
    #        for i in range(steps):
    #            print(d[i*s:i*s + s])
    #            features.append(d[i*s:i*s + s])
    #prefix_sizes = [4,]

    #ed = False
    #ing = False
    #offset = 0
    #if "ed" == ldata[0][-2:]:
    #    ed = True
    #    offset = -2
    #elif "ing" == ldata[0][-3:]:
    #    ing = True
    #features.append(("V-ed", ed))
    #features.append(("V-ing", ing))
    #features.append(("V_count", len(ldata[0])-offset))



    #+ dfunc([p, n2], ["P", "N2"])
    #n1p = dfunc([n1, p], ["N1", "P"])
    #n2p = dfunc([n2, p], ["N2", "P"])
    #n3p = [x + y for x,y in zip(dfunc2([n1, p], ["N1", "P"]), dfunc2([n2, p], ["N2", "P"]))]

    #features = features + dfunc2([v, n1], ["V", "N1"]) + dfunc2([v, n2], ["V", "N2"]) + dfunc2([p, v], ["P", "V"]) + n3p + dfunc2([v, n1, p, n2], ["V", "N1", "P", "N2"]) + dfunc2([n2, p, n1, v], ["N2","P","N1","V"]) #+ dfunc2([v, n1, p], ["V", "N1", "P"]) + dfunc2([n1, p, n2], ["N1", "P", "N2"]) #+ dfunc([v, n1, p, n2], ["V", "N1", "P", "N2"])

    #raise NotImplementedError  # remove when you finish defining this function
    #print(len(features))
    return features


# Question 9.2 [10 marks]
def open_question_9():
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick 3 examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.
    :rtype: str
    :return: Your answer of 1000 characters maximum.
    """
    return inspect.cleandoc("""
    Feature templates:
    Due to the fact models perform best with numerical data I wanted to create a way
    to represent words and combinations of words numerically. In order to do this I
    decided to use ord() to retrieve the ASCII code for each letter in a word and use
    it to calculate values that would form as representations for the words to which
    these letters belong.

    """)[:1000]


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""

def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm
    global best10_ents, worst10_ents, mean, std, best10_ascci_ents, worst10_ascci_ents
    global best10_non_eng_ents, worst10_non_eng_ents
    global answer_open_question_4, answer_open_question_3, answer_open_question_6,\
        answer_open_question_8, answer_open_question_9
    global ascci_ents, non_eng_ents

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features
    """
    print("*** Part I***\n")

    print("*** Question 1 ***")
    print('Building brown bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 2 ***")
    ents = tweet_ent(twitter_file_ids, lm)
    print("Best 10 english entropies:")
    best10_ents = ents[:10]
    ppEandT(best10_ents)
    print("Worst 10 english entropies:")
    worst10_ents = ents[-10:]
    ppEandT(worst10_ents)
    
    print("*** Question 3 ***")
    answer_open_question_3 = open_question_3()
    print(answer_open_question_3)

    print("*** Question 4 ***")
    answer_open_question_4 = open_question_4()
    print(answer_open_question_4)

    print("*** Question 5 ***")
    mean, std, ascci_ents, non_eng_ents = tweet_filter(ents)
    print('Mean: {}'.format(mean))
    print('Standard Deviation: {}'.format(std))
    print('ASCII tweets ')
    print("Best 10 English entropies:")
    best10_ascci_ents = ascci_ents[:10]
    ppEandT(best10_ascci_ents)
    print("Worst 10 English entropies:")
    worst10_ascci_ents = ascci_ents[-10:]
    ppEandT(worst10_ascci_ents)
    print('--------')
    print('Tweets considered non-English')
    print("Best 10 English entropies:")
    best10_non_eng_ents = non_eng_ents[:10]
    ppEandT(best10_non_eng_ents)
    print("Worst 10 English entropies:")
    worst10_non_eng_ents = non_eng_ents[-10:]
    ppEandT(worst10_non_eng_ents)

    print("*** Question 6 ***")
    answer_open_question_6 = open_question_6()
    print(answer_open_question_6)
    """
    """
    print("*** Part II***\n")

    print("*** Question 7 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 8 ***")
    answer_open_question_8 = open_question_8()
    print(answer_open_question_8)
    """
    # This is the code that generated the results in the table of the CW:

    # A single iteration of suffices for logistic regression for the simple feature extractors.
    #
    # extractors_and_iterations = [feature_extractor_1, feature_extractor_2, feature_extractor_3, eature_extractor_4, feature_extractor_5]
    #
    # print("Extractor    |  Accuracy")
    # print("------------------------")
    #
    # for i, ex_f in enumerate(extractors, start=1):
    #     training_features = apply_extractor(ex_f, ppattach.tuples("training"))
    #     dev_features = apply_extractor(ex_f, ppattach.tuples("devset"))
    #
    #     a_logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=6, trace=0)
    #     lr_acc = compute_accuracy(a_logistic_regression_model, dev_features)
    #     print(f"Extractor {i}  |  {lr_acc*100}")
    
    
    print("*** Question 9 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc*100}")

    answer_open_question_9 = open_question_9()
    print("Answer to open question:")
    print(answer_open_question_9)
    



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
