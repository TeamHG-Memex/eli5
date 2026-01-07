
Named Entity Recognition using sklearn-crfsuite
===============================================

In this notebook we train a basic CRF model for Named Entity Recognition
on CoNLL2002 data (following
https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb)
and check its weights to see what it learned.

To follow this tutorial you need NLTK > 3.x and sklearn-crfsuite Python
packages. The tutorial uses Python 3.

.. code:: ipython3

    import nltk
    import sklearn_crfsuite
    import eli5

1. Training data
----------------

CoNLL 2002 datasets contains a list of Spanish sentences, with Named
Entities annotated. It uses
`IOB2 <https://en.wikipedia.org/wiki/Inside_Outside_Beginning>`__
encoding. CoNLL 2002 data also provide POS tags.

.. code:: ipython3

    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
    train_sents[0]




.. parsed-literal::

    [('Melbourne', 'NP', 'B-LOC'),
     ('(', 'Fpa', 'O'),
     ('Australia', 'NP', 'B-LOC'),
     (')', 'Fpt', 'O'),
     (',', 'Fc', 'O'),
     ('25', 'Z', 'O'),
     ('may', 'NC', 'O'),
     ('(', 'Fpa', 'O'),
     ('EFE', 'NC', 'B-ORG'),
     (')', 'Fpt', 'O'),
     ('.', 'Fp', 'O')]



2. Feature extraction
---------------------

POS tags can be seen as pre-extracted features. Let’s extract more
features (word parts, simplified POS tags, lower/title/upper flags,
features of nearby words) and convert them to sklear-crfsuite format -
each sentence should be converted to a list of dicts. This is a very
simple baseline; you certainly can do better.

.. code:: ipython3

    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],        
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
            
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
                    
        return features
    
    
    def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]
    
    def sent2labels(sent):
        return [label for token, postag, label in sent]
    
    def sent2tokens(sent):
        return [token for token, postag, label in sent]
    
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

This is how features extracted from a single token look like:

.. code:: ipython3

    X_train[0][1]




.. parsed-literal::

    {'+1:postag': 'NP',
     '+1:postag[:2]': 'NP',
     '+1:word.istitle()': True,
     '+1:word.isupper()': False,
     '+1:word.lower()': 'australia',
     '-1:postag': 'NP',
     '-1:postag[:2]': 'NP',
     '-1:word.istitle()': True,
     '-1:word.isupper()': False,
     '-1:word.lower()': 'melbourne',
     'bias': 1.0,
     'postag': 'Fpa',
     'postag[:2]': 'Fp',
     'word.isdigit()': False,
     'word.istitle()': False,
     'word.isupper()': False,
     'word.lower()': '(',
     'word[-3:]': '('}



3. Train a CRF model
--------------------

Once we have features in a right format we can train a linear-chain CRF
(Conditional Random Fields) model using sklearn_crfsuite.CRF:

.. code:: ipython3

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, 
        c2=0.1, 
        max_iterations=20,
        all_possible_transitions=False,
    )
    crf.fit(X_train, y_train);

4. Inspect model weights
------------------------

CRFsuite CRF models use two kinds of features: state features and
transition features. Let’s check their weights using
eli5.explain_weights:

.. code:: ipython3

    eli5.show_weights(crf, top=30)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
            
    
    
    <table class="docutils" style="margin-bottom: 0.5em;">
        <thead>
            <tr>
                <td>From \ To</td>
                
                    <th>O</th>
                
                    <th>B-LOC</th>
                
                    <th>I-LOC</th>
                
                    <th>B-MISC</th>
                
                    <th>I-MISC</th>
                
                    <th>B-ORG</th>
                
                    <th>I-ORG</th>
                
                    <th>B-PER</th>
                
                    <th>I-PER</th>
                
            </tr>
        </thead>
        <tbody>
            
                
                    <tr>
                        <th>O</th>
                        
                            <td style="background-color: hsl(120, 100.00%, 85.24%)" title="O &rArr; O">
                                3.281
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 88.83%)" title="O &rArr; B-LOC">
                                2.204
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="O &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 89.19%)" title="O &rArr; B-MISC">
                                2.101
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="O &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 84.65%)" title="O &rArr; B-ORG">
                                3.468
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="O &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 88.40%)" title="O &rArr; B-PER">
                                2.325
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="O &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-LOC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.50%)" title="B-LOC &rArr; O">
                                -0.259
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.73%)" title="B-LOC &rArr; B-LOC">
                                -0.098
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 82.87%)" title="B-LOC &rArr; I-LOC">
                                4.058
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.83%)" title="B-LOC &rArr; B-PER">
                                -0.212
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-LOC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.12%)" title="I-LOC &rArr; O">
                                -0.173
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.46%)" title="I-LOC &rArr; B-LOC">
                                -0.609
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 84.75%)" title="I-LOC &rArr; I-LOC">
                                3.436
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-MISC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.13%)" title="B-MISC &rArr; O">
                                -0.673
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.98%)" title="B-MISC &rArr; B-LOC">
                                -0.341
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 82.84%)" title="B-MISC &rArr; I-MISC">
                                4.069
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.18%)" title="B-MISC &rArr; B-ORG">
                                -0.308
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.04%)" title="B-MISC &rArr; B-PER">
                                -0.331
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-MISC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.49%)" title="I-MISC &rArr; O">
                                -0.803
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.58%)" title="I-MISC &rArr; B-LOC">
                                -0.998
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.94%)" title="I-MISC &rArr; B-MISC">
                                -0.519
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 80.24%)" title="I-MISC &rArr; I-MISC">
                                4.977
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.42%)" title="I-MISC &rArr; B-ORG">
                                -0.817
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.45%)" title="I-MISC &rArr; B-PER">
                                -0.611
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-ORG</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.75%)" title="B-ORG &rArr; O">
                                -0.096
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.62%)" title="B-ORG &rArr; B-LOC">
                                -0.242
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.66%)" title="B-ORG &rArr; B-MISC">
                                -0.57
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.52%)" title="B-ORG &rArr; B-ORG">
                                -1.012
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 80.90%)" title="B-ORG &rArr; I-ORG">
                                4.739
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.19%)" title="B-ORG &rArr; B-PER">
                                -0.306
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-ORG</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.99%)" title="I-ORG &rArr; O">
                                -0.339
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 90.46%)" title="I-ORG &rArr; B-LOC">
                                -1.758
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.31%)" title="I-ORG &rArr; B-MISC">
                                -0.841
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 91.94%)" title="I-ORG &rArr; B-ORG">
                                -1.382
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 80.00%)" title="I-ORG &rArr; I-ORG">
                                5.062
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.20%)" title="I-ORG &rArr; B-PER">
                                -0.472
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-PER</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.62%)" title="B-PER &rArr; O">
                                -0.4
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.26%)" title="B-PER &rArr; B-LOC">
                                -0.851
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.51%)" title="B-PER &rArr; B-ORG">
                                -1.013
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.86%)" title="B-PER &rArr; B-PER">
                                -0.937
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 82.08%)" title="B-PER &rArr; I-PER">
                                4.329
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-PER</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.11%)" title="I-PER &rArr; O">
                                -0.676
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.21%)" title="I-PER &rArr; B-LOC">
                                -0.47
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.20%)" title="I-PER &rArr; B-PER">
                                -0.659
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 83.78%)" title="I-PER &rArr; I-PER">
                                3.754
                            </td>
                        
                    </tr>
                
            
        </tbody>
    </table>
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
            <table class="eli5-weights-wrapper" style="border-collapse: collapse; border: none; margin-bottom: 1.5em;">
                <tr>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=O
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-LOC
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-LOC
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-MISC
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-MISC
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-ORG
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-ORG
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-PER
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-PER
        
    </b>
    
    top features
                        </td>
                    
                </tr>
                <tr>
                    
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 84.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +4.416
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.60%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.116
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            BOS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.401
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.297
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.297
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.297
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.297
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.51%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.124
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.51%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.124
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.96%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.984
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            EOS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 91.36%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.859
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():y
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 91.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.684
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:RG
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 91.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.684
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:RG
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.610
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():-
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.610
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fg
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.610
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:-
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.610
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Fg
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.582
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.582
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:.
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.582
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():.
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.01%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.372
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:y
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.187
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:CS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.187
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:CS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:(
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Fpa
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():(
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 93.83%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 16444 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 90.57%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 3771 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 90.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.106
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 90.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.106
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 85.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.723
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -6.166
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 89.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.530
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.224
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():en
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.906
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:rid
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.905
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():madrid
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.88%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.646
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():españa
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.640
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:ona
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.11%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.595
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:aña
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.11%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.595
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.48%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.515
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():parís
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.514
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:rís
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.93%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.424
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():barcelona
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.420
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fg
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.420
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():-
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.420
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fg
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.99%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.413
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.390
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.11%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.389
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fpa
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.11%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.389
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():(
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.11%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.388
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():san
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.385
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NC
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 97.13%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 2282 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 97.11%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 413 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.11%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.389
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.11%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.389
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.11%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.389
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.02%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.406
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.88%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.646
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:ión
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.38%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.759
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():del
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.14%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.818
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 94.46%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.986
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 94.46%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.986
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.354
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 94.86%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.886
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.80%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.664
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.582
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.578
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.529
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():san
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.444
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.441
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.335
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():la
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.262
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.262
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.97%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.235
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:la
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.01%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.228
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:iro
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.03%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.226
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:oja
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.218
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:del
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.215
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():del
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.213
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.213
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.15%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.205
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():nueva
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 98.15%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 1665 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 98.15%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 258 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.15%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.206
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.15%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.206
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.213
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.213
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.219
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():en
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.222
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.97%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.235
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.36%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.342
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.23%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.366
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.23%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.366
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.392
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 91.92%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.690
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            BOS
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 91.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.770
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.693
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.06%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.606
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.06%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.606
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.06%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.606
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.06%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.606
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.538
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.508
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.508
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.508
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.484
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.484
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.479
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.76%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.457
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.76%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.457
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.400
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():liga
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.399
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:iga
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.22%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.367
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():la
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.29%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.354
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.29%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.354
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.332
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():del
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.286
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.286
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.284
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.284
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:NC
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 97.68%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 2284 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 97.54%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 314 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.308
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            BOS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.377
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 94.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.908
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 94.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.908
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 94.04%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.094
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 93.04%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.364
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.675
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.597
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.597
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.lower():&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.597
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.369
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.369
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.46%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.324
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():liga
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.318
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.304
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.303
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.261
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.261
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.258
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():copa
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.240
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():campeones
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.97%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.235
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:000
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.234
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.234
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.01%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.229
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():2000
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 98.01%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 3675 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 97.97%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 573 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.97%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.235
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            EOS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.264
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():y
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.265
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():y
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.265
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.74%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.274
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.306
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.306
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.48%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.320
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.48%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.320
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.370
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.90%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.641
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 88.80%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.695
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():efe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.31%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.519
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.64%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.084
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:EFE
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.74%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.174
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():gobierno
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.86%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.142
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.33%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.018
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():del
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.958
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:rno
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.671
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:PP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.671
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():pp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.667
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():al
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.30%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.555
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():el
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.499
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:eal
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.99%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.413
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():real
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.393
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():ayuntamiento
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.391
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.391
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:AQ
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 3518 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 96.90%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 619 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 96.90%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.430
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.90%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.430
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.80%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.450
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.455
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.455
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.55%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.500
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.89%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.642
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():los
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.80%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.664
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.61%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.707
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.44%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.746
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():en
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.44%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.747
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 94.02%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.100
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.31%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.289
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.31%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.289
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 92.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.499
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.64%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.200
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.539
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():real
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.511
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:rid
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.82%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.446
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.433
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.428
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.428
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.399
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():madrid
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.22%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.368
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:la
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.23%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.365
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():consejo
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.25%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.363
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.31%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.352
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():comisión
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.336
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.336
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.332
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Fpa
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.332
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.lower():(
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.53%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.311
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():estados
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.306
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():unidos
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 97.56%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 3473 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 97.57%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 703 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.304
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.304
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.306
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():a
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.384
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.384
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.391
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.507
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.507
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.535
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.36%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.540
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.195
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 91.89%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.698
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.71%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.683
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.601
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.589
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.589
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.14%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.589
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.565
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():a
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.46%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.520
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:osé
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.503
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():josé
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.476
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.472
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.472
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.452
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.452
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.452
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fc
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 96.79%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 4117 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 96.69%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 351 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 96.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.472
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():en
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.475
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.475
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.475
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.543
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():la
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.572
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.693
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.712
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.712
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.30%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.778
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():del
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.14%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.818
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.14%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.818
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 94.71%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.923
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():la
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.319
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.319
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:DA
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 88.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.742
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.48%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.736
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.82%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.660
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():josé
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.598
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.598
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.51%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.510
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.62%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.487
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():juan
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.419
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():maría
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.99%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.413
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.34%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.345
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():luis
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.319
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():manuel
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.51%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.315
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.51%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.315
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.309
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():carlos
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 97.54%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 3903 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 97.59%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 365 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.301
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.301
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.58%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.301
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:ión
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.305
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.305
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.305
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.305
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.305
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.lower():que
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.46%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.324
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():el
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.377
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.377
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Z
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.396
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.89%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.433
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.89%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.433
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.485
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 92.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.431
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                    
                </tr>
            </table>
        
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Transition features make sense: at least model learned that I-ENITITY
must follow B-ENTITY. It also learned that some transitions are
unlikely, e.g. it is not common in this dataset to have a location right
after an organization name (I-ORG -> B-LOC has a large negative weight).

Features don’t use gazetteers, so model had to remember some geographic
names from the training data, e.g. that España is a location.

If we regularize CRF more, we can expect that only features which are
generic will remain, and memoized tokens will go. With L1 regularization
(c1 parameter) coefficients of most features should be driven to zero.
Let’s check what effect does regularization have on CRF weights:

.. code:: ipython3

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=200,
        c2=0.1,
        max_iterations=20,
        all_possible_transitions=False,
    )
    crf.fit(X_train, y_train)
    eli5.show_weights(crf, top=30)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
            
    
    
    <table class="docutils" style="margin-bottom: 0.5em;">
        <thead>
            <tr>
                <td>From \ To</td>
                
                    <th>O</th>
                
                    <th>B-LOC</th>
                
                    <th>I-LOC</th>
                
                    <th>B-MISC</th>
                
                    <th>I-MISC</th>
                
                    <th>B-ORG</th>
                
                    <th>I-ORG</th>
                
                    <th>B-PER</th>
                
                    <th>I-PER</th>
                
            </tr>
        </thead>
        <tbody>
            
                
                    <tr>
                        <th>O</th>
                        
                            <td style="background-color: hsl(120, 100.00%, 84.74%)" title="O &rArr; O">
                                3.232
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 90.03%)" title="O &rArr; B-LOC">
                                1.76
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="O &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 89.00%)" title="O &rArr; B-MISC">
                                2.026
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="O &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 86.89%)" title="O &rArr; B-ORG">
                                2.603
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="O &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 90.70%)" title="O &rArr; B-PER">
                                1.593
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="O &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-LOC</th>
                        
                            <td style="background-color: hsl(120, 100.00%, 99.36%)" title="B-LOC &rArr; O">
                                0.035
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; B-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 86.30%)" title="B-LOC &rArr; I-LOC">
                                2.773
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-LOC &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-LOC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 99.56%)" title="I-LOC &rArr; O">
                                -0.02
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; B-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 85.19%)" title="I-LOC &rArr; I-LOC">
                                3.099
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-LOC &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-MISC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.58%)" title="B-MISC &rArr; O">
                                -0.382
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; B-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 80.00%)" title="B-MISC &rArr; I-MISC">
                                4.758
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-MISC &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-MISC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.42%)" title="I-MISC &rArr; O">
                                -0.256
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; B-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 81.81%)" title="I-MISC &rArr; I-MISC">
                                4.155
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-MISC &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-ORG</th>
                        
                            <td style="background-color: hsl(120, 100.00%, 98.13%)" title="B-ORG &rArr; O">
                                0.161
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; B-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 84.38%)" title="B-ORG &rArr; I-ORG">
                                3.344
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-ORG &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-ORG</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.42%)" title="I-ORG &rArr; O">
                                -0.126
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.84%)" title="I-ORG &rArr; B-LOC">
                                -0.081
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 82.14%)" title="I-ORG &rArr; I-ORG">
                                4.048
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-ORG &rArr; I-PER">
                                0.0
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-PER</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; O">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; B-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="B-PER &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 84.03%)" title="B-PER &rArr; I-PER">
                                3.449
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-PER</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.81%)" title="I-PER &rArr; O">
                                -0.085
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; B-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; I-LOC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; B-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; I-MISC">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; B-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; I-ORG">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 100.00%)" title="I-PER &rArr; B-PER">
                                0.0
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 88.15%)" title="I-PER &rArr; I-PER">
                                2.254
                            </td>
                        
                    </tr>
                
            
        </tbody>
    </table>
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
            <table class="eli5-weights-wrapper" style="border-collapse: collapse; border: none; margin-bottom: 1.5em;">
                <tr>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=O
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-LOC
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-LOC
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-MISC
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-MISC
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-ORG
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-ORG
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-PER
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-PER
        
    </b>
    
    top features
                        </td>
                    
                </tr>
                <tr>
                    
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 85.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.363
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            BOS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.31%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.842
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.47%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.478
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.665
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.439
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.439
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.400
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.400
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.400
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.400
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.391
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.391
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:CC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.99%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.365
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            EOS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.363
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.363
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.315
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.315
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.36%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.302
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.isupper()
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 97.36%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 15 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 97.91%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 14 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.216
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.216
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.334
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.334
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.417
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.417
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.547
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.547
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.547
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.600
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 85.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.552
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.446
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 92.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.417
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():en
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.183
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.25%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.498
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.38%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.lower():,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.38%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.38%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.80%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.098
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.081
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fpa
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.081
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():(
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.080
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.080
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.056
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.056
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.58%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.022
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.58%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.022
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.62%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.019
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            BOS
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.008
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.028
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():del
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.572
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 94.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.788
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.70%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.248
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.237
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.03%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.199
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.190
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.190
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.15%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.060
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.15%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.060
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.040
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.istitle()
        </td>
        
    </tr>
            
            
    
            
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.349
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.22%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.053
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.22%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.053
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.47%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.030
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.009
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.009
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.15%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.060
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.22%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.172
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 96.61%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.432
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.32%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.158
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.32%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.158
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.146
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.146
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.lower():&quot;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.146
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Fe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.47%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.030
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.47%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.030
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.89%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.087
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.094
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.119
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.61%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.120
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.61%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.121
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.211
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 91.22%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.681
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.20%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.507
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():del
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.350
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.350
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.48%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.282
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():efe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.234
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:EFE
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.195
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():(
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.195
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fpa
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.192
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.178
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Fpt
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.178
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.lower():)
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.173
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.136
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():el
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.110
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.110
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:NC
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.004
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.023
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.041
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.041
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:NC
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.210
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.515
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 92.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.318
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.762
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.185
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.185
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.185
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.058
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():de
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.32%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.043
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.58%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.267
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.06%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.536
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 94.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.800
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.44%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.463
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.44%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.463
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.44%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.463
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.148
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.58%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.125
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.095
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.007
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:AQ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.007
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:AQ
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.039
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.058
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.058
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.12%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.063
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.067
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.32%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.159
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.32%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.159
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.60%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.263
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:DA
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.60%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.263
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:DA
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 89.64%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.127
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.18%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.331
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.016
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.016
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:word.lower():,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.016
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:Fc
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.089
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.089
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            +1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.648
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                    
                </tr>
            </table>
        
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




As you can see, memoized tokens are mostly gone and model now relies on
word shapes and POS tags. There is only a few non-zero features
remaining. In our example the change probably made the quality worse,
but that’s a separate question.

Let’s focus on transition weights. We can expect that O -> I-ENTIRY
transitions to have large negative weights because they are impossible.
But these transitions have zero weights, not negative weights, both in
heavily regularized model and in our initial model. Something is going
on here.

The reason they are zero is that crfsuite haven’t seen these transitions
in training data, and assumed there is no need to learn weights for
them, to save some computation time. This is the default behavior, but
it is possible to turn it off using sklearn_crfsuite.CRF
``all_possible_transitions`` option. Let’s check how does it affect the
result:

.. code:: ipython3

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, 
        c2=0.1, 
        max_iterations=20, 
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train);

.. code:: ipython3

    eli5.show_weights(crf, top=5, show=['transition_features'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
            
    
    
    <table class="docutils" style="margin-bottom: 0.5em;">
        <thead>
            <tr>
                <td>From \ To</td>
                
                    <th>O</th>
                
                    <th>B-LOC</th>
                
                    <th>I-LOC</th>
                
                    <th>B-MISC</th>
                
                    <th>I-MISC</th>
                
                    <th>B-ORG</th>
                
                    <th>I-ORG</th>
                
                    <th>B-PER</th>
                
                    <th>I-PER</th>
                
            </tr>
        </thead>
        <tbody>
            
                
                    <tr>
                        <th>O</th>
                        
                            <td style="background-color: hsl(120, 100.00%, 88.77%)" title="O &rArr; O">
                                2.732
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 93.63%)" title="O &rArr; B-LOC">
                                1.217
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 83.65%)" title="O &rArr; I-LOC">
                                -4.675
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 92.57%)" title="O &rArr; B-MISC">
                                1.515
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 81.02%)" title="O &rArr; I-MISC">
                                -5.785
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 93.11%)" title="O &rArr; B-ORG">
                                1.36
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 80.10%)" title="O &rArr; I-ORG">
                                -6.19
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 94.57%)" title="O &rArr; B-PER">
                                0.968
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 80.00%)" title="O &rArr; I-PER">
                                -6.236
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-LOC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.04%)" title="B-LOC &rArr; O">
                                -0.226
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.96%)" title="B-LOC &rArr; B-LOC">
                                -0.091
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 86.98%)" title="B-LOC &rArr; I-LOC">
                                3.378
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.91%)" title="B-LOC &rArr; B-MISC">
                                -0.433
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.19%)" title="B-LOC &rArr; I-MISC">
                                -1.065
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.00%)" title="B-LOC &rArr; B-ORG">
                                -0.861
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 91.67%)" title="B-LOC &rArr; I-ORG">
                                -1.783
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.64%)" title="B-LOC &rArr; B-PER">
                                -0.295
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 92.39%)" title="B-LOC &rArr; I-PER">
                                -1.57
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-LOC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.30%)" title="I-LOC &rArr; O">
                                -0.184
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.18%)" title="I-LOC &rArr; B-LOC">
                                -0.585
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 89.74%)" title="I-LOC &rArr; I-LOC">
                                2.404
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.75%)" title="I-LOC &rArr; B-MISC">
                                -0.276
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.65%)" title="I-LOC &rArr; I-MISC">
                                -0.485
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.20%)" title="I-LOC &rArr; B-ORG">
                                -0.582
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.46%)" title="I-LOC &rArr; I-ORG">
                                -0.749
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.87%)" title="I-LOC &rArr; B-PER">
                                -0.442
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.91%)" title="I-LOC &rArr; I-PER">
                                -0.647
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-MISC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.61%)" title="B-MISC &rArr; O">
                                -0.714
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.32%)" title="B-MISC &rArr; B-LOC">
                                -0.353
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.40%)" title="B-MISC &rArr; I-LOC">
                                -0.539
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.73%)" title="B-MISC &rArr; B-MISC">
                                -0.278
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 86.62%)" title="B-MISC &rArr; I-MISC">
                                3.512
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.01%)" title="B-MISC &rArr; B-ORG">
                                -0.412
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.26%)" title="B-MISC &rArr; I-ORG">
                                -1.047
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.41%)" title="B-MISC &rArr; B-PER">
                                -0.336
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.86%)" title="B-MISC &rArr; I-PER">
                                -0.895
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-MISC</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.68%)" title="I-MISC &rArr; O">
                                -0.697
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.06%)" title="I-MISC &rArr; B-LOC">
                                -0.846
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.17%)" title="I-MISC &rArr; I-LOC">
                                -0.587
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.63%)" title="I-MISC &rArr; B-MISC">
                                -0.297
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 84.70%)" title="I-MISC &rArr; I-MISC">
                                4.252
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.09%)" title="I-MISC &rArr; B-ORG">
                                -0.84
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.67%)" title="I-MISC &rArr; I-ORG">
                                -1.206
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.47%)" title="I-MISC &rArr; B-PER">
                                -0.523
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.44%)" title="I-MISC &rArr; I-PER">
                                -1.001
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-ORG</th>
                        
                            <td style="background-color: hsl(120, 100.00%, 96.98%)" title="B-ORG &rArr; O">
                                0.419
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.28%)" title="B-ORG &rArr; B-LOC">
                                -0.187
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.16%)" title="B-ORG &rArr; I-LOC">
                                -1.074
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.27%)" title="B-ORG &rArr; B-MISC">
                                -0.567
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 92.26%)" title="B-ORG &rArr; I-MISC">
                                -1.607
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.95%)" title="B-ORG &rArr; B-ORG">
                                -1.13
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 81.94%)" title="B-ORG &rArr; I-ORG">
                                5.392
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.06%)" title="B-ORG &rArr; B-PER">
                                -0.223
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 90.59%)" title="B-ORG &rArr; I-PER">
                                -2.122
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-ORG</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.77%)" title="I-ORG &rArr; O">
                                -0.117
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 91.90%)" title="I-ORG &rArr; B-LOC">
                                -1.715
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.99%)" title="I-ORG &rArr; I-LOC">
                                -0.863
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.97%)" title="I-ORG &rArr; B-MISC">
                                -0.631
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.61%)" title="I-ORG &rArr; I-MISC">
                                -1.221
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 92.82%)" title="I-ORG &rArr; B-ORG">
                                -1.442
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 82.53%)" title="I-ORG &rArr; I-ORG">
                                5.141
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.09%)" title="I-ORG &rArr; B-PER">
                                -0.397
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 91.27%)" title="I-ORG &rArr; I-PER">
                                -1.908
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-PER</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.69%)" title="B-PER &rArr; O">
                                -0.127
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.22%)" title="B-PER &rArr; B-LOC">
                                -0.806
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.11%)" title="B-PER &rArr; I-LOC">
                                -0.834
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.49%)" title="B-PER &rArr; B-MISC">
                                -0.52
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.59%)" title="B-PER &rArr; I-MISC">
                                -1.228
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.11%)" title="B-PER &rArr; B-ORG">
                                -1.089
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 90.74%)" title="B-PER &rArr; I-ORG">
                                -2.076
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.41%)" title="B-PER &rArr; B-PER">
                                -1.01
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 85.24%)" title="B-PER &rArr; I-PER">
                                4.04
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-PER</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.39%)" title="I-PER &rArr; O">
                                -0.766
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 97.94%)" title="I-PER &rArr; B-LOC">
                                -0.242
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.80%)" title="I-PER &rArr; I-LOC">
                                -0.67
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 96.98%)" title="I-PER &rArr; B-MISC">
                                -0.418
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.02%)" title="I-PER &rArr; I-MISC">
                                -0.856
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 94.83%)" title="I-PER &rArr; B-ORG">
                                -0.903
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 92.72%)" title="I-PER &rArr; I-ORG">
                                -1.472
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 95.71%)" title="I-PER &rArr; B-PER">
                                -0.692
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 88.27%)" title="I-PER &rArr; I-PER">
                                2.909
                            </td>
                        
                    </tr>
                
            
        </tbody>
    </table>
        
    
        
    
        
    
        
    
    
    




With ``all_possible_transitions=True`` CRF learned large negative
weights for impossible transitions like O -> I-ORG.

5. Customization
----------------

The table above is large and kind of hard to inspect; eli5 provides
several options to look only at a part of features. You can check only a
subset of labels:

.. code:: ipython3

    eli5.show_weights(crf, top=10, targets=['O', 'B-ORG', 'I-ORG'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
            
    
    
    <table class="docutils" style="margin-bottom: 0.5em;">
        <thead>
            <tr>
                <td>From \ To</td>
                
                    <th>O</th>
                
                    <th>B-ORG</th>
                
                    <th>I-ORG</th>
                
            </tr>
        </thead>
        <tbody>
            
                
                    <tr>
                        <th>O</th>
                        
                            <td style="background-color: hsl(120, 100.00%, 88.72%)" title="O &rArr; O">
                                2.732
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 93.08%)" title="O &rArr; B-ORG">
                                1.36
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 80.00%)" title="O &rArr; I-ORG">
                                -6.19
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>B-ORG</th>
                        
                            <td style="background-color: hsl(120, 100.00%, 96.96%)" title="B-ORG &rArr; O">
                                0.419
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 93.92%)" title="B-ORG &rArr; B-ORG">
                                -1.13
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 81.84%)" title="B-ORG &rArr; I-ORG">
                                5.392
                            </td>
                        
                    </tr>
                
                    <tr>
                        <th>I-ORG</th>
                        
                            <td style="background-color: hsl(0, 100.00%, 98.76%)" title="I-ORG &rArr; O">
                                -0.117
                            </td>
                        
                            <td style="background-color: hsl(0, 100.00%, 92.79%)" title="I-ORG &rArr; B-ORG">
                                -1.442
                            </td>
                        
                            <td style="background-color: hsl(120, 100.00%, 82.44%)" title="I-ORG &rArr; I-ORG">
                                5.141
                            </td>
                        
                    </tr>
                
            
        </tbody>
    </table>
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
            <table class="eli5-weights-wrapper" style="border-collapse: collapse; border: none; margin-bottom: 1.5em;">
                <tr>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=O
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=B-ORG
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=I-ORG
        
    </b>
    
    top features
                        </td>
                    
                </tr>
                <tr>
                    
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 84.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +4.931
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            BOS
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.754
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.62%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.539
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.328
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.328
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():,
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.328
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:Fc
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.328
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:Fc
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 90.77%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 15039 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 91.16%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 3905 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 91.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.187
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:NP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 87.27%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.685
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -7.025
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 88.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.041
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.952
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():efe
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.14%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.851
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:EFE
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.93%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.278
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.lower():gobierno
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.033
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word[-3:]:rno
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.005
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.864
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():del
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 95.39%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 3524 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 95.47%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 621 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 95.47%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.842
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():en
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.48%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.416
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.48%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.416
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:SP
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                            <td style="padding: 0px; border: 1px solid black; vertical-align: top;">
                                
                                    
                                        
                                        
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; width: 100%;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 94.33%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.159
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():de
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.92%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.993
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.637
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag[:2]:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.637
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:postag:SP
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.55%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.570
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.lower():real
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.547
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 96.65%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 3517 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 96.95%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 676 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 96.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.480
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag:VMI
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.82%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.508
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            postag[:2]:VM
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.71%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.533
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            -1:word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.89%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.290
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bias
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                    
                </tr>
            </table>
        
    
        
            
    
    
        
            
    
    
        
            
    
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Another option is to check only some of the features - it helps to check
if a feature function works as intended. For example, let’s check how
word shape features are used by model using ``feature_re`` argument and
hide transition table:

.. code:: ipython3

    eli5.show_weights(crf, top=10, feature_re='^word\.is', 
                      horizontal_layout=False, show=['targets'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
            
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=O
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 87.27%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.685
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -7.025
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=B-LOC
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 90.58%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.397
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.99%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.099
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.152
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=I-LOC
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 97.03%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.460
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.018
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.345
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=B-MISC
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 91.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.017
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.603
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.76%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.012
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=I-MISC
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 97.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.271
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.072
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 98.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.106
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=B-ORG
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 88.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.041
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.005
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.42%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.044
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=I-ORG
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 96.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.547
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.014
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.012
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=B-PER
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 92.42%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.757
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.050
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.82%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.123
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=I-PER
        
    </b>
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                        Weight<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 94.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.976
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.istitle()
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.38%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.193
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isupper()
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.106
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            word.isdigit()
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
        
    
        
    
        
    
    
    




Looks fine - UPPERCASE and Titlecase words are likely to be entities of
some kind.

6. Formatting in console
------------------------

It is also possible to format the result as text (could be useful in
console):

.. code:: ipython3

    expl = eli5.explain_weights(crf, top=5, targets=['O', 'B-LOC', 'I-LOC'])
    print(eli5.format_as_text(expl))


.. parsed-literal::

    Explained as: CRF
    
    Transition features:
                O    B-LOC    I-LOC
    -----  ------  -------  -------
    O       2.732    1.217   -4.675
    B-LOC  -0.226   -0.091    3.378
    I-LOC  -0.184   -0.585    2.404
    
    y='O' top features
    Weight  Feature       
    ------  --------------
    +4.931  BOS           
    +3.754  postag[:2]:Fp 
    +3.539  bias          
    … 15043 more positive …
    … 3906 more negative …
    -3.685  word.isupper()
    -7.025  word.istitle()
    
    y='B-LOC' top features
    Weight  Feature           
    ------  ------------------
    +2.397  word.istitle()    
    +2.147  -1:word.lower():en
      … 2284 more positive …  
      … 433 more negative …   
    -1.080  postag[:2]:SP     
    -1.080  postag:SP         
    -1.273  -1:word.istitle() 
    
    y='I-LOC' top features
    Weight  Feature           
    ------  ------------------
    +0.882  -1:word.lower():de
    +0.780  -1:word.istitle() 
    +0.718  word[-3:]:de      
    +0.711  word.lower():de   
      … 1684 more positive …  
      … 268 more negative …   
    -1.965  BOS               
    


