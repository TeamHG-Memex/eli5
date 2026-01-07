
Debugging scikit-learn text classification pipeline
===================================================

scikit-learn docs provide a nice text classification
`tutorial <http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html>`__.
Make sure to read it first. We’ll be doing something similar to it,
while taking more detailed look at classifier weights and predictions.

1. Baseline model
-----------------

First, we need some data. Let’s load 20 Newsgroups data, keeping only 4
categories:

.. code:: ipython3

    from sklearn.datasets import fetch_20newsgroups
    
    categories = ['alt.atheism', 'soc.religion.christian', 
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42
    )
    twenty_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42
    )

A basic text processing pipeline - bag of words features and Logistic
Regression as a classifier:

.. code:: ipython3

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    
    vec = CountVectorizer()
    clf = LogisticRegressionCV()
    pipe = make_pipeline(vec, clf)
    pipe.fit(twenty_train.data, twenty_train.target);

We’re using LogisticRegressionCV here to adjust regularization parameter
C automatically. It allows to compare different vectorizers - optimal C
value could be different for different input features (e.g. for bigrams
or for character-level input). An alternative would be to use
GridSearchCV or RandomizedSearchCV.

Let’s check quality of this pipeline:

.. code:: ipython3

    from sklearn import metrics
    
    def print_report(pipe):
        y_test = twenty_test.target
        y_pred = pipe.predict(twenty_test.data)
        report = metrics.classification_report(y_test, y_pred, 
            target_names=twenty_test.target_names)
        print(report)
        print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))
        
    print_report(pipe)


.. parsed-literal::

                            precision    recall  f1-score   support
    
               alt.atheism       0.93      0.80      0.86       319
             comp.graphics       0.87      0.96      0.91       389
                   sci.med       0.94      0.81      0.87       396
    soc.religion.christian       0.85      0.98      0.91       398
    
               avg / total       0.90      0.89      0.89      1502
    
    accuracy: 0.891


Not bad. We can try other classifiers and preprocessing methods, but
let’s check first what the model learned using :func:`eli5.show_weights`
function:

.. code:: ipython3

    import eli5
    eli5.show_weights(clf, top=10)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
            <table class="eli5-weights-wrapper" style="border-collapse: collapse; border: none; margin-bottom: 1.5em;">
                <tr>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=0
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=1
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=2
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=3
        
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
            
                <tr style="background-color: hsl(120, 100.00%, 93.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.991
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x21167
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.88%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.925
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x19218
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.834
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x5714
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.813
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x23677
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.697
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x15511
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.696
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x26415
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.58%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.617
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x6440
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.64%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.594
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x26412
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 94.64%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 10174 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 94.42%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 25605 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 94.42%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.686
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x28473
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -10.453
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
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
            
                <tr style="background-color: hsl(120, 100.00%, 94.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.702
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x15699
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.62%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.825
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x17366
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.70%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.798
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x14281
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.786
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x30117
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.779
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x14277
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.773
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x17356
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.90%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.729
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x24267
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.724
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x7874
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.702
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x2148
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 96.98%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 11710 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 95.16%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 24069 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 95.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.379
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
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
            
                <tr style="background-color: hsl(120, 100.00%, 93.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.016
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x25234
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.82%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.951
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x12026
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.26%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.758
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x17854
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.697
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x11729
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.655
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x32847
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.522
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x22379
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.82%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.518
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x16328
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 94.82%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 15007 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 94.24%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 20772 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 94.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.764
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x15521
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.34%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.171
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x15699
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.04%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.013
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
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
            
                <tr style="background-color: hsl(120, 100.00%, 95.62%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.193
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x28473
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.030
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x8609
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.021
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x8559
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.946
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x8798
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.899
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x8544
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.70%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.797
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x8553
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 96.70%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 11122 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 96.54%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 24657 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 96.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.852
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x15699
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.42%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.894
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x25663
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.181
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x23122
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.243
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x16881
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                    
                </tr>
            </table>
        
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




The table above doesn’t make any sense; the problem is that eli5 was not
able to get feature and class names from the classifier object alone. We
can provide feature and target names explicitly:

.. code:: ipython3

    # eli5.show_weights(clf, 
    #                   feature_names=vec.get_feature_names(), 
    #                   target_names=twenty_test.target_names)

The code above works, but a better way is to provide vectorizer instead
and let eli5 figure out the details automatically:

.. code:: ipython3

    eli5.show_weights(clf, vec=vec, top=10, 
                      target_names=twenty_test.target_names)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
            <table class="eli5-weights-wrapper" style="border-collapse: collapse; border: none; margin-bottom: 1.5em;">
                <tr>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=alt.atheism
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=comp.graphics
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=sci.med
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=soc.religion.christian
        
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
            
                <tr style="background-color: hsl(120, 100.00%, 93.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.991
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            mathew
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.88%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.925
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            keith
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.834
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            atheism
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.813
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            okcforum
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.697
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            go
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.696
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            psuvm
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.58%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.617
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            believing
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.64%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.594
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            psu
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 94.64%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 10174 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 94.42%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 25605 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 94.42%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.686
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            rutgers
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -10.453
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
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
            
                <tr style="background-color: hsl(120, 100.00%, 94.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.702
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            graphics
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.62%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.825
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            images
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.70%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.798
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            files
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.786
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            software
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.779
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            file
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.773
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            image
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.90%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.729
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            package
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.724
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            card
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.702
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            3d
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 96.98%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 11710 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 95.16%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 24069 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 95.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.379
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
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
            
                <tr style="background-color: hsl(120, 100.00%, 93.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.016
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            pitt
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 93.82%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.951
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            doctor
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.26%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.758
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            information
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.697
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            disease
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.655
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            treatment
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.522
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            msg
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.82%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.518
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            health
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 94.82%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 15007 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 94.24%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 20772 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 94.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.764
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            god
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.34%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.171
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            graphics
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.04%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.013
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
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
            
                <tr style="background-color: hsl(120, 100.00%, 95.62%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.193
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            rutgers
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.030
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            church
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.021
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            christians
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.946
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            clh
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.899
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            christ
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.70%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.797
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            christian
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 96.70%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 11122 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 96.54%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 24657 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 96.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.852
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            graphics
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.42%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.894
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            posting
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.181
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            nntp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 95.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.243
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            host
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                    
                </tr>
            </table>
        
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




This starts to make more sense. Columns are target classes. In each
column there are features and their weights. Intercept (bias) feature is
shown as ``<BIAS>`` in the same table. We can inspect features and
weights because we’re using a bag-of-words vectorizer and a linear
classifier (so there is a direct mapping between individual words and
classifier coefficients). For other classifiers features can be harder
to inspect.

Some features look good, but some don’t. It seems model learned some
names specific to a dataset (email parts, etc.) though, instead of
learning topic-specific words. Let’s check prediction results on an
example:

.. code:: ipython3

    eli5.show_prediction(clf, twenty_test.data[0], vec=vec, 
                         target_names=twenty_test.target_names)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=alt.atheism
        
    </b>
    
        
        (probability <b>0.000</b>, score <b>-8.709</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 94.29%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.743
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -10.453
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 89.53%); opacity: 0.83" title="0.183">from</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 64.57%); opacity: 0.97" title="1.045">brian</span><span style="opacity: 0.80">@</span><span style="background-color: hsl(0, 100.00%, 82.17%); opacity: 0.86" title="-0.392">ucsd</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 77.11%); opacity: 0.89" title="0.560">edu</span><span style="opacity: 0.80"> (</span><span style="background-color: hsl(120, 100.00%, 64.57%); opacity: 0.97" title="1.045">brian</span><span style="opacity: 0.80"> kantor)
    </span><span style="background-color: hsl(0, 100.00%, 84.43%); opacity: 0.85" title="-0.323">subject</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 77.99%); opacity: 0.89" title="0.529">re</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 91.26%); opacity: 0.82" title="-0.142">help</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.70%); opacity: 0.82" title="-0.109">for</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.93%); opacity: 0.82" title="-0.149">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.23%); opacity: 0.82" title="-0.142">stones</span><span style="opacity: 0.80"> ..............
    </span><span style="background-color: hsl(0, 100.00%, 89.22%); opacity: 0.83" title="-0.191">organization</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 98.62%); opacity: 0.80" title="-0.010">the</span><span style="opacity: 0.80"> avant-garde </span><span style="background-color: hsl(0, 100.00%, 96.94%); opacity: 0.81" title="-0.032">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.62%); opacity: 0.80" title="-0.010">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.15%); opacity: 0.88" title="0.524">now</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.13%); opacity: 0.80" title="-0.016">ltd</span><span style="opacity: 0.80">.
    </span><span style="background-color: hsl(120, 100.00%, 75.30%); opacity: 0.90" title="0.624">lines</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 61.98%); opacity: 0.99" title="1.156">12</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 88.50%); opacity: 0.83" title="0.209">nntp</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 79.97%); opacity: 0.87" title="0.463">posting</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 85.11%); opacity: 0.85" title="0.303">host</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 82.17%); opacity: 0.86" title="-0.392">ucsd</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 77.11%); opacity: 0.89" title="0.560">edu</span><span style="opacity: 0.80">
    
    </span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.038">as</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 87.21%); opacity: 0.84" title="-0.244">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.53%); opacity: 0.83" title="0.183">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.47%); opacity: 0.83" title="-0.160">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.001">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.16%); opacity: 0.90" title="-0.593">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.93%); opacity: 0.82" title="-0.149">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.23%); opacity: 0.82" title="-0.142">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 78.55%); opacity: 0.88" title="0.510">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.96%); opacity: 0.89" title="0.565">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(0, 100.00%, 79.72%); opacity: 0.88" title="-0.471">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 91.97%); opacity: 0.82" title="-0.125">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.61%); opacity: 0.81" title="-0.053">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.88%); opacity: 0.85" title="-0.281">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.08%); opacity: 0.82" title="0.146">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.03%); opacity: 0.81" title="0.046">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 67.63%); opacity: 0.95" title="-0.918">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.26%); opacity: 0.85" title="-0.328">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.69%); opacity: 0.80" title="0.021">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.97%); opacity: 0.80" title="-0.000">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.62%); opacity: 0.80" title="-0.010">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.23%); opacity: 0.86" title="-0.359">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 89.41%); opacity: 0.83" title="0.186">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.42%); opacity: 0.84" title="-0.266">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.66%); opacity: 0.81" title="0.089">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 84.07%); opacity: 0.85" title="0.333">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.42%); opacity: 0.84" title="-0.266">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.05%); opacity: 0.84" title="-0.248">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.36%); opacity: 0.84" title="-0.267">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.123">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.32%); opacity: 0.82" title="-0.096">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 71.67%); opacity: 0.92" title="0.759">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.16%); opacity: 0.90" title="-0.593">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.01%); opacity: 0.83" title="-0.171">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 84.07%); opacity: 0.85" title="0.333">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.42%); opacity: 0.84" title="-0.266">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.05%); opacity: 0.84" title="-0.248">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 86.36%); opacity: 0.84" title="-0.267">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.123">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.07%); opacity: 0.81" title="-0.063">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.96%); opacity: 0.80" title="-0.000">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 79.73%); opacity: 0.88" title="-0.471">when</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(120, 100.00%, 87.91%); opacity: 0.84" title="0.225">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.21%); opacity: 0.80" title="-0.028">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.62%); opacity: 0.80" title="-0.010">the</span><span style="opacity: 0.80"> x-</span><span style="background-color: hsl(0, 100.00%, 97.33%); opacity: 0.80" title="-0.026">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.29%); opacity: 0.82" title="0.097">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.87%); opacity: 0.82" title="-0.151">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.36%); opacity: 0.84" title="-0.267">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.71%); opacity: 0.84" title="0.257">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.61%); opacity: 0.81" title="-0.053">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.25%); opacity: 0.83" title="0.165">she</span><span style="opacity: 0.80">'d </span><span style="background-color: hsl(0, 100.00%, 82.93%); opacity: 0.86" title="-0.368">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.93%); opacity: 0.82" title="-0.149">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 91.23%); opacity: 0.82" title="-0.142">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.06%); opacity: 0.81" title="-0.081">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.58%); opacity: 0.81" title="-0.091">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 94.06%); opacity: 0.81" title="-0.081">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.62%); opacity: 0.80" title="-0.010">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.25%); opacity: 0.80" title="-0.014">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.71%); opacity: 0.83" title="-0.179">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.12%); opacity: 0.81" title="-0.080">less</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 99.99%); opacity: 0.80" title="-0.000">demerol</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.43%); opacity: 0.81" title="-0.039">worked</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 88.03%); opacity: 0.84" title="-0.222">although</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 97.88%); opacity: 0.80" title="-0.019">nearly</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.072">got</span><span style="opacity: 0.80"> arrested </span><span style="background-color: hsl(0, 100.00%, 96.90%); opacity: 0.81" title="-0.032">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.47%); opacity: 0.83" title="-0.160">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.58%); opacity: 0.81" title="0.037">way</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.13%); opacity: 0.81" title="-0.061">home</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.73%); opacity: 0.88" title="-0.471">when</span><span style="opacity: 0.80"> i barfed
    </span><span style="background-color: hsl(120, 100.00%, 71.34%); opacity: 0.92" title="0.772">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.81%); opacity: 0.83" title="0.176">over</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.62%); opacity: 0.80" title="-0.010">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.22%); opacity: 0.81" title="0.043">police</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.94%); opacity: 0.84" title="0.224">car</span><span style="opacity: 0.80"> parked </span><span style="background-color: hsl(0, 100.00%, 83.50%); opacity: 0.86" title="-0.351">just</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.45%); opacity: 0.83" title="0.211">outside</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.62%); opacity: 0.80" title="-0.010">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.73%); opacity: 0.81" title="0.051">er</span><span style="opacity: 0.80">.
    	- </span><span style="background-color: hsl(120, 100.00%, 64.57%); opacity: 0.97" title="1.045">brian</span><span style="opacity: 0.80">
    </span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=comp.graphics
        
    </b>
    
        
        (probability <b>0.010</b>, score <b>-4.592</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 95.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.379
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 91.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.213
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 91.80%); opacity: 0.82" title="0.129">from</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 85.66%); opacity: 0.85" title="-0.287">brian</span><span style="opacity: 0.80">@</span><span style="background-color: hsl(0, 100.00%, 88.43%); opacity: 0.83" title="-0.211">ucsd</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(0, 100.00%, 90.52%); opacity: 0.83" title="-0.159">edu</span><span style="opacity: 0.80"> (</span><span style="background-color: hsl(0, 100.00%, 85.66%); opacity: 0.85" title="-0.287">brian</span><span style="opacity: 0.80"> kantor)
    </span><span style="background-color: hsl(0, 100.00%, 92.93%); opacity: 0.82" title="-0.105">subject</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 88.29%); opacity: 0.83" title="-0.215">re</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 90.22%); opacity: 0.83" title="0.166">help</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.73%); opacity: 0.85" title="0.285">for</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.87%); opacity: 0.82" title="-0.128">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.69%); opacity: 0.82" title="-0.110">stones</span><span style="opacity: 0.80"> ..............
    </span><span style="background-color: hsl(120, 100.00%, 88.42%); opacity: 0.83" title="0.212">organization</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.041">the</span><span style="opacity: 0.80"> avant-garde </span><span style="background-color: hsl(0, 100.00%, 88.75%); opacity: 0.83" title="-0.203">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.041">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.002">now</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 97.13%); opacity: 0.80" title="0.029">ltd</span><span style="opacity: 0.80">.
    </span><span style="background-color: hsl(120, 100.00%, 91.02%); opacity: 0.82" title="0.147">lines</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 86.10%); opacity: 0.84" title="-0.274">12</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 81.72%); opacity: 0.87" title="0.406">nntp</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 88.75%); opacity: 0.83" title="0.203">posting</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 80.88%); opacity: 0.87" title="0.433">host</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 88.43%); opacity: 0.83" title="-0.211">ucsd</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(0, 100.00%, 90.52%); opacity: 0.83" title="-0.159">edu</span><span style="opacity: 0.80">
    
    </span><span style="background-color: hsl(0, 100.00%, 87.59%); opacity: 0.84" title="-0.233">as</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 98.86%); opacity: 0.80" title="-0.008">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.80%); opacity: 0.82" title="0.129">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.37%); opacity: 0.88" title="-0.516">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 100.00%); opacity: 0.80" title="-0.000">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.46%); opacity: 0.82" title="0.137">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.87%); opacity: 0.82" title="-0.128">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.69%); opacity: 0.82" title="-0.110">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 89.89%); opacity: 0.83" title="0.174">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.054">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(0, 100.00%, 95.43%); opacity: 0.81" title="-0.056">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 94.78%); opacity: 0.81" title="-0.068">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.70%); opacity: 0.80" title="-0.009">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.44%); opacity: 0.82" title="0.137">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.37%); opacity: 0.81" title="-0.057">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.92%); opacity: 0.83" title="0.198">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.05%); opacity: 0.84" title="0.248">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.19%); opacity: 0.81" title="0.079">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.38%); opacity: 0.81" title="-0.040">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.06%); opacity: 0.80" title="-0.006">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.041">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.07%); opacity: 0.84" title="-0.248">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 95.61%); opacity: 0.81" title="0.053">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.176">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.22%); opacity: 0.80" title="-0.028">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 89.16%); opacity: 0.83" title="0.192">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.176">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.09%); opacity: 0.84" title="0.247">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.91%); opacity: 0.81" title="-0.065">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.81%); opacity: 0.84" title="-0.228">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.32%); opacity: 0.81" title="-0.041">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.45%); opacity: 0.81" title="-0.056">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.46%); opacity: 0.82" title="0.137">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.24%); opacity: 0.82" title="-0.119">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 89.16%); opacity: 0.83" title="0.192">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.176">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.09%); opacity: 0.84" title="0.247">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 94.91%); opacity: 0.81" title="-0.065">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.81%); opacity: 0.84" title="-0.228">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.20%); opacity: 0.80" title="-0.028">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.25%); opacity: 0.80" title="-0.027">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 91.84%); opacity: 0.82" title="-0.128">when</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 92.22%); opacity: 0.82" title="-0.120">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.36%); opacity: 0.83" title="-0.187">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.041">the</span><span style="opacity: 0.80"> x-</span><span style="background-color: hsl(120, 100.00%, 85.72%); opacity: 0.85" title="0.285">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.03%); opacity: 0.80" title="0.017">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.061">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.91%); opacity: 0.81" title="-0.065">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.019">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.70%); opacity: 0.80" title="-0.009">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.20%); opacity: 0.88" title="-0.522">she</span><span style="opacity: 0.80">'d </span><span style="background-color: hsl(0, 100.00%, 97.40%); opacity: 0.80" title="-0.025">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.87%); opacity: 0.82" title="-0.128">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 92.69%); opacity: 0.82" title="-0.110">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.39%); opacity: 0.83" title="-0.212">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.44%); opacity: 0.81" title="-0.074">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 88.39%); opacity: 0.83" title="-0.212">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.041">the</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.05%); opacity: 0.80" title="-0.030">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.20%); opacity: 0.82" title="-0.099">less</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 99.97%); opacity: 0.80" title="-0.000">demerol</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.68%); opacity: 0.81" title="-0.070">worked</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 92.93%); opacity: 0.82" title="0.105">although</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(120, 100.00%, 94.33%); opacity: 0.81" title="0.076">nearly</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.78%); opacity: 0.84" title="0.228">got</span><span style="opacity: 0.80"> arrested </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.207">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.37%); opacity: 0.88" title="-0.516">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.95%); opacity: 0.80" title="-0.007">way</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.19%); opacity: 0.81" title="0.079">home</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.84%); opacity: 0.82" title="-0.128">when</span><span style="opacity: 0.80"> i barfed
    </span><span style="background-color: hsl(120, 100.00%, 94.86%); opacity: 0.81" title="0.066">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.71%); opacity: 0.81" title="0.069">over</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.041">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.004">police</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.97%); opacity: 0.80" title="-0.000">car</span><span style="opacity: 0.80"> parked </span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.082">just</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.78%); opacity: 0.81" title="-0.050">outside</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.041">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.19%); opacity: 0.80" title="0.028">er</span><span style="opacity: 0.80">.
    	- </span><span style="background-color: hsl(0, 100.00%, 85.66%); opacity: 0.85" title="-0.287">brian</span><span style="opacity: 0.80">
    </span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.989</b>, score <b>3.945</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 82.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +8.958
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 88.04%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.013
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 94.37%); opacity: 0.81" title="-0.075">from</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 92.18%); opacity: 0.82" title="0.121">brian</span><span style="opacity: 0.80">@</span><span style="background-color: hsl(120, 100.00%, 63.75%); opacity: 0.97" title="1.080">ucsd</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 93.24%); opacity: 0.82" title="0.098">edu</span><span style="opacity: 0.80"> (</span><span style="background-color: hsl(120, 100.00%, 92.18%); opacity: 0.82" title="0.121">brian</span><span style="opacity: 0.80"> kantor)
    </span><span style="background-color: hsl(0, 100.00%, 90.16%); opacity: 0.83" title="-0.168">subject</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 79.54%); opacity: 0.88" title="0.477">re</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 75.86%); opacity: 0.90" title="0.604">help</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.31%); opacity: 0.82" title="-0.140">for</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.55%); opacity: 0.87" title="0.444">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.36%); opacity: 0.86" title="0.355">stones</span><span style="opacity: 0.80"> ..............
    </span><span style="background-color: hsl(120, 100.00%, 81.30%); opacity: 0.87" title="0.420">organization</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.153">the</span><span style="opacity: 0.80"> avant-garde </span><span style="background-color: hsl(120, 100.00%, 90.29%); opacity: 0.83" title="0.165">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.153">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.57%); opacity: 0.86" title="-0.349">now</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 91.30%); opacity: 0.82" title="0.141">ltd</span><span style="opacity: 0.80">.
    </span><span style="background-color: hsl(0, 100.00%, 76.27%); opacity: 0.89" title="-0.589">lines</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 95.89%); opacity: 0.81" title="-0.048">12</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 78.18%); opacity: 0.88" title="0.523">nntp</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 88.68%); opacity: 0.83" title="0.205">posting</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 76.67%); opacity: 0.89" title="0.575">host</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 63.75%); opacity: 0.97" title="1.080">ucsd</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 93.24%); opacity: 0.82" title="0.098">edu</span><span style="opacity: 0.80">
    
    </span><span style="background-color: hsl(120, 100.00%, 88.16%); opacity: 0.84" title="0.218">as</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(120, 100.00%, 81.09%); opacity: 0.87" title="0.426">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.37%); opacity: 0.81" title="-0.075">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.16%); opacity: 0.91" title="0.666">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.67%); opacity: 0.80" title="-0.001">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.44%); opacity: 0.82" title="-0.115">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.55%); opacity: 0.87" title="0.444">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.36%); opacity: 0.86" title="0.355">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 93.91%); opacity: 0.81" title="-0.085">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.74%); opacity: 0.88" title="-0.470">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.074">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 73.79%); opacity: 0.91" title="0.679">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.61%); opacity: 0.82" title="-0.134">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.10%); opacity: 0.86" title="0.363">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.79%); opacity: 0.93" title="-0.793">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.01%); opacity: 0.89" title="-0.563">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.53%); opacity: 0.84" title="0.235">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.01%); opacity: 0.80" title="-0.017">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.91%); opacity: 0.81" title="-0.048">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.006">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.153">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.77%); opacity: 0.91" title="0.680">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 86.29%); opacity: 0.84" title="-0.269">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.035">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.57%); opacity: 0.83" title="-0.182">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 84.77%); opacity: 0.85" title="-0.313">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.035">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.26%); opacity: 0.83" title="-0.216">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.60%); opacity: 0.80" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.05%); opacity: 0.81" title="0.063">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.06%); opacity: 0.81" title="0.045">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.011">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.44%); opacity: 0.82" title="-0.115">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.77%); opacity: 0.87" title="0.404">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 84.77%); opacity: 0.85" title="-0.313">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.035">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.26%); opacity: 0.83" title="-0.216">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 99.60%); opacity: 0.80" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.05%); opacity: 0.81" title="0.063">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.14%); opacity: 0.81" title="0.061">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.58%); opacity: 0.81" title="0.071">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 84.81%); opacity: 0.85" title="0.312">when</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 92.01%); opacity: 0.82" title="-0.125">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.55%); opacity: 0.81" title="-0.038">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.153">the</span><span style="opacity: 0.80"> x-</span><span style="background-color: hsl(120, 100.00%, 92.84%); opacity: 0.82" title="0.106">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.06%); opacity: 0.82" title="0.123">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.50%); opacity: 0.85" title="0.292">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.60%); opacity: 0.80" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.084">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.61%); opacity: 0.82" title="-0.134">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.17%); opacity: 0.91" title="0.665">she</span><span style="opacity: 0.80">'d </span><span style="background-color: hsl(120, 100.00%, 75.74%); opacity: 0.90" title="0.608">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.55%); opacity: 0.87" title="0.444">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 83.36%); opacity: 0.86" title="0.355">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.12%); opacity: 0.84" title="0.219">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.11%); opacity: 0.82" title="-0.101">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 88.12%); opacity: 0.84" title="0.219">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.153">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.95%); opacity: 0.80" title="-0.007">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.34%); opacity: 0.86" title="0.356">less</span><span style="opacity: 0.80">.
    
    </span><span style="opacity: 0.80">demerol</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.74%); opacity: 0.85" title="0.285">worked</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 94.98%); opacity: 0.81" title="-0.064">although</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 94.62%); opacity: 0.81" title="-0.071">nearly</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.67%); opacity: 0.87" title="-0.440">got</span><span style="opacity: 0.80"> arrested </span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.007">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.16%); opacity: 0.91" title="0.666">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.00%); opacity: 0.83" title="-0.172">way</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.58%); opacity: 0.81" title="-0.072">home</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.81%); opacity: 0.85" title="0.312">when</span><span style="opacity: 0.80"> i barfed
    </span><span style="background-color: hsl(120, 100.00%, 98.68%); opacity: 0.80" title="0.009">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.35%); opacity: 0.83" title="-0.213">over</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.153">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.22%); opacity: 0.80" title="-0.005">police</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.66%); opacity: 0.81" title="0.052">car</span><span style="opacity: 0.80"> parked </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.227">just</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.07%); opacity: 0.81" title="0.081">outside</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.153">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.09%); opacity: 0.81" title="-0.081">er</span><span style="opacity: 0.80">.
    	- </span><span style="background-color: hsl(120, 100.00%, 92.18%); opacity: 0.82" title="0.121">brian</span><span style="opacity: 0.80">
    </span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=soc.religion.christian
        
    </b>
    
        
        (probability <b>0.001</b>, score <b>-7.157</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.258
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 85.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -6.899
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 90.01%); opacity: 0.83" title="-0.171">from</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 90.13%); opacity: 0.83" title="-0.168">brian</span><span style="opacity: 0.80">@</span><span style="background-color: hsl(0, 100.00%, 95.31%); opacity: 0.81" title="-0.058">ucsd</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(0, 100.00%, 86.24%); opacity: 0.84" title="-0.270">edu</span><span style="opacity: 0.80"> (</span><span style="background-color: hsl(0, 100.00%, 90.13%); opacity: 0.83" title="-0.168">brian</span><span style="opacity: 0.80"> kantor)
    </span><span style="background-color: hsl(120, 100.00%, 86.64%); opacity: 0.84" title="0.259">subject</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 97.70%); opacity: 0.80" title="-0.021">re</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 78.87%); opacity: 0.88" title="-0.499">help</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.47%); opacity: 0.83" title="-0.210">for</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.98%); opacity: 0.81" title="-0.064">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.025">stones</span><span style="opacity: 0.80"> ..............
    </span><span style="background-color: hsl(0, 100.00%, 83.61%); opacity: 0.86" title="-0.347">organization</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.063">the</span><span style="opacity: 0.80"> avant-garde </span><span style="background-color: hsl(120, 100.00%, 98.48%); opacity: 0.80" title="0.012">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.063">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.021">now</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 93.09%); opacity: 0.82" title="-0.101">ltd</span><span style="opacity: 0.80">.
    </span><span style="background-color: hsl(0, 100.00%, 90.32%); opacity: 0.83" title="-0.164">lines</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 90.30%); opacity: 0.83" title="-0.164">12</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 61.39%); opacity: 0.99" title="-1.181">nntp</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 68.24%); opacity: 0.94" title="-0.894">posting</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-1.243">host</span><span style="opacity: 0.80">: </span><span style="background-color: hsl(0, 100.00%, 95.31%); opacity: 0.81" title="-0.058">ucsd</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(0, 100.00%, 86.24%); opacity: 0.84" title="-0.270">edu</span><span style="opacity: 0.80">
    
    </span><span style="background-color: hsl(120, 100.00%, 91.95%); opacity: 0.82" title="0.126">as</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.116">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.01%); opacity: 0.83" title="-0.171">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.080">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.96%); opacity: 0.80" title="-0.000">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.57%); opacity: 0.80" title="-0.011">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.98%); opacity: 0.81" title="-0.064">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.025">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 88.40%); opacity: 0.83" title="-0.212">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.65%); opacity: 0.81" title="0.036">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(0, 100.00%, 93.87%); opacity: 0.81" title="-0.085">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 91.68%); opacity: 0.82" title="-0.132">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.23%); opacity: 0.81" title="0.078">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.85%); opacity: 0.85" title="-0.310">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.34%); opacity: 0.83" title="0.188">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.005">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.99%); opacity: 0.80" title="-0.007">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.20%); opacity: 0.80" title="-0.005">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.48%); opacity: 0.80" title="0.012">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.76%); opacity: 0.80" title="-0.009">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.063">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.36%); opacity: 0.81" title="-0.057">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.019">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.28%); opacity: 0.83" title="0.165">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.26%); opacity: 0.81" title="0.042">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 88.99%); opacity: 0.83" title="-0.197">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.28%); opacity: 0.83" title="0.165">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.82%); opacity: 0.80" title="-0.019">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.74%); opacity: 0.81" title="0.051">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.43%); opacity: 0.80" title="-0.003">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.05%); opacity: 0.81" title="0.063">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.52%); opacity: 0.83" title="-0.209">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.57%); opacity: 0.80" title="-0.011">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.67%); opacity: 0.80" title="-0.021">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 88.99%); opacity: 0.83" title="-0.197">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.28%); opacity: 0.83" title="0.165">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.82%); opacity: 0.80" title="-0.019">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 95.74%); opacity: 0.81" title="0.051">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.43%); opacity: 0.80" title="-0.003">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.45%); opacity: 0.80" title="-0.003">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.16%); opacity: 0.80" title="-0.005">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 97.97%); opacity: 0.80" title="0.018">when</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.034">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.98%); opacity: 0.81" title="0.047">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.063">the</span><span style="opacity: 0.80"> x-</span><span style="background-color: hsl(0, 100.00%, 89.70%); opacity: 0.83" title="-0.179">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.02%); opacity: 0.82" title="-0.124">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.82%); opacity: 0.80" title="-0.019">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.74%); opacity: 0.81" title="0.051">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.01%); opacity: 0.81" title="-0.082">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.23%); opacity: 0.81" title="0.078">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.74%); opacity: 0.82" title="-0.109">she</span><span style="opacity: 0.80">'d </span><span style="background-color: hsl(0, 100.00%, 92.49%); opacity: 0.82" title="-0.114">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.98%); opacity: 0.81" title="-0.064">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.025">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.45%); opacity: 0.81" title="-0.039">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.94%); opacity: 0.82" title="0.126">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 96.45%); opacity: 0.81" title="-0.039">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.063">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.96%); opacity: 0.80" title="-0.000">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.92%); opacity: 0.81" title="0.048">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.71%); opacity: 0.81" title="-0.035">less</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 99.99%); opacity: 0.80" title="-0.000">demerol</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.85%); opacity: 0.80" title="-0.000">worked</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 94.42%); opacity: 0.81" title="0.074">although</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 98.51%); opacity: 0.80" title="-0.011">nearly</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.61%); opacity: 0.81" title="0.053">got</span><span style="opacity: 0.80"> arrested </span><span style="background-color: hsl(0, 100.00%, 87.11%); opacity: 0.84" title="-0.246">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.080">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.82%); opacity: 0.81" title="0.033">way</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.96%); opacity: 0.80" title="-0.018">home</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.97%); opacity: 0.80" title="0.018">when</span><span style="opacity: 0.80"> i barfed
    </span><span style="background-color: hsl(0, 100.00%, 86.85%); opacity: 0.84" title="-0.254">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.70%); opacity: 0.81" title="-0.069">over</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.063">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.77%); opacity: 0.80" title="-0.020">police</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.38%); opacity: 0.80" title="-0.025">car</span><span style="opacity: 0.80"> parked </span><span style="background-color: hsl(120, 100.00%, 91.36%); opacity: 0.82" title="0.139">just</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.84%); opacity: 0.80" title="-0.008">outside</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.063">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.99%); opacity: 0.80" title="-0.017">er</span><span style="opacity: 0.80">.
    	- </span><span style="background-color: hsl(0, 100.00%, 90.13%); opacity: 0.83" title="-0.168">brian</span><span style="opacity: 0.80">
    </span>
        </p>
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




What can be highlighted in text is highlighted in text. There is also a
separate table for features which can’t be highlighted in text -
``<BIAS>`` in this case. If you hover mouse on a highlighted word it
shows you a weight of this word in a title. Words are colored according
to their weights.

2. Baseline model, improved data
--------------------------------

Aha, from the highlighting above it can be seen that a classifier
learned some non-interesting stuff indeed, e.g. it remembered parts of
email addresses. We should probably clean the data first to make it more
interesting; improving model (trying different classifiers, etc.)
doesn’t make sense at this point - it may just learn to leverage these
email addresses better.

In practice we’d have to do cleaning yourselves; in this example 20
newsgroups dataset provides an option to remove footers and headers from
the messages. Nice. Let’s clean up the data and re-train a classifier.

.. code:: ipython3

    twenty_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=['headers', 'footers'],
    )
    twenty_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=['headers', 'footers'],
    )
    
    vec = CountVectorizer()
    clf = LogisticRegressionCV()
    pipe = make_pipeline(vec, clf)
    pipe.fit(twenty_train.data, twenty_train.target);

We just made the task harder and more realistic for a classifier.

.. code:: ipython3

    print_report(pipe)


.. parsed-literal::

                            precision    recall  f1-score   support
    
               alt.atheism       0.83      0.78      0.80       319
             comp.graphics       0.82      0.96      0.88       389
                   sci.med       0.89      0.80      0.84       396
    soc.religion.christian       0.88      0.86      0.87       398
    
               avg / total       0.85      0.85      0.85      1502
    
    accuracy: 0.852


A great result - we just made quality worse! Does it mean pipeline is
worse now? No, likely it has a better quality on unseen messages. It is
evaluation which is more fair now. Inspecting features used by
classifier allowed us to notice a problem with the data and made a good
change, despite of numbers which told us not to do that.

Instead of removing headers and footers we could have improved
evaluation setup directly, using e.g. GroupKFold from scikit-learn. Then
quality of old model would have dropped, we could have removed
headers/footers and see increased accuracy, so the numbers would have
told us to remove headers and footers. It is not obvious how to split
data though, what groups to use with GroupKFold.

So, what have the updated classifier learned? (output is less verbose
because only a subset of classes is shown - see “targets” argument):

.. code:: ipython3

    eli5.show_prediction(clf, twenty_test.data[0], vec=vec, 
                         target_names=twenty_test.target_names,
                         targets=['sci.med'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.732</b>, score <b>0.031</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.747
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.25%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.716
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.027">as</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(120, 100.00%, 60.22%); opacity: 1.00" title="0.443">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.32%); opacity: 0.83" title="-0.068">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.20%); opacity: 0.97" title="0.381">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.85%); opacity: 0.80" title="-0.003">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.24%); opacity: 0.84" title="-0.097">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.07%); opacity: 0.84" title="0.089">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.46%); opacity: 0.83" title="-0.066">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 81.53%); opacity: 0.87" title="-0.148">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.18%); opacity: 0.90" title="-0.213">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(120, 100.00%, 89.42%); opacity: 0.83" title="0.067">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 67.91%); opacity: 0.95" title="0.326">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.96%); opacity: 0.82" title="-0.045">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 65.61%); opacity: 0.96" title="0.360">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.77%); opacity: 0.99" title="-0.434">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.50%); opacity: 0.91" title="-0.235">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.30%); opacity: 0.93" title="0.292">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.18%); opacity: 0.83" title="-0.060">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.32%); opacity: 0.81" title="-0.015">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.12%); opacity: 0.80" title="0.006">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.09%); opacity: 0.87" title="-0.153">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.446">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 80.22%); opacity: 0.87" title="-0.163">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.84%); opacity: 0.83" title="0.063">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.82%); opacity: 0.86" title="-0.133">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 76.35%); opacity: 0.89" title="-0.211">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.84%); opacity: 0.83" title="0.063">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.23%); opacity: 0.80" title="-0.005">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.67%); opacity: 0.86" title="0.135">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.34%); opacity: 0.83" title="-0.077">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.05%); opacity: 0.82" title="-0.044">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.001">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.24%); opacity: 0.84" title="-0.097">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 61.87%); opacity: 0.99" title="0.417">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 76.35%); opacity: 0.89" title="-0.211">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.84%); opacity: 0.83" title="0.063">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.23%); opacity: 0.80" title="-0.005">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 82.67%); opacity: 0.86" title="0.135">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.34%); opacity: 0.83" title="-0.077">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.028">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.78%); opacity: 0.81" title="0.018">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 70.41%); opacity: 0.93" title="0.290">when</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 77.79%); opacity: 0.89" title="-0.192">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.09%); opacity: 0.84" title="0.089">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 81.09%); opacity: 0.87" title="-0.153">the</span><span style="opacity: 0.80"> x-</span><span style="background-color: hsl(0, 100.00%, 77.94%); opacity: 0.89" title="-0.191">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.09%); opacity: 0.81" title="0.029">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.21%); opacity: 0.84" title="0.088">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.67%); opacity: 0.86" title="0.135">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.75%); opacity: 0.81" title="-0.025">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.96%); opacity: 0.82" title="-0.045">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.52%); opacity: 1.00" title="0.438">she</span><span style="opacity: 0.80">'d </span><span style="background-color: hsl(120, 100.00%, 64.83%); opacity: 0.97" title="0.371">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.07%); opacity: 0.84" title="0.089">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 89.46%); opacity: 0.83" title="-0.066">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.85%); opacity: 0.83" title="0.063">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.52%); opacity: 0.83" title="-0.075">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 89.85%); opacity: 0.83" title="0.063">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.09%); opacity: 0.87" title="-0.153">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.001">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.11%); opacity: 0.80" title="-0.006">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.20%); opacity: 0.89" title="0.200">less</span><span style="opacity: 0.80">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Hm, it no longer uses email addresses, but it still doesn’t look good:
classifier assigns high weights to seemingly unrelated words like ‘do’
or ‘my’. These words appear in many texts, so maybe classifier uses them
as a proxy for bias. Or maybe some of them are more common in some of
classes.

3. Pipeline improvements
------------------------

To help classifier we may filter out stop words:

.. code:: ipython3

    vec = CountVectorizer(stop_words='english')
    clf = LogisticRegressionCV()
    pipe = make_pipeline(vec, clf)
    pipe.fit(twenty_train.data, twenty_train.target)
    
    print_report(pipe)


.. parsed-literal::

                            precision    recall  f1-score   support
    
               alt.atheism       0.87      0.76      0.81       319
             comp.graphics       0.85      0.95      0.90       389
                   sci.med       0.93      0.85      0.89       396
    soc.religion.christian       0.85      0.89      0.87       398
    
               avg / total       0.87      0.87      0.87      1502
    
    accuracy: 0.871


.. code:: ipython3

    eli5.show_prediction(clf, twenty_test.data[0], vec=vec, 
                         target_names=twenty_test.target_names,
                         targets=['sci.med'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.714</b>, score <b>0.510</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.184
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 83.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.674
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as i </span><span style="background-color: hsl(120, 100.00%, 68.54%); opacity: 0.94" title="0.431">recall</span><span style="opacity: 0.80"> from my </span><span style="background-color: hsl(0, 100.00%, 99.94%); opacity: 0.80" title="-0.000">bout</span><span style="opacity: 0.80"> with </span><span style="background-color: hsl(120, 100.00%, 79.43%); opacity: 0.88" title="0.235">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.60%); opacity: 0.83" title="0.077">stones</span><span style="opacity: 0.80">, there </span><span style="background-color: hsl(0, 100.00%, 78.57%); opacity: 0.88" title="-0.249">isn</span><span style="opacity: 0.80">'t any
    </span><span style="background-color: hsl(120, 100.00%, 64.28%); opacity: 0.97" title="0.516">medication</span><span style="opacity: 0.80"> that can do anything about them except </span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.006">relieve</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.607">pain</span><span style="opacity: 0.80">.
    
    either they </span><span style="background-color: hsl(0, 100.00%, 87.49%); opacity: 0.84" title="-0.115">pass</span><span style="opacity: 0.80">, or they have to be </span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.037">broken</span><span style="opacity: 0.80"> up with </span><span style="background-color: hsl(120, 100.00%, 65.54%); opacity: 0.96" title="0.491">sound</span><span style="opacity: 0.80">, or they have
    to be </span><span style="background-color: hsl(120, 100.00%, 98.20%); opacity: 0.80" title="0.007">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.49%); opacity: 0.81" title="0.036">surgically</span><span style="opacity: 0.80">.
    
    when i was in, the x-</span><span style="background-color: hsl(0, 100.00%, 78.25%); opacity: 0.88" title="-0.254">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.15%); opacity: 0.82" title="0.059">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.19%); opacity: 0.85" title="0.161">happened</span><span style="opacity: 0.80"> to </span><span style="background-color: hsl(0, 100.00%, 93.00%); opacity: 0.82" title="-0.050">mention</span><span style="opacity: 0.80"> that she'd had </span><span style="background-color: hsl(120, 100.00%, 79.43%); opacity: 0.88" title="0.235">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 90.60%); opacity: 0.83" title="0.077">stones</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(0, 100.00%, 90.01%); opacity: 0.83" title="-0.084">children</span><span style="opacity: 0.80">, and the </span><span style="background-color: hsl(120, 100.00%, 98.43%); opacity: 0.80" title="0.006">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.01%); opacity: 0.81" title="0.031">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Looks better, isn’t it?

Alternatively, we can use TF*IDF scheme; it should give a somewhat
similar effect.

Note that we’re cross-validating LogisticRegression regularisation
parameter here, like in other examples (LogisticRegressionCV, not
LogisticRegression). TF*IDF values are different from word count values,
so optimal C value can be different. We could draw a wrong conclusion if
a classifier with fixed regularization strength is used - the chosen C
value could have worked better for one kind of data.

.. code:: ipython3

    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vec = TfidfVectorizer()
    clf = LogisticRegressionCV()
    pipe = make_pipeline(vec, clf)
    pipe.fit(twenty_train.data, twenty_train.target)
    
    print_report(pipe)


.. parsed-literal::

                            precision    recall  f1-score   support
    
               alt.atheism       0.91      0.79      0.85       319
             comp.graphics       0.83      0.97      0.90       389
                   sci.med       0.95      0.87      0.91       396
    soc.religion.christian       0.90      0.91      0.91       398
    
               avg / total       0.90      0.89      0.89      1502
    
    accuracy: 0.892


.. code:: ipython3

    eli5.show_prediction(clf, twenty_test.data[0], vec=vec, 
                         target_names=twenty_test.target_names,
                         targets=['sci.med'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.987</b>, score <b>1.585</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +6.788
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 83.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.203
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 94.87%); opacity: 0.81" title="-0.069">as</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(120, 100.00%, 70.25%); opacity: 0.93" title="0.849">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.71%); opacity: 0.81" title="-0.092">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.87%); opacity: 0.88" title="0.521">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.98%); opacity: 0.80" title="-0.007">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.003">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.64%); opacity: 0.93" title="0.833">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.06%); opacity: 0.81" title="0.085">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 96.94%); opacity: 0.81" title="-0.033">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.26%); opacity: 0.83" title="-0.198">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(120, 100.00%, 93.72%); opacity: 0.81" title="0.092">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 63.42%); opacity: 0.98" title="1.140">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.77%); opacity: 0.81" title="-0.091">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.84%); opacity: 0.83" title="0.209">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.59%); opacity: 0.86" title="-0.363">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.45%); opacity: 0.84" title="-0.247">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.05%); opacity: 0.83" title="0.204">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.056">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.014">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.12%); opacity: 0.81" title="0.084">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.34%); opacity: 0.82" title="-0.122">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="1.296">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 89.34%); opacity: 0.83" title="-0.196">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.19%); opacity: 0.80" title="0.029">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.43%); opacity: 0.83" title="-0.220">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.71%); opacity: 0.80" title="-0.010">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.19%); opacity: 0.80" title="0.029">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.14%); opacity: 0.82" title="-0.104">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.35%); opacity: 0.82" title="0.100">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.01%); opacity: 0.81" title="-0.066">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.91%); opacity: 0.81" title="-0.068">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.003">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.003">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.18%); opacity: 0.90" title="0.618">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.71%); opacity: 0.80" title="-0.010">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.19%); opacity: 0.80" title="0.029">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.14%); opacity: 0.82" title="-0.104">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 93.35%); opacity: 0.82" title="0.100">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.01%); opacity: 0.81" title="-0.066">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.24%); opacity: 0.82" title="0.102">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.49%); opacity: 0.83" title="0.192">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 86.45%); opacity: 0.84" title="0.276">when</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 96.69%); opacity: 0.81" title="-0.037">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.70%); opacity: 0.83" title="0.186">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 92.34%); opacity: 0.82" title="-0.122">the</span><span style="opacity: 0.80"> x-</span><span style="background-color: hsl(0, 100.00%, 81.36%); opacity: 0.87" title="-0.435">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.78%); opacity: 0.85" title="0.296">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.23%); opacity: 0.81" title="0.044">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.35%); opacity: 0.82" title="0.100">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.81%); opacity: 0.82" title="-0.135">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.77%); opacity: 0.81" title="-0.091">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.12%); opacity: 0.89" title="0.583">she</span><span style="opacity: 0.80">'d </span><span style="background-color: hsl(120, 100.00%, 83.49%); opacity: 0.86" title="0.366">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.64%); opacity: 0.93" title="0.833">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 94.06%); opacity: 0.81" title="0.085">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.43%); opacity: 0.82" title="0.143">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.94%); opacity: 0.84" title="-0.291">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 91.43%); opacity: 0.82" title="0.143">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.34%); opacity: 0.82" title="-0.122">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.73%); opacity: 0.82" title="0.113">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.35%); opacity: 0.81" title="0.060">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.05%); opacity: 0.86" title="0.380">less</span><span style="opacity: 0.80">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




It helped, but didn’t have quite the same effect. Why not do both?

.. code:: ipython3

    vec = TfidfVectorizer(stop_words='english')
    clf = LogisticRegressionCV()
    pipe = make_pipeline(vec, clf)
    pipe.fit(twenty_train.data, twenty_train.target)
    
    print_report(pipe)


.. parsed-literal::

                            precision    recall  f1-score   support
    
               alt.atheism       0.93      0.77      0.84       319
             comp.graphics       0.84      0.97      0.90       389
                   sci.med       0.95      0.89      0.92       396
    soc.religion.christian       0.88      0.92      0.90       398
    
               avg / total       0.90      0.89      0.89      1502
    
    accuracy: 0.893


.. code:: ipython3

    eli5.show_prediction(clf, twenty_test.data[0], vec=vec, 
                         target_names=twenty_test.target_names,
                         targets=['sci.med'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.939</b>, score <b>1.910</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +5.488
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 85.18%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.578
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as i </span><span style="background-color: hsl(120, 100.00%, 74.21%); opacity: 0.91" title="0.667">recall</span><span style="opacity: 0.80"> from my </span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.007">bout</span><span style="opacity: 0.80"> with </span><span style="background-color: hsl(120, 100.00%, 70.15%); opacity: 0.93" title="0.823">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.35%); opacity: 0.84" title="0.241">stones</span><span style="opacity: 0.80">, there </span><span style="background-color: hsl(0, 100.00%, 89.90%); opacity: 0.83" title="-0.175">isn</span><span style="opacity: 0.80">'t any
    </span><span style="background-color: hsl(120, 100.00%, 64.10%); opacity: 0.97" title="1.071">medication</span><span style="opacity: 0.80"> that can do anything about them except </span><span style="background-color: hsl(120, 100.00%, 95.88%); opacity: 0.81" title="0.049">relieve</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="1.250">pain</span><span style="opacity: 0.80">.
    
    either they </span><span style="background-color: hsl(0, 100.00%, 89.63%); opacity: 0.83" title="-0.182">pass</span><span style="opacity: 0.80">, or they have to be </span><span style="background-color: hsl(0, 100.00%, 95.71%); opacity: 0.81" title="-0.051">broken</span><span style="opacity: 0.80"> up with </span><span style="background-color: hsl(120, 100.00%, 77.92%); opacity: 0.89" title="0.535">sound</span><span style="opacity: 0.80">, or they have
    to be </span><span style="background-color: hsl(120, 100.00%, 94.71%); opacity: 0.81" title="0.069">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.08%); opacity: 0.83" title="0.170">surgically</span><span style="opacity: 0.80">.
    
    when i was in, the x-</span><span style="background-color: hsl(0, 100.00%, 82.10%); opacity: 0.86" title="-0.396">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.03%); opacity: 0.85" title="0.337">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.049">happened</span><span style="opacity: 0.80"> to </span><span style="background-color: hsl(0, 100.00%, 93.20%); opacity: 0.82" title="-0.099">mention</span><span style="opacity: 0.80"> that she'd had </span><span style="background-color: hsl(120, 100.00%, 70.15%); opacity: 0.93" title="0.823">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 87.35%); opacity: 0.84" title="0.241">stones</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(0, 100.00%, 90.32%); opacity: 0.83" title="-0.165">children</span><span style="opacity: 0.80">, and the </span><span style="background-color: hsl(120, 100.00%, 92.84%); opacity: 0.82" title="0.107">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.32%); opacity: 0.82" title="0.118">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




This starts to look good!

4. Char-based pipeline
----------------------

Maybe we can get somewhat better quality by choosing a different
classifier, but let’s skip it for now. Let’s try other analysers instead
- use char n-grams instead of words:

.. code:: ipython3

    vec = TfidfVectorizer(stop_words='english', analyzer='char', 
                          ngram_range=(3,5))
    clf = LogisticRegressionCV()
    pipe = make_pipeline(vec, clf)
    pipe.fit(twenty_train.data, twenty_train.target)
    
    print_report(pipe)


.. parsed-literal::

                            precision    recall  f1-score   support
    
               alt.atheism       0.93      0.79      0.85       319
             comp.graphics       0.81      0.97      0.89       389
                   sci.med       0.95      0.86      0.90       396
    soc.religion.christian       0.89      0.91      0.90       398
    
               avg / total       0.89      0.89      0.89      1502
    
    accuracy: 0.888


.. code:: ipython3

    eli5.show_prediction(clf, twenty_test.data[0], vec=vec, 
                         target_names=twenty_test.target_names)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=alt.atheism
        
    </b>
    
        
        (probability <b>0.002</b>, score <b>-7.318</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 95.22%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.838
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -6.480
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 99.44%); opacity: 0.80" title="-0.001">a</span><span style="background-color: hsl(0, 100.00%, 96.44%); opacity: 0.81" title="-0.010">s</span><span style="background-color: hsl(0, 100.00%, 96.60%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 97.74%); opacity: 0.80" title="-0.005">i</span><span style="background-color: hsl(0, 100.00%, 95.72%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 95.31%); opacity: 0.81" title="-0.015">r</span><span style="background-color: hsl(0, 100.00%, 92.58%); opacity: 0.82" title="-0.028">e</span><span style="background-color: hsl(0, 100.00%, 91.31%); opacity: 0.82" title="-0.035">c</span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.009">a</span><span style="background-color: hsl(120, 100.00%, 96.50%); opacity: 0.81" title="0.010">l</span><span style="background-color: hsl(120, 100.00%, 96.03%); opacity: 0.81" title="0.011">l</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(0, 100.00%, 99.01%); opacity: 0.80" title="-0.002">f</span><span style="background-color: hsl(0, 100.00%, 97.35%); opacity: 0.80" title="-0.006">r</span><span style="background-color: hsl(0, 100.00%, 95.13%); opacity: 0.81" title="-0.015">o</span><span style="background-color: hsl(0, 100.00%, 93.16%); opacity: 0.82" title="-0.025">m</span><span style="background-color: hsl(0, 100.00%, 88.88%); opacity: 0.83" title="-0.050"> </span><span style="background-color: hsl(0, 100.00%, 87.38%); opacity: 0.84" title="-0.060">m</span><span style="background-color: hsl(0, 100.00%, 89.73%); opacity: 0.83" title="-0.045">y</span><span style="background-color: hsl(0, 100.00%, 91.72%); opacity: 0.82" title="-0.033"> </span><span style="background-color: hsl(0, 100.00%, 91.44%); opacity: 0.82" title="-0.034">b</span><span style="background-color: hsl(0, 100.00%, 93.85%); opacity: 0.81" title="-0.021">o</span><span style="background-color: hsl(0, 100.00%, 90.83%); opacity: 0.82" title="-0.038">u</span><span style="background-color: hsl(0, 100.00%, 91.34%); opacity: 0.82" title="-0.035">t</span><span style="background-color: hsl(0, 100.00%, 89.81%); opacity: 0.83" title="-0.044"> </span><span style="background-color: hsl(0, 100.00%, 87.92%); opacity: 0.84" title="-0.056">w</span><span style="background-color: hsl(0, 100.00%, 87.92%); opacity: 0.84" title="-0.056">i</span><span style="background-color: hsl(0, 100.00%, 88.03%); opacity: 0.84" title="-0.056">t</span><span style="background-color: hsl(0, 100.00%, 91.96%); opacity: 0.82" title="-0.031">h</span><span style="background-color: hsl(0, 100.00%, 98.00%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 94.41%); opacity: 0.81" title="0.019">k</span><span style="background-color: hsl(120, 100.00%, 97.23%); opacity: 0.80" title="0.007">i</span><span style="background-color: hsl(0, 100.00%, 94.16%); opacity: 0.81" title="-0.020">d</span><span style="background-color: hsl(0, 100.00%, 91.22%); opacity: 0.82" title="-0.036">n</span><span style="background-color: hsl(0, 100.00%, 90.53%); opacity: 0.83" title="-0.040">e</span><span style="background-color: hsl(0, 100.00%, 91.18%); opacity: 0.82" title="-0.036">y</span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 90.05%); opacity: 0.83" title="-0.043">s</span><span style="background-color: hsl(0, 100.00%, 88.58%); opacity: 0.83" title="-0.052">t</span><span style="background-color: hsl(0, 100.00%, 90.80%); opacity: 0.82" title="-0.038">o</span><span style="background-color: hsl(0, 100.00%, 89.14%); opacity: 0.83" title="-0.048">n</span><span style="background-color: hsl(0, 100.00%, 94.09%); opacity: 0.81" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 97.86%); opacity: 0.80" title="-0.005">s</span><span style="background-color: hsl(120, 100.00%, 95.59%); opacity: 0.81" title="0.013">,</span><span style="background-color: hsl(120, 100.00%, 92.88%); opacity: 0.82" title="0.026"> </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.036">t</span><span style="background-color: hsl(120, 100.00%, 91.75%); opacity: 0.82" title="0.033">h</span><span style="background-color: hsl(120, 100.00%, 91.84%); opacity: 0.82" title="0.032">e</span><span style="background-color: hsl(120, 100.00%, 95.40%); opacity: 0.81" title="0.014">r</span><span style="background-color: hsl(120, 100.00%, 93.89%); opacity: 0.81" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 90.83%); opacity: 0.82" title="0.038"> </span><span style="background-color: hsl(120, 100.00%, 87.43%); opacity: 0.84" title="0.060">i</span><span style="background-color: hsl(120, 100.00%, 82.83%); opacity: 0.86" title="0.093">s</span><span style="background-color: hsl(120, 100.00%, 79.98%); opacity: 0.87" title="0.116">n</span><span style="background-color: hsl(120, 100.00%, 79.31%); opacity: 0.88" title="0.121">'</span><span style="background-color: hsl(120, 100.00%, 83.58%); opacity: 0.86" title="0.087">t</span><span style="background-color: hsl(120, 100.00%, 90.58%); opacity: 0.83" title="0.039"> </span><span style="background-color: hsl(0, 100.00%, 97.60%); opacity: 0.80" title="-0.006">a</span><span style="background-color: hsl(0, 100.00%, 94.29%); opacity: 0.81" title="-0.019">n</span><span style="background-color: hsl(0, 100.00%, 94.32%); opacity: 0.81" title="-0.019">y</span><span style="background-color: hsl(0, 100.00%, 94.52%); opacity: 0.81" title="-0.018">
    </span><span style="background-color: hsl(0, 100.00%, 87.94%); opacity: 0.84" title="-0.056">m</span><span style="background-color: hsl(0, 100.00%, 84.30%); opacity: 0.85" title="-0.082">e</span><span style="background-color: hsl(0, 100.00%, 83.94%); opacity: 0.85" title="-0.084">d</span><span style="background-color: hsl(0, 100.00%, 85.15%); opacity: 0.85" title="-0.076">i</span><span style="background-color: hsl(0, 100.00%, 86.87%); opacity: 0.84" title="-0.063">c</span><span style="background-color: hsl(0, 100.00%, 87.03%); opacity: 0.84" title="-0.062">a</span><span style="background-color: hsl(0, 100.00%, 88.70%); opacity: 0.83" title="-0.051">t</span><span style="background-color: hsl(0, 100.00%, 92.54%); opacity: 0.82" title="-0.028">i</span><span style="background-color: hsl(0, 100.00%, 95.37%); opacity: 0.81" title="-0.014">o</span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.009">n</span><span style="background-color: hsl(120, 100.00%, 95.73%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 94.94%); opacity: 0.81" title="0.016">t</span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.008">h</span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.021">a</span><span style="background-color: hsl(0, 100.00%, 92.90%); opacity: 0.82" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 86.58%); opacity: 0.84" title="-0.065"> </span><span style="background-color: hsl(0, 100.00%, 83.80%); opacity: 0.85" title="-0.085">c</span><span style="background-color: hsl(0, 100.00%, 86.65%); opacity: 0.84" title="-0.065">a</span><span style="background-color: hsl(0, 100.00%, 88.99%); opacity: 0.83" title="-0.049">n</span><span style="background-color: hsl(0, 100.00%, 94.58%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(120, 100.00%, 96.79%); opacity: 0.81" title="0.008">d</span><span style="background-color: hsl(120, 100.00%, 94.22%); opacity: 0.81" title="0.020">o</span><span style="background-color: hsl(120, 100.00%, 94.68%); opacity: 0.81" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 98.10%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.014">n</span><span style="background-color: hsl(0, 100.00%, 96.62%); opacity: 0.81" title="-0.009">y</span><span style="background-color: hsl(120, 100.00%, 96.85%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(120, 100.00%, 94.03%); opacity: 0.81" title="0.021">h</span><span style="background-color: hsl(120, 100.00%, 93.78%); opacity: 0.81" title="0.022">i</span><span style="background-color: hsl(120, 100.00%, 93.06%); opacity: 0.82" title="0.025">n</span><span style="background-color: hsl(120, 100.00%, 94.08%); opacity: 0.81" title="0.020">g</span><span style="background-color: hsl(0, 100.00%, 97.30%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 93.22%); opacity: 0.82" title="-0.025">a</span><span style="background-color: hsl(0, 100.00%, 88.66%); opacity: 0.83" title="-0.051">b</span><span style="background-color: hsl(0, 100.00%, 88.80%); opacity: 0.83" title="-0.050">o</span><span style="background-color: hsl(0, 100.00%, 90.23%); opacity: 0.83" title="-0.042">u</span><span style="background-color: hsl(0, 100.00%, 95.50%); opacity: 0.81" title="-0.014">t</span><span style="background-color: hsl(120, 100.00%, 97.16%); opacity: 0.80" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.038">t</span><span style="background-color: hsl(120, 100.00%, 90.34%); opacity: 0.83" title="0.041">h</span><span style="background-color: hsl(120, 100.00%, 91.07%); opacity: 0.82" title="0.037">e</span><span style="background-color: hsl(120, 100.00%, 93.94%); opacity: 0.81" title="0.021">m</span><span style="background-color: hsl(120, 100.00%, 97.51%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.010">x</span><span style="background-color: hsl(120, 100.00%, 92.40%); opacity: 0.82" title="0.029">c</span><span style="background-color: hsl(120, 100.00%, 93.28%); opacity: 0.82" title="0.024">e</span><span style="background-color: hsl(120, 100.00%, 93.25%); opacity: 0.82" title="0.025">p</span><span style="background-color: hsl(120, 100.00%, 90.80%); opacity: 0.82" title="0.038">t</span><span style="background-color: hsl(120, 100.00%, 86.37%); opacity: 0.84" title="0.067"> </span><span style="background-color: hsl(120, 100.00%, 81.07%); opacity: 0.87" title="0.107">r</span><span style="background-color: hsl(120, 100.00%, 80.33%); opacity: 0.87" title="0.113">e</span><span style="background-color: hsl(120, 100.00%, 80.54%); opacity: 0.87" title="0.111">l</span><span style="background-color: hsl(120, 100.00%, 87.16%); opacity: 0.84" title="0.061">i</span><span style="background-color: hsl(0, 100.00%, 95.68%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 96.37%); opacity: 0.81" title="-0.010">v</span><span style="background-color: hsl(0, 100.00%, 96.61%); opacity: 0.81" title="-0.009">e</span><span style="background-color: hsl(120, 100.00%, 97.37%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 93.79%); opacity: 0.81" title="0.022">t</span><span style="background-color: hsl(120, 100.00%, 94.59%); opacity: 0.81" title="0.018">h</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 91.87%); opacity: 0.82" title="-0.032">p</span><span style="background-color: hsl(0, 100.00%, 90.91%); opacity: 0.82" title="-0.037">a</span><span style="background-color: hsl(0, 100.00%, 96.37%); opacity: 0.81" title="-0.010">i</span><span style="background-color: hsl(0, 100.00%, 95.61%); opacity: 0.81" title="-0.013">n</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.016">.</span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.038">e</span><span style="background-color: hsl(120, 100.00%, 91.60%); opacity: 0.82" title="0.033">i</span><span style="background-color: hsl(120, 100.00%, 91.27%); opacity: 0.82" title="0.035">t</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(0, 100.00%, 93.85%); opacity: 0.81" title="-0.021">e</span><span style="background-color: hsl(0, 100.00%, 92.04%); opacity: 0.82" title="-0.031">r</span><span style="background-color: hsl(0, 100.00%, 93.35%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(120, 100.00%, 97.84%); opacity: 0.80" title="0.005">t</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(0, 100.00%, 98.11%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(0, 100.00%, 95.77%); opacity: 0.81" title="-0.013">y</span><span style="background-color: hsl(0, 100.00%, 94.63%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 96.42%); opacity: 0.81" title="-0.010">p</span><span style="background-color: hsl(0, 100.00%, 96.28%); opacity: 0.81" title="-0.010">a</span><span style="background-color: hsl(0, 100.00%, 95.35%); opacity: 0.81" title="-0.014">s</span><span style="background-color: hsl(0, 100.00%, 96.69%); opacity: 0.81" title="-0.009">s</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001">,</span><span style="background-color: hsl(120, 100.00%, 91.86%); opacity: 0.82" title="0.032"> </span><span style="background-color: hsl(120, 100.00%, 90.68%); opacity: 0.82" title="0.039">o</span><span style="background-color: hsl(120, 100.00%, 91.17%); opacity: 0.82" title="0.036">r</span><span style="background-color: hsl(120, 100.00%, 93.17%); opacity: 0.82" title="0.025"> </span><span style="background-color: hsl(120, 100.00%, 93.69%); opacity: 0.81" title="0.022">t</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 91.80%); opacity: 0.82" title="-0.032">y</span><span style="background-color: hsl(0, 100.00%, 95.67%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.003">h</span><span style="background-color: hsl(120, 100.00%, 96.04%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 94.61%); opacity: 0.81" title="0.018">v</span><span style="background-color: hsl(120, 100.00%, 96.35%); opacity: 0.81" title="0.010">e</span><span style="background-color: hsl(0, 100.00%, 98.15%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.016">t</span><span style="background-color: hsl(0, 100.00%, 95.65%); opacity: 0.81" title="-0.013">o</span><span style="background-color: hsl(120, 100.00%, 97.56%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.025">b</span><span style="background-color: hsl(120, 100.00%, 91.83%); opacity: 0.82" title="0.032">e</span><span style="background-color: hsl(120, 100.00%, 96.45%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.003">b</span><span style="background-color: hsl(0, 100.00%, 91.67%); opacity: 0.82" title="-0.033">r</span><span style="background-color: hsl(0, 100.00%, 93.51%); opacity: 0.81" title="-0.023">o</span><span style="background-color: hsl(120, 100.00%, 94.34%); opacity: 0.81" title="0.019">k</span><span style="background-color: hsl(120, 100.00%, 92.86%); opacity: 0.82" title="0.027">e</span><span style="background-color: hsl(120, 100.00%, 88.99%); opacity: 0.83" title="0.049">n</span><span style="background-color: hsl(120, 100.00%, 90.01%); opacity: 0.83" title="0.043"> </span><span style="background-color: hsl(120, 100.00%, 93.08%); opacity: 0.82" title="0.025">u</span><span style="background-color: hsl(120, 100.00%, 93.01%); opacity: 0.82" title="0.026">p</span><span style="background-color: hsl(0, 100.00%, 93.19%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 89.45%); opacity: 0.83" title="-0.046">w</span><span style="background-color: hsl(0, 100.00%, 88.01%); opacity: 0.84" title="-0.056">i</span><span style="background-color: hsl(0, 100.00%, 86.21%); opacity: 0.84" title="-0.068">t</span><span style="background-color: hsl(0, 100.00%, 86.78%); opacity: 0.84" title="-0.064">h</span><span style="background-color: hsl(0, 100.00%, 88.17%); opacity: 0.84" title="-0.055"> </span><span style="background-color: hsl(0, 100.00%, 87.04%); opacity: 0.84" title="-0.062">s</span><span style="background-color: hsl(0, 100.00%, 88.55%); opacity: 0.83" title="-0.052">o</span><span style="background-color: hsl(0, 100.00%, 89.96%); opacity: 0.83" title="-0.043">u</span><span style="background-color: hsl(120, 100.00%, 97.95%); opacity: 0.80" title="0.004">n</span><span style="background-color: hsl(120, 100.00%, 96.04%); opacity: 0.81" title="0.011">d</span><span style="background-color: hsl(120, 100.00%, 91.58%); opacity: 0.82" title="0.034">,</span><span style="background-color: hsl(120, 100.00%, 89.83%); opacity: 0.83" title="0.044"> </span><span style="background-color: hsl(120, 100.00%, 90.54%); opacity: 0.83" title="0.040">o</span><span style="background-color: hsl(120, 100.00%, 91.33%); opacity: 0.82" title="0.035">r</span><span style="background-color: hsl(120, 100.00%, 93.17%); opacity: 0.82" title="0.025"> </span><span style="background-color: hsl(120, 100.00%, 93.69%); opacity: 0.81" title="0.022">t</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 91.80%); opacity: 0.82" title="-0.032">y</span><span style="background-color: hsl(0, 100.00%, 95.67%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 98.12%); opacity: 0.80" title="-0.004">h</span><span style="background-color: hsl(0, 100.00%, 96.86%); opacity: 0.81" title="-0.008">a</span><span style="background-color: hsl(0, 100.00%, 93.79%); opacity: 0.81" title="-0.022">v</span><span style="background-color: hsl(0, 100.00%, 92.40%); opacity: 0.82" title="-0.029">e</span><span style="background-color: hsl(0, 100.00%, 93.32%); opacity: 0.82" title="-0.024">
    </span><span style="background-color: hsl(0, 100.00%, 96.83%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(120, 100.00%, 98.32%); opacity: 0.80" title="0.003">o</span><span style="background-color: hsl(120, 100.00%, 95.81%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 94.36%); opacity: 0.81" title="0.019">b</span><span style="background-color: hsl(0, 100.00%, 99.92%); opacity: 0.80" title="-0.000">e</span><span style="background-color: hsl(0, 100.00%, 96.27%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 99.84%); opacity: 0.80" title="-0.000">e</span><span style="background-color: hsl(0, 100.00%, 99.09%); opacity: 0.80" title="-0.001">x</span><span style="background-color: hsl(120, 100.00%, 95.56%); opacity: 0.81" title="0.013">t</span><span style="background-color: hsl(0, 100.00%, 96.39%); opacity: 0.81" title="-0.010">r</span><span style="background-color: hsl(0, 100.00%, 95.91%); opacity: 0.81" title="-0.012">a</span><span style="background-color: hsl(0, 100.00%, 90.17%); opacity: 0.83" title="-0.042">c</span><span style="background-color: hsl(0, 100.00%, 93.02%); opacity: 0.82" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 90.51%); opacity: 0.83" title="-0.040">e</span><span style="background-color: hsl(0, 100.00%, 93.90%); opacity: 0.81" title="-0.021">d</span><span style="background-color: hsl(0, 100.00%, 93.55%); opacity: 0.81" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 90.50%); opacity: 0.83" title="-0.040">s</span><span style="background-color: hsl(0, 100.00%, 87.98%); opacity: 0.84" title="-0.056">u</span><span style="background-color: hsl(0, 100.00%, 84.78%); opacity: 0.85" title="-0.078">r</span><span style="background-color: hsl(0, 100.00%, 91.52%); opacity: 0.82" title="-0.034">g</span><span style="background-color: hsl(0, 100.00%, 98.23%); opacity: 0.80" title="-0.004">i</span><span style="background-color: hsl(120, 100.00%, 93.99%); opacity: 0.81" title="0.021">c</span><span style="background-color: hsl(120, 100.00%, 92.52%); opacity: 0.82" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 92.57%); opacity: 0.82" title="0.028">l</span><span style="background-color: hsl(120, 100.00%, 93.36%); opacity: 0.82" title="0.024">l</span><span style="background-color: hsl(120, 100.00%, 94.73%); opacity: 0.81" title="0.017">y</span><span style="background-color: hsl(120, 100.00%, 91.97%); opacity: 0.82" title="0.031">.</span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(0, 100.00%, 96.89%); opacity: 0.81" title="-0.008">w</span><span style="background-color: hsl(0, 100.00%, 91.51%); opacity: 0.82" title="-0.034">h</span><span style="background-color: hsl(0, 100.00%, 90.46%); opacity: 0.83" title="-0.040">e</span><span style="background-color: hsl(0, 100.00%, 90.91%); opacity: 0.82" title="-0.037">n</span><span style="background-color: hsl(0, 100.00%, 89.27%); opacity: 0.83" title="-0.047"> </span><span style="background-color: hsl(0, 100.00%, 86.89%); opacity: 0.84" title="-0.063">i</span><span style="background-color: hsl(0, 100.00%, 87.19%); opacity: 0.84" title="-0.061"> </span><span style="background-color: hsl(0, 100.00%, 91.46%); opacity: 0.82" title="-0.034">w</span><span style="background-color: hsl(0, 100.00%, 93.58%); opacity: 0.81" title="-0.023">a</span><span style="background-color: hsl(0, 100.00%, 96.47%); opacity: 0.81" title="-0.010">s</span><span style="background-color: hsl(0, 100.00%, 97.57%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 95.04%); opacity: 0.81" title="-0.016">i</span><span style="background-color: hsl(0, 100.00%, 97.31%); opacity: 0.80" title="-0.007">n</span><span style="background-color: hsl(120, 100.00%, 94.87%); opacity: 0.81" title="0.017">,</span><span style="background-color: hsl(120, 100.00%, 92.64%); opacity: 0.82" title="0.028"> </span><span style="background-color: hsl(120, 100.00%, 91.52%); opacity: 0.82" title="0.034">t</span><span style="background-color: hsl(120, 100.00%, 94.02%); opacity: 0.81" title="0.021">h</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.006">e</span><span style="background-color: hsl(0, 100.00%, 94.91%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 95.01%); opacity: 0.81" title="-0.016">x</span><span style="background-color: hsl(0, 100.00%, 95.08%); opacity: 0.81" title="-0.016">-</span><span style="background-color: hsl(0, 100.00%, 90.74%); opacity: 0.82" title="-0.038">r</span><span style="background-color: hsl(0, 100.00%, 91.36%); opacity: 0.82" title="-0.035">a</span><span style="background-color: hsl(0, 100.00%, 92.00%); opacity: 0.82" title="-0.031">y</span><span style="background-color: hsl(0, 100.00%, 93.36%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 94.53%); opacity: 0.81" title="-0.018">t</span><span style="background-color: hsl(0, 100.00%, 96.89%); opacity: 0.81" title="-0.008">e</span><span style="background-color: hsl(0, 100.00%, 99.69%); opacity: 0.80" title="-0.000">c</span><span style="background-color: hsl(120, 100.00%, 87.74%); opacity: 0.84" title="0.057">h</span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.061"> </span><span style="background-color: hsl(120, 100.00%, 86.86%); opacity: 0.84" title="0.063">h</span><span style="background-color: hsl(120, 100.00%, 92.23%); opacity: 0.82" title="0.030">a</span><span style="background-color: hsl(0, 100.00%, 95.74%); opacity: 0.81" title="-0.013">p</span><span style="background-color: hsl(0, 100.00%, 98.92%); opacity: 0.80" title="-0.002">p</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 95.93%); opacity: 0.81" title="0.012">n</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(0, 100.00%, 92.89%); opacity: 0.82" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 92.43%); opacity: 0.82" title="-0.029">t</span><span style="background-color: hsl(0, 100.00%, 95.21%); opacity: 0.81" title="-0.015">o</span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 91.18%); opacity: 0.82" title="0.036">m</span><span style="background-color: hsl(120, 100.00%, 88.56%); opacity: 0.83" title="0.052">e</span><span style="background-color: hsl(120, 100.00%, 83.69%); opacity: 0.86" title="0.086">n</span><span style="background-color: hsl(120, 100.00%, 86.45%); opacity: 0.84" title="0.066">t</span><span style="background-color: hsl(120, 100.00%, 89.43%); opacity: 0.83" title="0.046">i</span><span style="background-color: hsl(120, 100.00%, 92.56%); opacity: 0.82" title="0.028">o</span><span style="background-color: hsl(120, 100.00%, 94.08%); opacity: 0.81" title="0.020">n</span><span style="background-color: hsl(120, 100.00%, 95.73%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 94.94%); opacity: 0.81" title="0.016">t</span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.022">h</span><span style="background-color: hsl(120, 100.00%, 94.65%); opacity: 0.81" title="0.018">a</span><span style="background-color: hsl(120, 100.00%, 93.09%); opacity: 0.82" title="0.025">t</span><span style="background-color: hsl(120, 100.00%, 94.10%); opacity: 0.81" title="0.020"> </span><span style="background-color: hsl(120, 100.00%, 92.47%); opacity: 0.82" title="0.029">s</span><span style="background-color: hsl(120, 100.00%, 93.08%); opacity: 0.82" title="0.025">h</span><span style="background-color: hsl(120, 100.00%, 93.00%); opacity: 0.82" title="0.026">e</span><span style="background-color: hsl(120, 100.00%, 90.35%); opacity: 0.83" title="0.041">'</span><span style="background-color: hsl(120, 100.00%, 95.21%); opacity: 0.81" title="0.015">d</span><span style="background-color: hsl(120, 100.00%, 94.40%); opacity: 0.81" title="0.019"> </span><span style="background-color: hsl(120, 100.00%, 96.70%); opacity: 0.81" title="0.009">h</span><span style="background-color: hsl(120, 100.00%, 98.11%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(120, 100.00%, 91.95%); opacity: 0.82" title="0.031">d</span><span style="background-color: hsl(120, 100.00%, 89.85%); opacity: 0.83" title="0.044"> </span><span style="background-color: hsl(120, 100.00%, 89.96%); opacity: 0.83" title="0.043">k</span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.007">i</span><span style="background-color: hsl(0, 100.00%, 94.59%); opacity: 0.81" title="-0.018">d</span><span style="background-color: hsl(0, 100.00%, 92.24%); opacity: 0.82" title="-0.030">n</span><span style="background-color: hsl(0, 100.00%, 92.38%); opacity: 0.82" title="-0.029">e</span><span style="background-color: hsl(0, 100.00%, 96.67%); opacity: 0.81" title="-0.009">y</span><span style="background-color: hsl(120, 100.00%, 93.25%); opacity: 0.82" title="0.025">
    </span><span style="background-color: hsl(120, 100.00%, 96.17%); opacity: 0.81" title="0.011">s</span><span style="background-color: hsl(0, 100.00%, 96.47%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 94.81%); opacity: 0.81" title="-0.017">o</span><span style="background-color: hsl(0, 100.00%, 92.97%); opacity: 0.82" title="-0.026">n</span><span style="background-color: hsl(0, 100.00%, 93.66%); opacity: 0.81" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 94.14%); opacity: 0.81" title="-0.020">s</span><span style="background-color: hsl(0, 100.00%, 91.55%); opacity: 0.82" title="-0.034"> </span><span style="background-color: hsl(0, 100.00%, 89.38%); opacity: 0.83" title="-0.047">a</span><span style="background-color: hsl(0, 100.00%, 87.43%); opacity: 0.84" title="-0.060">n</span><span style="background-color: hsl(0, 100.00%, 88.38%); opacity: 0.83" title="-0.053">d</span><span style="background-color: hsl(0, 100.00%, 90.80%); opacity: 0.82" title="-0.038"> </span><span style="background-color: hsl(0, 100.00%, 93.06%); opacity: 0.82" title="-0.025">c</span><span style="background-color: hsl(120, 100.00%, 96.98%); opacity: 0.80" title="0.008">h</span><span style="background-color: hsl(120, 100.00%, 92.03%); opacity: 0.82" title="0.031">i</span><span style="background-color: hsl(120, 100.00%, 91.37%); opacity: 0.82" title="0.035">l</span><span style="background-color: hsl(120, 100.00%, 96.97%); opacity: 0.81" title="0.008">d</span><span style="background-color: hsl(0, 100.00%, 97.36%); opacity: 0.80" title="-0.006">r</span><span style="background-color: hsl(0, 100.00%, 97.78%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(120, 100.00%, 95.23%); opacity: 0.81" title="0.015">n</span><span style="background-color: hsl(120, 100.00%, 97.13%); opacity: 0.80" title="0.007">,</span><span style="background-color: hsl(0, 100.00%, 97.23%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 94.32%); opacity: 0.81" title="-0.019">a</span><span style="background-color: hsl(0, 100.00%, 94.46%); opacity: 0.81" title="-0.018">n</span><span style="background-color: hsl(0, 100.00%, 96.92%); opacity: 0.81" title="-0.008">d</span><span style="background-color: hsl(120, 100.00%, 97.78%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 94.44%); opacity: 0.81" title="0.019">t</span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.004">h</span><span style="background-color: hsl(0, 100.00%, 99.89%); opacity: 0.80" title="-0.000">e</span><span style="background-color: hsl(0, 100.00%, 95.83%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 98.22%); opacity: 0.80" title="-0.004">c</span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.023">h</span><span style="background-color: hsl(120, 100.00%, 91.91%); opacity: 0.82" title="0.032">i</span><span style="background-color: hsl(120, 100.00%, 92.94%); opacity: 0.82" title="0.026">l</span><span style="background-color: hsl(120, 100.00%, 97.88%); opacity: 0.80" title="0.005">d</span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.002">b</span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.006">i</span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.009">r</span><span style="background-color: hsl(120, 100.00%, 98.32%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 92.65%); opacity: 0.82" title="0.028">h</span><span style="background-color: hsl(120, 100.00%, 93.69%); opacity: 0.81" title="0.022"> </span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.007">h</span><span style="background-color: hsl(0, 100.00%, 95.45%); opacity: 0.81" title="-0.014">u</span><span style="background-color: hsl(0, 100.00%, 90.57%); opacity: 0.83" title="-0.039">r</span><span style="background-color: hsl(0, 100.00%, 99.91%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 96.29%); opacity: 0.81" title="0.010">l</span><span style="background-color: hsl(0, 100.00%, 95.37%); opacity: 0.81" title="-0.014">e</span><span style="background-color: hsl(0, 100.00%, 92.92%); opacity: 0.82" title="-0.026">s</span><span style="background-color: hsl(0, 100.00%, 93.32%); opacity: 0.82" title="-0.024">s</span><span style="background-color: hsl(0, 100.00%, 96.99%); opacity: 0.80" title="-0.008">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=comp.graphics
        
    </b>
    
        
        (probability <b>0.017</b>, score <b>-5.118</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 94.85%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.934
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.93%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -6.052
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 98.12%); opacity: 0.80" title="-0.004">a</span><span style="background-color: hsl(0, 100.00%, 98.82%); opacity: 0.80" title="-0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(0, 100.00%, 98.52%); opacity: 0.80" title="-0.003">i</span><span style="background-color: hsl(0, 100.00%, 95.88%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 95.49%); opacity: 0.81" title="-0.014">r</span><span style="background-color: hsl(0, 100.00%, 94.15%); opacity: 0.81" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 94.34%); opacity: 0.81" title="-0.019">c</span><span style="background-color: hsl(0, 100.00%, 95.18%); opacity: 0.81" title="-0.015">a</span><span style="background-color: hsl(0, 100.00%, 95.88%); opacity: 0.81" title="-0.012">l</span><span style="background-color: hsl(0, 100.00%, 99.75%); opacity: 0.80" title="-0.000">l</span><span style="background-color: hsl(120, 100.00%, 96.21%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 94.70%); opacity: 0.81" title="0.017">f</span><span style="background-color: hsl(120, 100.00%, 95.12%); opacity: 0.81" title="0.015">r</span><span style="background-color: hsl(120, 100.00%, 96.69%); opacity: 0.81" title="0.009">o</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.003">m</span><span style="background-color: hsl(0, 100.00%, 94.22%); opacity: 0.81" title="-0.020"> </span><span style="background-color: hsl(0, 100.00%, 91.51%); opacity: 0.82" title="-0.034">m</span><span style="background-color: hsl(0, 100.00%, 90.06%); opacity: 0.83" title="-0.043">y</span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.020"> </span><span style="background-color: hsl(0, 100.00%, 99.58%); opacity: 0.80" title="-0.000">b</span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.016">o</span><span style="background-color: hsl(120, 100.00%, 94.71%); opacity: 0.81" title="0.017">u</span><span style="background-color: hsl(120, 100.00%, 97.03%); opacity: 0.80" title="0.008">t</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(0, 100.00%, 99.06%); opacity: 0.80" title="-0.001">w</span><span style="background-color: hsl(0, 100.00%, 98.95%); opacity: 0.80" title="-0.002">i</span><span style="background-color: hsl(0, 100.00%, 98.88%); opacity: 0.80" title="-0.002">t</span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.003">h</span><span style="background-color: hsl(0, 100.00%, 97.68%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.014">k</span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.021">i</span><span style="background-color: hsl(0, 100.00%, 93.92%); opacity: 0.81" title="-0.021">d</span><span style="background-color: hsl(0, 100.00%, 95.06%); opacity: 0.81" title="-0.016">n</span><span style="background-color: hsl(0, 100.00%, 94.10%); opacity: 0.81" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 94.49%); opacity: 0.81" title="-0.018">y</span><span style="background-color: hsl(0, 100.00%, 92.85%); opacity: 0.82" title="-0.027"> </span><span style="background-color: hsl(0, 100.00%, 94.19%); opacity: 0.81" title="-0.020">s</span><span style="background-color: hsl(0, 100.00%, 92.90%); opacity: 0.82" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 94.26%); opacity: 0.81" title="-0.019">o</span><span style="background-color: hsl(0, 100.00%, 95.01%); opacity: 0.81" title="-0.016">n</span><span style="background-color: hsl(0, 100.00%, 98.61%); opacity: 0.80" title="-0.003">e</span><span style="background-color: hsl(0, 100.00%, 98.23%); opacity: 0.80" title="-0.004">s</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">,</span><span style="background-color: hsl(0, 100.00%, 95.70%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 93.31%); opacity: 0.82" title="-0.024">t</span><span style="background-color: hsl(0, 100.00%, 94.34%); opacity: 0.81" title="-0.019">h</span><span style="background-color: hsl(0, 100.00%, 97.82%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(120, 100.00%, 97.93%); opacity: 0.80" title="0.005">r</span><span style="background-color: hsl(120, 100.00%, 98.38%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(0, 100.00%, 96.66%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 95.39%); opacity: 0.81" title="-0.014">i</span><span style="background-color: hsl(0, 100.00%, 94.07%); opacity: 0.81" title="-0.020">s</span><span style="background-color: hsl(0, 100.00%, 94.72%); opacity: 0.81" title="-0.017">n</span><span style="background-color: hsl(0, 100.00%, 93.86%); opacity: 0.81" title="-0.021">'</span><span style="background-color: hsl(0, 100.00%, 95.37%); opacity: 0.81" title="-0.014">t</span><span style="background-color: hsl(0, 100.00%, 97.26%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 99.61%); opacity: 0.80" title="-0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(120, 100.00%, 97.48%); opacity: 0.80" title="0.006">y</span><span style="background-color: hsl(0, 100.00%, 98.35%); opacity: 0.80" title="-0.003">
    </span><span style="background-color: hsl(0, 100.00%, 92.99%); opacity: 0.82" title="-0.026">m</span><span style="background-color: hsl(0, 100.00%, 88.37%); opacity: 0.83" title="-0.053">e</span><span style="background-color: hsl(0, 100.00%, 84.70%); opacity: 0.85" title="-0.079">d</span><span style="background-color: hsl(0, 100.00%, 85.64%); opacity: 0.85" title="-0.072">i</span><span style="background-color: hsl(0, 100.00%, 87.98%); opacity: 0.84" title="-0.056">c</span><span style="background-color: hsl(0, 100.00%, 94.49%); opacity: 0.81" title="-0.018">a</span><span style="background-color: hsl(0, 100.00%, 99.02%); opacity: 0.80" title="-0.002">t</span><span style="background-color: hsl(120, 100.00%, 97.61%); opacity: 0.80" title="0.006">i</span><span style="background-color: hsl(0, 100.00%, 99.27%); opacity: 0.80" title="-0.001">o</span><span style="background-color: hsl(0, 100.00%, 95.37%); opacity: 0.81" title="-0.014">n</span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 89.84%); opacity: 0.83" title="-0.044">t</span><span style="background-color: hsl(0, 100.00%, 89.33%); opacity: 0.83" title="-0.047">h</span><span style="background-color: hsl(0, 100.00%, 90.75%); opacity: 0.82" title="-0.038">a</span><span style="background-color: hsl(0, 100.00%, 91.86%); opacity: 0.82" title="-0.032">t</span><span style="background-color: hsl(0, 100.00%, 94.81%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 97.63%); opacity: 0.80" title="-0.005">c</span><span style="background-color: hsl(0, 100.00%, 96.96%); opacity: 0.81" title="-0.008">a</span><span style="background-color: hsl(0, 100.00%, 99.75%); opacity: 0.80" title="-0.000">n</span><span style="background-color: hsl(0, 100.00%, 97.12%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 97.20%); opacity: 0.80" title="-0.007">d</span><span style="background-color: hsl(0, 100.00%, 94.67%); opacity: 0.81" title="-0.017">o</span><span style="background-color: hsl(0, 100.00%, 95.72%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 96.92%); opacity: 0.81" title="-0.008">a</span><span style="background-color: hsl(0, 100.00%, 98.04%); opacity: 0.80" title="-0.004">n</span><span style="background-color: hsl(0, 100.00%, 98.54%); opacity: 0.80" title="-0.003">y</span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.014">t</span><span style="background-color: hsl(0, 100.00%, 94.62%); opacity: 0.81" title="-0.018">h</span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.017">i</span><span style="background-color: hsl(0, 100.00%, 95.37%); opacity: 0.81" title="-0.014">n</span><span style="background-color: hsl(0, 100.00%, 97.24%); opacity: 0.80" title="-0.007">g</span><span style="background-color: hsl(0, 100.00%, 97.93%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 98.03%); opacity: 0.80" title="-0.004">a</span><span style="background-color: hsl(0, 100.00%, 97.91%); opacity: 0.80" title="-0.005">b</span><span style="background-color: hsl(120, 100.00%, 98.77%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.011">u</span><span style="background-color: hsl(120, 100.00%, 96.96%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(0, 100.00%, 97.33%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 94.81%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(0, 100.00%, 94.16%); opacity: 0.81" title="-0.020">h</span><span style="background-color: hsl(0, 100.00%, 97.49%); opacity: 0.80" title="-0.006">e</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(0, 100.00%, 99.10%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 95.22%); opacity: 0.81" title="-0.015">e</span><span style="background-color: hsl(0, 100.00%, 94.33%); opacity: 0.81" title="-0.019">x</span><span style="background-color: hsl(0, 100.00%, 92.98%); opacity: 0.82" title="-0.026">c</span><span style="background-color: hsl(0, 100.00%, 91.13%); opacity: 0.82" title="-0.036">e</span><span style="background-color: hsl(0, 100.00%, 91.35%); opacity: 0.82" title="-0.035">p</span><span style="background-color: hsl(0, 100.00%, 92.74%); opacity: 0.82" title="-0.027">t</span><span style="background-color: hsl(0, 100.00%, 91.56%); opacity: 0.82" title="-0.034"> </span><span style="background-color: hsl(0, 100.00%, 90.11%); opacity: 0.83" title="-0.042">r</span><span style="background-color: hsl(0, 100.00%, 85.71%); opacity: 0.85" title="-0.072">e</span><span style="background-color: hsl(0, 100.00%, 84.22%); opacity: 0.85" title="-0.082">l</span><span style="background-color: hsl(0, 100.00%, 84.44%); opacity: 0.85" title="-0.081">i</span><span style="background-color: hsl(0, 100.00%, 88.10%); opacity: 0.84" title="-0.055">e</span><span style="background-color: hsl(0, 100.00%, 90.26%); opacity: 0.83" title="-0.041">v</span><span style="background-color: hsl(0, 100.00%, 92.88%); opacity: 0.82" title="-0.026">e</span><span style="background-color: hsl(0, 100.00%, 92.46%); opacity: 0.82" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 91.94%); opacity: 0.82" title="-0.032">t</span><span style="background-color: hsl(0, 100.00%, 90.66%); opacity: 0.83" title="-0.039">h</span><span style="background-color: hsl(0, 100.00%, 92.37%); opacity: 0.82" title="-0.029">e</span><span style="background-color: hsl(0, 100.00%, 93.14%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 91.89%); opacity: 0.82" title="-0.032">p</span><span style="background-color: hsl(0, 100.00%, 90.55%); opacity: 0.83" title="-0.040">a</span><span style="background-color: hsl(0, 100.00%, 89.43%); opacity: 0.83" title="-0.046">i</span><span style="background-color: hsl(0, 100.00%, 91.17%); opacity: 0.82" title="-0.036">n</span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.020">.</span><span style="background-color: hsl(0, 100.00%, 97.08%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 97.66%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(0, 100.00%, 97.91%); opacity: 0.80" title="-0.005">i</span><span style="background-color: hsl(0, 100.00%, 94.93%); opacity: 0.81" title="-0.016">t</span><span style="background-color: hsl(0, 100.00%, 95.25%); opacity: 0.81" title="-0.015">h</span><span style="background-color: hsl(0, 100.00%, 95.67%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 98.48%); opacity: 0.80" title="-0.003">r</span><span style="background-color: hsl(0, 100.00%, 94.77%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 91.87%); opacity: 0.82" title="-0.032">t</span><span style="background-color: hsl(0, 100.00%, 90.59%); opacity: 0.83" title="-0.039">h</span><span style="background-color: hsl(0, 100.00%, 91.54%); opacity: 0.82" title="-0.034">e</span><span style="background-color: hsl(0, 100.00%, 94.19%); opacity: 0.81" title="-0.020">y</span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 99.19%); opacity: 0.80" title="-0.001">p</span><span style="background-color: hsl(0, 100.00%, 96.13%); opacity: 0.81" title="-0.011">a</span><span style="background-color: hsl(0, 100.00%, 96.11%); opacity: 0.81" title="-0.011">s</span><span style="background-color: hsl(0, 100.00%, 97.62%); opacity: 0.80" title="-0.006">s</span><span style="background-color: hsl(0, 100.00%, 97.28%); opacity: 0.80" title="-0.007">,</span><span style="background-color: hsl(0, 100.00%, 96.28%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 97.64%); opacity: 0.80" title="-0.005">o</span><span style="background-color: hsl(0, 100.00%, 99.92%); opacity: 0.80" title="-0.000">r</span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 92.03%); opacity: 0.82" title="-0.031">t</span><span style="background-color: hsl(0, 100.00%, 90.24%); opacity: 0.83" title="-0.041">h</span><span style="background-color: hsl(0, 100.00%, 89.88%); opacity: 0.83" title="-0.044">e</span><span style="background-color: hsl(0, 100.00%, 91.28%); opacity: 0.82" title="-0.035">y</span><span style="background-color: hsl(0, 100.00%, 93.68%); opacity: 0.81" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 95.89%); opacity: 0.81" title="-0.012">h</span><span style="background-color: hsl(0, 100.00%, 96.83%); opacity: 0.81" title="-0.008">a</span><span style="background-color: hsl(0, 100.00%, 97.46%); opacity: 0.80" title="-0.006">v</span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(0, 100.00%, 97.23%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 96.47%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 96.16%); opacity: 0.81" title="-0.011">o</span><span style="background-color: hsl(0, 100.00%, 94.69%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 95.02%); opacity: 0.81" title="-0.016">b</span><span style="background-color: hsl(0, 100.00%, 92.92%); opacity: 0.82" title="-0.026">e</span><span style="background-color: hsl(0, 100.00%, 91.52%); opacity: 0.82" title="-0.034"> </span><span style="background-color: hsl(0, 100.00%, 90.40%); opacity: 0.83" title="-0.041">b</span><span style="background-color: hsl(0, 100.00%, 91.62%); opacity: 0.82" title="-0.033">r</span><span style="background-color: hsl(0, 100.00%, 93.50%); opacity: 0.81" title="-0.023">o</span><span style="background-color: hsl(0, 100.00%, 95.50%); opacity: 0.81" title="-0.014">k</span><span style="background-color: hsl(0, 100.00%, 95.60%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 94.95%); opacity: 0.81" title="-0.016">n</span><span style="background-color: hsl(0, 100.00%, 96.54%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 97.50%); opacity: 0.80" title="-0.006">u</span><span style="background-color: hsl(0, 100.00%, 97.27%); opacity: 0.80" title="-0.007">p</span><span style="background-color: hsl(0, 100.00%, 98.14%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 97.73%); opacity: 0.80" title="-0.005">w</span><span style="background-color: hsl(0, 100.00%, 98.33%); opacity: 0.80" title="-0.003">i</span><span style="background-color: hsl(120, 100.00%, 97.60%); opacity: 0.80" title="0.006">t</span><span style="background-color: hsl(120, 100.00%, 95.92%); opacity: 0.81" title="0.012">h</span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 95.47%); opacity: 0.81" title="0.014">s</span><span style="background-color: hsl(120, 100.00%, 98.22%); opacity: 0.80" title="0.004">o</span><span style="background-color: hsl(0, 100.00%, 97.98%); opacity: 0.80" title="-0.004">u</span><span style="background-color: hsl(0, 100.00%, 94.65%); opacity: 0.81" title="-0.018">n</span><span style="background-color: hsl(0, 100.00%, 94.55%); opacity: 0.81" title="-0.018">d</span><span style="background-color: hsl(0, 100.00%, 93.03%); opacity: 0.82" title="-0.026">,</span><span style="background-color: hsl(0, 100.00%, 93.14%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 96.64%); opacity: 0.81" title="-0.009">o</span><span style="background-color: hsl(0, 100.00%, 99.06%); opacity: 0.80" title="-0.001">r</span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 92.03%); opacity: 0.82" title="-0.031">t</span><span style="background-color: hsl(0, 100.00%, 90.24%); opacity: 0.83" title="-0.041">h</span><span style="background-color: hsl(0, 100.00%, 89.88%); opacity: 0.83" title="-0.044">e</span><span style="background-color: hsl(0, 100.00%, 91.28%); opacity: 0.82" title="-0.035">y</span><span style="background-color: hsl(0, 100.00%, 93.68%); opacity: 0.81" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 95.36%); opacity: 0.81" title="-0.014">h</span><span style="background-color: hsl(0, 100.00%, 95.81%); opacity: 0.81" title="-0.012">a</span><span style="background-color: hsl(0, 100.00%, 96.25%); opacity: 0.81" title="-0.011">v</span><span style="background-color: hsl(0, 100.00%, 94.98%); opacity: 0.81" title="-0.016">e</span><span style="background-color: hsl(0, 100.00%, 93.79%); opacity: 0.81" title="-0.022">
    </span><span style="background-color: hsl(0, 100.00%, 94.06%); opacity: 0.81" title="-0.020">t</span><span style="background-color: hsl(0, 100.00%, 93.91%); opacity: 0.81" title="-0.021">o</span><span style="background-color: hsl(0, 100.00%, 93.80%); opacity: 0.81" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 94.02%); opacity: 0.81" title="-0.021">b</span><span style="background-color: hsl(0, 100.00%, 94.16%); opacity: 0.81" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 96.04%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 96.37%); opacity: 0.81" title="-0.010">e</span><span style="background-color: hsl(0, 100.00%, 96.99%); opacity: 0.80" title="-0.008">x</span><span style="background-color: hsl(120, 100.00%, 98.48%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 95.70%); opacity: 0.81" title="0.013">r</span><span style="background-color: hsl(120, 100.00%, 95.50%); opacity: 0.81" title="0.014">a</span><span style="background-color: hsl(120, 100.00%, 96.07%); opacity: 0.81" title="0.011">c</span><span style="background-color: hsl(0, 100.00%, 96.91%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(0, 100.00%, 96.00%); opacity: 0.81" title="-0.012">e</span><span style="background-color: hsl(0, 100.00%, 94.50%); opacity: 0.81" title="-0.018">d</span><span style="background-color: hsl(0, 100.00%, 94.13%); opacity: 0.81" title="-0.020"> </span><span style="background-color: hsl(0, 100.00%, 93.00%); opacity: 0.82" title="-0.026">s</span><span style="background-color: hsl(0, 100.00%, 91.41%); opacity: 0.82" title="-0.035">u</span><span style="background-color: hsl(0, 100.00%, 91.04%); opacity: 0.82" title="-0.037">r</span><span style="background-color: hsl(0, 100.00%, 90.43%); opacity: 0.83" title="-0.040">g</span><span style="background-color: hsl(0, 100.00%, 91.43%); opacity: 0.82" title="-0.034">i</span><span style="background-color: hsl(0, 100.00%, 90.75%); opacity: 0.82" title="-0.038">c</span><span style="background-color: hsl(0, 100.00%, 91.25%); opacity: 0.82" title="-0.035">a</span><span style="background-color: hsl(0, 100.00%, 92.45%); opacity: 0.82" title="-0.029">l</span><span style="background-color: hsl(0, 100.00%, 95.28%); opacity: 0.81" title="-0.015">l</span><span style="background-color: hsl(0, 100.00%, 95.95%); opacity: 0.81" title="-0.012">y</span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.014">.</span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 94.94%); opacity: 0.81" title="-0.016">w</span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.015">h</span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.014">e</span><span style="background-color: hsl(0, 100.00%, 94.51%); opacity: 0.81" title="-0.018">n</span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 97.11%); opacity: 0.80" title="-0.007">i</span><span style="background-color: hsl(0, 100.00%, 95.22%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 94.03%); opacity: 0.81" title="-0.021">w</span><span style="background-color: hsl(0, 100.00%, 94.16%); opacity: 0.81" title="-0.020">a</span><span style="background-color: hsl(0, 100.00%, 95.13%); opacity: 0.81" title="-0.015">s</span><span style="background-color: hsl(0, 100.00%, 94.51%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 94.46%); opacity: 0.81" title="-0.018">i</span><span style="background-color: hsl(0, 100.00%, 95.22%); opacity: 0.81" title="-0.015">n</span><span style="background-color: hsl(0, 100.00%, 96.75%); opacity: 0.81" title="-0.009">,</span><span style="background-color: hsl(0, 100.00%, 93.75%); opacity: 0.81" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 92.35%); opacity: 0.82" title="-0.029">t</span><span style="background-color: hsl(0, 100.00%, 92.61%); opacity: 0.82" title="-0.028">h</span><span style="background-color: hsl(0, 100.00%, 96.63%); opacity: 0.81" title="-0.009">e</span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 94.22%); opacity: 0.81" title="0.020">x</span><span style="background-color: hsl(120, 100.00%, 97.15%); opacity: 0.80" title="0.007">-</span><span style="background-color: hsl(120, 100.00%, 90.38%); opacity: 0.83" title="0.041">r</span><span style="background-color: hsl(120, 100.00%, 90.43%); opacity: 0.83" title="0.040">a</span><span style="background-color: hsl(120, 100.00%, 90.95%); opacity: 0.82" title="0.037">y</span><span style="background-color: hsl(120, 100.00%, 91.39%); opacity: 0.82" title="0.035"> </span><span style="background-color: hsl(120, 100.00%, 91.60%); opacity: 0.82" title="0.033">t</span><span style="background-color: hsl(120, 100.00%, 92.88%); opacity: 0.82" title="0.026">e</span><span style="background-color: hsl(120, 100.00%, 93.54%); opacity: 0.81" title="0.023">c</span><span style="background-color: hsl(120, 100.00%, 98.05%); opacity: 0.80" title="0.004">h</span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 93.22%); opacity: 0.82" title="-0.025">h</span><span style="background-color: hsl(0, 100.00%, 91.85%); opacity: 0.82" title="-0.032">a</span><span style="background-color: hsl(0, 100.00%, 90.29%); opacity: 0.83" title="-0.041">p</span><span style="background-color: hsl(0, 100.00%, 91.54%); opacity: 0.82" title="-0.034">p</span><span style="background-color: hsl(0, 100.00%, 92.24%); opacity: 0.82" title="-0.030">e</span><span style="background-color: hsl(0, 100.00%, 94.28%); opacity: 0.81" title="-0.019">n</span><span style="background-color: hsl(0, 100.00%, 94.98%); opacity: 0.81" title="-0.016">e</span><span style="background-color: hsl(0, 100.00%, 94.38%); opacity: 0.81" title="-0.019">d</span><span style="background-color: hsl(0, 100.00%, 92.16%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 90.22%); opacity: 0.83" title="-0.042">t</span><span style="background-color: hsl(0, 100.00%, 89.60%); opacity: 0.83" title="-0.045">o</span><span style="background-color: hsl(0, 100.00%, 90.22%); opacity: 0.83" title="-0.042"> </span><span style="background-color: hsl(0, 100.00%, 90.52%); opacity: 0.83" title="-0.040">m</span><span style="background-color: hsl(0, 100.00%, 89.81%); opacity: 0.83" title="-0.044">e</span><span style="background-color: hsl(0, 100.00%, 89.44%); opacity: 0.83" title="-0.046">n</span><span style="background-color: hsl(0, 100.00%, 90.35%); opacity: 0.83" title="-0.041">t</span><span style="background-color: hsl(0, 100.00%, 92.74%); opacity: 0.82" title="-0.027">i</span><span style="background-color: hsl(0, 100.00%, 95.03%); opacity: 0.81" title="-0.016">o</span><span style="background-color: hsl(0, 100.00%, 94.37%); opacity: 0.81" title="-0.019">n</span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 89.84%); opacity: 0.83" title="-0.044">t</span><span style="background-color: hsl(0, 100.00%, 88.83%); opacity: 0.83" title="-0.050">h</span><span style="background-color: hsl(0, 100.00%, 89.26%); opacity: 0.83" title="-0.048">a</span><span style="background-color: hsl(0, 100.00%, 88.76%); opacity: 0.83" title="-0.051">t</span><span style="background-color: hsl(0, 100.00%, 88.23%); opacity: 0.83" title="-0.054"> </span><span style="background-color: hsl(0, 100.00%, 88.15%); opacity: 0.84" title="-0.055">s</span><span style="background-color: hsl(0, 100.00%, 89.96%); opacity: 0.83" title="-0.043">h</span><span style="background-color: hsl(0, 100.00%, 93.77%); opacity: 0.81" title="-0.022">e</span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.004">'</span><span style="background-color: hsl(120, 100.00%, 97.25%); opacity: 0.80" title="0.007">d</span><span style="background-color: hsl(0, 100.00%, 99.34%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 99.17%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 97.66%); opacity: 0.80" title="-0.005">a</span><span style="background-color: hsl(0, 100.00%, 96.41%); opacity: 0.81" title="-0.010">d</span><span style="background-color: hsl(0, 100.00%, 94.07%); opacity: 0.81" title="-0.020"> </span><span style="background-color: hsl(0, 100.00%, 93.75%); opacity: 0.81" title="-0.022">k</span><span style="background-color: hsl(0, 100.00%, 94.12%); opacity: 0.81" title="-0.020">i</span><span style="background-color: hsl(0, 100.00%, 94.08%); opacity: 0.81" title="-0.020">d</span><span style="background-color: hsl(0, 100.00%, 96.07%); opacity: 0.81" title="-0.011">n</span><span style="background-color: hsl(0, 100.00%, 95.40%); opacity: 0.81" title="-0.014">e</span><span style="background-color: hsl(0, 100.00%, 97.37%); opacity: 0.80" title="-0.006">y</span><span style="background-color: hsl(0, 100.00%, 96.02%); opacity: 0.81" title="-0.012">
    </span><span style="background-color: hsl(0, 100.00%, 97.84%); opacity: 0.80" title="-0.005">s</span><span style="background-color: hsl(0, 100.00%, 94.29%); opacity: 0.81" title="-0.019">t</span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.019">o</span><span style="background-color: hsl(0, 100.00%, 94.92%); opacity: 0.81" title="-0.016">n</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(0, 100.00%, 99.79%); opacity: 0.80" title="-0.000">s</span><span style="background-color: hsl(120, 100.00%, 98.52%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(0, 100.00%, 96.91%); opacity: 0.81" title="-0.008">a</span><span style="background-color: hsl(0, 100.00%, 95.46%); opacity: 0.81" title="-0.014">n</span><span style="background-color: hsl(0, 100.00%, 94.08%); opacity: 0.81" title="-0.020">d</span><span style="background-color: hsl(0, 100.00%, 90.81%); opacity: 0.82" title="-0.038"> </span><span style="background-color: hsl(0, 100.00%, 89.88%); opacity: 0.83" title="-0.044">c</span><span style="background-color: hsl(0, 100.00%, 87.92%); opacity: 0.84" title="-0.056">h</span><span style="background-color: hsl(0, 100.00%, 88.76%); opacity: 0.83" title="-0.051">i</span><span style="background-color: hsl(0, 100.00%, 87.93%); opacity: 0.84" title="-0.056">l</span><span style="background-color: hsl(0, 100.00%, 88.92%); opacity: 0.83" title="-0.050">d</span><span style="background-color: hsl(0, 100.00%, 91.73%); opacity: 0.82" title="-0.033">r</span><span style="background-color: hsl(0, 100.00%, 92.68%); opacity: 0.82" title="-0.027">e</span><span style="background-color: hsl(0, 100.00%, 94.43%); opacity: 0.81" title="-0.019">n</span><span style="background-color: hsl(0, 100.00%, 95.20%); opacity: 0.81" title="-0.015">,</span><span style="background-color: hsl(0, 100.00%, 94.55%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 93.63%); opacity: 0.81" title="-0.023">a</span><span style="background-color: hsl(0, 100.00%, 93.94%); opacity: 0.81" title="-0.021">n</span><span style="background-color: hsl(0, 100.00%, 93.85%); opacity: 0.81" title="-0.021">d</span><span style="background-color: hsl(0, 100.00%, 92.58%); opacity: 0.82" title="-0.028"> </span><span style="background-color: hsl(0, 100.00%, 91.59%); opacity: 0.82" title="-0.034">t</span><span style="background-color: hsl(0, 100.00%, 89.95%); opacity: 0.83" title="-0.043">h</span><span style="background-color: hsl(0, 100.00%, 89.85%); opacity: 0.83" title="-0.044">e</span><span style="background-color: hsl(0, 100.00%, 87.75%); opacity: 0.84" title="-0.057"> </span><span style="background-color: hsl(0, 100.00%, 87.67%); opacity: 0.84" title="-0.058">c</span><span style="background-color: hsl(0, 100.00%, 86.55%); opacity: 0.84" title="-0.066">h</span><span style="background-color: hsl(0, 100.00%, 89.76%); opacity: 0.83" title="-0.044">i</span><span style="background-color: hsl(0, 100.00%, 90.74%); opacity: 0.82" title="-0.038">l</span><span style="background-color: hsl(0, 100.00%, 93.74%); opacity: 0.81" title="-0.022">d</span><span style="background-color: hsl(0, 100.00%, 96.76%); opacity: 0.81" title="-0.009">b</span><span style="background-color: hsl(0, 100.00%, 99.05%); opacity: 0.80" title="-0.001">i</span><span style="background-color: hsl(0, 100.00%, 98.19%); opacity: 0.80" title="-0.004">r</span><span style="background-color: hsl(0, 100.00%, 98.61%); opacity: 0.80" title="-0.003">t</span><span style="background-color: hsl(0, 100.00%, 95.62%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 94.99%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 92.50%); opacity: 0.82" title="-0.028">h</span><span style="background-color: hsl(0, 100.00%, 93.00%); opacity: 0.82" title="-0.026">u</span><span style="background-color: hsl(0, 100.00%, 94.98%); opacity: 0.81" title="-0.016">r</span><span style="background-color: hsl(0, 100.00%, 97.52%); opacity: 0.80" title="-0.006">t</span><span style="background-color: hsl(0, 100.00%, 95.95%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.010">l</span><span style="background-color: hsl(0, 100.00%, 95.33%); opacity: 0.81" title="-0.014">e</span><span style="background-color: hsl(0, 100.00%, 96.02%); opacity: 0.81" title="-0.012">s</span><span style="background-color: hsl(0, 100.00%, 96.18%); opacity: 0.81" title="-0.011">s</span><span style="background-color: hsl(0, 100.00%, 98.66%); opacity: 0.80" title="-0.002">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.963</b>, score <b>-0.656</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 84.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +4.493
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 82.97%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.149
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 98.28%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.003">s</span><span style="background-color: hsl(0, 100.00%, 99.25%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.015">i</span><span style="background-color: hsl(120, 100.00%, 89.44%); opacity: 0.83" title="0.046"> </span><span style="background-color: hsl(120, 100.00%, 84.25%); opacity: 0.85" title="0.082">r</span><span style="background-color: hsl(120, 100.00%, 80.80%); opacity: 0.87" title="0.109">e</span><span style="background-color: hsl(120, 100.00%, 78.36%); opacity: 0.88" title="0.129">c</span><span style="background-color: hsl(120, 100.00%, 83.29%); opacity: 0.86" title="0.089">a</span><span style="background-color: hsl(120, 100.00%, 87.84%); opacity: 0.84" title="0.057">l</span><span style="background-color: hsl(120, 100.00%, 93.95%); opacity: 0.81" title="0.021">l</span><span style="background-color: hsl(0, 100.00%, 96.44%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 95.31%); opacity: 0.81" title="-0.015">f</span><span style="background-color: hsl(0, 100.00%, 92.84%); opacity: 0.82" title="-0.027">r</span><span style="background-color: hsl(0, 100.00%, 95.40%); opacity: 0.81" title="-0.014">o</span><span style="background-color: hsl(0, 100.00%, 96.73%); opacity: 0.81" title="-0.009">m</span><span style="background-color: hsl(120, 100.00%, 90.17%); opacity: 0.83" title="0.042"> </span><span style="background-color: hsl(120, 100.00%, 84.87%); opacity: 0.85" title="0.078">m</span><span style="background-color: hsl(120, 100.00%, 85.23%); opacity: 0.85" title="0.075">y</span><span style="background-color: hsl(120, 100.00%, 91.17%); opacity: 0.82" title="0.036"> </span><span style="background-color: hsl(120, 100.00%, 97.52%); opacity: 0.80" title="0.006">b</span><span style="background-color: hsl(0, 100.00%, 95.85%); opacity: 0.81" title="-0.012">o</span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.003">u</span><span style="background-color: hsl(120, 100.00%, 96.86%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(0, 100.00%, 98.32%); opacity: 0.80" title="-0.003">i</span><span style="background-color: hsl(0, 100.00%, 97.02%); opacity: 0.80" title="-0.008">t</span><span style="background-color: hsl(0, 100.00%, 94.66%); opacity: 0.81" title="-0.018">h</span><span style="background-color: hsl(0, 100.00%, 97.67%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 93.07%); opacity: 0.82" title="0.025">k</span><span style="background-color: hsl(120, 100.00%, 89.01%); opacity: 0.83" title="0.049">i</span><span style="background-color: hsl(120, 100.00%, 83.35%); opacity: 0.86" title="0.089">d</span><span style="background-color: hsl(120, 100.00%, 81.66%); opacity: 0.87" title="0.102">n</span><span style="background-color: hsl(120, 100.00%, 81.15%); opacity: 0.87" title="0.106">e</span><span style="background-color: hsl(120, 100.00%, 84.55%); opacity: 0.85" title="0.080">y</span><span style="background-color: hsl(120, 100.00%, 88.38%); opacity: 0.83" title="0.053"> </span><span style="background-color: hsl(120, 100.00%, 90.60%); opacity: 0.83" title="0.039">s</span><span style="background-color: hsl(120, 100.00%, 94.77%); opacity: 0.81" title="0.017">t</span><span style="background-color: hsl(0, 100.00%, 99.66%); opacity: 0.80" title="-0.000">o</span><span style="background-color: hsl(120, 100.00%, 95.47%); opacity: 0.81" title="0.014">n</span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 96.27%); opacity: 0.81" title="0.010">s</span><span style="background-color: hsl(0, 100.00%, 91.30%); opacity: 0.82" title="-0.035">,</span><span style="background-color: hsl(0, 100.00%, 90.49%); opacity: 0.83" title="-0.040"> </span><span style="background-color: hsl(0, 100.00%, 88.10%); opacity: 0.84" title="-0.055">t</span><span style="background-color: hsl(0, 100.00%, 87.21%); opacity: 0.84" title="-0.061">h</span><span style="background-color: hsl(0, 100.00%, 88.18%); opacity: 0.84" title="-0.055">e</span><span style="background-color: hsl(0, 100.00%, 91.85%); opacity: 0.82" title="-0.032">r</span><span style="background-color: hsl(0, 100.00%, 96.52%); opacity: 0.81" title="-0.009">e</span><span style="background-color: hsl(0, 100.00%, 98.37%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(0, 100.00%, 96.68%); opacity: 0.81" title="-0.009">i</span><span style="background-color: hsl(0, 100.00%, 92.56%); opacity: 0.82" title="-0.028">s</span><span style="background-color: hsl(0, 100.00%, 91.27%); opacity: 0.82" title="-0.035">n</span><span style="background-color: hsl(0, 100.00%, 91.50%); opacity: 0.82" title="-0.034">'</span><span style="background-color: hsl(0, 100.00%, 90.73%); opacity: 0.82" title="-0.039">t</span><span style="background-color: hsl(0, 100.00%, 92.23%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 93.37%); opacity: 0.82" title="-0.024">a</span><span style="background-color: hsl(0, 100.00%, 96.86%); opacity: 0.81" title="-0.008">n</span><span style="background-color: hsl(120, 100.00%, 98.13%); opacity: 0.80" title="0.004">y</span><span style="background-color: hsl(120, 100.00%, 90.29%); opacity: 0.83" title="0.041">
    </span><span style="background-color: hsl(120, 100.00%, 76.97%); opacity: 0.89" title="0.141">m</span><span style="background-color: hsl(120, 100.00%, 65.95%); opacity: 0.96" title="0.247">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.311">d</span><span style="background-color: hsl(120, 100.00%, 62.52%); opacity: 0.98" title="0.283">i</span><span style="background-color: hsl(120, 100.00%, 69.10%); opacity: 0.94" title="0.215">c</span><span style="background-color: hsl(120, 100.00%, 81.75%); opacity: 0.87" title="0.101">a</span><span style="background-color: hsl(120, 100.00%, 92.20%); opacity: 0.82" title="0.030">t</span><span style="background-color: hsl(0, 100.00%, 98.33%); opacity: 0.80" title="-0.003">i</span><span style="background-color: hsl(0, 100.00%, 98.18%); opacity: 0.80" title="-0.004">o</span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(0, 100.00%, 97.59%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 96.17%); opacity: 0.81" title="-0.011">t</span><span style="background-color: hsl(0, 100.00%, 94.23%); opacity: 0.81" title="-0.020">h</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 92.22%); opacity: 0.82" title="0.030">t</span><span style="background-color: hsl(120, 100.00%, 88.24%); opacity: 0.83" title="0.054"> </span><span style="background-color: hsl(120, 100.00%, 86.22%); opacity: 0.84" title="0.068">c</span><span style="background-color: hsl(120, 100.00%, 92.17%); opacity: 0.82" title="0.030">a</span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.012">n</span><span style="background-color: hsl(0, 100.00%, 92.16%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 91.24%); opacity: 0.82" title="-0.036">d</span><span style="background-color: hsl(0, 100.00%, 96.67%); opacity: 0.81" title="-0.009">o</span><span style="background-color: hsl(0, 100.00%, 95.43%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 94.93%); opacity: 0.81" title="-0.016">a</span><span style="background-color: hsl(0, 100.00%, 90.22%); opacity: 0.83" title="-0.042">n</span><span style="background-color: hsl(0, 100.00%, 86.36%); opacity: 0.84" title="-0.067">y</span><span style="background-color: hsl(0, 100.00%, 88.16%); opacity: 0.84" title="-0.055">t</span><span style="background-color: hsl(0, 100.00%, 91.77%); opacity: 0.82" title="-0.032">h</span><span style="background-color: hsl(0, 100.00%, 94.43%); opacity: 0.81" title="-0.019">i</span><span style="background-color: hsl(0, 100.00%, 94.77%); opacity: 0.81" title="-0.017">n</span><span style="background-color: hsl(0, 100.00%, 92.32%); opacity: 0.82" title="-0.029">g</span><span style="background-color: hsl(0, 100.00%, 96.23%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 97.01%); opacity: 0.80" title="0.008">a</span><span style="background-color: hsl(120, 100.00%, 93.91%); opacity: 0.81" title="0.021">b</span><span style="background-color: hsl(120, 100.00%, 92.79%); opacity: 0.82" title="0.027">o</span><span style="background-color: hsl(120, 100.00%, 95.62%); opacity: 0.81" title="0.013">u</span><span style="background-color: hsl(0, 100.00%, 95.91%); opacity: 0.81" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 91.29%); opacity: 0.82" title="-0.035"> </span><span style="background-color: hsl(0, 100.00%, 87.68%); opacity: 0.84" title="-0.058">t</span><span style="background-color: hsl(0, 100.00%, 90.11%); opacity: 0.83" title="-0.042">h</span><span style="background-color: hsl(0, 100.00%, 92.38%); opacity: 0.82" title="-0.029">e</span><span style="background-color: hsl(120, 100.00%, 96.29%); opacity: 0.81" title="0.010">m</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 95.01%); opacity: 0.81" title="0.016">e</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">x</span><span style="background-color: hsl(0, 100.00%, 96.49%); opacity: 0.81" title="-0.010">c</span><span style="background-color: hsl(120, 100.00%, 94.97%); opacity: 0.81" title="0.016">e</span><span style="background-color: hsl(120, 100.00%, 96.99%); opacity: 0.80" title="0.008">p</span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.009">t</span><span style="background-color: hsl(0, 100.00%, 94.97%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 90.51%); opacity: 0.83" title="-0.040">r</span><span style="background-color: hsl(0, 100.00%, 88.00%); opacity: 0.84" title="-0.056">e</span><span style="background-color: hsl(0, 100.00%, 86.87%); opacity: 0.84" title="-0.063">l</span><span style="background-color: hsl(0, 100.00%, 89.81%); opacity: 0.83" title="-0.044">i</span><span style="background-color: hsl(0, 100.00%, 97.41%); opacity: 0.80" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 97.27%); opacity: 0.80" title="-0.007">v</span><span style="background-color: hsl(120, 100.00%, 96.62%); opacity: 0.81" title="0.009">e</span><span style="background-color: hsl(0, 100.00%, 98.45%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(0, 100.00%, 94.59%); opacity: 0.81" title="-0.018">t</span><span style="background-color: hsl(0, 100.00%, 92.70%); opacity: 0.82" title="-0.027">h</span><span style="background-color: hsl(0, 100.00%, 95.07%); opacity: 0.81" title="-0.016">e</span><span style="background-color: hsl(120, 100.00%, 90.81%); opacity: 0.82" title="0.038"> </span><span style="background-color: hsl(120, 100.00%, 80.66%); opacity: 0.87" title="0.110">p</span><span style="background-color: hsl(120, 100.00%, 77.30%); opacity: 0.89" title="0.138">a</span><span style="background-color: hsl(120, 100.00%, 75.29%); opacity: 0.90" title="0.156">i</span><span style="background-color: hsl(120, 100.00%, 81.56%); opacity: 0.87" title="0.103">n</span><span style="background-color: hsl(120, 100.00%, 89.37%); opacity: 0.83" title="0.047">.</span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.007"> </span><span style="background-color: hsl(0, 100.00%, 93.81%); opacity: 0.81" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 91.13%); opacity: 0.82" title="-0.036">i</span><span style="background-color: hsl(0, 100.00%, 88.85%); opacity: 0.83" title="-0.050">t</span><span style="background-color: hsl(0, 100.00%, 88.42%); opacity: 0.83" title="-0.053">h</span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.036">e</span><span style="background-color: hsl(0, 100.00%, 94.22%); opacity: 0.81" title="-0.020">r</span><span style="background-color: hsl(0, 100.00%, 95.43%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 93.04%); opacity: 0.82" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.017">h</span><span style="background-color: hsl(0, 100.00%, 99.08%); opacity: 0.80" title="-0.001">e</span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.017">y</span><span style="background-color: hsl(0, 100.00%, 98.76%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(0, 100.00%, 91.58%); opacity: 0.82" title="-0.034">p</span><span style="background-color: hsl(0, 100.00%, 88.89%); opacity: 0.83" title="-0.050">a</span><span style="background-color: hsl(0, 100.00%, 87.98%); opacity: 0.84" title="-0.056">s</span><span style="background-color: hsl(0, 100.00%, 91.25%); opacity: 0.82" title="-0.035">s</span><span style="background-color: hsl(0, 100.00%, 99.59%); opacity: 0.80" title="-0.000">,</span><span style="background-color: hsl(120, 100.00%, 98.16%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 97.29%); opacity: 0.80" title="0.007">o</span><span style="background-color: hsl(120, 100.00%, 97.83%); opacity: 0.80" title="0.005">r</span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.018">t</span><span style="background-color: hsl(0, 100.00%, 95.85%); opacity: 0.81" title="-0.012">h</span><span style="background-color: hsl(120, 100.00%, 97.65%); opacity: 0.80" title="0.005">e</span><span style="background-color: hsl(120, 100.00%, 92.14%); opacity: 0.82" title="0.030">y</span><span style="background-color: hsl(120, 100.00%, 94.79%); opacity: 0.81" title="0.017"> </span><span style="background-color: hsl(0, 100.00%, 98.77%); opacity: 0.80" title="-0.002">h</span><span style="background-color: hsl(0, 100.00%, 92.76%); opacity: 0.82" title="-0.027">a</span><span style="background-color: hsl(0, 100.00%, 91.58%); opacity: 0.82" title="-0.034">v</span><span style="background-color: hsl(0, 100.00%, 91.23%); opacity: 0.82" title="-0.036">e</span><span style="background-color: hsl(0, 100.00%, 95.20%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 95.70%); opacity: 0.81" title="-0.013">t</span><span style="background-color: hsl(0, 100.00%, 91.34%); opacity: 0.82" title="-0.035">o</span><span style="background-color: hsl(0, 100.00%, 88.28%); opacity: 0.83" title="-0.054"> </span><span style="background-color: hsl(0, 100.00%, 86.79%); opacity: 0.84" title="-0.064">b</span><span style="background-color: hsl(0, 100.00%, 90.17%); opacity: 0.83" title="-0.042">e</span><span style="background-color: hsl(0, 100.00%, 97.51%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(120, 100.00%, 96.69%); opacity: 0.81" title="0.009">b</span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.017">r</span><span style="background-color: hsl(120, 100.00%, 96.65%); opacity: 0.81" title="0.009">o</span><span style="background-color: hsl(0, 100.00%, 96.95%); opacity: 0.81" title="-0.008">k</span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.012">e</span><span style="background-color: hsl(0, 100.00%, 91.16%); opacity: 0.82" title="-0.036">n</span><span style="background-color: hsl(0, 100.00%, 94.51%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(120, 100.00%, 97.59%); opacity: 0.80" title="0.006">u</span><span style="background-color: hsl(120, 100.00%, 91.80%); opacity: 0.82" title="0.032">p</span><span style="background-color: hsl(120, 100.00%, 89.46%); opacity: 0.83" title="0.046"> </span><span style="background-color: hsl(120, 100.00%, 89.68%); opacity: 0.83" title="0.045">w</span><span style="background-color: hsl(120, 100.00%, 96.80%); opacity: 0.81" title="0.008">i</span><span style="background-color: hsl(0, 100.00%, 95.59%); opacity: 0.81" title="-0.013">t</span><span style="background-color: hsl(0, 100.00%, 91.59%); opacity: 0.82" title="-0.034">h</span><span style="background-color: hsl(0, 100.00%, 92.87%); opacity: 0.82" title="-0.026"> </span><span style="background-color: hsl(120, 100.00%, 95.40%); opacity: 0.81" title="0.014">s</span><span style="background-color: hsl(120, 100.00%, 91.14%); opacity: 0.82" title="0.036">o</span><span style="background-color: hsl(120, 100.00%, 90.99%); opacity: 0.82" title="0.037">u</span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.012">n</span><span style="background-color: hsl(0, 100.00%, 94.70%); opacity: 0.81" title="-0.017">d</span><span style="background-color: hsl(0, 100.00%, 95.21%); opacity: 0.81" title="-0.015">,</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.005">o</span><span style="background-color: hsl(120, 100.00%, 98.14%); opacity: 0.80" title="0.004">r</span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.018">t</span><span style="background-color: hsl(0, 100.00%, 95.85%); opacity: 0.81" title="-0.012">h</span><span style="background-color: hsl(120, 100.00%, 97.65%); opacity: 0.80" title="0.005">e</span><span style="background-color: hsl(120, 100.00%, 92.14%); opacity: 0.82" title="0.030">y</span><span style="background-color: hsl(120, 100.00%, 94.79%); opacity: 0.81" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 97.62%); opacity: 0.80" title="0.006">h</span><span style="background-color: hsl(0, 100.00%, 98.59%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(120, 100.00%, 96.03%); opacity: 0.81" title="0.011">v</span><span style="background-color: hsl(120, 100.00%, 93.63%); opacity: 0.81" title="0.023">e</span><span style="background-color: hsl(120, 100.00%, 92.62%); opacity: 0.82" title="0.028">
    </span><span style="background-color: hsl(0, 100.00%, 99.15%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(0, 100.00%, 92.86%); opacity: 0.82" title="-0.027">o</span><span style="background-color: hsl(0, 100.00%, 90.06%); opacity: 0.83" title="-0.043"> </span><span style="background-color: hsl(0, 100.00%, 90.66%); opacity: 0.83" title="-0.039">b</span><span style="background-color: hsl(0, 100.00%, 93.87%); opacity: 0.81" title="-0.021">e</span><span style="background-color: hsl(0, 100.00%, 98.12%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 93.96%); opacity: 0.81" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 91.70%); opacity: 0.82" title="0.033">x</span><span style="background-color: hsl(120, 100.00%, 89.58%); opacity: 0.83" title="0.046">t</span><span style="background-color: hsl(120, 100.00%, 88.17%); opacity: 0.84" title="0.055">r</span><span style="background-color: hsl(120, 100.00%, 88.53%); opacity: 0.83" title="0.052">a</span><span style="background-color: hsl(120, 100.00%, 90.36%); opacity: 0.83" title="0.041">c</span><span style="background-color: hsl(120, 100.00%, 87.76%); opacity: 0.84" title="0.057">t</span><span style="background-color: hsl(120, 100.00%, 86.91%); opacity: 0.84" title="0.063">e</span><span style="background-color: hsl(120, 100.00%, 87.22%); opacity: 0.84" title="0.061">d</span><span style="background-color: hsl(120, 100.00%, 82.45%); opacity: 0.86" title="0.096"> </span><span style="background-color: hsl(120, 100.00%, 76.23%); opacity: 0.90" title="0.148">s</span><span style="background-color: hsl(120, 100.00%, 73.46%); opacity: 0.91" title="0.173">u</span><span style="background-color: hsl(120, 100.00%, 73.14%); opacity: 0.91" title="0.176">r</span><span style="background-color: hsl(120, 100.00%, 77.85%); opacity: 0.89" title="0.134">g</span><span style="background-color: hsl(120, 100.00%, 84.78%); opacity: 0.85" title="0.078">i</span><span style="background-color: hsl(120, 100.00%, 84.66%); opacity: 0.85" title="0.079">c</span><span style="background-color: hsl(120, 100.00%, 86.90%); opacity: 0.84" title="0.063">a</span><span style="background-color: hsl(120, 100.00%, 88.90%); opacity: 0.83" title="0.050">l</span><span style="background-color: hsl(120, 100.00%, 93.36%); opacity: 0.82" title="0.024">l</span><span style="background-color: hsl(120, 100.00%, 94.60%); opacity: 0.81" title="0.018">y</span><span style="background-color: hsl(120, 100.00%, 95.10%); opacity: 0.81" title="0.015">.</span><span style="background-color: hsl(120, 100.00%, 96.61%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 93.51%); opacity: 0.81" title="0.023">w</span><span style="background-color: hsl(120, 100.00%, 92.73%); opacity: 0.82" title="0.027">h</span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.047">e</span><span style="background-color: hsl(120, 100.00%, 87.17%); opacity: 0.84" title="0.061">n</span><span style="background-color: hsl(120, 100.00%, 87.81%); opacity: 0.84" title="0.057"> </span><span style="background-color: hsl(120, 100.00%, 87.19%); opacity: 0.84" title="0.061">i</span><span style="background-color: hsl(120, 100.00%, 88.67%); opacity: 0.83" title="0.051"> </span><span style="background-color: hsl(120, 100.00%, 93.01%); opacity: 0.82" title="0.026">w</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.009">a</span><span style="background-color: hsl(120, 100.00%, 98.77%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 94.44%); opacity: 0.81" title="0.019"> </span><span style="background-color: hsl(120, 100.00%, 94.29%); opacity: 0.81" title="0.019">i</span><span style="background-color: hsl(0, 100.00%, 96.67%); opacity: 0.81" title="-0.009">n</span><span style="background-color: hsl(0, 100.00%, 89.19%); opacity: 0.83" title="-0.048">,</span><span style="background-color: hsl(0, 100.00%, 87.09%); opacity: 0.84" title="-0.062"> </span><span style="background-color: hsl(0, 100.00%, 86.07%); opacity: 0.84" title="-0.069">t</span><span style="background-color: hsl(0, 100.00%, 88.11%); opacity: 0.84" title="-0.055">h</span><span style="background-color: hsl(0, 100.00%, 89.74%); opacity: 0.83" title="-0.045">e</span><span style="background-color: hsl(0, 100.00%, 96.10%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 97.17%); opacity: 0.80" title="0.007">x</span><span style="background-color: hsl(120, 100.00%, 90.33%); opacity: 0.83" title="0.041">-</span><span style="background-color: hsl(0, 100.00%, 93.84%); opacity: 0.81" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 89.76%); opacity: 0.83" title="-0.044">a</span><span style="background-color: hsl(0, 100.00%, 87.41%); opacity: 0.84" title="-0.060">y</span><span style="background-color: hsl(0, 100.00%, 89.70%); opacity: 0.83" title="-0.045"> </span><span style="background-color: hsl(0, 100.00%, 95.88%); opacity: 0.81" title="-0.012">t</span><span style="background-color: hsl(120, 100.00%, 92.86%); opacity: 0.82" title="0.027">e</span><span style="background-color: hsl(120, 100.00%, 92.96%); opacity: 0.82" title="0.026">c</span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.012">h</span><span style="background-color: hsl(0, 100.00%, 98.68%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 95.17%); opacity: 0.81" title="0.015">h</span><span style="background-color: hsl(120, 100.00%, 90.84%); opacity: 0.82" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 86.57%); opacity: 0.84" title="0.065">p</span><span style="background-color: hsl(120, 100.00%, 88.71%); opacity: 0.83" title="0.051">p</span><span style="background-color: hsl(120, 100.00%, 91.92%); opacity: 0.82" title="0.032">e</span><span style="background-color: hsl(0, 100.00%, 97.74%); opacity: 0.80" title="-0.005">n</span><span style="background-color: hsl(0, 100.00%, 96.48%); opacity: 0.81" title="-0.010">e</span><span style="background-color: hsl(120, 100.00%, 96.60%); opacity: 0.81" title="0.009">d</span><span style="background-color: hsl(120, 100.00%, 90.42%); opacity: 0.83" title="0.040"> </span><span style="background-color: hsl(120, 100.00%, 87.31%); opacity: 0.84" title="0.060">t</span><span style="background-color: hsl(120, 100.00%, 85.64%); opacity: 0.85" title="0.072">o</span><span style="background-color: hsl(120, 100.00%, 85.92%); opacity: 0.85" title="0.070"> </span><span style="background-color: hsl(120, 100.00%, 87.09%); opacity: 0.84" title="0.062">m</span><span style="background-color: hsl(120, 100.00%, 86.43%); opacity: 0.84" title="0.066">e</span><span style="background-color: hsl(120, 100.00%, 88.57%); opacity: 0.83" title="0.052">n</span><span style="background-color: hsl(120, 100.00%, 88.98%); opacity: 0.83" title="0.049">t</span><span style="background-color: hsl(120, 100.00%, 92.46%); opacity: 0.82" title="0.029">i</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.003">o</span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.007">n</span><span style="background-color: hsl(0, 100.00%, 97.59%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 96.17%); opacity: 0.81" title="-0.011">t</span><span style="background-color: hsl(0, 100.00%, 92.99%); opacity: 0.82" title="-0.026">h</span><span style="background-color: hsl(0, 100.00%, 94.53%); opacity: 0.81" title="-0.018">a</span><span style="background-color: hsl(120, 100.00%, 97.65%); opacity: 0.80" title="0.005">t</span><span style="background-color: hsl(120, 100.00%, 87.43%); opacity: 0.84" title="0.060"> </span><span style="background-color: hsl(120, 100.00%, 84.16%); opacity: 0.85" title="0.083">s</span><span style="background-color: hsl(120, 100.00%, 85.53%); opacity: 0.85" title="0.073">h</span><span style="background-color: hsl(120, 100.00%, 90.70%); opacity: 0.82" title="0.039">e</span><span style="background-color: hsl(0, 100.00%, 97.89%); opacity: 0.80" title="-0.005">'</span><span style="background-color: hsl(0, 100.00%, 92.16%); opacity: 0.82" title="-0.030">d</span><span style="background-color: hsl(0, 100.00%, 93.82%); opacity: 0.81" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 96.74%); opacity: 0.81" title="-0.009">h</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.002">a</span><span style="background-color: hsl(120, 100.00%, 97.09%); opacity: 0.80" title="0.007">d</span><span style="background-color: hsl(120, 100.00%, 95.24%); opacity: 0.81" title="0.015"> </span><span style="background-color: hsl(120, 100.00%, 92.08%); opacity: 0.82" title="0.031">k</span><span style="background-color: hsl(120, 100.00%, 88.36%); opacity: 0.83" title="0.053">i</span><span style="background-color: hsl(120, 100.00%, 84.38%); opacity: 0.85" title="0.081">d</span><span style="background-color: hsl(120, 100.00%, 84.33%); opacity: 0.85" title="0.082">n</span><span style="background-color: hsl(120, 100.00%, 86.89%); opacity: 0.84" title="0.063">e</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.011">y</span><span style="background-color: hsl(0, 100.00%, 95.06%); opacity: 0.81" title="-0.016">
    </span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.013">s</span><span style="background-color: hsl(0, 100.00%, 98.48%); opacity: 0.80" title="-0.003">t</span><span style="background-color: hsl(120, 100.00%, 97.56%); opacity: 0.80" title="0.006">o</span><span style="background-color: hsl(120, 100.00%, 91.38%); opacity: 0.82" title="0.035">n</span><span style="background-color: hsl(120, 100.00%, 91.43%); opacity: 0.82" title="0.034">e</span><span style="background-color: hsl(120, 100.00%, 87.30%); opacity: 0.84" title="0.060">s</span><span style="background-color: hsl(120, 100.00%, 89.78%); opacity: 0.83" title="0.044"> </span><span style="background-color: hsl(120, 100.00%, 90.61%); opacity: 0.83" title="0.039">a</span><span style="background-color: hsl(120, 100.00%, 89.90%); opacity: 0.83" title="0.044">n</span><span style="background-color: hsl(120, 100.00%, 89.72%); opacity: 0.83" title="0.045">d</span><span style="background-color: hsl(120, 100.00%, 92.15%); opacity: 0.82" title="0.030"> </span><span style="background-color: hsl(120, 100.00%, 93.91%); opacity: 0.81" title="0.021">c</span><span style="background-color: hsl(0, 100.00%, 99.26%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 95.73%); opacity: 0.81" title="-0.013">i</span><span style="background-color: hsl(0, 100.00%, 89.40%); opacity: 0.83" title="-0.047">l</span><span style="background-color: hsl(0, 100.00%, 91.18%); opacity: 0.82" title="-0.036">d</span><span style="background-color: hsl(0, 100.00%, 93.93%); opacity: 0.81" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 92.95%); opacity: 0.82" title="-0.026">e</span><span style="background-color: hsl(0, 100.00%, 91.94%); opacity: 0.82" title="-0.032">n</span><span style="background-color: hsl(0, 100.00%, 94.51%); opacity: 0.81" title="-0.018">,</span><span style="background-color: hsl(120, 100.00%, 98.23%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 92.80%); opacity: 0.82" title="0.027">a</span><span style="background-color: hsl(120, 100.00%, 94.40%); opacity: 0.81" title="0.019">n</span><span style="background-color: hsl(120, 100.00%, 95.10%); opacity: 0.81" title="0.015">d</span><span style="background-color: hsl(0, 100.00%, 97.59%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 92.87%); opacity: 0.82" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 93.08%); opacity: 0.82" title="-0.025">h</span><span style="background-color: hsl(0, 100.00%, 96.79%); opacity: 0.81" title="-0.008">e</span><span style="background-color: hsl(120, 100.00%, 97.54%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 95.89%); opacity: 0.81" title="0.012">c</span><span style="background-color: hsl(120, 100.00%, 98.21%); opacity: 0.80" title="0.004">h</span><span style="background-color: hsl(120, 100.00%, 97.37%); opacity: 0.80" title="0.006">i</span><span style="background-color: hsl(0, 100.00%, 97.18%); opacity: 0.80" title="-0.007">l</span><span style="background-color: hsl(120, 100.00%, 94.41%); opacity: 0.81" title="0.019">d</span><span style="background-color: hsl(120, 100.00%, 88.80%); opacity: 0.83" title="0.050">b</span><span style="background-color: hsl(120, 100.00%, 87.77%); opacity: 0.84" title="0.057">i</span><span style="background-color: hsl(120, 100.00%, 91.85%); opacity: 0.82" title="0.032">r</span><span style="background-color: hsl(120, 100.00%, 95.71%); opacity: 0.81" title="0.013">t</span><span style="background-color: hsl(0, 100.00%, 99.14%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 95.32%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 90.17%); opacity: 0.83" title="-0.042">h</span><span style="background-color: hsl(0, 100.00%, 89.94%); opacity: 0.83" title="-0.043">u</span><span style="background-color: hsl(0, 100.00%, 91.09%); opacity: 0.82" title="-0.036">r</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 93.86%); opacity: 0.81" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 92.93%); opacity: 0.82" title="0.026">l</span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.035">e</span><span style="background-color: hsl(120, 100.00%, 91.77%); opacity: 0.82" title="0.032">s</span><span style="background-color: hsl(120, 100.00%, 93.65%); opacity: 0.81" title="0.022">s</span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.007">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=soc.religion.christian
        
    </b>
    
        
        (probability <b>0.018</b>, score <b>-5.048</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 96.22%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.600
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 81.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.648
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 96.63%); opacity: 0.81" title="0.009">a</span><span style="background-color: hsl(120, 100.00%, 97.81%); opacity: 0.80" title="0.005">s</span><span style="background-color: hsl(120, 100.00%, 97.15%); opacity: 0.80" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 97.17%); opacity: 0.80" title="0.007">i</span><span style="background-color: hsl(120, 100.00%, 96.81%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(0, 100.00%, 93.88%); opacity: 0.81" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 93.63%); opacity: 0.81" title="-0.023">e</span><span style="background-color: hsl(0, 100.00%, 90.90%); opacity: 0.82" title="-0.038">c</span><span style="background-color: hsl(0, 100.00%, 90.08%); opacity: 0.83" title="-0.042">a</span><span style="background-color: hsl(0, 100.00%, 91.23%); opacity: 0.82" title="-0.036">l</span><span style="background-color: hsl(0, 100.00%, 92.25%); opacity: 0.82" title="-0.030">l</span><span style="background-color: hsl(0, 100.00%, 94.40%); opacity: 0.81" title="-0.019"> </span><span style="background-color: hsl(0, 100.00%, 95.22%); opacity: 0.81" title="-0.015">f</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 97.20%); opacity: 0.80" title="0.007">o</span><span style="background-color: hsl(120, 100.00%, 94.34%); opacity: 0.81" title="0.019">m</span><span style="background-color: hsl(120, 100.00%, 92.35%); opacity: 0.82" title="0.029"> </span><span style="background-color: hsl(120, 100.00%, 94.01%); opacity: 0.81" title="0.021">m</span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.027">y</span><span style="background-color: hsl(120, 100.00%, 95.33%); opacity: 0.81" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 95.31%); opacity: 0.81" title="0.015">b</span><span style="background-color: hsl(0, 100.00%, 97.94%); opacity: 0.80" title="-0.004">o</span><span style="background-color: hsl(0, 100.00%, 96.74%); opacity: 0.81" title="-0.009">u</span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.009">t</span><span style="background-color: hsl(120, 100.00%, 92.61%); opacity: 0.82" title="0.028"> </span><span style="background-color: hsl(120, 100.00%, 92.17%); opacity: 0.82" title="0.030">w</span><span style="background-color: hsl(120, 100.00%, 93.08%); opacity: 0.82" title="0.025">i</span><span style="background-color: hsl(120, 100.00%, 93.49%); opacity: 0.81" title="0.023">t</span><span style="background-color: hsl(120, 100.00%, 96.44%); opacity: 0.81" title="0.010">h</span><span style="background-color: hsl(0, 100.00%, 99.47%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 92.42%); opacity: 0.82" title="-0.029">k</span><span style="background-color: hsl(0, 100.00%, 91.51%); opacity: 0.82" title="-0.034">i</span><span style="background-color: hsl(0, 100.00%, 88.07%); opacity: 0.84" title="-0.055">d</span><span style="background-color: hsl(0, 100.00%, 87.62%); opacity: 0.84" title="-0.058">n</span><span style="background-color: hsl(0, 100.00%, 89.14%); opacity: 0.83" title="-0.048">e</span><span style="background-color: hsl(0, 100.00%, 92.88%); opacity: 0.82" title="-0.026">y</span><span style="background-color: hsl(120, 100.00%, 95.80%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 92.72%); opacity: 0.82" title="0.027">s</span><span style="background-color: hsl(120, 100.00%, 83.29%); opacity: 0.86" title="0.089">t</span><span style="background-color: hsl(120, 100.00%, 85.76%); opacity: 0.85" title="0.071">o</span><span style="background-color: hsl(120, 100.00%, 87.06%); opacity: 0.84" title="0.062">n</span><span style="background-color: hsl(120, 100.00%, 96.88%); opacity: 0.81" title="0.008">e</span><span style="background-color: hsl(0, 100.00%, 93.78%); opacity: 0.81" title="-0.022">s</span><span style="background-color: hsl(0, 100.00%, 96.03%); opacity: 0.81" title="-0.011">,</span><span style="background-color: hsl(0, 100.00%, 97.49%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.016">t</span><span style="background-color: hsl(120, 100.00%, 94.65%); opacity: 0.81" title="0.018">h</span><span style="background-color: hsl(0, 100.00%, 97.83%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(0, 100.00%, 93.85%); opacity: 0.81" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.044">e</span><span style="background-color: hsl(0, 100.00%, 88.57%); opacity: 0.83" title="-0.052"> </span><span style="background-color: hsl(0, 100.00%, 87.55%); opacity: 0.84" title="-0.059">i</span><span style="background-color: hsl(0, 100.00%, 87.22%); opacity: 0.84" title="-0.061">s</span><span style="background-color: hsl(0, 100.00%, 85.33%); opacity: 0.85" title="-0.074">n</span><span style="background-color: hsl(0, 100.00%, 85.25%); opacity: 0.85" title="-0.075">'</span><span style="background-color: hsl(0, 100.00%, 88.37%); opacity: 0.83" title="-0.053">t</span><span style="background-color: hsl(0, 100.00%, 91.49%); opacity: 0.82" title="-0.034"> </span><span style="background-color: hsl(0, 100.00%, 96.46%); opacity: 0.81" title="-0.010">a</span><span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.006">n</span><span style="background-color: hsl(0, 100.00%, 97.91%); opacity: 0.80" title="-0.005">y</span><span style="background-color: hsl(0, 100.00%, 97.07%); opacity: 0.80" title="-0.007">
    </span><span style="background-color: hsl(0, 100.00%, 89.33%); opacity: 0.83" title="-0.047">m</span><span style="background-color: hsl(0, 100.00%, 81.93%); opacity: 0.86" title="-0.100">e</span><span style="background-color: hsl(0, 100.00%, 78.70%); opacity: 0.88" title="-0.126">d</span><span style="background-color: hsl(0, 100.00%, 81.08%); opacity: 0.87" title="-0.107">i</span><span style="background-color: hsl(0, 100.00%, 86.35%); opacity: 0.84" title="-0.067">c</span><span style="background-color: hsl(0, 100.00%, 94.58%); opacity: 0.81" title="-0.018">a</span><span style="background-color: hsl(120, 100.00%, 96.91%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(0, 100.00%, 96.64%); opacity: 0.81" title="-0.009">i</span><span style="background-color: hsl(0, 100.00%, 97.10%); opacity: 0.80" title="-0.007">o</span><span style="background-color: hsl(0, 100.00%, 97.07%); opacity: 0.80" title="-0.007">n</span><span style="background-color: hsl(120, 100.00%, 96.01%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 93.32%); opacity: 0.82" title="0.024">t</span><span style="background-color: hsl(120, 100.00%, 91.88%); opacity: 0.82" title="0.032">h</span><span style="background-color: hsl(120, 100.00%, 93.54%); opacity: 0.81" title="0.023">a</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.78%); opacity: 0.80" title="0.005">c</span><span style="background-color: hsl(120, 100.00%, 91.68%); opacity: 0.82" title="0.033">a</span><span style="background-color: hsl(120, 100.00%, 90.12%); opacity: 0.83" title="0.042">n</span><span style="background-color: hsl(120, 100.00%, 91.18%); opacity: 0.82" title="0.036"> </span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.017">d</span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.007">o</span><span style="background-color: hsl(120, 100.00%, 98.28%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.015">a</span><span style="background-color: hsl(120, 100.00%, 88.85%); opacity: 0.83" title="0.050">n</span><span style="background-color: hsl(120, 100.00%, 86.72%); opacity: 0.84" title="0.064">y</span><span style="background-color: hsl(120, 100.00%, 86.55%); opacity: 0.84" title="0.066">t</span><span style="background-color: hsl(120, 100.00%, 91.40%); opacity: 0.82" title="0.035">h</span><span style="background-color: hsl(120, 100.00%, 94.33%); opacity: 0.81" title="0.019">i</span><span style="background-color: hsl(120, 100.00%, 97.55%); opacity: 0.80" title="0.006">n</span><span style="background-color: hsl(120, 100.00%, 96.20%); opacity: 0.81" title="0.011">g</span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 95.84%); opacity: 0.81" title="0.012">a</span><span style="background-color: hsl(120, 100.00%, 93.75%); opacity: 0.81" title="0.022">b</span><span style="background-color: hsl(120, 100.00%, 97.67%); opacity: 0.80" title="0.005">o</span><span style="background-color: hsl(0, 100.00%, 97.37%); opacity: 0.80" title="-0.006">u</span><span style="background-color: hsl(0, 100.00%, 97.34%); opacity: 0.80" title="-0.006">t</span><span style="background-color: hsl(120, 100.00%, 96.79%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 96.26%); opacity: 0.81" title="0.011">t</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(0, 100.00%, 94.55%); opacity: 0.81" title="-0.018">e</span><span style="background-color: hsl(0, 100.00%, 90.78%); opacity: 0.82" title="-0.038">m</span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 97.75%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.002">x</span><span style="background-color: hsl(120, 100.00%, 97.66%); opacity: 0.80" title="0.005">c</span><span style="background-color: hsl(120, 100.00%, 98.03%); opacity: 0.80" title="0.004">e</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.011">p</span><span style="background-color: hsl(0, 100.00%, 94.52%); opacity: 0.81" title="-0.018">t</span><span style="background-color: hsl(0, 100.00%, 94.38%); opacity: 0.81" title="-0.019"> </span><span style="background-color: hsl(0, 100.00%, 92.45%); opacity: 0.82" title="-0.029">r</span><span style="background-color: hsl(0, 100.00%, 98.10%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(120, 100.00%, 97.83%); opacity: 0.80" title="0.005">l</span><span style="background-color: hsl(120, 100.00%, 93.68%); opacity: 0.81" title="0.022">i</span><span style="background-color: hsl(120, 100.00%, 90.81%); opacity: 0.82" title="0.038">e</span><span style="background-color: hsl(120, 100.00%, 92.87%); opacity: 0.82" title="0.026">v</span><span style="background-color: hsl(120, 100.00%, 97.60%); opacity: 0.80" title="0.006">e</span><span style="background-color: hsl(120, 100.00%, 96.03%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 95.14%); opacity: 0.81" title="0.015">t</span><span style="background-color: hsl(120, 100.00%, 90.75%); opacity: 0.82" title="0.038">h</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.026">e</span><span style="background-color: hsl(0, 100.00%, 98.20%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 86.54%); opacity: 0.84" title="-0.066">p</span><span style="background-color: hsl(0, 100.00%, 84.39%); opacity: 0.85" title="-0.081">a</span><span style="background-color: hsl(0, 100.00%, 81.24%); opacity: 0.87" title="-0.105">i</span><span style="background-color: hsl(0, 100.00%, 87.54%); opacity: 0.84" title="-0.059">n</span><span style="background-color: hsl(0, 100.00%, 96.52%); opacity: 0.81" title="-0.009">.</span><span style="background-color: hsl(120, 100.00%, 96.22%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 97.47%); opacity: 0.80" title="0.006">e</span><span style="background-color: hsl(120, 100.00%, 93.43%); opacity: 0.82" title="0.024">i</span><span style="background-color: hsl(120, 100.00%, 90.21%); opacity: 0.83" title="0.042">t</span><span style="background-color: hsl(120, 100.00%, 86.56%); opacity: 0.84" title="0.065">h</span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.069">e</span><span style="background-color: hsl(120, 100.00%, 89.96%); opacity: 0.83" title="0.043">r</span><span style="background-color: hsl(120, 100.00%, 89.80%); opacity: 0.83" title="0.044"> </span><span style="background-color: hsl(120, 100.00%, 88.45%); opacity: 0.83" title="0.053">t</span><span style="background-color: hsl(120, 100.00%, 86.81%); opacity: 0.84" title="0.064">h</span><span style="background-color: hsl(120, 100.00%, 88.69%); opacity: 0.83" title="0.051">e</span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.025">y</span><span style="background-color: hsl(120, 100.00%, 93.96%); opacity: 0.81" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 91.84%); opacity: 0.82" title="0.032">p</span><span style="background-color: hsl(120, 100.00%, 86.97%); opacity: 0.84" title="0.063">a</span><span style="background-color: hsl(120, 100.00%, 86.17%); opacity: 0.84" title="0.068">s</span><span style="background-color: hsl(120, 100.00%, 93.37%); opacity: 0.82" title="0.024">s</span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.012">,</span><span style="background-color: hsl(0, 100.00%, 91.11%); opacity: 0.82" title="-0.036"> </span><span style="background-color: hsl(0, 100.00%, 89.05%); opacity: 0.83" title="-0.049">o</span><span style="background-color: hsl(0, 100.00%, 88.96%); opacity: 0.83" title="-0.049">r</span><span style="background-color: hsl(0, 100.00%, 96.13%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 92.18%); opacity: 0.82" title="0.030">t</span><span style="background-color: hsl(120, 100.00%, 87.03%); opacity: 0.84" title="0.062">h</span><span style="background-color: hsl(120, 100.00%, 86.62%); opacity: 0.84" title="0.065">e</span><span style="background-color: hsl(120, 100.00%, 88.42%); opacity: 0.83" title="0.053">y</span><span style="background-color: hsl(120, 100.00%, 93.88%); opacity: 0.81" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.005">h</span><span style="background-color: hsl(120, 100.00%, 95.71%); opacity: 0.81" title="0.013">a</span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.009">v</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.77%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 95.70%); opacity: 0.81" title="0.013">t</span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.032">o</span><span style="background-color: hsl(120, 100.00%, 90.49%); opacity: 0.83" title="0.040"> </span><span style="background-color: hsl(120, 100.00%, 90.27%); opacity: 0.83" title="0.041">b</span><span style="background-color: hsl(120, 100.00%, 91.70%); opacity: 0.82" title="0.033">e</span><span style="background-color: hsl(120, 100.00%, 90.02%); opacity: 0.83" title="0.043"> </span><span style="background-color: hsl(120, 100.00%, 90.44%); opacity: 0.83" title="0.040">b</span><span style="background-color: hsl(120, 100.00%, 89.24%); opacity: 0.83" title="0.048">r</span><span style="background-color: hsl(120, 100.00%, 94.61%); opacity: 0.81" title="0.018">o</span><span style="background-color: hsl(0, 100.00%, 92.20%); opacity: 0.82" title="-0.030">k</span><span style="background-color: hsl(0, 100.00%, 91.53%); opacity: 0.82" title="-0.034">e</span><span style="background-color: hsl(0, 100.00%, 93.48%); opacity: 0.81" title="-0.023">n</span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 92.97%); opacity: 0.82" title="-0.026">u</span><span style="background-color: hsl(0, 100.00%, 89.06%); opacity: 0.83" title="-0.049">p</span><span style="background-color: hsl(0, 100.00%, 92.55%); opacity: 0.82" title="-0.028"> </span><span style="background-color: hsl(0, 100.00%, 95.17%); opacity: 0.81" title="-0.015">w</span><span style="background-color: hsl(120, 100.00%, 94.89%); opacity: 0.81" title="0.016">i</span><span style="background-color: hsl(120, 100.00%, 91.19%); opacity: 0.82" title="0.036">t</span><span style="background-color: hsl(120, 100.00%, 88.26%); opacity: 0.83" title="0.054">h</span><span style="background-color: hsl(120, 100.00%, 91.03%); opacity: 0.82" title="0.037"> </span><span style="background-color: hsl(120, 100.00%, 93.82%); opacity: 0.81" title="0.022">s</span><span style="background-color: hsl(0, 100.00%, 99.17%); opacity: 0.80" title="-0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.003">u</span><span style="background-color: hsl(120, 100.00%, 96.49%); opacity: 0.81" title="0.010">n</span><span style="background-color: hsl(120, 100.00%, 91.71%); opacity: 0.82" title="0.033">d</span><span style="background-color: hsl(120, 100.00%, 92.06%); opacity: 0.82" title="0.031">,</span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 91.36%); opacity: 0.82" title="-0.035">o</span><span style="background-color: hsl(0, 100.00%, 90.02%); opacity: 0.83" title="-0.043">r</span><span style="background-color: hsl(0, 100.00%, 96.13%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 92.18%); opacity: 0.82" title="0.030">t</span><span style="background-color: hsl(120, 100.00%, 87.03%); opacity: 0.84" title="0.062">h</span><span style="background-color: hsl(120, 100.00%, 86.62%); opacity: 0.84" title="0.065">e</span><span style="background-color: hsl(120, 100.00%, 88.42%); opacity: 0.83" title="0.053">y</span><span style="background-color: hsl(120, 100.00%, 93.88%); opacity: 0.81" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 95.92%); opacity: 0.81" title="0.012">h</span><span style="background-color: hsl(120, 100.00%, 92.78%); opacity: 0.82" title="0.027">a</span><span style="background-color: hsl(120, 100.00%, 92.74%); opacity: 0.82" title="0.027">v</span><span style="background-color: hsl(120, 100.00%, 90.16%); opacity: 0.83" title="0.042">e</span><span style="background-color: hsl(120, 100.00%, 89.34%); opacity: 0.83" title="0.047">
    </span><span style="background-color: hsl(120, 100.00%, 91.35%); opacity: 0.82" title="0.035">t</span><span style="background-color: hsl(120, 100.00%, 89.89%); opacity: 0.83" title="0.044">o</span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.036"> </span><span style="background-color: hsl(120, 100.00%, 91.77%); opacity: 0.82" title="0.032">b</span><span style="background-color: hsl(120, 100.00%, 90.79%); opacity: 0.82" title="0.038">e</span><span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.019"> </span><span style="background-color: hsl(0, 100.00%, 92.68%); opacity: 0.82" title="-0.027">e</span><span style="background-color: hsl(0, 100.00%, 88.94%); opacity: 0.83" title="-0.050">x</span><span style="background-color: hsl(0, 100.00%, 83.38%); opacity: 0.86" title="-0.089">t</span><span style="background-color: hsl(0, 100.00%, 83.01%); opacity: 0.86" title="-0.092">r</span><span style="background-color: hsl(0, 100.00%, 83.20%); opacity: 0.86" title="-0.090">a</span><span style="background-color: hsl(0, 100.00%, 89.45%); opacity: 0.83" title="-0.046">c</span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.036">t</span><span style="background-color: hsl(0, 100.00%, 93.80%); opacity: 0.81" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 96.59%); opacity: 0.81" title="-0.009">d</span><span style="background-color: hsl(0, 100.00%, 90.29%); opacity: 0.83" title="-0.041"> </span><span style="background-color: hsl(0, 100.00%, 86.68%); opacity: 0.84" title="-0.065">s</span><span style="background-color: hsl(0, 100.00%, 86.38%); opacity: 0.84" title="-0.067">u</span><span style="background-color: hsl(0, 100.00%, 88.23%); opacity: 0.83" title="-0.054">r</span><span style="background-color: hsl(0, 100.00%, 89.55%); opacity: 0.83" title="-0.046">g</span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.023">i</span><span style="background-color: hsl(0, 100.00%, 91.47%); opacity: 0.82" title="-0.034">c</span><span style="background-color: hsl(0, 100.00%, 93.32%); opacity: 0.82" title="-0.024">a</span><span style="background-color: hsl(0, 100.00%, 94.21%); opacity: 0.81" title="-0.020">l</span><span style="background-color: hsl(0, 100.00%, 94.25%); opacity: 0.81" title="-0.019">l</span><span style="background-color: hsl(0, 100.00%, 93.62%); opacity: 0.81" title="-0.023">y</span><span style="background-color: hsl(0, 100.00%, 91.26%); opacity: 0.82" title="-0.035">.</span><span style="background-color: hsl(0, 100.00%, 93.87%); opacity: 0.81" title="-0.021"> </span><span style="background-color: hsl(0, 100.00%, 94.76%); opacity: 0.81" title="-0.017">w</span><span style="background-color: hsl(0, 100.00%, 99.49%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 95.68%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 94.23%); opacity: 0.81" title="-0.020">n</span><span style="background-color: hsl(0, 100.00%, 97.29%); opacity: 0.80" title="-0.007"> </span><span style="background-color: hsl(120, 100.00%, 97.51%); opacity: 0.80" title="0.006">i</span><span style="background-color: hsl(120, 100.00%, 95.85%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 97.44%); opacity: 0.80" title="0.006">w</span><span style="background-color: hsl(0, 100.00%, 99.39%); opacity: 0.80" title="-0.001">a</span><span style="background-color: hsl(0, 100.00%, 95.83%); opacity: 0.81" title="-0.012">s</span><span style="background-color: hsl(0, 100.00%, 94.97%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.007">i</span><span style="background-color: hsl(120, 100.00%, 93.57%); opacity: 0.81" title="0.023">n</span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.025">,</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.042"> </span><span style="background-color: hsl(120, 100.00%, 89.09%); opacity: 0.83" title="0.049">t</span><span style="background-color: hsl(120, 100.00%, 90.21%); opacity: 0.83" title="0.042">h</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(0, 100.00%, 92.14%); opacity: 0.82" title="-0.030">x</span><span style="background-color: hsl(0, 100.00%, 92.28%); opacity: 0.82" title="-0.030">-</span><span style="background-color: hsl(0, 100.00%, 96.19%); opacity: 0.81" title="-0.011">r</span><span style="background-color: hsl(120, 100.00%, 98.66%); opacity: 0.80" title="0.002">a</span><span style="background-color: hsl(120, 100.00%, 94.31%); opacity: 0.81" title="0.019">y</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(0, 100.00%, 93.47%); opacity: 0.82" title="-0.023">t</span><span style="background-color: hsl(0, 100.00%, 88.49%); opacity: 0.83" title="-0.053">e</span><span style="background-color: hsl(0, 100.00%, 85.03%); opacity: 0.85" title="-0.076">c</span><span style="background-color: hsl(0, 100.00%, 84.56%); opacity: 0.85" title="-0.080">h</span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.036"> </span><span style="background-color: hsl(0, 100.00%, 94.48%); opacity: 0.81" title="-0.018">h</span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.020">a</span><span style="background-color: hsl(120, 100.00%, 89.44%); opacity: 0.83" title="0.046">p</span><span style="background-color: hsl(120, 100.00%, 91.73%); opacity: 0.82" title="0.033">p</span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.030">e</span><span style="background-color: hsl(120, 100.00%, 93.02%); opacity: 0.82" title="0.026">n</span><span style="background-color: hsl(120, 100.00%, 93.64%); opacity: 0.81" title="0.023">e</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.003">d</span><span style="background-color: hsl(120, 100.00%, 95.45%); opacity: 0.81" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 93.83%); opacity: 0.81" title="0.022">t</span><span style="background-color: hsl(120, 100.00%, 96.04%); opacity: 0.81" title="0.011">o</span><span style="background-color: hsl(0, 100.00%, 95.41%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 89.83%); opacity: 0.83" title="-0.044">m</span><span style="background-color: hsl(0, 100.00%, 86.88%); opacity: 0.84" title="-0.063">e</span><span style="background-color: hsl(0, 100.00%, 84.33%); opacity: 0.85" title="-0.082">n</span><span style="background-color: hsl(0, 100.00%, 86.60%); opacity: 0.84" title="-0.065">t</span><span style="background-color: hsl(0, 100.00%, 88.28%); opacity: 0.83" title="-0.054">i</span><span style="background-color: hsl(0, 100.00%, 93.54%); opacity: 0.81" title="-0.023">o</span><span style="background-color: hsl(0, 100.00%, 95.97%); opacity: 0.81" title="-0.012">n</span><span style="background-color: hsl(120, 100.00%, 96.01%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 93.32%); opacity: 0.82" title="0.024">t</span><span style="background-color: hsl(120, 100.00%, 92.07%); opacity: 0.82" title="0.031">h</span><span style="background-color: hsl(120, 100.00%, 93.47%); opacity: 0.82" title="0.023">a</span><span style="background-color: hsl(120, 100.00%, 94.83%); opacity: 0.81" title="0.017">t</span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 91.40%); opacity: 0.82" title="-0.035">s</span><span style="background-color: hsl(0, 100.00%, 90.63%); opacity: 0.83" title="-0.039">h</span><span style="background-color: hsl(0, 100.00%, 89.98%); opacity: 0.83" title="-0.043">e</span><span style="background-color: hsl(0, 100.00%, 87.85%); opacity: 0.84" title="-0.057">'</span><span style="background-color: hsl(0, 100.00%, 97.97%); opacity: 0.80" title="-0.004">d</span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.003">h</span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.009">a</span><span style="background-color: hsl(0, 100.00%, 96.12%); opacity: 0.81" title="-0.011">d</span><span style="background-color: hsl(0, 100.00%, 94.99%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 91.34%); opacity: 0.82" title="-0.035">k</span><span style="background-color: hsl(0, 100.00%, 89.85%); opacity: 0.83" title="-0.044">i</span><span style="background-color: hsl(0, 100.00%, 89.14%); opacity: 0.83" title="-0.048">d</span><span style="background-color: hsl(0, 100.00%, 88.86%); opacity: 0.83" title="-0.050">n</span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.019">e</span><span style="background-color: hsl(120, 100.00%, 96.72%); opacity: 0.81" title="0.009">y</span><span style="background-color: hsl(120, 100.00%, 95.19%); opacity: 0.81" title="0.015">
    </span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.005">s</span><span style="background-color: hsl(120, 100.00%, 87.76%); opacity: 0.84" title="0.057">t</span><span style="background-color: hsl(120, 100.00%, 89.75%); opacity: 0.83" title="0.044">o</span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.029">n</span><span style="background-color: hsl(0, 100.00%, 93.94%); opacity: 0.81" title="-0.021">e</span><span style="background-color: hsl(0, 100.00%, 89.08%); opacity: 0.83" title="-0.049">s</span><span style="background-color: hsl(0, 100.00%, 90.69%); opacity: 0.82" title="-0.039"> </span><span style="background-color: hsl(0, 100.00%, 95.88%); opacity: 0.81" title="-0.012">a</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 96.97%); opacity: 0.81" title="0.008">d</span><span style="background-color: hsl(120, 100.00%, 90.85%); opacity: 0.82" title="0.038"> </span><span style="background-color: hsl(120, 100.00%, 88.28%); opacity: 0.83" title="0.054">c</span><span style="background-color: hsl(120, 100.00%, 83.49%); opacity: 0.86" title="0.088">h</span><span style="background-color: hsl(120, 100.00%, 84.38%); opacity: 0.85" title="0.081">i</span><span style="background-color: hsl(120, 100.00%, 79.46%); opacity: 0.88" title="0.120">l</span><span style="background-color: hsl(120, 100.00%, 81.01%); opacity: 0.87" title="0.107">d</span><span style="background-color: hsl(120, 100.00%, 86.93%); opacity: 0.84" title="0.063">r</span><span style="background-color: hsl(120, 100.00%, 87.43%); opacity: 0.84" title="0.059">e</span><span style="background-color: hsl(120, 100.00%, 92.06%); opacity: 0.82" title="0.031">n</span><span style="background-color: hsl(120, 100.00%, 97.34%); opacity: 0.80" title="0.006">,</span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 94.14%); opacity: 0.81" title="-0.020">a</span><span style="background-color: hsl(0, 100.00%, 95.94%); opacity: 0.81" title="-0.012">n</span><span style="background-color: hsl(0, 100.00%, 96.33%); opacity: 0.81" title="-0.010">d</span><span style="background-color: hsl(120, 100.00%, 97.01%); opacity: 0.80" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 94.18%); opacity: 0.81" title="0.020">t</span><span style="background-color: hsl(120, 100.00%, 88.71%); opacity: 0.83" title="0.051">h</span><span style="background-color: hsl(120, 100.00%, 90.50%); opacity: 0.83" title="0.040">e</span><span style="background-color: hsl(120, 100.00%, 86.74%); opacity: 0.84" title="0.064"> </span><span style="background-color: hsl(120, 100.00%, 87.18%); opacity: 0.84" title="0.061">c</span><span style="background-color: hsl(120, 100.00%, 85.16%); opacity: 0.85" title="0.075">h</span><span style="background-color: hsl(120, 100.00%, 88.35%); opacity: 0.83" title="0.053">i</span><span style="background-color: hsl(120, 100.00%, 86.30%); opacity: 0.84" title="0.067">l</span><span style="background-color: hsl(120, 100.00%, 92.59%); opacity: 0.82" title="0.028">d</span><span style="background-color: hsl(0, 100.00%, 91.51%); opacity: 0.82" title="-0.034">b</span><span style="background-color: hsl(0, 100.00%, 86.48%); opacity: 0.84" title="-0.066">i</span><span style="background-color: hsl(0, 100.00%, 89.80%); opacity: 0.83" title="-0.044">r</span><span style="background-color: hsl(0, 100.00%, 91.68%); opacity: 0.82" title="-0.033">t</span><span style="background-color: hsl(0, 100.00%, 94.61%); opacity: 0.81" title="-0.018">h</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 87.13%); opacity: 0.84" title="0.062">h</span><span style="background-color: hsl(120, 100.00%, 84.66%); opacity: 0.85" title="0.079">u</span><span style="background-color: hsl(120, 100.00%, 85.41%); opacity: 0.85" title="0.074">r</span><span style="background-color: hsl(0, 100.00%, 96.71%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(0, 100.00%, 97.50%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 94.92%); opacity: 0.81" title="-0.016">l</span><span style="background-color: hsl(120, 100.00%, 97.80%); opacity: 0.80" title="0.005">e</span><span style="background-color: hsl(120, 100.00%, 97.03%); opacity: 0.80" title="0.008">s</span><span style="background-color: hsl(120, 100.00%, 95.01%); opacity: 0.81" title="0.016">s</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.002">.</span>
        </p>
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




It works, but quality is a bit worse. Also, it takes ages to train.

It looks like stop_words have no effect now - in fact, this is
documented in scikit-learn docs, so our stop_words=‘english’ was
useless. But at least it is now more obvious how the text looks like for
a char ngram-based classifier. Grab a cup of tea and see how char_wb
looks like:

.. code:: ipython3

    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
    clf = LogisticRegressionCV()
    pipe = make_pipeline(vec, clf)
    pipe.fit(twenty_train.data, twenty_train.target)
    
    print_report(pipe)


.. parsed-literal::

                            precision    recall  f1-score   support
    
               alt.atheism       0.93      0.79      0.85       319
             comp.graphics       0.87      0.96      0.91       389
                   sci.med       0.91      0.90      0.90       396
    soc.religion.christian       0.89      0.91      0.90       398
    
               avg / total       0.90      0.89      0.89      1502
    
    accuracy: 0.894


.. code:: ipython3

    eli5.show_prediction(clf, twenty_test.data[0], vec=vec, 
                         target_names=twenty_test.target_names)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=alt.atheism
        
    </b>
    
        
        (probability <b>0.000</b>, score <b>-8.878</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 90.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.560
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 81.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -6.318
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.006">as</span><span style="background-color: hsl(0, 100.00%, 90.68%); opacity: 0.82" title="-0.032"> </span><span style="background-color: hsl(0, 100.00%, 91.78%); opacity: 0.82" title="-0.027">i</span><span style="background-color: hsl(0, 100.00%, 88.72%); opacity: 0.83" title="-0.043"> </span><span style="background-color: hsl(0, 100.00%, 89.87%); opacity: 0.83" title="-0.037">r</span><span style="background-color: hsl(0, 100.00%, 88.78%); opacity: 0.83" title="-0.042">e</span><span style="background-color: hsl(0, 100.00%, 90.13%); opacity: 0.83" title="-0.035">c</span><span style="background-color: hsl(120, 100.00%, 95.41%); opacity: 0.81" title="0.012">a</span><span style="background-color: hsl(120, 100.00%, 90.90%); opacity: 0.82" title="0.031">l</span><span style="background-color: hsl(120, 100.00%, 89.79%); opacity: 0.83" title="0.037">l</span><span style="background-color: hsl(120, 100.00%, 91.35%); opacity: 0.82" title="0.029"> </span><span style="background-color: hsl(120, 100.00%, 95.92%); opacity: 0.81" title="0.010">f</span><span style="background-color: hsl(120, 100.00%, 95.47%); opacity: 0.81" title="0.012">r</span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.010">o</span><span style="background-color: hsl(120, 100.00%, 97.32%); opacity: 0.80" title="0.005">m</span><span style="background-color: hsl(0, 100.00%, 89.97%); opacity: 0.83" title="-0.036"> </span><span style="background-color: hsl(0, 100.00%, 86.72%); opacity: 0.84" title="-0.054">my</span><span style="background-color: hsl(0, 100.00%, 90.51%); opacity: 0.83" title="-0.033"> </span><span style="background-color: hsl(0, 100.00%, 90.37%); opacity: 0.83" title="-0.034">b</span><span style="background-color: hsl(0, 100.00%, 89.70%); opacity: 0.83" title="-0.037">o</span><span style="background-color: hsl(0, 100.00%, 87.17%); opacity: 0.84" title="-0.051">u</span><span style="background-color: hsl(0, 100.00%, 90.11%); opacity: 0.83" title="-0.035">t</span><span style="background-color: hsl(0, 100.00%, 87.40%); opacity: 0.84" title="-0.050"> </span><span style="background-color: hsl(0, 100.00%, 83.97%); opacity: 0.85" title="-0.070">w</span><span style="background-color: hsl(0, 100.00%, 81.11%); opacity: 0.87" title="-0.089">i</span><span style="background-color: hsl(0, 100.00%, 80.26%); opacity: 0.87" title="-0.095">t</span><span style="background-color: hsl(0, 100.00%, 84.67%); opacity: 0.85" title="-0.066">h</span><span style="background-color: hsl(0, 100.00%, 92.94%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(120, 100.00%, 91.92%); opacity: 0.82" title="0.026">k</span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.023">i</span><span style="background-color: hsl(0, 100.00%, 94.92%); opacity: 0.81" title="-0.014">d</span><span style="background-color: hsl(0, 100.00%, 89.08%); opacity: 0.83" title="-0.041">n</span><span style="background-color: hsl(0, 100.00%, 89.04%); opacity: 0.83" title="-0.041">e</span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.017">y</span><span style="background-color: hsl(0, 100.00%, 99.85%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(0, 100.00%, 93.64%); opacity: 0.81" title="-0.019">s</span><span style="background-color: hsl(0, 100.00%, 88.82%); opacity: 0.83" title="-0.042">t</span><span style="background-color: hsl(0, 100.00%, 90.18%); opacity: 0.83" title="-0.035">o</span><span style="background-color: hsl(0, 100.00%, 85.97%); opacity: 0.84" title="-0.058">n</span><span style="background-color: hsl(0, 100.00%, 92.52%); opacity: 0.82" title="-0.024">e</span><span style="background-color: hsl(0, 100.00%, 97.69%); opacity: 0.80" title="-0.004">s</span><span style="background-color: hsl(0, 100.00%, 95.35%); opacity: 0.81" title="-0.012">,</span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 93.64%); opacity: 0.81" title="0.019">t</span><span style="background-color: hsl(120, 100.00%, 92.48%); opacity: 0.82" title="0.024">h</span><span style="background-color: hsl(120, 100.00%, 89.68%); opacity: 0.83" title="0.038">e</span><span style="background-color: hsl(120, 100.00%, 93.35%); opacity: 0.82" title="0.020">r</span><span style="background-color: hsl(120, 100.00%, 91.80%); opacity: 0.82" title="0.027">e</span><span style="background-color: hsl(120, 100.00%, 89.04%); opacity: 0.83" title="0.041"> </span><span style="background-color: hsl(120, 100.00%, 84.29%); opacity: 0.85" title="0.068">i</span><span style="background-color: hsl(120, 100.00%, 77.57%); opacity: 0.89" title="0.114">s</span><span style="background-color: hsl(120, 100.00%, 73.41%); opacity: 0.91" title="0.145">n</span><span style="background-color: hsl(120, 100.00%, 73.65%); opacity: 0.91" title="0.143">'</span><span style="background-color: hsl(120, 100.00%, 79.15%); opacity: 0.88" title="0.103">t</span><span style="background-color: hsl(120, 100.00%, 91.47%); opacity: 0.82" title="0.029"> </span><span style="background-color: hsl(0, 100.00%, 89.43%); opacity: 0.83" title="-0.039">a</span><span style="background-color: hsl(0, 100.00%, 87.85%); opacity: 0.84" title="-0.047">n</span><span style="background-color: hsl(0, 100.00%, 90.05%); opacity: 0.83" title="-0.036">y</span><span style="background-color: hsl(0, 100.00%, 86.05%); opacity: 0.84" title="-0.058">
    </span><span style="background-color: hsl(0, 100.00%, 81.40%); opacity: 0.87" title="-0.087">m</span><span style="background-color: hsl(0, 100.00%, 74.51%); opacity: 0.91" title="-0.137">e</span><span style="background-color: hsl(0, 100.00%, 75.60%); opacity: 0.90" title="-0.128">d</span><span style="background-color: hsl(0, 100.00%, 79.20%); opacity: 0.88" title="-0.102">i</span><span style="background-color: hsl(0, 100.00%, 82.65%); opacity: 0.86" title="-0.079">c</span><span style="background-color: hsl(0, 100.00%, 81.16%); opacity: 0.87" title="-0.089">a</span><span style="background-color: hsl(0, 100.00%, 82.81%); opacity: 0.86" title="-0.078">t</span><span style="background-color: hsl(0, 100.00%, 87.43%); opacity: 0.84" title="-0.050">i</span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.024">o</span><span style="background-color: hsl(0, 100.00%, 96.48%); opacity: 0.81" title="-0.008">n</span><span style="background-color: hsl(0, 100.00%, 99.65%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.013">h</span><span style="background-color: hsl(120, 100.00%, 95.71%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 95.30%); opacity: 0.81" title="0.012">t</span><span style="background-color: hsl(0, 100.00%, 87.53%); opacity: 0.84" title="-0.049"> </span><span style="background-color: hsl(0, 100.00%, 82.13%); opacity: 0.86" title="-0.082">c</span><span style="background-color: hsl(0, 100.00%, 82.40%); opacity: 0.86" title="-0.081">a</span><span style="background-color: hsl(0, 100.00%, 85.80%); opacity: 0.85" title="-0.059">n</span><span style="background-color: hsl(0, 100.00%, 91.70%); opacity: 0.82" title="-0.028"> </span><span style="background-color: hsl(0, 100.00%, 98.46%); opacity: 0.80" title="-0.002">do</span><span style="background-color: hsl(0, 100.00%, 94.61%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 90.73%); opacity: 0.82" title="-0.032">a</span><span style="background-color: hsl(0, 100.00%, 89.59%); opacity: 0.83" title="-0.038">n</span><span style="background-color: hsl(0, 100.00%, 90.01%); opacity: 0.83" title="-0.036">y</span><span style="background-color: hsl(0, 100.00%, 99.54%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(120, 100.00%, 93.60%); opacity: 0.81" title="0.019">h</span><span style="background-color: hsl(120, 100.00%, 92.23%); opacity: 0.82" title="0.025">i</span><span style="background-color: hsl(120, 100.00%, 91.33%); opacity: 0.82" title="0.029">n</span><span style="background-color: hsl(120, 100.00%, 93.92%); opacity: 0.81" title="0.018">g</span><span style="background-color: hsl(0, 100.00%, 92.94%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 86.76%); opacity: 0.84" title="-0.054">a</span><span style="background-color: hsl(0, 100.00%, 81.43%); opacity: 0.87" title="-0.087">b</span><span style="background-color: hsl(0, 100.00%, 82.47%); opacity: 0.86" title="-0.080">o</span><span style="background-color: hsl(0, 100.00%, 84.22%); opacity: 0.85" title="-0.069">u</span><span style="background-color: hsl(0, 100.00%, 89.92%); opacity: 0.83" title="-0.036">t</span><span style="background-color: hsl(0, 100.00%, 95.73%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 90.31%); opacity: 0.83" title="0.034">t</span><span style="background-color: hsl(120, 100.00%, 87.38%); opacity: 0.84" title="0.050">h</span><span style="background-color: hsl(120, 100.00%, 86.21%); opacity: 0.84" title="0.057">e</span><span style="background-color: hsl(120, 100.00%, 88.74%); opacity: 0.83" title="0.043">m</span><span style="background-color: hsl(120, 100.00%, 90.75%); opacity: 0.82" title="0.032"> </span><span style="background-color: hsl(120, 100.00%, 90.92%); opacity: 0.82" title="0.031">e</span><span style="background-color: hsl(120, 100.00%, 87.68%); opacity: 0.84" title="0.048">x</span><span style="background-color: hsl(120, 100.00%, 83.12%); opacity: 0.86" title="0.076">c</span><span style="background-color: hsl(120, 100.00%, 86.46%); opacity: 0.84" title="0.055">e</span><span style="background-color: hsl(120, 100.00%, 87.07%); opacity: 0.84" title="0.052">p</span><span style="background-color: hsl(120, 100.00%, 91.41%); opacity: 0.82" title="0.029">t</span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.051"> </span><span style="background-color: hsl(120, 100.00%, 81.42%); opacity: 0.87" title="0.087">r</span><span style="background-color: hsl(120, 100.00%, 77.44%); opacity: 0.89" title="0.115">e</span><span style="background-color: hsl(120, 100.00%, 75.08%); opacity: 0.90" title="0.132">l</span><span style="background-color: hsl(120, 100.00%, 81.83%); opacity: 0.86" title="0.084">i</span><span style="background-color: hsl(0, 100.00%, 97.15%); opacity: 0.80" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 94.95%); opacity: 0.81" title="-0.014">v</span><span style="background-color: hsl(0, 100.00%, 93.30%); opacity: 0.82" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 98.30%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 94.84%); opacity: 0.81" title="0.014">t</span><span style="background-color: hsl(120, 100.00%, 94.73%); opacity: 0.81" title="0.014">h</span><span style="background-color: hsl(120, 100.00%, 95.66%); opacity: 0.81" title="0.011">e</span><span style="background-color: hsl(0, 100.00%, 90.07%); opacity: 0.83" title="-0.036"> </span><span style="background-color: hsl(0, 100.00%, 88.91%); opacity: 0.83" title="-0.042">p</span><span style="background-color: hsl(0, 100.00%, 91.51%); opacity: 0.82" title="-0.028">a</span><span style="background-color: hsl(0, 100.00%, 93.36%); opacity: 0.82" title="-0.020">i</span><span style="background-color: hsl(0, 100.00%, 92.83%); opacity: 0.82" title="-0.022">n</span><span style="background-color: hsl(0, 100.00%, 93.03%); opacity: 0.82" title="-0.021">.</span><span style="background-color: hsl(0, 100.00%, 98.40%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 86.03%); opacity: 0.84" title="0.058">e</span><span style="background-color: hsl(120, 100.00%, 87.96%); opacity: 0.84" title="0.047">i</span><span style="background-color: hsl(120, 100.00%, 88.45%); opacity: 0.83" title="0.044">t</span><span style="background-color: hsl(0, 100.00%, 99.33%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 90.45%); opacity: 0.83" title="-0.034">e</span><span style="background-color: hsl(0, 100.00%, 88.05%); opacity: 0.84" title="-0.046">r</span><span style="background-color: hsl(0, 100.00%, 89.08%); opacity: 0.83" title="-0.041"> </span><span style="background-color: hsl(0, 100.00%, 99.51%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 93.40%); opacity: 0.82" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 90.35%); opacity: 0.83" title="-0.034">y</span><span style="background-color: hsl(0, 100.00%, 89.06%); opacity: 0.83" title="-0.041"> </span><span style="background-color: hsl(0, 100.00%, 91.72%); opacity: 0.82" title="-0.027">p</span><span style="background-color: hsl(0, 100.00%, 90.84%); opacity: 0.82" title="-0.032">a</span><span style="background-color: hsl(0, 100.00%, 92.91%); opacity: 0.82" title="-0.022">s</span><span style="background-color: hsl(0, 100.00%, 94.20%); opacity: 0.81" title="-0.016">s</span><span style="background-color: hsl(0, 100.00%, 93.85%); opacity: 0.81" title="-0.018">,</span><span style="background-color: hsl(120, 100.00%, 94.56%); opacity: 0.81" title="0.015"> </span><span style="background-color: hsl(120, 100.00%, 92.25%); opacity: 0.82" title="0.025">or</span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(0, 100.00%, 99.51%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 93.40%); opacity: 0.82" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 90.35%); opacity: 0.83" title="-0.034">y</span><span style="background-color: hsl(0, 100.00%, 97.50%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 94.48%); opacity: 0.81" title="0.015">h</span><span style="background-color: hsl(120, 100.00%, 95.52%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 95.99%); opacity: 0.81" title="0.010">v</span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.005">e</span><span style="background-color: hsl(0, 100.00%, 94.32%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 89.91%); opacity: 0.83" title="-0.036">to</span><span style="background-color: hsl(0, 100.00%, 92.54%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 97.13%); opacity: 0.80" title="-0.006">be</span><span style="background-color: hsl(0, 100.00%, 89.85%); opacity: 0.83" title="-0.037"> </span><span style="background-color: hsl(0, 100.00%, 88.95%); opacity: 0.83" title="-0.041">b</span><span style="background-color: hsl(0, 100.00%, 86.44%); opacity: 0.84" title="-0.055">r</span><span style="background-color: hsl(0, 100.00%, 89.00%); opacity: 0.83" title="-0.041">o</span><span style="background-color: hsl(120, 100.00%, 90.31%); opacity: 0.83" title="0.034">k</span><span style="background-color: hsl(120, 100.00%, 87.06%); opacity: 0.84" title="0.052">e</span><span style="background-color: hsl(120, 100.00%, 85.68%); opacity: 0.85" title="0.060">n</span><span style="background-color: hsl(120, 100.00%, 85.88%); opacity: 0.85" title="0.059"> </span><span style="background-color: hsl(120, 100.00%, 89.03%); opacity: 0.83" title="0.041">up</span><span style="background-color: hsl(0, 100.00%, 94.13%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 83.97%); opacity: 0.85" title="-0.070">w</span><span style="background-color: hsl(0, 100.00%, 81.11%); opacity: 0.87" title="-0.089">i</span><span style="background-color: hsl(0, 100.00%, 80.26%); opacity: 0.87" title="-0.095">t</span><span style="background-color: hsl(0, 100.00%, 84.67%); opacity: 0.85" title="-0.066">h</span><span style="background-color: hsl(0, 100.00%, 86.77%); opacity: 0.84" title="-0.054"> </span><span style="background-color: hsl(0, 100.00%, 86.28%); opacity: 0.84" title="-0.056">s</span><span style="background-color: hsl(0, 100.00%, 86.80%); opacity: 0.84" title="-0.053">o</span><span style="background-color: hsl(0, 100.00%, 86.66%); opacity: 0.84" title="-0.054">u</span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(120, 100.00%, 93.93%); opacity: 0.81" title="0.018">d</span><span style="background-color: hsl(120, 100.00%, 90.84%); opacity: 0.82" title="0.032">,</span><span style="background-color: hsl(120, 100.00%, 90.66%); opacity: 0.83" title="0.033"> </span><span style="background-color: hsl(120, 100.00%, 92.25%); opacity: 0.82" title="0.025">or</span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(0, 100.00%, 99.51%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 93.40%); opacity: 0.82" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 90.35%); opacity: 0.83" title="-0.034">y</span><span style="background-color: hsl(0, 100.00%, 97.50%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 94.48%); opacity: 0.81" title="0.015">h</span><span style="background-color: hsl(120, 100.00%, 95.52%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 95.99%); opacity: 0.81" title="0.010">v</span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.005">e</span><span style="background-color: hsl(0, 100.00%, 94.32%); opacity: 0.81" title="-0.016">
    </span><span style="background-color: hsl(0, 100.00%, 89.91%); opacity: 0.83" title="-0.036">to</span><span style="background-color: hsl(0, 100.00%, 92.54%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 97.13%); opacity: 0.80" title="-0.006">be</span><span style="background-color: hsl(0, 100.00%, 94.07%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(120, 100.00%, 96.07%); opacity: 0.81" title="0.009">e</span><span style="background-color: hsl(120, 100.00%, 98.53%); opacity: 0.80" title="0.002">x</span><span style="background-color: hsl(120, 100.00%, 92.22%); opacity: 0.82" title="0.025">t</span><span style="background-color: hsl(0, 100.00%, 96.34%); opacity: 0.81" title="-0.009">r</span><span style="background-color: hsl(0, 100.00%, 95.17%); opacity: 0.81" title="-0.013">a</span><span style="background-color: hsl(0, 100.00%, 85.76%); opacity: 0.85" title="-0.059">c</span><span style="background-color: hsl(0, 100.00%, 90.52%); opacity: 0.83" title="-0.033">t</span><span style="background-color: hsl(0, 100.00%, 86.08%); opacity: 0.84" title="-0.058">e</span><span style="background-color: hsl(0, 100.00%, 91.95%); opacity: 0.82" title="-0.026">d</span><span style="background-color: hsl(0, 100.00%, 91.29%); opacity: 0.82" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 86.63%); opacity: 0.84" title="-0.054">s</span><span style="background-color: hsl(0, 100.00%, 80.95%); opacity: 0.87" title="-0.090">u</span><span style="background-color: hsl(0, 100.00%, 77.28%); opacity: 0.89" title="-0.116">r</span><span style="background-color: hsl(0, 100.00%, 87.98%); opacity: 0.84" title="-0.047">g</span><span style="background-color: hsl(0, 100.00%, 97.68%); opacity: 0.80" title="-0.004">i</span><span style="background-color: hsl(120, 100.00%, 91.73%); opacity: 0.82" title="0.027">c</span><span style="background-color: hsl(120, 100.00%, 89.05%); opacity: 0.83" title="0.041">a</span><span style="background-color: hsl(120, 100.00%, 89.57%); opacity: 0.83" title="0.038">l</span><span style="background-color: hsl(120, 100.00%, 90.84%); opacity: 0.82" title="0.032">l</span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.025">y</span><span style="background-color: hsl(120, 100.00%, 92.14%); opacity: 0.82" title="0.025">.</span><span style="background-color: hsl(0, 100.00%, 96.39%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 89.17%); opacity: 0.83" title="-0.040">w</span><span style="background-color: hsl(0, 100.00%, 87.00%); opacity: 0.84" title="-0.052">h</span><span style="background-color: hsl(0, 100.00%, 88.90%); opacity: 0.83" title="-0.042">e</span><span style="background-color: hsl(0, 100.00%, 92.74%); opacity: 0.82" title="-0.023">n</span><span style="background-color: hsl(0, 100.00%, 91.73%); opacity: 0.82" title="-0.027"> </span><span style="background-color: hsl(0, 100.00%, 91.78%); opacity: 0.82" title="-0.027">i</span><span style="background-color: hsl(0, 100.00%, 89.66%); opacity: 0.83" title="-0.038"> </span><span style="background-color: hsl(0, 100.00%, 94.98%); opacity: 0.81" title="-0.013">w</span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.013">a</span><span style="background-color: hsl(0, 100.00%, 97.94%); opacity: 0.80" title="-0.004">s</span><span style="background-color: hsl(0, 100.00%, 97.67%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 92.44%); opacity: 0.82" title="-0.024">i</span><span style="background-color: hsl(0, 100.00%, 95.71%); opacity: 0.81" title="-0.011">n</span><span style="background-color: hsl(0, 100.00%, 95.96%); opacity: 0.81" title="-0.010">,</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 94.84%); opacity: 0.81" title="0.014">t</span><span style="background-color: hsl(120, 100.00%, 94.73%); opacity: 0.81" title="0.014">h</span><span style="background-color: hsl(120, 100.00%, 95.66%); opacity: 0.81" title="0.011">e</span><span style="background-color: hsl(0, 100.00%, 95.99%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 94.74%); opacity: 0.81" title="-0.014">x</span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.024">-</span><span style="background-color: hsl(0, 100.00%, 87.18%); opacity: 0.84" title="-0.051">r</span><span style="background-color: hsl(0, 100.00%, 89.07%); opacity: 0.83" title="-0.041">a</span><span style="background-color: hsl(0, 100.00%, 91.04%); opacity: 0.82" title="-0.031">y</span><span style="background-color: hsl(0, 100.00%, 93.65%); opacity: 0.81" title="-0.019"> </span><span style="background-color: hsl(0, 100.00%, 93.39%); opacity: 0.82" title="-0.020">t</span><span style="background-color: hsl(0, 100.00%, 92.89%); opacity: 0.82" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 92.64%); opacity: 0.82" title="-0.023">c</span><span style="background-color: hsl(0, 100.00%, 94.13%); opacity: 0.81" title="-0.017">h</span><span style="background-color: hsl(0, 100.00%, 98.49%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 97.25%); opacity: 0.80" title="0.006">h</span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(0, 100.00%, 93.82%); opacity: 0.81" title="-0.018">p</span><span style="background-color: hsl(0, 100.00%, 98.20%); opacity: 0.80" title="-0.003">p</span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.022">n</span><span style="background-color: hsl(120, 100.00%, 93.16%); opacity: 0.82" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 93.58%); opacity: 0.81" title="0.019">d</span><span style="background-color: hsl(0, 100.00%, 91.47%); opacity: 0.82" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 89.91%); opacity: 0.83" title="-0.036">to</span><span style="background-color: hsl(0, 100.00%, 96.21%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(120, 100.00%, 89.20%); opacity: 0.83" title="0.040">m</span><span style="background-color: hsl(120, 100.00%, 83.87%); opacity: 0.85" title="0.071">e</span><span style="background-color: hsl(120, 100.00%, 76.28%); opacity: 0.89" title="0.123">n</span><span style="background-color: hsl(120, 100.00%, 79.44%); opacity: 0.88" title="0.101">t</span><span style="background-color: hsl(120, 100.00%, 84.04%); opacity: 0.85" title="0.070">i</span><span style="background-color: hsl(120, 100.00%, 88.78%); opacity: 0.83" title="0.042">o</span><span style="background-color: hsl(120, 100.00%, 95.55%); opacity: 0.81" title="0.011">n</span><span style="background-color: hsl(0, 100.00%, 99.65%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.013">h</span><span style="background-color: hsl(120, 100.00%, 95.71%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 95.30%); opacity: 0.81" title="0.012">t</span><span style="background-color: hsl(120, 100.00%, 92.99%); opacity: 0.82" title="0.022"> </span><span style="background-color: hsl(120, 100.00%, 87.74%); opacity: 0.84" title="0.048">s</span><span style="background-color: hsl(120, 100.00%, 84.93%); opacity: 0.85" title="0.065">h</span><span style="background-color: hsl(120, 100.00%, 84.65%); opacity: 0.85" title="0.066">e</span><span style="background-color: hsl(120, 100.00%, 87.78%); opacity: 0.84" title="0.048">'</span><span style="background-color: hsl(0, 100.00%, 95.22%); opacity: 0.81" title="-0.013">d</span><span style="background-color: hsl(0, 100.00%, 97.49%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 94.67%); opacity: 0.81" title="-0.015">h</span><span style="background-color: hsl(120, 100.00%, 97.86%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(0, 100.00%, 98.41%); opacity: 0.80" title="-0.003">d</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.022"> </span><span style="background-color: hsl(120, 100.00%, 91.92%); opacity: 0.82" title="0.026">k</span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.023">i</span><span style="background-color: hsl(0, 100.00%, 94.92%); opacity: 0.81" title="-0.014">d</span><span style="background-color: hsl(0, 100.00%, 89.08%); opacity: 0.83" title="-0.041">n</span><span style="background-color: hsl(0, 100.00%, 89.04%); opacity: 0.83" title="-0.041">e</span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.017">y</span><span style="background-color: hsl(0, 100.00%, 99.85%); opacity: 0.80" title="-0.000">
    </span><span style="background-color: hsl(0, 100.00%, 93.64%); opacity: 0.81" title="-0.019">s</span><span style="background-color: hsl(0, 100.00%, 88.82%); opacity: 0.83" title="-0.042">t</span><span style="background-color: hsl(0, 100.00%, 92.40%); opacity: 0.82" title="-0.024">o</span><span style="background-color: hsl(0, 100.00%, 92.43%); opacity: 0.82" title="-0.024">n</span><span style="background-color: hsl(120, 100.00%, 96.67%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 92.34%); opacity: 0.82" title="0.025">s</span><span style="background-color: hsl(0, 100.00%, 97.07%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 91.03%); opacity: 0.82" title="-0.031">a</span><span style="background-color: hsl(0, 100.00%, 89.06%); opacity: 0.83" title="-0.041">n</span><span style="background-color: hsl(0, 100.00%, 91.37%); opacity: 0.82" title="-0.029">d</span><span style="background-color: hsl(0, 100.00%, 96.06%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(120, 100.00%, 94.40%); opacity: 0.81" title="0.016">c</span><span style="background-color: hsl(120, 100.00%, 90.71%); opacity: 0.82" title="0.032">h</span><span style="background-color: hsl(120, 100.00%, 87.87%); opacity: 0.84" title="0.047">i</span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.043">l</span><span style="background-color: hsl(0, 100.00%, 98.73%); opacity: 0.80" title="-0.002">d</span><span style="background-color: hsl(0, 100.00%, 92.25%); opacity: 0.82" title="-0.025">r</span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.016">e</span><span style="background-color: hsl(120, 100.00%, 96.29%); opacity: 0.81" title="0.009">n</span><span style="background-color: hsl(120, 100.00%, 96.42%); opacity: 0.81" title="0.008">,</span><span style="background-color: hsl(0, 100.00%, 95.33%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 91.03%); opacity: 0.82" title="-0.031">a</span><span style="background-color: hsl(0, 100.00%, 89.06%); opacity: 0.83" title="-0.041">n</span><span style="background-color: hsl(0, 100.00%, 91.37%); opacity: 0.82" title="-0.029">d</span><span style="background-color: hsl(0, 100.00%, 94.06%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(120, 100.00%, 94.84%); opacity: 0.81" title="0.014">t</span><span style="background-color: hsl(120, 100.00%, 94.73%); opacity: 0.81" title="0.014">h</span><span style="background-color: hsl(120, 100.00%, 95.66%); opacity: 0.81" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 97.26%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 94.40%); opacity: 0.81" title="0.016">c</span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.030">h</span><span style="background-color: hsl(120, 100.00%, 89.02%); opacity: 0.83" title="0.041">i</span><span style="background-color: hsl(120, 100.00%, 90.06%); opacity: 0.83" title="0.036">l</span><span style="background-color: hsl(120, 100.00%, 98.01%); opacity: 0.80" title="0.004">d</span><span style="background-color: hsl(0, 100.00%, 97.49%); opacity: 0.80" title="-0.005">b</span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.005">i</span><span style="background-color: hsl(0, 100.00%, 97.49%); opacity: 0.80" title="-0.005">r</span><span style="background-color: hsl(0, 100.00%, 92.65%); opacity: 0.82" title="-0.023">t</span><span style="background-color: hsl(0, 100.00%, 92.84%); opacity: 0.82" title="-0.022">h</span><span style="background-color: hsl(0, 100.00%, 91.23%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 89.04%); opacity: 0.83" title="-0.041">h</span><span style="background-color: hsl(0, 100.00%, 92.22%); opacity: 0.82" title="-0.025">u</span><span style="background-color: hsl(0, 100.00%, 88.14%); opacity: 0.84" title="-0.046">r</span><span style="background-color: hsl(0, 100.00%, 95.30%); opacity: 0.81" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(0, 100.00%, 90.65%); opacity: 0.83" title="-0.033">e</span><span style="background-color: hsl(0, 100.00%, 88.48%); opacity: 0.83" title="-0.044">s</span><span style="background-color: hsl(0, 100.00%, 87.84%); opacity: 0.84" title="-0.047">s</span><span style="background-color: hsl(0, 100.00%, 92.58%); opacity: 0.82" title="-0.023">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=comp.graphics
        
    </b>
    
        
        (probability <b>0.005</b>, score <b>-6.007</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 94.96%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.974
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -6.981
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 93.91%); opacity: 0.81" title="-0.018">as</span><span style="background-color: hsl(0, 100.00%, 97.59%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 94.81%); opacity: 0.81" title="-0.014">r</span><span style="background-color: hsl(0, 100.00%, 94.05%); opacity: 0.81" title="-0.017">e</span><span style="background-color: hsl(0, 100.00%, 93.66%); opacity: 0.81" title="-0.019">c</span><span style="background-color: hsl(0, 100.00%, 94.32%); opacity: 0.81" title="-0.016">a</span><span style="background-color: hsl(0, 100.00%, 94.63%); opacity: 0.81" title="-0.015">l</span><span style="background-color: hsl(0, 100.00%, 96.63%); opacity: 0.81" title="-0.008">l</span><span style="background-color: hsl(120, 100.00%, 95.60%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 92.98%); opacity: 0.82" title="0.022">f</span><span style="background-color: hsl(120, 100.00%, 92.69%); opacity: 0.82" title="0.023">r</span><span style="background-color: hsl(120, 100.00%, 94.46%); opacity: 0.81" title="0.015">o</span><span style="background-color: hsl(120, 100.00%, 97.10%); opacity: 0.80" title="0.006">m</span><span style="background-color: hsl(0, 100.00%, 91.52%); opacity: 0.82" title="-0.028"> </span><span style="background-color: hsl(0, 100.00%, 88.75%); opacity: 0.83" title="-0.042">my</span><span style="background-color: hsl(0, 100.00%, 96.73%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.024">b</span><span style="background-color: hsl(120, 100.00%, 91.20%); opacity: 0.82" title="0.030">o</span><span style="background-color: hsl(120, 100.00%, 91.53%); opacity: 0.82" title="0.028">u</span><span style="background-color: hsl(120, 100.00%, 94.58%); opacity: 0.81" title="0.015">t</span><span style="background-color: hsl(120, 100.00%, 97.64%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 98.43%); opacity: 0.80" title="0.003">i</span><span style="background-color: hsl(120, 100.00%, 98.08%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 98.07%); opacity: 0.80" title="0.003">h</span><span style="background-color: hsl(0, 100.00%, 96.71%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 93.62%); opacity: 0.81" title="-0.019">k</span><span style="background-color: hsl(0, 100.00%, 92.69%); opacity: 0.82" title="-0.023">i</span><span style="background-color: hsl(0, 100.00%, 92.66%); opacity: 0.82" title="-0.023">d</span><span style="background-color: hsl(0, 100.00%, 96.25%); opacity: 0.81" title="-0.009">n</span><span style="background-color: hsl(0, 100.00%, 94.60%); opacity: 0.81" title="-0.015">e</span><span style="background-color: hsl(0, 100.00%, 97.68%); opacity: 0.80" title="-0.004">y</span><span style="background-color: hsl(0, 100.00%, 92.74%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 95.25%); opacity: 0.81" title="-0.012">s</span><span style="background-color: hsl(0, 100.00%, 90.42%); opacity: 0.83" title="-0.034">t</span><span style="background-color: hsl(0, 100.00%, 92.04%); opacity: 0.82" title="-0.026">o</span><span style="background-color: hsl(0, 100.00%, 92.15%); opacity: 0.82" title="-0.025">n</span><span style="background-color: hsl(0, 100.00%, 97.72%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(0, 100.00%, 97.60%); opacity: 0.80" title="-0.005">s</span><span style="background-color: hsl(120, 100.00%, 96.42%); opacity: 0.81" title="0.008">,</span><span style="background-color: hsl(0, 100.00%, 95.42%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 91.62%); opacity: 0.82" title="-0.028">t</span><span style="background-color: hsl(0, 100.00%, 92.90%); opacity: 0.82" title="-0.022">h</span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(120, 100.00%, 96.12%); opacity: 0.81" title="0.009">r</span><span style="background-color: hsl(120, 100.00%, 95.81%); opacity: 0.81" title="0.010">e</span><span style="background-color: hsl(0, 100.00%, 96.87%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 94.99%); opacity: 0.81" title="-0.013">i</span><span style="background-color: hsl(0, 100.00%, 92.29%); opacity: 0.82" title="-0.025">s</span><span style="background-color: hsl(0, 100.00%, 93.66%); opacity: 0.81" title="-0.019">n</span><span style="background-color: hsl(0, 100.00%, 94.21%); opacity: 0.81" title="-0.016">'</span><span style="background-color: hsl(0, 100.00%, 96.63%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(0, 100.00%, 99.77%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(120, 100.00%, 97.88%); opacity: 0.80" title="0.004">y</span><span style="background-color: hsl(0, 100.00%, 90.29%); opacity: 0.83" title="-0.034">
    </span><span style="background-color: hsl(0, 100.00%, 84.97%); opacity: 0.85" title="-0.064">m</span><span style="background-color: hsl(0, 100.00%, 79.51%); opacity: 0.88" title="-0.100">e</span><span style="background-color: hsl(0, 100.00%, 76.36%); opacity: 0.89" title="-0.123">d</span><span style="background-color: hsl(0, 100.00%, 79.06%); opacity: 0.88" title="-0.103">i</span><span style="background-color: hsl(0, 100.00%, 83.45%); opacity: 0.86" title="-0.074">c</span><span style="background-color: hsl(0, 100.00%, 92.69%); opacity: 0.82" title="-0.023">a</span><span style="background-color: hsl(0, 100.00%, 99.16%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(120, 100.00%, 95.85%); opacity: 0.81" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 97.18%); opacity: 0.80" title="0.006">o</span><span style="background-color: hsl(120, 100.00%, 98.44%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(0, 100.00%, 93.79%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 90.25%); opacity: 0.83" title="-0.035">t</span><span style="background-color: hsl(0, 100.00%, 87.26%); opacity: 0.84" title="-0.051">h</span><span style="background-color: hsl(0, 100.00%, 87.99%); opacity: 0.84" title="-0.047">a</span><span style="background-color: hsl(0, 100.00%, 89.88%); opacity: 0.83" title="-0.037">t</span><span style="background-color: hsl(0, 100.00%, 92.97%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.005">c</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013">a</span><span style="background-color: hsl(0, 100.00%, 96.05%); opacity: 0.81" title="-0.010">n</span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.019"> </span><span style="background-color: hsl(0, 100.00%, 94.49%); opacity: 0.81" title="-0.015">do</span><span style="background-color: hsl(0, 100.00%, 96.27%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 99.43%); opacity: 0.80" title="-0.001">a</span><span style="background-color: hsl(0, 100.00%, 97.18%); opacity: 0.80" title="-0.006">n</span><span style="background-color: hsl(0, 100.00%, 98.10%); opacity: 0.80" title="-0.003">y</span><span style="background-color: hsl(0, 100.00%, 93.45%); opacity: 0.82" title="-0.020">t</span><span style="background-color: hsl(0, 100.00%, 92.39%); opacity: 0.82" title="-0.024">h</span><span style="background-color: hsl(0, 100.00%, 92.40%); opacity: 0.82" title="-0.024">i</span><span style="background-color: hsl(0, 100.00%, 92.84%); opacity: 0.82" title="-0.022">n</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013">g</span><span style="background-color: hsl(0, 100.00%, 97.17%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 99.72%); opacity: 0.80" title="-0.000">a</span><span style="background-color: hsl(120, 100.00%, 97.61%); opacity: 0.80" title="0.005">b</span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.010">o</span><span style="background-color: hsl(120, 100.00%, 94.58%); opacity: 0.81" title="0.015">u</span><span style="background-color: hsl(120, 100.00%, 95.74%); opacity: 0.81" title="0.011">t</span><span style="background-color: hsl(0, 100.00%, 95.50%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 91.76%); opacity: 0.82" title="-0.027">t</span><span style="background-color: hsl(0, 100.00%, 91.28%); opacity: 0.82" title="-0.030">h</span><span style="background-color: hsl(0, 100.00%, 95.91%); opacity: 0.81" title="-0.010">e</span><span style="background-color: hsl(120, 100.00%, 96.82%); opacity: 0.81" title="0.007">m</span><span style="background-color: hsl(0, 100.00%, 99.14%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 92.99%); opacity: 0.82" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 90.41%); opacity: 0.83" title="-0.034">x</span><span style="background-color: hsl(0, 100.00%, 88.58%); opacity: 0.83" title="-0.043">c</span><span style="background-color: hsl(0, 100.00%, 86.00%); opacity: 0.84" title="-0.058">e</span><span style="background-color: hsl(0, 100.00%, 86.64%); opacity: 0.84" title="-0.054">p</span><span style="background-color: hsl(0, 100.00%, 89.00%); opacity: 0.83" title="-0.041">t</span><span style="background-color: hsl(0, 100.00%, 88.11%); opacity: 0.84" title="-0.046"> </span><span style="background-color: hsl(0, 100.00%, 86.40%); opacity: 0.84" title="-0.056">r</span><span style="background-color: hsl(0, 100.00%, 79.93%); opacity: 0.87" title="-0.097">e</span><span style="background-color: hsl(0, 100.00%, 77.66%); opacity: 0.89" title="-0.113">l</span><span style="background-color: hsl(0, 100.00%, 77.94%); opacity: 0.89" title="-0.111">i</span><span style="background-color: hsl(0, 100.00%, 83.61%); opacity: 0.86" title="-0.073">e</span><span style="background-color: hsl(0, 100.00%, 87.50%); opacity: 0.84" title="-0.049">v</span><span style="background-color: hsl(0, 100.00%, 91.93%); opacity: 0.82" title="-0.026">e</span><span style="background-color: hsl(0, 100.00%, 91.08%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 90.18%); opacity: 0.83" title="-0.035">t</span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.046">h</span><span style="background-color: hsl(0, 100.00%, 90.48%); opacity: 0.83" title="-0.033">e</span><span style="background-color: hsl(0, 100.00%, 92.38%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 91.06%); opacity: 0.82" title="-0.031">p</span><span style="background-color: hsl(0, 100.00%, 88.23%); opacity: 0.83" title="-0.045">a</span><span style="background-color: hsl(0, 100.00%, 85.63%); opacity: 0.85" title="-0.060">i</span><span style="background-color: hsl(0, 100.00%, 86.90%); opacity: 0.84" title="-0.053">n</span><span style="background-color: hsl(0, 100.00%, 90.02%); opacity: 0.83" title="-0.036">.</span><span style="background-color: hsl(0, 100.00%, 93.97%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 96.28%); opacity: 0.81" title="-0.009">e</span><span style="background-color: hsl(0, 100.00%, 96.42%); opacity: 0.81" title="-0.008">i</span><span style="background-color: hsl(0, 100.00%, 92.61%); opacity: 0.82" title="-0.023">t</span><span style="background-color: hsl(0, 100.00%, 93.14%); opacity: 0.82" title="-0.021">h</span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.018">e</span><span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.005">r</span><span style="background-color: hsl(0, 100.00%, 92.23%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 87.92%); opacity: 0.84" title="-0.047">t</span><span style="background-color: hsl(0, 100.00%, 85.03%); opacity: 0.85" title="-0.064">h</span><span style="background-color: hsl(0, 100.00%, 85.49%); opacity: 0.85" title="-0.061">e</span><span style="background-color: hsl(0, 100.00%, 88.45%); opacity: 0.83" title="-0.044">y</span><span style="background-color: hsl(0, 100.00%, 91.63%); opacity: 0.82" title="-0.028"> </span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.016">p</span><span style="background-color: hsl(0, 100.00%, 93.83%); opacity: 0.81" title="-0.018">a</span><span style="background-color: hsl(0, 100.00%, 93.86%); opacity: 0.81" title="-0.018">s</span><span style="background-color: hsl(0, 100.00%, 96.79%); opacity: 0.81" title="-0.007">s</span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.007">,</span><span style="background-color: hsl(120, 100.00%, 95.47%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 94.85%); opacity: 0.81" title="0.014">or</span><span style="background-color: hsl(0, 100.00%, 95.38%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 87.92%); opacity: 0.84" title="-0.047">t</span><span style="background-color: hsl(0, 100.00%, 85.03%); opacity: 0.85" title="-0.064">h</span><span style="background-color: hsl(0, 100.00%, 85.49%); opacity: 0.85" title="-0.061">e</span><span style="background-color: hsl(0, 100.00%, 88.45%); opacity: 0.83" title="-0.044">y</span><span style="background-color: hsl(0, 100.00%, 92.81%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 98.51%); opacity: 0.80" title="-0.002">h</span><span style="background-color: hsl(0, 100.00%, 97.32%); opacity: 0.80" title="-0.005">a</span><span style="background-color: hsl(0, 100.00%, 96.09%); opacity: 0.81" title="-0.009">v</span><span style="background-color: hsl(0, 100.00%, 96.42%); opacity: 0.81" title="-0.008">e</span><span style="background-color: hsl(0, 100.00%, 92.66%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 92.32%); opacity: 0.82" title="-0.025">to</span><span style="background-color: hsl(0, 100.00%, 92.23%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.016">be</span><span style="background-color: hsl(0, 100.00%, 91.11%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 89.17%); opacity: 0.83" title="-0.040">b</span><span style="background-color: hsl(0, 100.00%, 88.97%); opacity: 0.83" title="-0.041">r</span><span style="background-color: hsl(0, 100.00%, 90.74%); opacity: 0.82" title="-0.032">o</span><span style="background-color: hsl(0, 100.00%, 93.66%); opacity: 0.81" title="-0.019">k</span><span style="background-color: hsl(0, 100.00%, 93.57%); opacity: 0.81" title="-0.019">e</span><span style="background-color: hsl(0, 100.00%, 94.07%); opacity: 0.81" title="-0.017">n</span><span style="background-color: hsl(0, 100.00%, 97.86%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">up</span><span style="background-color: hsl(0, 100.00%, 99.84%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 98.43%); opacity: 0.80" title="0.003">i</span><span style="background-color: hsl(120, 100.00%, 98.08%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 98.07%); opacity: 0.80" title="0.003">h</span><span style="background-color: hsl(120, 100.00%, 96.53%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(0, 100.00%, 98.59%); opacity: 0.80" title="-0.002">o</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">u</span><span style="background-color: hsl(0, 100.00%, 95.38%); opacity: 0.81" title="-0.012">n</span><span style="background-color: hsl(0, 100.00%, 96.81%); opacity: 0.81" title="-0.007">d</span><span style="background-color: hsl(0, 100.00%, 98.06%); opacity: 0.80" title="-0.003">,</span><span style="background-color: hsl(120, 100.00%, 98.26%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 94.85%); opacity: 0.81" title="0.014">or</span><span style="background-color: hsl(0, 100.00%, 95.38%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 87.92%); opacity: 0.84" title="-0.047">t</span><span style="background-color: hsl(0, 100.00%, 85.03%); opacity: 0.85" title="-0.064">h</span><span style="background-color: hsl(0, 100.00%, 85.49%); opacity: 0.85" title="-0.061">e</span><span style="background-color: hsl(0, 100.00%, 88.45%); opacity: 0.83" title="-0.044">y</span><span style="background-color: hsl(0, 100.00%, 92.81%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 98.51%); opacity: 0.80" title="-0.002">h</span><span style="background-color: hsl(0, 100.00%, 97.32%); opacity: 0.80" title="-0.005">a</span><span style="background-color: hsl(0, 100.00%, 96.09%); opacity: 0.81" title="-0.009">v</span><span style="background-color: hsl(0, 100.00%, 96.42%); opacity: 0.81" title="-0.008">e</span><span style="background-color: hsl(0, 100.00%, 92.66%); opacity: 0.82" title="-0.023">
    </span><span style="background-color: hsl(0, 100.00%, 92.32%); opacity: 0.82" title="-0.025">to</span><span style="background-color: hsl(0, 100.00%, 92.23%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.016">be</span><span style="background-color: hsl(0, 100.00%, 96.50%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 96.61%); opacity: 0.81" title="-0.008">e</span><span style="background-color: hsl(0, 100.00%, 94.73%); opacity: 0.81" title="-0.014">x</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 93.81%); opacity: 0.81" title="0.018">r</span><span style="background-color: hsl(120, 100.00%, 93.10%); opacity: 0.82" title="0.021">a</span><span style="background-color: hsl(120, 100.00%, 93.85%); opacity: 0.81" title="0.018">c</span><span style="background-color: hsl(0, 100.00%, 96.00%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 95.36%); opacity: 0.81" title="-0.012">e</span><span style="background-color: hsl(0, 100.00%, 94.94%); opacity: 0.81" title="-0.014">d</span><span style="background-color: hsl(0, 100.00%, 93.20%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(0, 100.00%, 90.88%); opacity: 0.82" title="-0.031">s</span><span style="background-color: hsl(0, 100.00%, 87.50%); opacity: 0.84" title="-0.049">u</span><span style="background-color: hsl(0, 100.00%, 87.08%); opacity: 0.84" title="-0.052">r</span><span style="background-color: hsl(0, 100.00%, 87.28%); opacity: 0.84" title="-0.051">g</span><span style="background-color: hsl(0, 100.00%, 89.05%); opacity: 0.83" title="-0.041">i</span><span style="background-color: hsl(0, 100.00%, 87.77%); opacity: 0.84" title="-0.048">c</span><span style="background-color: hsl(0, 100.00%, 88.05%); opacity: 0.84" title="-0.046">a</span><span style="background-color: hsl(0, 100.00%, 89.43%); opacity: 0.83" title="-0.039">l</span><span style="background-color: hsl(0, 100.00%, 92.97%); opacity: 0.82" title="-0.022">l</span><span style="background-color: hsl(0, 100.00%, 93.88%); opacity: 0.81" title="-0.018">y</span><span style="background-color: hsl(0, 100.00%, 96.33%); opacity: 0.81" title="-0.009">.</span><span style="background-color: hsl(0, 100.00%, 94.83%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 95.32%); opacity: 0.81" title="-0.012">w</span><span style="background-color: hsl(0, 100.00%, 94.00%); opacity: 0.81" title="-0.017">h</span><span style="background-color: hsl(0, 100.00%, 95.01%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 93.87%); opacity: 0.81" title="-0.018">n</span><span style="background-color: hsl(0, 100.00%, 96.87%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 96.15%); opacity: 0.81" title="-0.009">w</span><span style="background-color: hsl(0, 100.00%, 95.80%); opacity: 0.81" title="-0.010">a</span><span style="background-color: hsl(0, 100.00%, 97.53%); opacity: 0.80" title="-0.005">s</span><span style="background-color: hsl(0, 100.00%, 94.44%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 94.38%); opacity: 0.81" title="-0.016">i</span><span style="background-color: hsl(0, 100.00%, 93.93%); opacity: 0.81" title="-0.018">n</span><span style="background-color: hsl(0, 100.00%, 98.25%); opacity: 0.80" title="-0.003">,</span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 90.18%); opacity: 0.83" title="-0.035">t</span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.046">h</span><span style="background-color: hsl(0, 100.00%, 90.48%); opacity: 0.83" title="-0.033">e</span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(120, 100.00%, 98.41%); opacity: 0.80" title="0.003">x</span><span style="background-color: hsl(120, 100.00%, 97.93%); opacity: 0.80" title="0.004">-</span><span style="background-color: hsl(120, 100.00%, 88.19%); opacity: 0.83" title="0.046">r</span><span style="background-color: hsl(120, 100.00%, 88.97%); opacity: 0.83" title="0.041">a</span><span style="background-color: hsl(120, 100.00%, 88.95%); opacity: 0.83" title="0.041">y</span><span style="background-color: hsl(120, 100.00%, 89.08%); opacity: 0.83" title="0.041"> </span><span style="background-color: hsl(120, 100.00%, 89.14%); opacity: 0.83" title="0.040">t</span><span style="background-color: hsl(120, 100.00%, 88.87%); opacity: 0.83" title="0.042">e</span><span style="background-color: hsl(120, 100.00%, 90.77%); opacity: 0.82" title="0.032">c</span><span style="background-color: hsl(120, 100.00%, 95.52%); opacity: 0.81" title="0.011">h</span><span style="background-color: hsl(0, 100.00%, 95.86%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 91.28%); opacity: 0.82" title="-0.030">h</span><span style="background-color: hsl(0, 100.00%, 87.61%); opacity: 0.84" title="-0.049">a</span><span style="background-color: hsl(0, 100.00%, 84.69%); opacity: 0.85" title="-0.066">p</span><span style="background-color: hsl(0, 100.00%, 86.70%); opacity: 0.84" title="-0.054">p</span><span style="background-color: hsl(0, 100.00%, 88.45%); opacity: 0.83" title="-0.044">e</span><span style="background-color: hsl(0, 100.00%, 92.19%); opacity: 0.82" title="-0.025">n</span><span style="background-color: hsl(0, 100.00%, 94.97%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 98.17%); opacity: 0.80" title="-0.003">d</span><span style="background-color: hsl(0, 100.00%, 93.75%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 92.32%); opacity: 0.82" title="-0.025">to</span><span style="background-color: hsl(0, 100.00%, 92.23%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 92.15%); opacity: 0.82" title="-0.025">m</span><span style="background-color: hsl(0, 100.00%, 87.78%); opacity: 0.84" title="-0.048">e</span><span style="background-color: hsl(0, 100.00%, 85.17%); opacity: 0.85" title="-0.063">n</span><span style="background-color: hsl(0, 100.00%, 86.27%); opacity: 0.84" title="-0.056">t</span><span style="background-color: hsl(0, 100.00%, 89.77%); opacity: 0.83" title="-0.037">i</span><span style="background-color: hsl(0, 100.00%, 94.32%); opacity: 0.81" title="-0.016">o</span><span style="background-color: hsl(0, 100.00%, 97.68%); opacity: 0.80" title="-0.004">n</span><span style="background-color: hsl(0, 100.00%, 93.79%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 90.25%); opacity: 0.83" title="-0.035">t</span><span style="background-color: hsl(0, 100.00%, 87.26%); opacity: 0.84" title="-0.051">h</span><span style="background-color: hsl(0, 100.00%, 87.99%); opacity: 0.84" title="-0.047">a</span><span style="background-color: hsl(0, 100.00%, 89.88%); opacity: 0.83" title="-0.037">t</span><span style="background-color: hsl(0, 100.00%, 88.51%); opacity: 0.83" title="-0.044"> </span><span style="background-color: hsl(0, 100.00%, 88.98%); opacity: 0.83" title="-0.041">s</span><span style="background-color: hsl(0, 100.00%, 88.54%); opacity: 0.83" title="-0.044">h</span><span style="background-color: hsl(0, 100.00%, 92.50%); opacity: 0.82" title="-0.024">e</span><span style="background-color: hsl(120, 100.00%, 97.39%); opacity: 0.80" title="0.005">'</span><span style="background-color: hsl(120, 100.00%, 96.30%); opacity: 0.81" title="0.009">d</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(0, 100.00%, 97.17%); opacity: 0.80" title="-0.006">a</span><span style="background-color: hsl(0, 100.00%, 97.34%); opacity: 0.80" title="-0.005">d</span><span style="background-color: hsl(0, 100.00%, 93.45%); opacity: 0.82" title="-0.020"> </span><span style="background-color: hsl(0, 100.00%, 93.62%); opacity: 0.81" title="-0.019">k</span><span style="background-color: hsl(0, 100.00%, 92.69%); opacity: 0.82" title="-0.023">i</span><span style="background-color: hsl(0, 100.00%, 92.66%); opacity: 0.82" title="-0.023">d</span><span style="background-color: hsl(0, 100.00%, 96.25%); opacity: 0.81" title="-0.009">n</span><span style="background-color: hsl(0, 100.00%, 94.60%); opacity: 0.81" title="-0.015">e</span><span style="background-color: hsl(0, 100.00%, 97.68%); opacity: 0.80" title="-0.004">y</span><span style="background-color: hsl(0, 100.00%, 92.74%); opacity: 0.82" title="-0.023">
    </span><span style="background-color: hsl(0, 100.00%, 95.25%); opacity: 0.81" title="-0.012">s</span><span style="background-color: hsl(0, 100.00%, 90.42%); opacity: 0.83" title="-0.034">t</span><span style="background-color: hsl(0, 100.00%, 91.39%); opacity: 0.82" title="-0.029">o</span><span style="background-color: hsl(0, 100.00%, 91.62%); opacity: 0.82" title="-0.028">n</span><span style="background-color: hsl(0, 100.00%, 96.85%); opacity: 0.81" title="-0.007">e</span><span style="background-color: hsl(0, 100.00%, 96.24%); opacity: 0.81" title="-0.009">s</span><span style="background-color: hsl(0, 100.00%, 98.25%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.016">a</span><span style="background-color: hsl(0, 100.00%, 93.27%); opacity: 0.82" title="-0.020">n</span><span style="background-color: hsl(0, 100.00%, 93.69%); opacity: 0.81" title="-0.019">d</span><span style="background-color: hsl(0, 100.00%, 87.42%); opacity: 0.84" title="-0.050"> </span><span style="background-color: hsl(0, 100.00%, 86.26%); opacity: 0.84" title="-0.057">c</span><span style="background-color: hsl(0, 100.00%, 82.65%); opacity: 0.86" title="-0.079">h</span><span style="background-color: hsl(0, 100.00%, 82.95%); opacity: 0.86" title="-0.077">i</span><span style="background-color: hsl(0, 100.00%, 81.86%); opacity: 0.86" title="-0.084">l</span><span style="background-color: hsl(0, 100.00%, 83.05%); opacity: 0.86" title="-0.076">d</span><span style="background-color: hsl(0, 100.00%, 87.17%); opacity: 0.84" title="-0.051">r</span><span style="background-color: hsl(0, 100.00%, 88.83%); opacity: 0.83" title="-0.042">e</span><span style="background-color: hsl(0, 100.00%, 92.58%); opacity: 0.82" title="-0.023">n</span><span style="background-color: hsl(0, 100.00%, 96.65%); opacity: 0.81" title="-0.008">,</span><span style="background-color: hsl(0, 100.00%, 95.60%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.016">a</span><span style="background-color: hsl(0, 100.00%, 93.27%); opacity: 0.82" title="-0.020">n</span><span style="background-color: hsl(0, 100.00%, 93.69%); opacity: 0.81" title="-0.019">d</span><span style="background-color: hsl(0, 100.00%, 90.89%); opacity: 0.82" title="-0.031"> </span><span style="background-color: hsl(0, 100.00%, 90.18%); opacity: 0.83" title="-0.035">t</span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.046">h</span><span style="background-color: hsl(0, 100.00%, 90.48%); opacity: 0.83" title="-0.033">e</span><span style="background-color: hsl(0, 100.00%, 86.65%); opacity: 0.84" title="-0.054"> </span><span style="background-color: hsl(0, 100.00%, 86.26%); opacity: 0.84" title="-0.057">c</span><span style="background-color: hsl(0, 100.00%, 83.24%); opacity: 0.86" title="-0.075">h</span><span style="background-color: hsl(0, 100.00%, 84.92%); opacity: 0.85" title="-0.065">i</span><span style="background-color: hsl(0, 100.00%, 86.18%); opacity: 0.84" title="-0.057">l</span><span style="background-color: hsl(0, 100.00%, 90.68%); opacity: 0.82" title="-0.032">d</span><span style="background-color: hsl(0, 100.00%, 95.21%); opacity: 0.81" title="-0.013">b</span><span style="background-color: hsl(0, 100.00%, 99.29%); opacity: 0.80" title="-0.001">i</span><span style="background-color: hsl(0, 100.00%, 98.27%); opacity: 0.80" title="-0.003">r</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(0, 100.00%, 95.90%); opacity: 0.81" title="-0.010">h</span><span style="background-color: hsl(0, 100.00%, 94.51%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 91.03%); opacity: 0.82" title="-0.031">h</span><span style="background-color: hsl(0, 100.00%, 90.46%); opacity: 0.83" title="-0.034">u</span><span style="background-color: hsl(0, 100.00%, 93.14%); opacity: 0.82" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 98.37%); opacity: 0.80" title="-0.003">t</span><span style="background-color: hsl(0, 100.00%, 94.94%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 95.82%); opacity: 0.81" title="-0.010">l</span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.014">e</span><span style="background-color: hsl(0, 100.00%, 95.86%); opacity: 0.81" title="-0.010">s</span><span style="background-color: hsl(0, 100.00%, 94.15%); opacity: 0.81" title="-0.017">s</span><span style="background-color: hsl(0, 100.00%, 97.35%); opacity: 0.80" title="-0.005">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.834</b>, score <b>-0.440</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 91.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.134
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 90.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.573
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 95.99%); opacity: 0.81" title="-0.010">as</span><span style="background-color: hsl(0, 100.00%, 96.77%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 99.04%); opacity: 0.80" title="-0.001">i</span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.024"> </span><span style="background-color: hsl(120, 100.00%, 86.87%); opacity: 0.84" title="0.053">r</span><span style="background-color: hsl(120, 100.00%, 83.39%); opacity: 0.86" title="0.074">e</span><span style="background-color: hsl(120, 100.00%, 81.51%); opacity: 0.87" title="0.086">c</span><span style="background-color: hsl(120, 100.00%, 84.88%); opacity: 0.85" title="0.065">a</span><span style="background-color: hsl(120, 100.00%, 88.92%); opacity: 0.83" title="0.042">l</span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.010">l</span><span style="background-color: hsl(0, 100.00%, 95.25%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 94.73%); opacity: 0.81" title="-0.014">f</span><span style="background-color: hsl(0, 100.00%, 94.46%); opacity: 0.81" title="-0.015">r</span><span style="background-color: hsl(0, 100.00%, 96.35%); opacity: 0.81" title="-0.009">o</span><span style="background-color: hsl(0, 100.00%, 97.65%); opacity: 0.80" title="-0.005">m</span><span style="background-color: hsl(120, 100.00%, 90.30%); opacity: 0.83" title="0.034"> </span><span style="background-color: hsl(120, 100.00%, 86.82%); opacity: 0.84" title="0.053">my</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.019"> </span><span style="background-color: hsl(0, 100.00%, 95.95%); opacity: 0.81" title="-0.010">b</span><span style="background-color: hsl(0, 100.00%, 96.17%); opacity: 0.81" title="-0.009">o</span><span style="background-color: hsl(0, 100.00%, 99.26%); opacity: 0.80" title="-0.001">u</span><span style="background-color: hsl(120, 100.00%, 97.86%); opacity: 0.80" title="0.004">t</span><span style="background-color: hsl(0, 100.00%, 97.52%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 96.72%); opacity: 0.81" title="-0.007">w</span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.017">i</span><span style="background-color: hsl(0, 100.00%, 94.46%); opacity: 0.81" title="-0.015">t</span><span style="background-color: hsl(0, 100.00%, 94.85%); opacity: 0.81" title="-0.014">h</span><span style="background-color: hsl(0, 100.00%, 98.93%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(120, 100.00%, 91.85%); opacity: 0.82" title="0.027">k</span><span style="background-color: hsl(120, 100.00%, 89.22%); opacity: 0.83" title="0.040">i</span><span style="background-color: hsl(120, 100.00%, 84.09%); opacity: 0.85" title="0.070">d</span><span style="background-color: hsl(120, 100.00%, 84.13%); opacity: 0.85" title="0.069">n</span><span style="background-color: hsl(120, 100.00%, 85.54%); opacity: 0.85" title="0.061">e</span><span style="background-color: hsl(120, 100.00%, 92.19%); opacity: 0.82" title="0.025">y</span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(0, 100.00%, 95.72%); opacity: 0.81" title="-0.011">s</span><span style="background-color: hsl(0, 100.00%, 94.43%); opacity: 0.81" title="-0.016">t</span><span style="background-color: hsl(0, 100.00%, 94.01%); opacity: 0.81" title="-0.017">o</span><span style="background-color: hsl(120, 100.00%, 98.52%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(0, 100.00%, 99.17%); opacity: 0.80" title="-0.001">e</span><span style="background-color: hsl(120, 100.00%, 95.57%); opacity: 0.81" title="0.011">s</span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.006">,</span><span style="background-color: hsl(0, 100.00%, 95.57%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 92.17%); opacity: 0.82" title="-0.025">t</span><span style="background-color: hsl(0, 100.00%, 89.50%); opacity: 0.83" title="-0.039">h</span><span style="background-color: hsl(0, 100.00%, 89.28%); opacity: 0.83" title="-0.040">e</span><span style="background-color: hsl(0, 100.00%, 91.61%); opacity: 0.82" title="-0.028">r</span><span style="background-color: hsl(0, 100.00%, 94.04%); opacity: 0.81" title="-0.017">e</span><span style="background-color: hsl(0, 100.00%, 95.47%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 94.27%); opacity: 0.81" title="-0.016">i</span><span style="background-color: hsl(0, 100.00%, 93.06%); opacity: 0.82" title="-0.021">s</span><span style="background-color: hsl(0, 100.00%, 92.25%); opacity: 0.82" title="-0.025">n</span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.018">'</span><span style="background-color: hsl(0, 100.00%, 95.97%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 97.41%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 98.29%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(0, 100.00%, 99.37%); opacity: 0.80" title="-0.001">n</span><span style="background-color: hsl(0, 100.00%, 99.27%); opacity: 0.80" title="-0.001">y</span><span style="background-color: hsl(120, 100.00%, 83.42%); opacity: 0.86" title="0.074">
    </span><span style="background-color: hsl(120, 100.00%, 73.01%); opacity: 0.91" title="0.148">m</span><span style="background-color: hsl(120, 100.00%, 64.18%); opacity: 0.97" title="0.222">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.260">d</span><span style="background-color: hsl(120, 100.00%, 65.31%); opacity: 0.96" title="0.212">i</span><span style="background-color: hsl(120, 100.00%, 73.51%); opacity: 0.91" title="0.144">c</span><span style="background-color: hsl(120, 100.00%, 85.01%); opacity: 0.85" title="0.064">a</span><span style="background-color: hsl(120, 100.00%, 94.89%); opacity: 0.81" title="0.014">t</span><span style="background-color: hsl(0, 100.00%, 96.37%); opacity: 0.81" title="-0.008">i</span><span style="background-color: hsl(0, 100.00%, 94.70%); opacity: 0.81" title="-0.014">o</span><span style="background-color: hsl(0, 100.00%, 95.22%); opacity: 0.81" title="-0.013">n</span><span style="background-color: hsl(0, 100.00%, 94.03%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 93.24%); opacity: 0.82" title="-0.021">t</span><span style="background-color: hsl(0, 100.00%, 92.06%); opacity: 0.82" title="-0.026">h</span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.019">a</span><span style="background-color: hsl(0, 100.00%, 95.04%); opacity: 0.81" title="-0.013">t</span><span style="background-color: hsl(120, 100.00%, 97.35%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.016">c</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(0, 100.00%, 96.08%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 96.01%); opacity: 0.81" title="-0.010">do</span><span style="background-color: hsl(0, 100.00%, 93.66%); opacity: 0.81" title="-0.019"> </span><span style="background-color: hsl(0, 100.00%, 92.70%); opacity: 0.82" title="-0.023">a</span><span style="background-color: hsl(0, 100.00%, 88.73%); opacity: 0.83" title="-0.043">n</span><span style="background-color: hsl(0, 100.00%, 85.34%); opacity: 0.85" title="-0.062">y</span><span style="background-color: hsl(0, 100.00%, 86.84%); opacity: 0.84" title="-0.053">t</span><span style="background-color: hsl(0, 100.00%, 91.09%); opacity: 0.82" title="-0.030">h</span><span style="background-color: hsl(0, 100.00%, 94.02%); opacity: 0.81" title="-0.017">i</span><span style="background-color: hsl(0, 100.00%, 96.06%); opacity: 0.81" title="-0.009">n</span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.011">g</span><span style="background-color: hsl(0, 100.00%, 99.24%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(120, 100.00%, 95.76%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 94.22%); opacity: 0.81" title="0.016">b</span><span style="background-color: hsl(120, 100.00%, 94.20%); opacity: 0.81" title="0.016">o</span><span style="background-color: hsl(120, 100.00%, 95.35%); opacity: 0.81" title="0.012">u</span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.006">t</span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 91.56%); opacity: 0.82" title="-0.028">t</span><span style="background-color: hsl(0, 100.00%, 91.67%); opacity: 0.82" title="-0.028">h</span><span style="background-color: hsl(0, 100.00%, 91.89%); opacity: 0.82" title="-0.027">e</span><span style="background-color: hsl(0, 100.00%, 94.97%); opacity: 0.81" title="-0.013">m</span><span style="background-color: hsl(0, 100.00%, 93.81%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 97.34%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.001">x</span><span style="background-color: hsl(0, 100.00%, 97.64%); opacity: 0.80" title="-0.005">c</span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.016">e</span><span style="background-color: hsl(120, 100.00%, 95.33%); opacity: 0.81" title="0.012">p</span><span style="background-color: hsl(120, 100.00%, 94.97%); opacity: 0.81" title="0.013">t</span><span style="background-color: hsl(0, 100.00%, 95.97%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 91.81%); opacity: 0.82" title="-0.027">r</span><span style="background-color: hsl(0, 100.00%, 88.63%); opacity: 0.83" title="-0.043">e</span><span style="background-color: hsl(0, 100.00%, 87.38%); opacity: 0.84" title="-0.050">l</span><span style="background-color: hsl(0, 100.00%, 89.71%); opacity: 0.83" title="-0.037">i</span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 95.04%); opacity: 0.81" title="-0.013">v</span><span style="background-color: hsl(0, 100.00%, 97.71%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(0, 100.00%, 94.29%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 91.94%); opacity: 0.82" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.030">h</span><span style="background-color: hsl(0, 100.00%, 92.59%); opacity: 0.82" title="-0.023">e</span><span style="background-color: hsl(120, 100.00%, 93.69%); opacity: 0.81" title="0.019"> </span><span style="background-color: hsl(120, 100.00%, 83.51%); opacity: 0.86" title="0.073">p</span><span style="background-color: hsl(120, 100.00%, 80.41%); opacity: 0.87" title="0.094">a</span><span style="background-color: hsl(120, 100.00%, 77.90%); opacity: 0.89" title="0.111">i</span><span style="background-color: hsl(120, 100.00%, 83.19%); opacity: 0.86" title="0.075">n</span><span style="background-color: hsl(120, 100.00%, 90.24%); opacity: 0.83" title="0.035">.</span><span style="background-color: hsl(120, 100.00%, 97.40%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(0, 100.00%, 92.05%); opacity: 0.82" title="-0.026">e</span><span style="background-color: hsl(0, 100.00%, 89.59%); opacity: 0.83" title="-0.038">i</span><span style="background-color: hsl(0, 100.00%, 87.56%); opacity: 0.84" title="-0.049">t</span><span style="background-color: hsl(0, 100.00%, 87.92%); opacity: 0.84" title="-0.047">h</span><span style="background-color: hsl(0, 100.00%, 91.32%); opacity: 0.82" title="-0.029">e</span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.014">r</span><span style="background-color: hsl(0, 100.00%, 96.17%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 97.72%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.009">y</span><span style="background-color: hsl(0, 100.00%, 97.54%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 91.82%); opacity: 0.82" title="-0.027">p</span><span style="background-color: hsl(0, 100.00%, 89.46%); opacity: 0.83" title="-0.039">a</span><span style="background-color: hsl(0, 100.00%, 88.71%); opacity: 0.83" title="-0.043">s</span><span style="background-color: hsl(0, 100.00%, 91.70%); opacity: 0.82" title="-0.027">s</span><span style="background-color: hsl(0, 100.00%, 95.82%); opacity: 0.81" title="-0.010">,</span><span style="background-color: hsl(0, 100.00%, 96.09%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 96.22%); opacity: 0.81" title="-0.009">or</span><span style="background-color: hsl(0, 100.00%, 94.56%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 97.72%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.009">y</span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.018">h</span><span style="background-color: hsl(0, 100.00%, 91.90%); opacity: 0.82" title="-0.027">a</span><span style="background-color: hsl(0, 100.00%, 91.83%); opacity: 0.82" title="-0.027">v</span><span style="background-color: hsl(0, 100.00%, 93.16%); opacity: 0.82" title="-0.021">e</span><span style="background-color: hsl(0, 100.00%, 99.73%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.77%); opacity: 0.81" title="0.010">to</span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 94.60%); opacity: 0.81" title="-0.015">be</span><span style="background-color: hsl(120, 100.00%, 97.91%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 94.93%); opacity: 0.81" title="0.014">b</span><span style="background-color: hsl(120, 100.00%, 94.91%); opacity: 0.81" title="0.014">r</span><span style="background-color: hsl(120, 100.00%, 96.06%); opacity: 0.81" title="0.009">o</span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.002">k</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(0, 100.00%, 96.68%); opacity: 0.81" title="-0.007">n</span><span style="background-color: hsl(0, 100.00%, 99.97%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.65%); opacity: 0.80" title="0.002">up</span><span style="background-color: hsl(0, 100.00%, 99.11%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 96.72%); opacity: 0.81" title="-0.007">w</span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.017">i</span><span style="background-color: hsl(0, 100.00%, 94.46%); opacity: 0.81" title="-0.015">t</span><span style="background-color: hsl(0, 100.00%, 94.85%); opacity: 0.81" title="-0.014">h</span><span style="background-color: hsl(0, 100.00%, 97.75%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 90.69%); opacity: 0.82" title="0.032">s</span><span style="background-color: hsl(120, 100.00%, 89.38%); opacity: 0.83" title="0.039">o</span><span style="background-color: hsl(120, 100.00%, 91.14%); opacity: 0.82" title="0.030">u</span><span style="background-color: hsl(120, 100.00%, 96.32%); opacity: 0.81" title="0.009">n</span><span style="background-color: hsl(0, 100.00%, 93.33%); opacity: 0.82" title="-0.020">d</span><span style="background-color: hsl(0, 100.00%, 91.30%); opacity: 0.82" title="-0.029">,</span><span style="background-color: hsl(0, 100.00%, 94.05%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 96.22%); opacity: 0.81" title="-0.009">or</span><span style="background-color: hsl(0, 100.00%, 94.56%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 97.72%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.009">y</span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.018">h</span><span style="background-color: hsl(0, 100.00%, 91.90%); opacity: 0.82" title="-0.027">a</span><span style="background-color: hsl(0, 100.00%, 91.83%); opacity: 0.82" title="-0.027">v</span><span style="background-color: hsl(0, 100.00%, 93.16%); opacity: 0.82" title="-0.021">e</span><span style="background-color: hsl(0, 100.00%, 99.73%); opacity: 0.80" title="-0.000">
    </span><span style="background-color: hsl(120, 100.00%, 95.77%); opacity: 0.81" title="0.010">to</span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 94.60%); opacity: 0.81" title="-0.015">be</span><span style="background-color: hsl(0, 100.00%, 96.21%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.008">e</span><span style="background-color: hsl(120, 100.00%, 91.82%); opacity: 0.82" title="0.027">x</span><span style="background-color: hsl(120, 100.00%, 90.11%); opacity: 0.83" title="0.035">t</span><span style="background-color: hsl(120, 100.00%, 89.08%); opacity: 0.83" title="0.041">r</span><span style="background-color: hsl(120, 100.00%, 89.62%); opacity: 0.83" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 91.08%); opacity: 0.82" title="0.031">c</span><span style="background-color: hsl(120, 100.00%, 88.98%); opacity: 0.83" title="0.041">t</span><span style="background-color: hsl(120, 100.00%, 89.87%); opacity: 0.83" title="0.037">e</span><span style="background-color: hsl(120, 100.00%, 92.26%); opacity: 0.82" title="0.025">d</span><span style="background-color: hsl(120, 100.00%, 86.29%); opacity: 0.84" title="0.056"> </span><span style="background-color: hsl(120, 100.00%, 79.89%); opacity: 0.87" title="0.097">s</span><span style="background-color: hsl(120, 100.00%, 74.78%); opacity: 0.90" title="0.135">u</span><span style="background-color: hsl(120, 100.00%, 73.87%); opacity: 0.91" title="0.142">r</span><span style="background-color: hsl(120, 100.00%, 79.24%); opacity: 0.88" title="0.102">g</span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.055">i</span><span style="background-color: hsl(120, 100.00%, 86.61%); opacity: 0.84" title="0.054">c</span><span style="background-color: hsl(120, 100.00%, 88.50%); opacity: 0.83" title="0.044">a</span><span style="background-color: hsl(120, 100.00%, 89.82%); opacity: 0.83" title="0.037">l</span><span style="background-color: hsl(120, 100.00%, 92.97%); opacity: 0.82" title="0.022">l</span><span style="background-color: hsl(120, 100.00%, 94.38%); opacity: 0.81" title="0.016">y</span><span style="background-color: hsl(120, 100.00%, 97.50%); opacity: 0.80" title="0.005">.</span><span style="background-color: hsl(120, 100.00%, 96.88%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 94.61%); opacity: 0.81" title="0.015">w</span><span style="background-color: hsl(120, 100.00%, 93.53%); opacity: 0.81" title="0.019">h</span><span style="background-color: hsl(120, 100.00%, 93.06%); opacity: 0.82" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 94.06%); opacity: 0.81" title="0.017">n</span><span style="background-color: hsl(120, 100.00%, 97.48%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(0, 100.00%, 99.04%); opacity: 0.80" title="-0.001">i</span><span style="background-color: hsl(0, 100.00%, 98.47%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(0, 100.00%, 97.28%); opacity: 0.80" title="-0.006">w</span><span style="background-color: hsl(0, 100.00%, 96.39%); opacity: 0.81" title="-0.008">a</span><span style="background-color: hsl(0, 100.00%, 95.98%); opacity: 0.81" title="-0.010">s</span><span style="background-color: hsl(120, 100.00%, 96.14%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.012">i</span><span style="background-color: hsl(120, 100.00%, 97.84%); opacity: 0.80" title="0.004">n</span><span style="background-color: hsl(0, 100.00%, 96.65%); opacity: 0.81" title="-0.008">,</span><span style="background-color: hsl(0, 100.00%, 93.42%); opacity: 0.82" title="-0.020"> </span><span style="background-color: hsl(0, 100.00%, 91.94%); opacity: 0.82" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.030">h</span><span style="background-color: hsl(0, 100.00%, 92.59%); opacity: 0.82" title="-0.023">e</span><span style="background-color: hsl(0, 100.00%, 99.19%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(120, 100.00%, 93.75%); opacity: 0.81" title="0.018">x</span><span style="background-color: hsl(120, 100.00%, 90.25%); opacity: 0.83" title="0.035">-</span><span style="background-color: hsl(0, 100.00%, 96.53%); opacity: 0.81" title="-0.008">r</span><span style="background-color: hsl(0, 100.00%, 93.27%); opacity: 0.82" title="-0.020">a</span><span style="background-color: hsl(0, 100.00%, 89.70%); opacity: 0.83" title="-0.037">y</span><span style="background-color: hsl(0, 100.00%, 91.83%); opacity: 0.82" title="-0.027"> </span><span style="background-color: hsl(0, 100.00%, 97.43%); opacity: 0.80" title="-0.005">t</span><span style="background-color: hsl(120, 100.00%, 96.99%); opacity: 0.80" title="0.006">e</span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.010">c</span><span style="background-color: hsl(120, 100.00%, 95.80%); opacity: 0.81" title="0.010">h</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 95.97%); opacity: 0.81" title="0.010">h</span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.025">a</span><span style="background-color: hsl(120, 100.00%, 89.32%); opacity: 0.83" title="0.039">p</span><span style="background-color: hsl(120, 100.00%, 91.66%); opacity: 0.82" title="0.028">p</span><span style="background-color: hsl(120, 100.00%, 95.31%); opacity: 0.81" title="0.012">e</span><span style="background-color: hsl(0, 100.00%, 94.84%); opacity: 0.81" title="-0.014">n</span><span style="background-color: hsl(0, 100.00%, 92.87%); opacity: 0.82" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 94.18%); opacity: 0.81" title="-0.017">d</span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 95.77%); opacity: 0.81" title="0.010">to</span><span style="background-color: hsl(120, 100.00%, 93.46%); opacity: 0.82" title="0.020"> </span><span style="background-color: hsl(120, 100.00%, 93.95%); opacity: 0.81" title="0.018">m</span><span style="background-color: hsl(120, 100.00%, 89.80%); opacity: 0.83" title="0.037">e</span><span style="background-color: hsl(120, 100.00%, 89.19%); opacity: 0.83" title="0.040">n</span><span style="background-color: hsl(120, 100.00%, 89.88%); opacity: 0.83" title="0.037">t</span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.021">i</span><span style="background-color: hsl(0, 100.00%, 96.88%); opacity: 0.81" title="-0.007">o</span><span style="background-color: hsl(0, 100.00%, 96.41%); opacity: 0.81" title="-0.008">n</span><span style="background-color: hsl(0, 100.00%, 94.03%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 93.24%); opacity: 0.82" title="-0.021">t</span><span style="background-color: hsl(0, 100.00%, 92.06%); opacity: 0.82" title="-0.026">h</span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.019">a</span><span style="background-color: hsl(0, 100.00%, 95.04%); opacity: 0.81" title="-0.013">t</span><span style="background-color: hsl(120, 100.00%, 91.68%); opacity: 0.82" title="0.028"> </span><span style="background-color: hsl(120, 100.00%, 88.34%); opacity: 0.83" title="0.045">s</span><span style="background-color: hsl(120, 100.00%, 87.07%); opacity: 0.84" title="0.052">h</span><span style="background-color: hsl(120, 100.00%, 91.36%); opacity: 0.82" title="0.029">e</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.002">'</span><span style="background-color: hsl(0, 100.00%, 96.93%); opacity: 0.81" title="-0.007">d</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.97%); opacity: 0.81" title="0.010">h</span><span style="background-color: hsl(120, 100.00%, 95.42%); opacity: 0.81" title="0.012">a</span><span style="background-color: hsl(120, 100.00%, 94.59%); opacity: 0.81" title="0.015">d</span><span style="background-color: hsl(120, 100.00%, 93.98%); opacity: 0.81" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 91.85%); opacity: 0.82" title="0.027">k</span><span style="background-color: hsl(120, 100.00%, 89.22%); opacity: 0.83" title="0.040">i</span><span style="background-color: hsl(120, 100.00%, 84.09%); opacity: 0.85" title="0.070">d</span><span style="background-color: hsl(120, 100.00%, 84.13%); opacity: 0.85" title="0.069">n</span><span style="background-color: hsl(120, 100.00%, 85.54%); opacity: 0.85" title="0.061">e</span><span style="background-color: hsl(120, 100.00%, 92.19%); opacity: 0.82" title="0.025">y</span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.003">
    </span><span style="background-color: hsl(0, 100.00%, 95.72%); opacity: 0.81" title="-0.011">s</span><span style="background-color: hsl(0, 100.00%, 94.43%); opacity: 0.81" title="-0.016">t</span><span style="background-color: hsl(0, 100.00%, 95.24%); opacity: 0.81" title="-0.012">o</span><span style="background-color: hsl(120, 100.00%, 94.50%); opacity: 0.81" title="0.015">n</span><span style="background-color: hsl(120, 100.00%, 93.35%); opacity: 0.82" title="0.020">e</span><span style="background-color: hsl(120, 100.00%, 90.63%); opacity: 0.83" title="0.033">s</span><span style="background-color: hsl(120, 100.00%, 93.82%); opacity: 0.81" title="0.018"> </span><span style="background-color: hsl(120, 100.00%, 97.13%); opacity: 0.80" title="0.006">a</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.011">n</span><span style="background-color: hsl(120, 100.00%, 95.80%); opacity: 0.81" title="0.010">d</span><span style="background-color: hsl(120, 100.00%, 96.70%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(0, 100.00%, 97.38%); opacity: 0.80" title="-0.005">c</span><span style="background-color: hsl(0, 100.00%, 96.67%); opacity: 0.81" title="-0.007">h</span><span style="background-color: hsl(0, 100.00%, 96.49%); opacity: 0.81" title="-0.008">i</span><span style="background-color: hsl(0, 100.00%, 91.55%); opacity: 0.82" title="-0.028">l</span><span style="background-color: hsl(0, 100.00%, 93.60%); opacity: 0.81" title="-0.019">d</span><span style="background-color: hsl(0, 100.00%, 95.95%); opacity: 0.81" title="-0.010">r</span><span style="background-color: hsl(0, 100.00%, 94.33%); opacity: 0.81" title="-0.016">e</span><span style="background-color: hsl(0, 100.00%, 92.75%); opacity: 0.82" title="-0.023">n</span><span style="background-color: hsl(0, 100.00%, 92.01%); opacity: 0.82" title="-0.026">,</span><span style="background-color: hsl(0, 100.00%, 95.30%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(120, 100.00%, 97.13%); opacity: 0.80" title="0.006">a</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.011">n</span><span style="background-color: hsl(120, 100.00%, 95.80%); opacity: 0.81" title="0.010">d</span><span style="background-color: hsl(0, 100.00%, 98.54%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(0, 100.00%, 91.94%); opacity: 0.82" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.030">h</span><span style="background-color: hsl(0, 100.00%, 92.59%); opacity: 0.82" title="-0.023">e</span><span style="background-color: hsl(0, 100.00%, 95.18%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 97.38%); opacity: 0.80" title="-0.005">c</span><span style="background-color: hsl(0, 100.00%, 98.06%); opacity: 0.80" title="-0.003">h</span><span style="background-color: hsl(120, 100.00%, 97.42%); opacity: 0.80" title="0.005">i</span><span style="background-color: hsl(0, 100.00%, 99.82%); opacity: 0.80" title="-0.000">l</span><span style="background-color: hsl(120, 100.00%, 94.57%); opacity: 0.81" title="0.015">d</span><span style="background-color: hsl(120, 100.00%, 89.24%); opacity: 0.83" title="0.040">b</span><span style="background-color: hsl(120, 100.00%, 87.79%); opacity: 0.84" title="0.048">i</span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.032">r</span><span style="background-color: hsl(120, 100.00%, 94.23%); opacity: 0.81" title="0.016">t</span><span style="background-color: hsl(120, 100.00%, 96.87%); opacity: 0.81" title="0.007">h</span><span style="background-color: hsl(0, 100.00%, 99.55%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(0, 100.00%, 92.86%); opacity: 0.82" title="-0.022">h</span><span style="background-color: hsl(0, 100.00%, 91.80%); opacity: 0.82" title="-0.027">u</span><span style="background-color: hsl(0, 100.00%, 91.76%); opacity: 0.82" title="-0.027">r</span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.005">t</span><span style="background-color: hsl(120, 100.00%, 95.35%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 94.31%); opacity: 0.81" title="0.016">l</span><span style="background-color: hsl(120, 100.00%, 92.36%); opacity: 0.82" title="0.024">e</span><span style="background-color: hsl(120, 100.00%, 93.57%); opacity: 0.81" title="0.019">s</span><span style="background-color: hsl(120, 100.00%, 93.84%); opacity: 0.81" title="0.018">s</span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.007">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=soc.religion.christian
        
    </b>
    
        
        (probability <b>0.160</b>, score <b>-2.510</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 88.26%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.263
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 82.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.773
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 89.09%); opacity: 0.83" title="0.041">as</span><span style="background-color: hsl(120, 100.00%, 87.95%); opacity: 0.84" title="0.047"> </span><span style="background-color: hsl(120, 100.00%, 90.87%); opacity: 0.82" title="0.032">i</span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.030"> </span><span style="background-color: hsl(0, 100.00%, 92.11%); opacity: 0.82" title="-0.026">r</span><span style="background-color: hsl(0, 100.00%, 87.58%); opacity: 0.84" title="-0.049">e</span><span style="background-color: hsl(0, 100.00%, 83.40%); opacity: 0.86" title="-0.074">c</span><span style="background-color: hsl(0, 100.00%, 80.88%); opacity: 0.87" title="-0.091">a</span><span style="background-color: hsl(0, 100.00%, 83.37%); opacity: 0.86" title="-0.074">l</span><span style="background-color: hsl(0, 100.00%, 89.78%); opacity: 0.83" title="-0.037">l</span><span style="background-color: hsl(0, 100.00%, 93.05%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(0, 100.00%, 94.68%); opacity: 0.81" title="-0.015">f</span><span style="background-color: hsl(0, 100.00%, 94.19%); opacity: 0.81" title="-0.017">r</span><span style="background-color: hsl(0, 100.00%, 95.12%); opacity: 0.81" title="-0.013">o</span><span style="background-color: hsl(0, 100.00%, 98.50%); opacity: 0.80" title="-0.002">m</span><span style="background-color: hsl(120, 100.00%, 94.15%); opacity: 0.81" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 96.27%); opacity: 0.81" title="0.009">my</span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 96.98%); opacity: 0.80" title="-0.006">b</span><span style="background-color: hsl(0, 100.00%, 93.15%); opacity: 0.82" title="-0.021">o</span><span style="background-color: hsl(0, 100.00%, 92.79%); opacity: 0.82" title="-0.022">u</span><span style="background-color: hsl(0, 100.00%, 95.33%); opacity: 0.81" title="-0.012">t</span><span style="background-color: hsl(120, 100.00%, 93.00%); opacity: 0.82" title="0.022"> </span><span style="background-color: hsl(120, 100.00%, 90.89%); opacity: 0.82" title="0.031">w</span><span style="background-color: hsl(120, 100.00%, 89.23%); opacity: 0.83" title="0.040">i</span><span style="background-color: hsl(120, 100.00%, 90.12%); opacity: 0.83" title="0.035">t</span><span style="background-color: hsl(120, 100.00%, 92.82%); opacity: 0.82" title="0.022">h</span><span style="background-color: hsl(120, 100.00%, 96.30%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(0, 100.00%, 88.58%); opacity: 0.83" title="-0.043">k</span><span style="background-color: hsl(0, 100.00%, 85.27%); opacity: 0.85" title="-0.062">i</span><span style="background-color: hsl(0, 100.00%, 80.72%); opacity: 0.87" title="-0.092">d</span><span style="background-color: hsl(0, 100.00%, 82.98%); opacity: 0.86" title="-0.077">n</span><span style="background-color: hsl(0, 100.00%, 89.57%); opacity: 0.83" title="-0.038">e</span><span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.007">y</span><span style="background-color: hsl(120, 100.00%, 86.47%); opacity: 0.84" title="0.055"> </span><span style="background-color: hsl(120, 100.00%, 84.11%); opacity: 0.85" title="0.070">s</span><span style="background-color: hsl(120, 100.00%, 69.82%); opacity: 0.93" title="0.174">t</span><span style="background-color: hsl(120, 100.00%, 73.48%); opacity: 0.91" title="0.145">o</span><span style="background-color: hsl(120, 100.00%, 75.41%); opacity: 0.90" title="0.130">n</span><span style="background-color: hsl(120, 100.00%, 89.55%); opacity: 0.83" title="0.038">e</span><span style="background-color: hsl(0, 100.00%, 92.85%); opacity: 0.82" title="-0.022">s</span><span style="background-color: hsl(0, 100.00%, 96.25%); opacity: 0.81" title="-0.009">,</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 91.45%); opacity: 0.82" title="0.029">t</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.026">h</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(0, 100.00%, 94.83%); opacity: 0.81" title="-0.014">r</span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.030">e</span><span style="background-color: hsl(0, 100.00%, 88.94%); opacity: 0.83" title="-0.041"> </span><span style="background-color: hsl(0, 100.00%, 86.48%); opacity: 0.84" title="-0.055">i</span><span style="background-color: hsl(0, 100.00%, 82.35%); opacity: 0.86" title="-0.081">s</span><span style="background-color: hsl(0, 100.00%, 78.46%); opacity: 0.88" title="-0.107">n</span><span style="background-color: hsl(0, 100.00%, 77.06%); opacity: 0.89" title="-0.118">'</span><span style="background-color: hsl(0, 100.00%, 80.31%); opacity: 0.87" title="-0.095">t</span><span style="background-color: hsl(0, 100.00%, 86.56%); opacity: 0.84" title="-0.055"> </span><span style="background-color: hsl(0, 100.00%, 97.59%); opacity: 0.80" title="-0.005">a</span><span style="background-color: hsl(0, 100.00%, 96.00%); opacity: 0.81" title="-0.010">n</span><span style="background-color: hsl(0, 100.00%, 96.15%); opacity: 0.81" title="-0.009">y</span><span style="background-color: hsl(0, 100.00%, 84.79%); opacity: 0.85" title="-0.065">
    </span><span style="background-color: hsl(0, 100.00%, 76.45%); opacity: 0.89" title="-0.122">m</span><span style="background-color: hsl(0, 100.00%, 69.82%); opacity: 0.93" title="-0.174">e</span><span style="background-color: hsl(0, 100.00%, 65.81%); opacity: 0.96" title="-0.208">d</span><span style="background-color: hsl(0, 100.00%, 72.28%); opacity: 0.92" title="-0.154">i</span><span style="background-color: hsl(0, 100.00%, 82.81%); opacity: 0.86" title="-0.078">c</span><span style="background-color: hsl(0, 100.00%, 98.28%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(120, 100.00%, 90.75%); opacity: 0.82" title="0.032">t</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(0, 100.00%, 96.62%); opacity: 0.81" title="-0.008">o</span><span style="background-color: hsl(0, 100.00%, 96.12%); opacity: 0.81" title="-0.009">n</span><span style="background-color: hsl(120, 100.00%, 94.70%); opacity: 0.81" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 90.32%); opacity: 0.83" title="0.034">t</span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.043">h</span><span style="background-color: hsl(120, 100.00%, 90.36%); opacity: 0.83" title="0.034">a</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.026">t</span><span style="background-color: hsl(120, 100.00%, 90.14%); opacity: 0.83" title="0.035"> </span><span style="background-color: hsl(120, 100.00%, 90.20%); opacity: 0.83" title="0.035">c</span><span style="background-color: hsl(120, 100.00%, 87.48%); opacity: 0.84" title="0.050">a</span><span style="background-color: hsl(120, 100.00%, 87.48%); opacity: 0.84" title="0.049">n</span><span style="background-color: hsl(120, 100.00%, 90.06%); opacity: 0.83" title="0.036"> </span><span style="background-color: hsl(120, 100.00%, 97.71%); opacity: 0.80" title="0.004">do</span><span style="background-color: hsl(120, 100.00%, 92.77%); opacity: 0.82" title="0.023"> </span><span style="background-color: hsl(120, 100.00%, 86.21%); opacity: 0.84" title="0.057">a</span><span style="background-color: hsl(120, 100.00%, 76.25%); opacity: 0.89" title="0.124">n</span><span style="background-color: hsl(120, 100.00%, 71.86%); opacity: 0.92" title="0.157">y</span><span style="background-color: hsl(120, 100.00%, 72.01%); opacity: 0.92" title="0.156">t</span><span style="background-color: hsl(120, 100.00%, 81.95%); opacity: 0.86" title="0.083">h</span><span style="background-color: hsl(120, 100.00%, 88.07%); opacity: 0.84" title="0.046">i</span><span style="background-color: hsl(120, 100.00%, 94.51%); opacity: 0.81" title="0.015">n</span><span style="background-color: hsl(120, 100.00%, 94.57%); opacity: 0.81" title="0.015">g</span><span style="background-color: hsl(120, 100.00%, 94.77%); opacity: 0.81" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 93.87%); opacity: 0.81" title="0.018">a</span><span style="background-color: hsl(120, 100.00%, 92.61%); opacity: 0.82" title="0.023">b</span><span style="background-color: hsl(120, 100.00%, 97.04%); opacity: 0.80" title="0.006">o</span><span style="background-color: hsl(0, 100.00%, 98.38%); opacity: 0.80" title="-0.003">u</span><span style="background-color: hsl(0, 100.00%, 96.60%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(120, 100.00%, 93.53%); opacity: 0.81" title="0.019"> </span><span style="background-color: hsl(120, 100.00%, 90.77%); opacity: 0.82" title="0.032">t</span><span style="background-color: hsl(120, 100.00%, 93.91%); opacity: 0.81" title="0.018">h</span><span style="background-color: hsl(0, 100.00%, 98.21%); opacity: 0.80" title="-0.003">e</span><span style="background-color: hsl(0, 100.00%, 92.69%); opacity: 0.82" title="-0.023">m</span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(0, 100.00%, 99.00%); opacity: 0.80" title="-0.001">e</span><span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.007">x</span><span style="background-color: hsl(0, 100.00%, 97.01%); opacity: 0.80" title="-0.006">c</span><span style="background-color: hsl(0, 100.00%, 98.18%); opacity: 0.80" title="-0.003">e</span><span style="background-color: hsl(0, 100.00%, 97.57%); opacity: 0.80" title="-0.005">p</span><span style="background-color: hsl(0, 100.00%, 97.42%); opacity: 0.80" title="-0.005">t</span><span style="background-color: hsl(0, 100.00%, 97.07%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 95.32%); opacity: 0.81" title="-0.012">r</span><span style="background-color: hsl(0, 100.00%, 97.61%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.017">l</span><span style="background-color: hsl(0, 100.00%, 95.20%); opacity: 0.81" title="-0.013">i</span><span style="background-color: hsl(120, 100.00%, 94.79%); opacity: 0.81" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 94.25%); opacity: 0.81" title="0.016">v</span><span style="background-color: hsl(120, 100.00%, 93.66%); opacity: 0.81" title="0.019">e</span><span style="background-color: hsl(120, 100.00%, 89.30%); opacity: 0.83" title="0.040"> </span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.051">t</span><span style="background-color: hsl(120, 100.00%, 84.61%); opacity: 0.85" title="0.066">h</span><span style="background-color: hsl(120, 100.00%, 87.33%); opacity: 0.84" title="0.050">e</span><span style="background-color: hsl(120, 100.00%, 97.76%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 82.45%); opacity: 0.86" title="-0.080">p</span><span style="background-color: hsl(0, 100.00%, 75.89%); opacity: 0.90" title="-0.126">a</span><span style="background-color: hsl(0, 100.00%, 72.94%); opacity: 0.91" title="-0.149">i</span><span style="background-color: hsl(0, 100.00%, 81.60%); opacity: 0.87" title="-0.086">n</span><span style="background-color: hsl(0, 100.00%, 94.85%); opacity: 0.81" title="-0.014">.</span><span style="background-color: hsl(120, 100.00%, 95.11%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 95.69%); opacity: 0.81" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 89.90%); opacity: 0.83" title="0.036">i</span><span style="background-color: hsl(120, 100.00%, 85.24%); opacity: 0.85" title="0.063">t</span><span style="background-color: hsl(120, 100.00%, 80.72%); opacity: 0.87" title="0.092">h</span><span style="background-color: hsl(120, 100.00%, 80.39%); opacity: 0.87" title="0.094">e</span><span style="background-color: hsl(120, 100.00%, 85.32%); opacity: 0.85" title="0.062">r</span><span style="background-color: hsl(120, 100.00%, 84.12%); opacity: 0.85" title="0.070"> </span><span style="background-color: hsl(120, 100.00%, 80.27%); opacity: 0.87" title="0.095">t</span><span style="background-color: hsl(120, 100.00%, 73.58%); opacity: 0.91" title="0.144">h</span><span style="background-color: hsl(120, 100.00%, 72.57%); opacity: 0.92" title="0.152">e</span><span style="background-color: hsl(120, 100.00%, 75.14%); opacity: 0.90" title="0.132">y</span><span style="background-color: hsl(120, 100.00%, 76.80%); opacity: 0.89" title="0.119"> </span><span style="background-color: hsl(120, 100.00%, 77.83%); opacity: 0.89" title="0.112">p</span><span style="background-color: hsl(120, 100.00%, 75.20%); opacity: 0.90" title="0.131">a</span><span style="background-color: hsl(120, 100.00%, 76.35%); opacity: 0.89" title="0.123">s</span><span style="background-color: hsl(120, 100.00%, 86.91%); opacity: 0.84" title="0.053">s</span><span style="background-color: hsl(0, 100.00%, 98.39%); opacity: 0.80" title="-0.003">,</span><span style="background-color: hsl(0, 100.00%, 88.66%); opacity: 0.83" title="-0.043"> </span><span style="background-color: hsl(0, 100.00%, 87.14%); opacity: 0.84" title="-0.051">or</span><span style="background-color: hsl(120, 100.00%, 95.89%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 80.27%); opacity: 0.87" title="0.095">t</span><span style="background-color: hsl(120, 100.00%, 73.58%); opacity: 0.91" title="0.144">h</span><span style="background-color: hsl(120, 100.00%, 72.57%); opacity: 0.92" title="0.152">e</span><span style="background-color: hsl(120, 100.00%, 75.14%); opacity: 0.90" title="0.132">y</span><span style="background-color: hsl(120, 100.00%, 83.22%); opacity: 0.86" title="0.075"> </span><span style="background-color: hsl(120, 100.00%, 92.96%); opacity: 0.82" title="0.022">h</span><span style="background-color: hsl(120, 100.00%, 89.25%); opacity: 0.83" title="0.040">a</span><span style="background-color: hsl(120, 100.00%, 89.95%); opacity: 0.83" title="0.036">v</span><span style="background-color: hsl(120, 100.00%, 90.71%); opacity: 0.82" title="0.032">e</span><span style="background-color: hsl(120, 100.00%, 91.52%); opacity: 0.82" title="0.028"> </span><span style="background-color: hsl(120, 100.00%, 92.19%); opacity: 0.82" title="0.025">to</span><span style="background-color: hsl(120, 100.00%, 88.77%); opacity: 0.83" title="0.042"> </span><span style="background-color: hsl(120, 100.00%, 88.17%); opacity: 0.84" title="0.046">be</span><span style="background-color: hsl(120, 100.00%, 83.45%); opacity: 0.86" title="0.074"> </span><span style="background-color: hsl(120, 100.00%, 85.62%); opacity: 0.85" title="0.060">b</span><span style="background-color: hsl(120, 100.00%, 85.92%); opacity: 0.84" title="0.059">r</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.008">o</span><span style="background-color: hsl(0, 100.00%, 84.43%); opacity: 0.85" title="-0.068">k</span><span style="background-color: hsl(0, 100.00%, 84.61%); opacity: 0.85" title="-0.066">e</span><span style="background-color: hsl(0, 100.00%, 88.10%); opacity: 0.84" title="-0.046">n</span><span style="background-color: hsl(0, 100.00%, 86.01%); opacity: 0.84" title="-0.058"> </span><span style="background-color: hsl(0, 100.00%, 89.84%); opacity: 0.83" title="-0.037">up</span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 90.89%); opacity: 0.82" title="0.031">w</span><span style="background-color: hsl(120, 100.00%, 89.23%); opacity: 0.83" title="0.040">i</span><span style="background-color: hsl(120, 100.00%, 90.12%); opacity: 0.83" title="0.035">t</span><span style="background-color: hsl(120, 100.00%, 92.82%); opacity: 0.82" title="0.022">h</span><span style="background-color: hsl(0, 100.00%, 98.30%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(0, 100.00%, 90.70%); opacity: 0.82" title="-0.032">s</span><span style="background-color: hsl(0, 100.00%, 89.30%); opacity: 0.83" title="-0.040">o</span><span style="background-color: hsl(0, 100.00%, 95.28%); opacity: 0.81" title="-0.012">u</span><span style="background-color: hsl(120, 100.00%, 94.47%); opacity: 0.81" title="0.015">n</span><span style="background-color: hsl(120, 100.00%, 88.84%); opacity: 0.83" title="0.042">d</span><span style="background-color: hsl(120, 100.00%, 87.16%); opacity: 0.84" title="0.051">,</span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 87.14%); opacity: 0.84" title="-0.051">or</span><span style="background-color: hsl(120, 100.00%, 95.89%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 80.27%); opacity: 0.87" title="0.095">t</span><span style="background-color: hsl(120, 100.00%, 73.58%); opacity: 0.91" title="0.144">h</span><span style="background-color: hsl(120, 100.00%, 72.57%); opacity: 0.92" title="0.152">e</span><span style="background-color: hsl(120, 100.00%, 75.14%); opacity: 0.90" title="0.132">y</span><span style="background-color: hsl(120, 100.00%, 83.22%); opacity: 0.86" title="0.075"> </span><span style="background-color: hsl(120, 100.00%, 92.96%); opacity: 0.82" title="0.022">h</span><span style="background-color: hsl(120, 100.00%, 89.25%); opacity: 0.83" title="0.040">a</span><span style="background-color: hsl(120, 100.00%, 89.95%); opacity: 0.83" title="0.036">v</span><span style="background-color: hsl(120, 100.00%, 90.71%); opacity: 0.82" title="0.032">e</span><span style="background-color: hsl(120, 100.00%, 91.52%); opacity: 0.82" title="0.028">
    </span><span style="background-color: hsl(120, 100.00%, 92.19%); opacity: 0.82" title="0.025">to</span><span style="background-color: hsl(120, 100.00%, 88.77%); opacity: 0.83" title="0.042"> </span><span style="background-color: hsl(120, 100.00%, 88.17%); opacity: 0.84" title="0.046">be</span><span style="background-color: hsl(120, 100.00%, 90.70%); opacity: 0.82" title="0.032"> </span><span style="background-color: hsl(0, 100.00%, 88.83%); opacity: 0.83" title="-0.042">e</span><span style="background-color: hsl(0, 100.00%, 84.30%); opacity: 0.85" title="-0.068">x</span><span style="background-color: hsl(0, 100.00%, 75.24%); opacity: 0.90" title="-0.131">t</span><span style="background-color: hsl(0, 100.00%, 74.15%); opacity: 0.91" title="-0.139">r</span><span style="background-color: hsl(0, 100.00%, 73.44%); opacity: 0.91" title="-0.145">a</span><span style="background-color: hsl(0, 100.00%, 82.32%); opacity: 0.86" title="-0.081">c</span><span style="background-color: hsl(0, 100.00%, 87.34%); opacity: 0.84" title="-0.050">t</span><span style="background-color: hsl(0, 100.00%, 92.81%); opacity: 0.82" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.009">d</span><span style="background-color: hsl(0, 100.00%, 85.85%); opacity: 0.85" title="-0.059"> </span><span style="background-color: hsl(0, 100.00%, 79.82%); opacity: 0.88" title="-0.098">s</span><span style="background-color: hsl(0, 100.00%, 78.63%); opacity: 0.88" title="-0.106">u</span><span style="background-color: hsl(0, 100.00%, 79.50%); opacity: 0.88" title="-0.100">r</span><span style="background-color: hsl(0, 100.00%, 81.24%); opacity: 0.87" title="-0.088">g</span><span style="background-color: hsl(0, 100.00%, 88.30%); opacity: 0.83" title="-0.045">i</span><span style="background-color: hsl(0, 100.00%, 87.32%); opacity: 0.84" title="-0.050">c</span><span style="background-color: hsl(0, 100.00%, 89.37%); opacity: 0.83" title="-0.039">a</span><span style="background-color: hsl(0, 100.00%, 90.83%); opacity: 0.82" title="-0.032">l</span><span style="background-color: hsl(0, 100.00%, 91.86%); opacity: 0.82" title="-0.027">l</span><span style="background-color: hsl(0, 100.00%, 91.57%); opacity: 0.82" title="-0.028">y</span><span style="background-color: hsl(0, 100.00%, 90.33%); opacity: 0.83" title="-0.034">.</span><span style="background-color: hsl(0, 100.00%, 94.25%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 96.37%); opacity: 0.81" title="-0.008">w</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(0, 100.00%, 95.95%); opacity: 0.81" title="-0.010">e</span><span style="background-color: hsl(120, 100.00%, 98.65%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(120, 100.00%, 91.67%); opacity: 0.82" title="0.028"> </span><span style="background-color: hsl(120, 100.00%, 90.87%); opacity: 0.82" title="0.032">i</span><span style="background-color: hsl(120, 100.00%, 93.10%); opacity: 0.82" title="0.021"> </span><span style="background-color: hsl(0, 100.00%, 91.59%); opacity: 0.82" title="-0.028">w</span><span style="background-color: hsl(0, 100.00%, 91.18%); opacity: 0.82" title="-0.030">a</span><span style="background-color: hsl(0, 100.00%, 90.31%); opacity: 0.83" title="-0.034">s</span><span style="background-color: hsl(0, 100.00%, 90.57%); opacity: 0.83" title="-0.033"> </span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.007">i</span><span style="background-color: hsl(120, 100.00%, 95.57%); opacity: 0.81" title="0.011">n</span><span style="background-color: hsl(120, 100.00%, 95.53%); opacity: 0.81" title="0.011">,</span><span style="background-color: hsl(120, 100.00%, 90.17%); opacity: 0.83" title="0.035"> </span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.051">t</span><span style="background-color: hsl(120, 100.00%, 84.61%); opacity: 0.85" title="0.066">h</span><span style="background-color: hsl(120, 100.00%, 87.33%); opacity: 0.84" title="0.050">e</span><span style="background-color: hsl(120, 100.00%, 96.68%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(0, 100.00%, 88.71%); opacity: 0.83" title="-0.043">x</span><span style="background-color: hsl(0, 100.00%, 84.22%); opacity: 0.85" title="-0.069">-</span><span style="background-color: hsl(0, 100.00%, 88.28%); opacity: 0.83" title="-0.045">r</span><span style="background-color: hsl(0, 100.00%, 92.89%); opacity: 0.82" title="-0.022">a</span><span style="background-color: hsl(0, 100.00%, 99.81%); opacity: 0.80" title="-0.000">y</span><span style="background-color: hsl(0, 100.00%, 96.01%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 86.33%); opacity: 0.84" title="-0.056">t</span><span style="background-color: hsl(0, 100.00%, 81.94%); opacity: 0.86" title="-0.084">e</span><span style="background-color: hsl(0, 100.00%, 82.15%); opacity: 0.86" title="-0.082">c</span><span style="background-color: hsl(0, 100.00%, 88.86%); opacity: 0.83" title="-0.042">h</span><span style="background-color: hsl(120, 100.00%, 91.16%); opacity: 0.82" title="0.030"> </span><span style="background-color: hsl(120, 100.00%, 85.43%); opacity: 0.85" title="0.061">h</span><span style="background-color: hsl(120, 100.00%, 82.55%); opacity: 0.86" title="0.080">a</span><span style="background-color: hsl(120, 100.00%, 80.14%); opacity: 0.87" title="0.096">p</span><span style="background-color: hsl(120, 100.00%, 83.20%); opacity: 0.86" title="0.075">p</span><span style="background-color: hsl(120, 100.00%, 84.48%); opacity: 0.85" title="0.067">e</span><span style="background-color: hsl(120, 100.00%, 87.59%); opacity: 0.84" title="0.049">n</span><span style="background-color: hsl(120, 100.00%, 89.97%); opacity: 0.83" title="0.036">e</span><span style="background-color: hsl(0, 100.00%, 99.94%); opacity: 0.80" title="-0.000">d</span><span style="background-color: hsl(120, 100.00%, 94.17%); opacity: 0.81" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 92.19%); opacity: 0.82" title="0.025">to</span><span style="background-color: hsl(0, 100.00%, 93.04%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(0, 100.00%, 84.38%); opacity: 0.85" title="-0.068">m</span><span style="background-color: hsl(0, 100.00%, 78.66%); opacity: 0.88" title="-0.106">e</span><span style="background-color: hsl(0, 100.00%, 74.84%); opacity: 0.90" title="-0.134">n</span><span style="background-color: hsl(0, 100.00%, 78.12%); opacity: 0.88" title="-0.110">t</span><span style="background-color: hsl(0, 100.00%, 81.35%); opacity: 0.87" title="-0.087">i</span><span style="background-color: hsl(0, 100.00%, 89.32%); opacity: 0.83" title="-0.039">o</span><span style="background-color: hsl(0, 100.00%, 93.90%); opacity: 0.81" title="-0.018">n</span><span style="background-color: hsl(120, 100.00%, 94.70%); opacity: 0.81" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 90.32%); opacity: 0.83" title="0.034">t</span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.043">h</span><span style="background-color: hsl(120, 100.00%, 90.36%); opacity: 0.83" title="0.034">a</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.026">t</span><span style="background-color: hsl(0, 100.00%, 91.05%); opacity: 0.82" title="-0.031"> </span><span style="background-color: hsl(0, 100.00%, 83.71%); opacity: 0.86" title="-0.072">s</span><span style="background-color: hsl(0, 100.00%, 79.07%); opacity: 0.88" title="-0.103">h</span><span style="background-color: hsl(0, 100.00%, 79.59%); opacity: 0.88" title="-0.099">e</span><span style="background-color: hsl(0, 100.00%, 81.84%); opacity: 0.86" title="-0.084">'</span><span style="background-color: hsl(0, 100.00%, 95.43%); opacity: 0.81" title="-0.012">d</span><span style="background-color: hsl(120, 100.00%, 96.18%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.021">h</span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.017">a</span><span style="background-color: hsl(120, 100.00%, 96.30%); opacity: 0.81" title="0.009">d</span><span style="background-color: hsl(0, 100.00%, 99.89%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(0, 100.00%, 88.58%); opacity: 0.83" title="-0.043">k</span><span style="background-color: hsl(0, 100.00%, 85.27%); opacity: 0.85" title="-0.062">i</span><span style="background-color: hsl(0, 100.00%, 80.72%); opacity: 0.87" title="-0.092">d</span><span style="background-color: hsl(0, 100.00%, 82.98%); opacity: 0.86" title="-0.077">n</span><span style="background-color: hsl(0, 100.00%, 89.57%); opacity: 0.83" title="-0.038">e</span><span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.007">y</span><span style="background-color: hsl(120, 100.00%, 86.47%); opacity: 0.84" title="0.055">
    </span><span style="background-color: hsl(120, 100.00%, 84.11%); opacity: 0.85" title="0.070">s</span><span style="background-color: hsl(120, 100.00%, 69.82%); opacity: 0.93" title="0.174">t</span><span style="background-color: hsl(120, 100.00%, 74.97%); opacity: 0.90" title="0.133">o</span><span style="background-color: hsl(120, 100.00%, 79.87%); opacity: 0.88" title="0.098">n</span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.011">e</span><span style="background-color: hsl(0, 100.00%, 85.70%); opacity: 0.85" title="-0.060">s</span><span style="background-color: hsl(0, 100.00%, 88.53%); opacity: 0.83" title="-0.044"> </span><span style="background-color: hsl(120, 100.00%, 98.02%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(120, 100.00%, 97.49%); opacity: 0.80" title="0.005">n</span><span style="background-color: hsl(120, 100.00%, 97.31%); opacity: 0.80" title="0.006">d</span><span style="background-color: hsl(120, 100.00%, 85.94%); opacity: 0.84" title="0.058"> </span><span style="background-color: hsl(120, 100.00%, 79.68%); opacity: 0.88" title="0.099">c</span><span style="background-color: hsl(120, 100.00%, 72.89%); opacity: 0.91" title="0.149">h</span><span style="background-color: hsl(120, 100.00%, 71.60%); opacity: 0.92" title="0.160">i</span><span style="background-color: hsl(120, 100.00%, 65.68%); opacity: 0.96" title="0.209">l</span><span style="background-color: hsl(120, 100.00%, 69.49%); opacity: 0.94" title="0.177">d</span><span style="background-color: hsl(120, 100.00%, 80.17%); opacity: 0.87" title="0.095">r</span><span style="background-color: hsl(120, 100.00%, 82.28%); opacity: 0.86" title="0.081">e</span><span style="background-color: hsl(120, 100.00%, 88.03%); opacity: 0.84" title="0.046">n</span><span style="background-color: hsl(120, 100.00%, 90.15%); opacity: 0.83" title="0.035">,</span><span style="background-color: hsl(120, 100.00%, 93.34%); opacity: 0.82" title="0.020"> </span><span style="background-color: hsl(120, 100.00%, 98.02%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(120, 100.00%, 97.49%); opacity: 0.80" title="0.005">n</span><span style="background-color: hsl(120, 100.00%, 97.31%); opacity: 0.80" title="0.006">d</span><span style="background-color: hsl(120, 100.00%, 91.07%); opacity: 0.82" title="0.031"> </span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.051">t</span><span style="background-color: hsl(120, 100.00%, 84.61%); opacity: 0.85" title="0.066">h</span><span style="background-color: hsl(120, 100.00%, 87.33%); opacity: 0.84" title="0.050">e</span><span style="background-color: hsl(120, 100.00%, 81.53%); opacity: 0.87" title="0.086"> </span><span style="background-color: hsl(120, 100.00%, 79.68%); opacity: 0.88" title="0.099">c</span><span style="background-color: hsl(120, 100.00%, 74.38%); opacity: 0.91" title="0.138">h</span><span style="background-color: hsl(120, 100.00%, 76.47%); opacity: 0.89" title="0.122">i</span><span style="background-color: hsl(120, 100.00%, 76.09%); opacity: 0.90" title="0.125">l</span><span style="background-color: hsl(120, 100.00%, 87.89%); opacity: 0.84" title="0.047">d</span><span style="background-color: hsl(0, 100.00%, 85.85%); opacity: 0.85" title="-0.059">b</span><span style="background-color: hsl(0, 100.00%, 77.80%); opacity: 0.89" title="-0.112">i</span><span style="background-color: hsl(0, 100.00%, 83.71%); opacity: 0.86" title="-0.072">r</span><span style="background-color: hsl(0, 100.00%, 88.20%); opacity: 0.83" title="-0.045">t</span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.007">h</span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.029"> </span><span style="background-color: hsl(120, 100.00%, 78.02%); opacity: 0.89" title="0.111">h</span><span style="background-color: hsl(120, 100.00%, 78.40%); opacity: 0.88" title="0.108">u</span><span style="background-color: hsl(120, 100.00%, 78.98%); opacity: 0.88" title="0.104">r</span><span style="background-color: hsl(120, 100.00%, 95.59%); opacity: 0.81" title="0.011">t</span><span style="background-color: hsl(120, 100.00%, 91.62%); opacity: 0.82" title="0.028"> </span><span style="background-color: hsl(120, 100.00%, 97.78%); opacity: 0.80" title="0.004">l</span><span style="background-color: hsl(120, 100.00%, 93.55%); opacity: 0.81" title="0.019">e</span><span style="background-color: hsl(120, 100.00%, 92.69%); opacity: 0.82" title="0.023">s</span><span style="background-color: hsl(120, 100.00%, 90.31%); opacity: 0.83" title="0.034">s</span><span style="background-color: hsl(120, 100.00%, 94.83%); opacity: 0.81" title="0.014">.</span>
        </p>
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




The result is similar, with some minor changes. Quality is better for
unknown reason; maybe cross-word dependencies are not that important.

5. Debugging HashingVectorizer
------------------------------

To check that we can try fitting word n-grams instead of char n-grams.
But let’s deal with efficiency first. To handle large vocabularies we
can use HashingVectorizer from scikit-learn; to make training faster we
can employ SGDCLassifier:

.. code:: ipython3

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    
    vec = HashingVectorizer(stop_words='english', ngram_range=(1,2))
    clf = SGDClassifier(n_iter=10, random_state=42)
    pipe = make_pipeline(vec, clf)
    pipe.fit(twenty_train.data, twenty_train.target)
    
    print_report(pipe)


.. parsed-literal::

                            precision    recall  f1-score   support
    
               alt.atheism       0.90      0.80      0.85       319
             comp.graphics       0.88      0.96      0.92       389
                   sci.med       0.93      0.90      0.92       396
    soc.religion.christian       0.89      0.91      0.90       398
    
               avg / total       0.90      0.90      0.90      1502
    
    accuracy: 0.899


It was super-fast! We’re not choosing regularization parameter using
cross-validation though. Let’s check what model learned:

.. code:: ipython3

    eli5.show_prediction(clf, twenty_test.data[0], vec=vec, 
                         target_names=twenty_test.target_names,
                         targets=['sci.med'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (score <b>0.097</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.678
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 82.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.581
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as i </span><span style="background-color: hsl(120, 100.00%, 78.29%); opacity: 0.88" title="0.102">recall</span><span style="opacity: 0.80"> from my </span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">bout</span><span style="opacity: 0.80"> with </span><span style="background-color: hsl(120, 100.00%, 80.98%); opacity: 0.87" title="0.084">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.76%); opacity: 0.82" title="0.021">stones</span><span style="opacity: 0.80">, there </span><span style="background-color: hsl(0, 100.00%, 85.92%); opacity: 0.84" title="-0.055">isn</span><span style="opacity: 0.80">'t any
    </span><span style="background-color: hsl(120, 100.00%, 75.59%); opacity: 0.90" title="0.120">medication</span><span style="opacity: 0.80"> that can do anything about them except </span><span style="background-color: hsl(120, 100.00%, 97.98%); opacity: 0.80" title="0.003">relieve</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.243">pain</span><span style="opacity: 0.80">.
    
    either they </span><span style="background-color: hsl(0, 100.00%, 91.23%); opacity: 0.82" title="-0.028">pass</span><span style="opacity: 0.80">, or they have to be </span><span style="background-color: hsl(120, 100.00%, 95.86%); opacity: 0.81" title="0.010">broken</span><span style="opacity: 0.80"> up with </span><span style="background-color: hsl(120, 100.00%, 78.81%); opacity: 0.88" title="0.098">sound</span><span style="opacity: 0.80">, or they have
    to be extracted </span><span style="background-color: hsl(120, 100.00%, 95.06%); opacity: 0.81" title="0.012">surgically</span><span style="opacity: 0.80">.
    
    when i was in, the x-</span><span style="background-color: hsl(0, 100.00%, 85.04%); opacity: 0.85" title="-0.060">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.82%); opacity: 0.83" title="0.039">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.74%); opacity: 0.82" title="0.021">happened</span><span style="opacity: 0.80"> to </span><span style="background-color: hsl(0, 100.00%, 93.25%); opacity: 0.82" title="-0.019">mention</span><span style="opacity: 0.80"> that she'd had </span><span style="background-color: hsl(120, 100.00%, 80.98%); opacity: 0.87" title="0.084">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 92.76%); opacity: 0.82" title="0.021">stones</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(0, 100.00%, 93.79%); opacity: 0.81" title="-0.017">children</span><span style="opacity: 0.80">, and the </span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.006">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.20%); opacity: 0.81" title="0.008">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Result looks similar to CountVectorizer. But with HashingVectorizer we
don’t even have a vocabulary! Why does it work?

.. code:: ipython3

    eli5.show_weights(clf, vec=vec, top=10,
                      target_names=twenty_test.target_names)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
            <table class="eli5-weights-wrapper" style="border-collapse: collapse; border: none; margin-bottom: 1.5em;">
                <tr>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=alt.atheism
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=comp.graphics
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=sci.med
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=soc.religion.christian
        
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
            
                <tr style="background-color: hsl(120, 100.00%, 83.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.836
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x199378
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 85.43%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.378
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x938889
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.12%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.776
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x718537
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.625
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x349126
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.18%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.554
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x242643
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.509
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x71928
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 89.40%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 50341 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 88.79%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 50567 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 88.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.634
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x683213
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.03%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.795
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x741207
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 87.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.872
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x199709
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.132
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x641063
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
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.737
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x580586
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.84%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.056
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x342790
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.29%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.956
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x771885
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.787
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x363686
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.717
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x111283
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 88.40%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 32081 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 88.20%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 31710 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 88.20%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.760
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x857427
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.779
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x85557
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 87.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.813
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x693269
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 87.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.021
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x120354
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 85.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.447
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x814572
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
            
                <tr style="background-color: hsl(120, 100.00%, 86.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.209
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x988761
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.23%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.194
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x337555
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.162
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x154565
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.92%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.818
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x806262
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 87.92%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 44124 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 88.46%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 43892 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 88.46%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.704
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x790864
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.750
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x580586
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 87.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.851
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x34701
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.71%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.085
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x85557
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.43%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.147
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x365313
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.42%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x494508
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
            
                <tr style="background-color: hsl(120, 100.00%, 82.71%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.034
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x641063
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 82.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.016
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x199709
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 82.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.977
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x741207
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.092
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x396081
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.901
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x274863
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 87.54%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 51475 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 87.26%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 51717 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 87.26%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.963
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x672777
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.096
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x199378
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.45%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.143
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x443433
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 83.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.963
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x718537
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 81.88%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.245
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            x970058
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                    
                </tr>
            </table>
        
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Ok, we don’t have a vocabulary, so we don’t have feature names. Are we
out of luck? Nope, eli5 has an answer for that:
:class:`~.InvertableHashingVectorizer`. It can be used to get feature names for
HahshingVectorizer without fitiing a huge vocabulary. It still needs
some data to learn words -> hashes mapping though; we can use a random
subset of data to fit it.

.. code:: ipython3

    from eli5.sklearn import InvertableHashingVectorizer
    import numpy as np

.. code:: ipython3

    ivec = InvertableHashingVectorizer(vec)
    sample_size = len(twenty_train.data) // 10
    X_sample = np.random.choice(twenty_train.data, size=sample_size)
    ivec.fit(X_sample);

.. code:: ipython3

    eli5.show_weights(clf, vec=ivec, top=20,
                      target_names=twenty_test.target_names)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
            <table class="eli5-weights-wrapper" style="border-collapse: collapse; border: none; margin-bottom: 1.5em;">
                <tr>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=alt.atheism
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=comp.graphics
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=sci.med
        
    </b>
    
    top features
                        </td>
                    
                        <td style="padding: 0.5em; border: 1px solid black; text-align: center;">
                            <b>
        
            y=soc.religion.christian
        
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
            
                <tr style="background-color: hsl(120, 100.00%, 83.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.836
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            atheism
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 85.43%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.378
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            writes
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.634
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            morality
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.625
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            motto <span title="(-)mail brl">&hellip;</span>
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.18%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.554
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            religion
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.509
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            islam
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.489
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            keith
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.56%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.476
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            religious
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.439
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            objective <span title="witnesses longer">&hellip;</span>
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.414
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            wrote
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.92%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.405
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            said
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.14%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.361
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            punishment
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.27%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.335
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            livesey
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.29%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.332
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            mathew
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.33%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.324
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            atheist
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.320
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            agree
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 90.35%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 47696 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 88.12%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 53202 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 88.12%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.776
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            rutgers edu
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.03%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.795
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            rutgers
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 87.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.872
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            christ <span title="wiggle writes">&hellip;</span>
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.132
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            christians
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
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.737
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            graphics
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 85.13%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.447
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            image
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.84%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.056
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            code
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.021
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            files
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.29%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.956
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            images
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.813
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            3d
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.07%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.787
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            software
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.717
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            file
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.47%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.701
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            ftp
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.02%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.587
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            video
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.572
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            keywords
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.572
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            card
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.509
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            points
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.44%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.500
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            line
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.47%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.494
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            need
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.53%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.483
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            computer
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.470
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            hi
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 89.59%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 30146 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 88.70%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 33635 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 88.70%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.654
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            people
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.20%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.760
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            keyboard
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.10%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.779
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            god
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
            
                <tr style="background-color: hsl(120, 100.00%, 86.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.209
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            health
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.23%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.194
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            msg
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.162
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            doctor
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.42%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            disease
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.43%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.147
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            treatment
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.851
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            medical
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.92%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.818
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            com
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.46%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.704
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            pain
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.663
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            effects
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.88%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.616
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            cancer
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.38%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.513
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            case
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.453
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            diet
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.70%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.447
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            blood
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.439
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            information
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.76%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.435
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            keyboard
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.407
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            pitt
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 89.91%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 42291 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 89.63%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 45715 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 89.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.462
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            church
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.697
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            FEATURE[354651]
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 88.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.750
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            graphics
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.71%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.085
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            god
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
            
                <tr style="background-color: hsl(120, 100.00%, 81.88%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.245
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            church
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 82.71%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.034
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            christians
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 82.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.016
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            christ <span title="wiggle writes">&hellip;</span>
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 82.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.977
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            rutgers
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 83.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.963
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            rutgers edu
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.45%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.143
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            christian
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.092
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            heaven <span title="(-)recommend finding">&hellip;</span>
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.26%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.963
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            love
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.901
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            athos rutgers
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.901
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            athos
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.741
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            satan
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.714
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            authority
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.70%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.653
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            faith
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.74%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.644
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            1993
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.643
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            article apr
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 88.80%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.633
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            understanding
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.541
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            sin <span title="(-)sight don">&hellip;</span>
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.40%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.509
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            god
        </td>
        
    </tr>
            
            
                <tr style="background-color: hsl(120, 100.00%, 89.40%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 49948 more positive &hellip;</i>
                    </td>
                </tr>
            
    
            
                <tr style="background-color: hsl(0, 100.00%, 89.32%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 53234 more negative &hellip;</i>
                    </td>
                </tr>
            
            
                <tr style="background-color: hsl(0, 100.00%, 89.32%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.525
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            graphics
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -2.096
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            atheism
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                                    
                                
                            </td>
                        
                    
                </tr>
            </table>
        
    
        
            
    
    
        
            
    
    
        
            
    
    
        
            
    
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




There are collisions (hover mouse over features with “…”), and there are
important features which were not seen in the random sample
(FEATURE[…]), but overall it looks fine.

“rutgers edu” bigram feature is suspicious though, it looks like a part
of URL.

.. code:: ipython3

    rutgers_example = [x for x in twenty_train.data if 'rutgers' in x.lower()][0]
    print(rutgers_example)


.. parsed-literal::

    In article <Apr.8.00.57.41.1993.28246@athos.rutgers.edu> REXLEX@fnal.gov writes:
    >In article <Apr.7.01.56.56.1993.22824@athos.rutgers.edu> shrum@hpfcso.fc.hp.com
    >Matt. 22:9-14 'Go therefore to the main highways, and as many as you find
    >there, invite to the wedding feast.'...
    
    >hmmmmmm.  Sounds like your theology and Christ's are at odds. Which one am I 
    >to believe?


Yep, it looks like model learned this address instead of learning
something useful.

.. code:: ipython3

    eli5.show_prediction(clf, rutgers_example, vec=vec, 
                         target_names=twenty_test.target_names, 
                         targets=['soc.religion.christian'])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=soc.religion.christian
        
    </b>
    
        
        (score <b>2.044</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +2.706
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 92.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.662
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">in </span><span style="background-color: hsl(120, 100.00%, 93.91%); opacity: 0.81" title="0.045">article</span><span style="opacity: 0.80"> &lt;</span><span style="background-color: hsl(120, 100.00%, 82.16%); opacity: 0.86" title="0.210">apr</span><span style="opacity: 0.80">.8.</span><span style="background-color: hsl(120, 100.00%, 95.31%); opacity: 0.81" title="0.031">00</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.004">57</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 98.52%); opacity: 0.80" title="0.006">41</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 85.10%); opacity: 0.85" title="0.162">1993</span><span style="opacity: 0.80">.28246@</span><span style="background-color: hsl(120, 100.00%, 73.62%); opacity: 0.91" title="0.367">athos</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.666">rutgers</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 93.54%); opacity: 0.81" title="0.049">edu</span><span style="opacity: 0.80">&gt; </span><span style="background-color: hsl(0, 100.00%, 99.91%); opacity: 0.80" title="-0.000">rexlex</span><span style="opacity: 0.80">@</span><span style="background-color: hsl(0, 100.00%, 97.69%); opacity: 0.80" title="-0.011">fnal</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(0, 100.00%, 98.47%); opacity: 0.80" title="-0.006">gov</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.12%); opacity: 0.80" title="-0.008">writes</span><span style="opacity: 0.80">:
    &gt;in </span><span style="background-color: hsl(120, 100.00%, 98.23%); opacity: 0.80" title="0.008">article</span><span style="opacity: 0.80"> &lt;</span><span style="background-color: hsl(120, 100.00%, 81.33%); opacity: 0.87" title="0.224">apr</span><span style="opacity: 0.80">.7.</span><span style="background-color: hsl(120, 100.00%, 93.62%); opacity: 0.81" title="0.048">01</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 97.83%); opacity: 0.80" title="0.010">56</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 97.12%); opacity: 0.80" title="0.016">56</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 85.00%); opacity: 0.85" title="0.164">1993</span><span style="opacity: 0.80">.22824@</span><span style="background-color: hsl(120, 100.00%, 73.62%); opacity: 0.91" title="0.367">athos</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.666">rutgers</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(120, 100.00%, 93.54%); opacity: 0.81" title="0.049">edu</span><span style="opacity: 0.80">&gt; shrum@hpfcso.fc.</span><span style="background-color: hsl(0, 100.00%, 98.17%); opacity: 0.80" title="-0.008">hp</span><span style="opacity: 0.80">.</span><span style="background-color: hsl(0, 100.00%, 86.02%); opacity: 0.84" title="-0.148">com</span><span style="opacity: 0.80">
    &gt;</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.005">matt</span><span style="opacity: 0.80">. </span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.007">22</span><span style="opacity: 0.80">:9-</span><span style="background-color: hsl(120, 100.00%, 91.09%); opacity: 0.82" title="0.078">14</span><span style="opacity: 0.80"> 'go therefore to the </span><span style="background-color: hsl(120, 100.00%, 96.89%); opacity: 0.81" title="0.017">main</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.03%); opacity: 0.80" title="-0.003">highways</span><span style="opacity: 0.80">, and as many as you find
    &gt;there, </span><span style="background-color: hsl(0, 100.00%, 99.92%); opacity: 0.80" title="-0.000">invite</span><span style="opacity: 0.80"> to the </span><span style="background-color: hsl(120, 100.00%, 93.26%); opacity: 0.82" title="0.052">wedding</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.66%); opacity: 0.80" title="0.012">feast</span><span style="opacity: 0.80">.'...
    
    &gt;hmmmmmm.  </span><span style="background-color: hsl(0, 100.00%, 98.89%); opacity: 0.80" title="-0.004">sounds</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.38%); opacity: 0.80" title="-0.002">like</span><span style="opacity: 0.80"> your </span><span style="background-color: hsl(0, 100.00%, 94.12%); opacity: 0.81" title="-0.043">theology</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(120, 100.00%, 77.57%); opacity: 0.89" title="0.292">christ</span><span style="opacity: 0.80">'s are at </span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">odds</span><span style="opacity: 0.80">. which one am i 
    &gt;to </span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.035">believe</span><span style="opacity: 0.80">?</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Quoted text makes it too easy for model to classify some of the
messages; that won’t generalize to new messages. So to improve the model
next step could be to process the data further, e.g. remove quoted text
or replace email addresses with a special token.

You get the idea: looking at features helps to understand how classifier
works. Maybe even more importantly, it helps to notice preprocessing
bugs, data leaks, issues with task specification - all these nasty
problems you get in a real world.
