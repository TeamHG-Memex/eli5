
Explaining XGBoost predictions on the Titanic dataset
=====================================================

This tutorial will show you how to analyze predictions of an XGBoost
classifier (regression for XGBoost and most scikit-learn tree ensembles
are also supported by eli5). We will use `Titanic
dataset <https://www.kaggle.com/c/titanic/data>`__, which is small and
has not too many features, but is still interesting enough.

We are using `XGBoost <https://xgboost.readthedocs.io/en/latest/>`__
0.81 and data downloaded from https://www.kaggle.com/c/titanic/data (it
is also bundled in the eli5 repo:
https://github.com/TeamHG-Memex/eli5/blob/master/notebooks/titanic-train.csv).

1. Training data
----------------

Let’s start by loading the data:

.. code:: ipython3

    import csv
    import numpy as np
    
    with open('titanic-train.csv', 'rt') as f:
        data = list(csv.DictReader(f))
    data[:1]




.. parsed-literal::

    [OrderedDict([('PassengerId', '1'),
                  ('Survived', '0'),
                  ('Pclass', '3'),
                  ('Name', 'Braund, Mr. Owen Harris'),
                  ('Sex', 'male'),
                  ('Age', '22'),
                  ('SibSp', '1'),
                  ('Parch', '0'),
                  ('Ticket', 'A/5 21171'),
                  ('Fare', '7.25'),
                  ('Cabin', ''),
                  ('Embarked', 'S')])]



Variable descriptions:

-  **Age:** Age
-  **Cabin:** Cabin
-  **Embarked:** Port of Embarkation (C = Cherbourg; Q = Queenstown; S =
   Southampton)
-  **Fare:** Passenger Fare
-  **Name:** Name
-  **Parch:** Number of Parents/Children Aboard
-  **Pclass:** Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
-  **Sex:** Sex
-  **Sibsp:** Number of Siblings/Spouses Aboard
-  **Survived:** Survival (0 = No; 1 = Yes)
-  **Ticket:** Ticket Number

Next, shuffle data and separate features from what we are trying to
predict: survival.

.. code:: ipython3

    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    
    _all_xs = [{k: v for k, v in row.items() if k != 'Survived'} for row in data]
    _all_ys = np.array([int(row['Survived']) for row in data])
    
    all_xs, all_ys = shuffle(_all_xs, _all_ys, random_state=0)
    train_xs, valid_xs, train_ys, valid_ys = train_test_split(
        all_xs, all_ys, test_size=0.25, random_state=0)
    print('{} items total, {:.1%} true'.format(len(all_xs), np.mean(all_ys)))


.. parsed-literal::

    891 items total, 38.4% true


We do just minimal preprocessing: convert obviously contiuous *Age* and
*Fare* variables to floats, and *SibSp*, *Parch* to integers. Missing
*Age* values are removed.

.. code:: ipython3

    for x in all_xs:
        if x['Age']:
            x['Age'] = float(x['Age'])
        else:
            x.pop('Age')
        x['Fare'] = float(x['Fare'])
        x['SibSp'] = int(x['SibSp'])
        x['Parch'] = int(x['Parch'])

2. Simple XGBoost classifier
----------------------------

Let’s first build a very simple classifier with
`xbgoost.XGBClassifier <http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier>`__
and
`sklearn.feature_extraction.DictVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`__,
and check its accuracy with 10-fold cross-validation:

.. code:: ipython3

    from xgboost import XGBClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    
    clf = XGBClassifier()
    vec = DictVectorizer()
    pipeline = make_pipeline(vec, clf)
    
    def evaluate(_clf):
        scores = cross_val_score(_clf, all_xs, all_ys, scoring='accuracy', cv=10)
        print('Accuracy: {:.3f} ± {:.3f}'.format(np.mean(scores), 2 * np.std(scores)))
        _clf.fit(train_xs, train_ys)  # so that parts of the original pipeline are fitted
         
    evaluate(pipeline)


.. parsed-literal::

    Accuracy: 0.823 ± 0.071


There is one tricky bit about the code above: one may be templed to just
pass ``dense=True`` to ``DictVectorizer``: after all, in this case the
matrixes are small. But this is not a great solution, because we will
loose the ability to distinguish features that are missing and features
that have zero value.

3. Explaining weights
---------------------

In order to calculate a prediction, XGBoost sums predictions of all its
trees. The number of trees is controlled by ``n_estimators`` argument
and is 100 by default. Each tree is not a great predictor on it’s own,
but by summing across all trees, XGBoost is able to provide a robust
estimate in many cases. Here is one of the trees:

.. code:: ipython3

    booster = clf.get_booster()
    original_feature_names = booster.feature_names
    booster.feature_names = vec.get_feature_names()
    print(booster.get_dump()[0])
    # recover original feature names
    booster.feature_names = original_feature_names


.. parsed-literal::

    0:[Sex=female<-9.53674316e-07] yes=1,no=2,missing=1
    	1:[Age<13] yes=3,no=4,missing=4
    		3:[SibSp<2] yes=7,no=8,missing=7
    			7:leaf=0.145454556
    			8:leaf=-0.125
    		4:[Fare<26.2687492] yes=9,no=10,missing=9
    			9:leaf=-0.151515156
    			10:leaf=-0.0727272779
    	2:[Pclass=3<-9.53674316e-07] yes=5,no=6,missing=5
    		5:[Fare<12.1750002] yes=11,no=12,missing=12
    			11:leaf=0.0500000007
    			12:leaf=0.175193802
    		6:[Fare<24.8083496] yes=13,no=14,missing=14
    			13:leaf=0.0365591422
    			14:leaf=-0.151999995
    


We see that this tree checks *Sex*, *Age*, *Pclass*, *Fare* and *SibSp*
features. ``leaf`` gives the decision of a single tree, and they are
summed over all trees in the ensemble.

Let’s check feature importances with :func:`eli5.show_weights`:

.. code:: ipython3

    from eli5 import show_weights
    show_weights(clf, vec=vec)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
            <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
        <thead>
        <tr style="border: none;">
            <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
            <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
        </tr>
        </thead>
        <tbody>
        
            <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.4278
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Sex=female
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 88.46%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.1949
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Pclass=3
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.57%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0665
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Embarked=S
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.49%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0510
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Pclass=2
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.06%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0420
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    SibSp
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.08%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0417
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Cabin=
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.29%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0385
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Embarked=C
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.47%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0358
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Ticket=1601
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.66%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0331
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Age
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.72%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0323
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Fare
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.49%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0220
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Pclass=1
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 98.15%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0143
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Parch
                </td>
            </tr>
        
            <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name=Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)
                </td>
            </tr>
        
            <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name=Roebling, Mr. Washington Augustus II
                </td>
            </tr>
        
            <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name=Rosblom, Mr. Viktor Richard
                </td>
            </tr>
        
            <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name=Ross, Mr. John Hugo
                </td>
            </tr>
        
            <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name=Rush, Mr. Alfred George John
                </td>
            </tr>
        
            <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name=Rouse, Mr. Richard Henry
                </td>
            </tr>
        
            <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name=Ryerson, Miss. Emily Borie
                </td>
            </tr>
        
            <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name=Ryerson, Miss. Susan Parker &quot;Suzette&quot;
                </td>
            </tr>
        
        
            
                <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 1972 more &hellip;</i>
                    </td>
                </tr>
            
        
        </tbody>
    </table>
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




There are several different ways to calculate feature importances. By
default, “gain” is used, that is the average gain of the feature when it
is used in trees. Other types are “weight” - the number of times a
feature is used to split the data, and “cover” - the average coverage of
the feature. You can pass it with ``importance_type`` argument.

Now we know that two most important features are *Sex=female* and
*Pclass=3*, but we still don’t know how XGBoost decides what prediction
to make based on their values.

4. Explaining predictions
-------------------------

To get a better idea of how our classifier works, let’s examine
individual predictions with :func:`eli5.show_prediction`:

.. code:: ipython3

    from eli5 import show_prediction
    show_prediction(clf, valid_xs[1], vec=vec, show_feature_values=True)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=1
        
    </b>
    
        
        (probability <b>0.566</b>, score <b>0.264</b>)
    
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
                
                    <th style="padding: 0 0.5em 0 1em; text-align: right; border: none;">Value</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.673
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Sex=female
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 91.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.479
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Embarked=S
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                Missing
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.070
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Fare
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                7.879
            </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.004
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Cabin=
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.006
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Parch
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.50%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.009
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Pclass=2
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                Missing
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.47%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.009
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Ticket=1601
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                Missing
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.38%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.012
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Embarked=C
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                Missing
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.071
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            SibSp
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.073
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Pclass=1
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                Missing
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.36%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.147
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Age
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                19.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 91.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.528
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 85.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.100
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Pclass=3
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Weight means how much each feature contributed to the final prediction
across all trees. The idea for weight calculation is described in
http://blog.datadive.net/interpreting-random-forests/; eli5 provides an
independent implementation of this algorithm for XGBoost and most
scikit-learn tree ensembles.

Here we see that classifier thinks it’s good to be a female, but bad to
travel third class. Some features have “Missing” as value (we are
passing ``show_feature_values=True`` to view the values): that means
that the feature was missing, so in this case it’s good to not have
embarked in Southampton. This is where our decision to go with sparse
matrices comes handy - we still see that *Parch* is zero, not missing.

It’s possible to show only features that are present using
``feature_filter`` argument: it’s a function that accepts feature name
and value, and returns True value for features that should be shown:

.. code:: ipython3

    no_missing = lambda feature_name, feature_value: not np.isnan(feature_value)
    show_prediction(clf, valid_xs[1], vec=vec, show_feature_values=True, feature_filter=no_missing)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
                
                    
                    
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=1
        
    </b>
    
        
        (probability <b>0.566</b>, score <b>0.264</b>)
    
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
                
                    <th style="padding: 0 0.5em 0 1em; text-align: right; border: none;">Value</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.673
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Sex=female
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.83%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.070
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Fare
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                7.879
            </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.004
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Cabin=
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 99.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.006
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Parch
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.81%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.071
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            SibSp
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.36%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.147
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Age
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                19.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 91.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.528
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 85.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -1.100
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Pclass=3
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
                
            
    
            
    
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




5. Adding text features
-----------------------

Right now we treat *Name* field as categorical, like other text
features. But in this dataset each name is unique, so XGBoost does not
use this feature at all, because it’s such a poor discriminator: it’s
absent from the weights table in section 3.

But *Name* still might contain some useful information. We don’t want to
guess how to best pre-process it and what features to extract, so let’s
use the most general character ngram vectorizer:

.. code:: ipython3

    from sklearn.pipeline import FeatureUnion
    from sklearn.feature_extraction.text import CountVectorizer
    
    vec2 = FeatureUnion([
        ('Name', CountVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 4),
            preprocessor=lambda x: x['Name'],
            max_features=100,
        )),
        ('All', DictVectorizer()),
    ])
    clf2 = XGBClassifier()
    pipeline2 = make_pipeline(vec2, clf2)
    evaluate(pipeline2)


.. parsed-literal::

    Accuracy: 0.839 ± 0.081


In this case the pipeline is more complex, we slightly improved our
result, but the improvement is not significant. Let’s look at feature
importances:

.. code:: ipython3

    show_weights(clf2, vec=vec2)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
            <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
        <thead>
        <tr style="border: none;">
            <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
            <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
        </tr>
        </thead>
        <tbody>
        
            <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.3138
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__<span style="background-color: hsl(120, 80%, 70%); margin: 0 0.1em 0 0.1em" title="A space symbol">&emsp;</span>Mr.
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 92.18%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0821
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__Pclass=3
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.92%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0443
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__sso
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.18%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0294
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__Sex=female
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.97%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0212
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__lia
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.04%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0205
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__Fare
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0203
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__Ticket=1601
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.12%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0197
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__Embarked=S
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.23%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0187
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__<span style="background-color: hsl(120, 80%, 70%); margin: 0 0.1em 0 0.1em" title="A space symbol">&emsp;</span>Ma
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.33%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0177
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__Cabin=
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0172
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__<span style="background-color: hsl(120, 80%, 70%); margin: 0 0.1em 0 0.1em" title="A space symbol">&emsp;</span>Mar
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.42%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0168
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__s,<span style="background-color: hsl(120, 80%, 70%); margin: 0 0 0 0.1em" title="A space symbol">&emsp;</span>
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.51%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0160
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__<span style="background-color: hsl(120, 80%, 70%); margin: 0 0.1em 0 0.1em" title="A space symbol">&emsp;</span>Mr
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.54%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0157
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__son
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.76%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0138
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__ne<span style="background-color: hsl(120, 80%, 70%); margin: 0 0 0 0.1em" title="A space symbol">&emsp;</span>
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.76%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0137
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__ber
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.77%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0136
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__SibSp
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.78%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0136
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    Name__e,<span style="background-color: hsl(120, 80%, 70%); margin: 0 0 0 0.1em" title="A space symbol">&emsp;</span>
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.80%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0134
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__Pclass=1
                </td>
            </tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.91%); border: none;">
                <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                    0.0125
                    
                </td>
                <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                    All__Embarked=C
                </td>
            </tr>
        
        
            
                <tr style="background-color: hsl(120, 100.00%, 97.91%); border: none;">
                    <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                        <i>&hellip; 2072 more &hellip;</i>
                    </td>
                </tr>
            
        
        </tbody>
    </table>
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




We see that now there is a lot of features that come from the *Name*
field (in fact, a classifier based on *Name* alone gives about 0.79
accuracy). Name features listed in this way are not very informative,
they make more sense when we check out predictions. We hide missing
features here because there is a lot of missing features in text, but
they are not very interesting:

.. code:: ipython3

    from IPython.display import display
    
    for idx in [4, 5, 7, 37, 81]:
        display(show_prediction(clf2, valid_xs[idx], vec=vec2,
                                show_feature_values=True, feature_filter=no_missing))



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=1
        
    </b>
    
        
        (probability <b>0.771</b>, score <b>1.215</b>)
    
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
                
                    <th style="padding: 0 0.5em 0 1em; text-align: right; border: none;">Value</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.995
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Name: Highlighted in text (sum)
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.43%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.347
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Fare
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                17.800
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.69%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.236
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Sex=female
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 95.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.109
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Age
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                18.000
            </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.32%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.029
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Cabin=
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.069
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Parch
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 94.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.150
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Embarked=S
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 93.15%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.215
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__SibSp
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.98%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.539
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.89%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.932
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Pclass=3
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <b>Name:</b> <span style="opacity: 0.80">Arnold-Franchi,</span><span style="background-color: hsl(120, 100.00%, 83.64%); opacity: 0.86" title="0.067"> Mrs</span><span style="opacity: 0.80">. Josef (Josefi</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.242">ne </span><span style="opacity: 0.80">Franchi)</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=0
        
    </b>
    
        
        (probability <b>0.905</b>, score <b>-2.248</b>)
    
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
                
                    <th style="padding: 0 0.5em 0 1em; text-align: right; border: none;">Value</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.948
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Name: Highlighted in text (sum)
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 86.54%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.539
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 89.33%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.387
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Parch
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.80%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.221
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Age
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                45.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.73%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.071
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Cabin=
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.94%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.037
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__SibSp
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 96.86%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.067
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Pclass=1
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 87.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.492
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Fare
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                26.550
            </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <b>Name:</b> <span style="opacity: 0.80">Romain</span><span style="background-color: hsl(120, 100.00%, 86.78%); opacity: 0.84" title="0.056">e,</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.270"> </span><span style="background-color: hsl(120, 100.00%, 65.95%); opacity: 0.96" title="0.215">Mr</span><span style="background-color: hsl(120, 100.00%, 65.63%); opacity: 0.96" title="0.218">.</span><span style="opacity: 0.80"> Ch</span><span style="background-color: hsl(0, 100.00%, 87.44%); opacity: 0.84" title="-0.052">arl</span><span style="background-color: hsl(120, 100.00%, 92.42%); opacity: 0.82" title="0.025">es </span><span style="opacity: 0.80">Hallace (&quot;Mr C Rolmane&quot;)</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=0
        
    </b>
    
        
        (probability <b>0.941</b>, score <b>-2.762</b>)
    
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
                
                    <th style="padding: 0 0.5em 0 1em; text-align: right; border: none;">Value</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.946
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__SibSp
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                8.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 87.97%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.942
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Fare
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                69.550
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 90.44%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.678
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Pclass=3
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 91.86%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.539
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 96.52%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.160
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Parch
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                2.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 97.97%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.074
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Embarked=S
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 98.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.029
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Cabin=
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 90.53%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.669
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Name: Highlighted in text (sum)
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                
            </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <b>Name:</b> <span style="opacity: 0.80">Sag</span><span style="background-color: hsl(120, 100.00%, 79.23%); opacity: 0.88" title="0.112">e,</span><span style="background-color: hsl(0, 100.00%, 71.77%); opacity: 0.92" title="-0.174"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.286">Ma</span><span style="background-color: hsl(0, 100.00%, 74.79%); opacity: 0.90" title="-0.148">s</span><span style="opacity: 0.80">ter. Thomas Henry</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=1
        
    </b>
    
        
        (probability <b>0.679</b>, score <b>0.750</b>)
    
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
                
                    <th style="padding: 0 0.5em 0 1em; text-align: right; border: none;">Value</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 92.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.236
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Sex=female
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 92.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.226
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Fare
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                7.879
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.67%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.141
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Name: Highlighted in text (sum)
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.010
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__SibSp
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.029
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Cabin=
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 97.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.041
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Parch
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.539
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.932
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Pclass=3
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <b>Name:</b> <span style="opacity: 0.80">Mockl</span><span style="background-color: hsl(120, 100.00%, 70.66%); opacity: 0.93" title="0.059">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.091">r,</span><span style="background-color: hsl(120, 100.00%, 80.52%); opacity: 0.87" title="0.033"> </span><span style="opacity: 0.80">Miss. Helen</span><span style="background-color: hsl(0, 100.00%, 87.51%); opacity: 0.84" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 75.91%); opacity: 0.90" title="-0.044">Ma</span><span style="background-color: hsl(0, 100.00%, 82.98%); opacity: 0.86" title="-0.027">r</span><span style="opacity: 0.80">y &quot;Ellie&quot;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=1
        
    </b>
    
        
        (probability <b>0.660</b>, score <b>0.663</b>)
    
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
                
                    <th style="padding: 0 0.5em 0 1em; text-align: right; border: none;">Value</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 92.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.236
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Sex=female
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.161
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Fare
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                23.250
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.21%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.158
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Name: Highlighted in text (sum)
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 94.39%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.152
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Embarked=Q
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.010
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__SibSp
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                2.000
            </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.029
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Cabin=
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 96.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.069
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Parch
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                0.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.539
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.932
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            All__Pclass=3
        </td>
        
            <td style="padding: 0 0.5em 0 1em; text-align: right; border: none;">
                1.000
            </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <b>Name:</b> <span style="opacity: 0.80">McCo</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.078">y, </span><span style="opacity: 0.80">Miss. Agn</span><span style="background-color: hsl(0, 100.00%, 81.90%); opacity: 0.86" title="-0.025">es</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



Text features from the *Name* field are highlighted directly in text,
and the sum of weights is shown in the weights table as “Name:
Highlighted in text (sum)”.

Looks like name classifier tried to infer both gender and status from
the title: “Mr.” is bad because women are saved first, and it’s better
to be “Mrs.” (married) than “Miss.”. Also name classifier is trying to
pick some parts of names and surnames, especially endings, perhaps as a
proxy for social status. It’s especially bad to be “Mary” if you are
from the third class.
