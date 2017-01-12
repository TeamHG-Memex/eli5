
TextExplainer: debugging black-box text classifiers
===================================================

While eli5 supports many classifiers and preprocessing methods, it can't
support them all.

If a library is not supported by eli5 directly, or the text processing
pipeline is too complex for eli5, eli5 can still help - it provides an
implementation of `LIME <http://arxiv.org/abs/1602.04938>`__ (Ribeiro et
al., 2016) algorithm which allows to explain predictions of arbitrary
classifiers, including text classifiers. ``eli5.lime`` can also help
when it is hard to get exact mapping between model coefficients and text
features, e.g. if there is dimension reduction involved.

Example problem: LSA+SVM for 20 Newsgroups dataset
--------------------------------------------------

Let's load "20 Newsgroups" dataset and create a text processing pipeline
which is hard to debug using conventional methods: SVM with RBF kernel
trained on
`LSA <https://en.wikipedia.org/wiki/Latent_semantic_analysis>`__
features.

.. code:: ipython3

    from sklearn.datasets import fetch_20newsgroups
    
    categories = ['alt.atheism', 'soc.religion.christian', 
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers'),
    )
    twenty_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers'),
    )

.. code:: ipython3

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline, make_pipeline
    
    vec = TfidfVectorizer(min_df=3, stop_words='english',
                          ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    lsa = make_pipeline(vec, svd)
    
    clf = SVC(C=150, gamma=2e-2, probability=True)
    pipe = make_pipeline(lsa, clf)
    pipe.fit(twenty_train.data, twenty_train.target)
    pipe.score(twenty_test.data, twenty_test.target)




.. parsed-literal::

    0.89014647137150471



The dimension of the input documents is reduced to 100, and then a
kernel SVM is used to classify the documents.

This is what the pipeline returns for a document - it is pretty sure the
first message in test data belongs to sci.med:

.. code:: ipython3

    def print_prediction(doc):
        y_pred = pipe.predict_proba([doc])[0]
        for target, prob in zip(twenty_train.target_names, y_pred):
            print("{:.3f} {}".format(prob, target))    
    
    doc = twenty_test.data[0]
    print_prediction(doc)


.. parsed-literal::

    0.001 alt.atheism
    0.001 comp.graphics
    0.995 sci.med
    0.004 soc.religion.christian


TextExplainer
-------------

Such pipelines are not supported by eli5 directly, but one can use
``eli5.lime.TextExplainer`` to debug the prediction - to check what was
important in the document to make this decision.

Create a :class:`~.TextExplainer` instance, then pass the document to explain
and a black-box classifier (a function which returns probabilities) to
the :meth:`~.TextExplainer.fit` method, then check the explanation:

.. code:: ipython3

    import eli5
    from eli5.lime import TextExplainer
    
    te = TextExplainer(random_state=42)
    te.fit(doc, pipe.predict_proba)
    te.show_prediction(target_names=twenty_train.target_names)




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
    
        
        (probability <b>0.000</b>, score <b>-9.583</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.93%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.360
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -9.223
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 96.48%); opacity: 0.81" title="-0.071">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.35%); opacity: 0.81" title="-0.075">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.13%); opacity: 0.80" title="-0.053">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.79%); opacity: 0.82" title="0.197">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.33%); opacity: 0.82" title="0.216">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.24%); opacity: 0.80" title="-0.008">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.18%); opacity: 0.80" title="-0.052">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.49%); opacity: 0.87" title="-0.759">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.63%); opacity: 0.86" title="-0.637">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.048">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.56%); opacity: 0.88" title="0.875">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 91.73%); opacity: 0.82" title="0.240">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.35%); opacity: 0.81" title="-0.139">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 85.82%); opacity: 0.85" title="-0.519">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.82%); opacity: 0.82" title="-0.279">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.73%); opacity: 0.81" title="-0.126">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.61%); opacity: 0.83" title="0.288">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.61%); opacity: 0.83" title="0.288">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.35%); opacity: 0.82" title="-0.215">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.90%); opacity: 0.82" title="-0.233">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.41%); opacity: 0.80" title="0.046">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.12%); opacity: 0.82" title="-0.185">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.72%); opacity: 0.81" title="-0.094">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.36%); opacity: 0.88" title="-0.887">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 98.48%); opacity: 0.80" title="-0.021">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.18%); opacity: 0.82" title="-0.182">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.57%); opacity: 0.82" title="-0.247">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 94.18%); opacity: 0.81" title="0.145">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.52%); opacity: 0.80" title="-0.021">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.98%); opacity: 0.80" title="-0.012">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.50%); opacity: 0.83" title="0.293">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.60%); opacity: 0.82" title="0.205">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.38%); opacity: 0.82" title="0.214">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.089">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.69%); opacity: 0.80" title="-0.017">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.11%); opacity: 0.82" title="-0.266">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 94.18%); opacity: 0.81" title="0.145">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.52%); opacity: 0.80" title="-0.021">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.98%); opacity: 0.80" title="-0.012">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 90.50%); opacity: 0.83" title="0.293">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.17%); opacity: 0.80" title="-0.009">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.006">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.53%); opacity: 0.81" title="0.100">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 93.11%); opacity: 0.82" title="-0.185">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.56%); opacity: 0.81" title="-0.168">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.77%); opacity: 0.82" title="-0.198">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.84%); opacity: 0.81" title="-0.090">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 94.46%); opacity: 0.81" title="-0.135">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.017">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 91.11%); opacity: 0.82" title="-0.266">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.67%); opacity: 0.82" title="-0.202">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.05%); opacity: 0.80" title="-0.011">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.70%); opacity: 0.81" title="-0.065">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.11%); opacity: 0.84" title="0.403">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.12%); opacity: 0.80" title="-0.029">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.55%); opacity: 0.81" title="-0.168">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 93.65%); opacity: 0.81" title="-0.165">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.45%); opacity: 0.82" title="-0.172">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.49%); opacity: 0.87" title="-0.759">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 83.63%); opacity: 0.86" title="-0.637">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.27%); opacity: 0.82" title="-0.179">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.24%); opacity: 0.82" title="-0.261">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 93.87%); opacity: 0.81" title="-0.157">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.71%); opacity: 0.80" title="0.038">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.76%); opacity: 0.81" title="0.093">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.85%); opacity: 0.82" title="-0.195">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.56%); opacity: 0.80" title="-0.042">less</span><span style="opacity: 0.80">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=comp.graphics
        
    </b>
    
        
        (probability <b>0.000</b>, score <b>-8.285</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.213
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 81.78%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -8.073
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.024">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.024">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.00%); opacity: 0.84" title="-0.458">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.22%); opacity: 0.81" title="0.078">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.88%); opacity: 0.81" title="0.156">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.36%); opacity: 0.82" title="0.176">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.55%); opacity: 0.81" title="-0.132">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.27%); opacity: 0.88" title="-0.955">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.59%); opacity: 0.87" title="-0.812">stones</span><span style="opacity: 0.80">, there </span><span style="background-color: hsl(0, 100.00%, 89.92%); opacity: 0.83" title="-0.318">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 94.55%); opacity: 0.81" title="0.132">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.85%); opacity: 0.81" title="0.061">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 82.07%); opacity: 0.86" title="-0.725">medication</span><span style="opacity: 0.80"> that </span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.045">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.91%); opacity: 0.80" title="-0.013">do</span><span style="opacity: 0.80"> anything about </span><span style="background-color: hsl(120, 100.00%, 93.92%); opacity: 0.81" title="0.155">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.54%); opacity: 0.81" title="0.099">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.46%); opacity: 0.81" title="-0.135">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.15%); opacity: 0.83" title="-0.308">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 62.21%); opacity: 0.98" title="-2.104">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 94.70%); opacity: 0.81" title="-0.127">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.072">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.30%); opacity: 0.81" title="0.076">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 95.80%); opacity: 0.81" title="-0.091">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.74%); opacity: 0.81" title="-0.093">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.009">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.39%); opacity: 0.81" title="0.138">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.64%); opacity: 0.80" title="-0.003">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.010">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.37%); opacity: 0.81" title="0.139">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.10%); opacity: 0.81" title="-0.148">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.90%); opacity: 0.81" title="-0.059">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 95.80%); opacity: 0.81" title="-0.091">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.74%); opacity: 0.81" title="-0.093">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.009">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 94.39%); opacity: 0.81" title="0.138">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.64%); opacity: 0.80" title="-0.003">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.002">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.02%); opacity: 0.80" title="0.031">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.063">when</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(0, 100.00%, 92.46%); opacity: 0.82" title="-0.211">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.41%); opacity: 0.80" title="0.046">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 99.35%); opacity: 0.80" title="-0.006">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.090">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 73.23%); opacity: 0.91" title="1.286">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.22%); opacity: 0.80" title="-0.008">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.70%); opacity: 0.83" title="-0.329">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.150">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.44%); opacity: 0.82" title="0.211">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.129">that</span><span style="opacity: 0.80"> she'</span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.023">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.021">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.27%); opacity: 0.88" title="-0.955">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 80.59%); opacity: 0.87" title="-0.812">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.21%); opacity: 0.81" title="-0.144">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.87%); opacity: 0.85" title="-0.624">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.003">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.47%); opacity: 0.82" title="0.251">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.89%); opacity: 0.81" title="0.156">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.08%); opacity: 0.83" title="-0.357">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.996</b>, score <b>5.846</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 85.27%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +5.959
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.08%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.113
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as i </span><span style="background-color: hsl(120, 100.00%, 85.80%); opacity: 0.85" title="0.520">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.94%); opacity: 0.81" title="-0.058">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.72%); opacity: 0.81" title="-0.094">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.08%); opacity: 0.82" title="-0.186">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.25%); opacity: 0.80" title="-0.050">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.80%); opacity: 0.89" title="0.984">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.49%); opacity: 0.88" title="0.879">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 96.69%); opacity: 0.81" title="-0.065">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.11%); opacity: 0.81" title="-0.148">isn</span><span style="opacity: 0.80">'t any
    </span><span style="background-color: hsl(120, 100.00%, 80.17%); opacity: 0.87" title="0.838">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.93%); opacity: 0.81" title="-0.058">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.74%); opacity: 0.81" title="-0.161">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.88%); opacity: 0.81" title="-0.156">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.47%); opacity: 0.81" title="-0.135">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.53%); opacity: 0.80" title="0.043">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.55%); opacity: 0.81" title="-0.132">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.42%); opacity: 0.80" title="-0.046">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.182">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.24%); opacity: 0.82" title="0.219">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="2.282">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 98.10%); opacity: 0.80" title="-0.029">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.92%); opacity: 0.81" title="-0.059">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.52%); opacity: 0.81" title="-0.133">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.95%); opacity: 0.80" title="-0.013">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.04%); opacity: 0.80" title="-0.011">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.40%); opacity: 0.80" title="0.046">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.081">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.16%); opacity: 0.80" title="-0.009">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.34%); opacity: 0.83" title="-0.300">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.45%); opacity: 0.82" title="-0.211">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.66%); opacity: 0.81" title="0.096">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.65%); opacity: 0.81" title="0.066">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.95%); opacity: 0.80" title="-0.013">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.04%); opacity: 0.80" title="-0.011">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.40%); opacity: 0.80" title="0.046">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.081">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.24%); opacity: 0.81" title="-0.109">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.53%); opacity: 0.82" title="-0.208">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.96%); opacity: 0.81" title="0.086">surgically</span><span style="opacity: 0.80">.
    
    when </span><span style="background-color: hsl(120, 100.00%, 97.83%); opacity: 0.80" title="0.036">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.83%); opacity: 0.80" title="0.036">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.81%); opacity: 0.80" title="-0.001">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.05%); opacity: 0.80" title="-0.055">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.42%); opacity: 0.81" title="-0.073">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 82.68%); opacity: 0.86" title="-0.690">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.276">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.65%); opacity: 0.82" title="-0.203">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.121">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.04%); opacity: 0.82" title="-0.188">mention</span><span style="opacity: 0.80"> that she'd had </span><span style="background-color: hsl(120, 100.00%, 77.80%); opacity: 0.89" title="0.984">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 79.49%); opacity: 0.88" title="0.879">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.65%); opacity: 0.81" title="0.066">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.98%); opacity: 0.81" title="-0.086">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 97.47%); opacity: 0.80" title="0.044">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.76%); opacity: 0.82" title="-0.198">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.76%); opacity: 0.82" title="-0.198">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.50%); opacity: 0.82" title="0.250">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=soc.religion.christian
        
    </b>
    
        
        (probability <b>0.004</b>, score <b>-5.484</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.99%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.346
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.72%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.137
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 95.75%); opacity: 0.81" title="-0.093">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.29%); opacity: 0.80" title="0.049">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.61%); opacity: 0.82" title="-0.245">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.61%); opacity: 0.81" title="0.130">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.44%); opacity: 0.80" title="-0.005">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.87%); opacity: 0.82" title="0.194">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.40%); opacity: 0.81" title="0.104">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.17%); opacity: 0.87" title="-0.838">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.96%); opacity: 0.86" title="-0.675">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 95.02%); opacity: 0.81" title="0.116">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.54%); opacity: 0.82" title="0.248">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.035">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.08%); opacity: 0.80" title="0.054">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 84.53%); opacity: 0.85" title="-0.587">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.16%); opacity: 0.80" title="0.052">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.12%); opacity: 0.82" title="0.266">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.34%); opacity: 0.82" title="0.256">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.78%); opacity: 0.82" title="0.198">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.86%); opacity: 0.80" title="-0.014">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.83%); opacity: 0.81" title="0.158">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.53%); opacity: 0.80" title="-0.004">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.49%); opacity: 0.81" title="-0.135">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.72%); opacity: 0.82" title="-0.240">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 66.53%); opacity: 0.96" title="-1.769">pain</span><span style="opacity: 0.80">.
    
    either </span><span style="background-color: hsl(0, 100.00%, 96.81%); opacity: 0.81" title="-0.062">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.38%); opacity: 0.82" title="0.175">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.65%); opacity: 0.80" title="-0.040">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.133">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.056">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.129">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.17%); opacity: 0.80" title="-0.009">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.06%); opacity: 0.83" title="0.358">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.20%); opacity: 0.81" title="0.145">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.137">with</span><span style="opacity: 0.80"> sound, </span><span style="background-color: hsl(0, 100.00%, 97.65%); opacity: 0.80" title="-0.040">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.133">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.056">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.129">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.58%); opacity: 0.82" title="0.206">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.22%); opacity: 0.82" title="0.262">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.93%); opacity: 0.81" title="-0.119">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 96.46%); opacity: 0.81" title="-0.071">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.55%); opacity: 0.81" title="-0.069">i</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(0, 100.00%, 95.43%); opacity: 0.81" title="-0.103">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 96.34%); opacity: 0.81" title="0.075">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.182">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 94.25%); opacity: 0.81" title="0.143">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.49%); opacity: 0.81" title="-0.171">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.23%); opacity: 0.84" title="0.447">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.35%); opacity: 0.82" title="0.215">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.31%); opacity: 0.83" title="0.346">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.53%); opacity: 0.81" title="0.100">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.43%); opacity: 0.80" title="-0.005">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.009">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.83%); opacity: 0.80" title="0.035">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.17%); opacity: 0.87" title="-0.838">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 82.96%); opacity: 0.86" title="-0.675">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.20%); opacity: 0.80" title="-0.027">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.60%); opacity: 0.84" title="0.479">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 94.17%); opacity: 0.81" title="-0.146">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.10%); opacity: 0.81" title="0.082">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.92%); opacity: 0.81" title="0.120">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.59%); opacity: 0.81" title="-0.098">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.08%); opacity: 0.81" title="-0.149">less</span><span style="opacity: 0.80">.</span>
        </p>
    
        
    
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Why it works
------------

Explanation makes sense - we expect reasonable classifier to take
highlighted words in account. But how can we be sure this is how the
pipeline works, not just a nice-looking lie? A simple sanity check is to
remove or change the highlighted words, to confirm that they change the
outcome:

.. code:: ipython3

    import re
    doc2 = re.sub(r'(recall|kidney|stones|medication|pain|tech)', '', doc, flags=re.I)
    print_prediction(doc2)


.. parsed-literal::

    0.068 alt.atheism
    0.149 comp.graphics
    0.369 sci.med
    0.414 soc.religion.christian


Predicted probabilities changed a lot indeed.

And in fact, :class:`~.TextExplainer` did something similar to get the
explanation. :class:`~.TextExplainer` generated a lot of texts similar to the
document (by removing some of the words), and then trained a white-box
classifier which predicts the output of the black-box classifier (not
the true labels!). The explanation we saw is for this white-box
classifier.

This approach follows the LIME algorithm; for text data the algorithm is
actually pretty straightforward:

1. generate distorted versions of the text;
2. predict probabilities for these distorted texts using the black-box
   classifier;
3. train another classifier (one of those eli5 supports) which tries to
   predict output of a black-box classifier on these texts.

The algorithm works because even though it could be hard or impossible
to approximate a black-box classifier globally (for every possible
text), approximating it in a small neighbourhood near a given text often
works well, even with simple white-box classifiers.

Generated samples (distorted texts) are available in ``samples_``
attribute:

.. code:: ipython3

    print(te.samples_[0])


.. parsed-literal::

    As    my   kidney ,  isn' any
      can        .
    
    Either they ,     be    ,   
    to   .
    
       ,  - tech  to mention  ' had kidney
     and ,     .


By default :class:`~.TextExplainer` generates 5000 distorted texts (use
``n_samples`` argument to change the amount):

.. code:: ipython3

    len(te.samples_)




.. parsed-literal::

    5000



Trained white-box classifier and vectorizer are available as ``vec_``
and ``clf_`` attributes:

.. code:: ipython3

    te.vec_, te.clf_




.. parsed-literal::

    (CountVectorizer(analyzer='word', binary=False, decode_error='strict',
             dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
             lowercase=True, max_df=1.0, max_features=None, min_df=1,
             ngram_range=(1, 2), preprocessor=None, stop_words=None,
             strip_accents=None, token_pattern='(?u)\\b\\w+\\b', tokenizer=None,
             vocabulary=None),
     SGDClassifier(alpha=0.001, average=False, class_weight=None, epsilon=0.1,
            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
            learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
            penalty='elasticnet', power_t=0.5,
            random_state=<mtrand.RandomState object at 0x112f6eab0>,
            shuffle=True, verbose=0, warm_start=False))



Should we trust the explanation?
--------------------------------

Ok, this sounds fine, but how can we be sure that this simple text
classification pipeline approximated the black-box classifier well?

One way to do that is to check the quality on a held-out dataset (which
is also generated). :class:`~.TextExplainer` does that by default and stores
metrics in ``metrics_`` attribute:

.. code:: ipython3

    te.metrics_




.. parsed-literal::

    {'mean_KL_divergence': 0.020277596015756863, 'score': 0.98684669657535129}



-  'score' is an accuracy score weighted by cosine distance between
   generated sample and the original document (i.e. texts which are
   closer to the example are more important). Accuracy shows how good
   are 'top 1' predictions.
-  'mean\_KL\_divergence' is a mean `Kullback–Leibler
   divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__
   for all target classes; it is also weighted by distance. KL
   divergence shows how well are probabilities approximated; 0.0 means a
   perfect match.

In this example both accuracy and KL divergence are good; it means our
white-box classifier usually assigns the same labels as the black-box
classifier on the dataset we generated, and its predicted probabilities
are close to those predicted by our LSA+SVM pipeline. So it is likely
(though not guaranteed, we'll discuss it later) that the explanation is
correct and can be trusted.

When working with LIME (e.g. via :class:`~.TextExplainer`) it is always a good
idea to check these scores. If they are not good then you can tell that
something is not right.

Let's make it fail
------------------

By default :class:`~.TextExplainer` uses a very basic text processing pipeline:
Logistic Regression trained on bag-of-words and bag-of-bigrams features
(see ``te.clf_`` and ``te.vec_`` attributes). It limits a set of
black-box classifiers it can explain: because the text is seen as "bag
of words/ngrams", the default white-box pipeline can't distinguish e.g.
between the same word in the beginning of the document and in the end of
the document. Bigrams help to alleviate the problem in practice, but not
completely.

Black-box classifiers which use features like "text length" (not
directly related to tokens) can be also hard to approximate using the
default bag-of-words/ngrams model.

This kind of failure is usually detectable though - scores (accuracy and
KL divergence) will be low. Let's check it on a completely synthetic
example - a black-box classifier which assigns a class based on oddity
of document length and on a presence of 'medication' word.

.. code:: ipython3

    import numpy as np
    
    def predict_proba_len(docs):
        # nasty predict_proba - the result is based on document length,
        # and also on a presence of "medication"
        proba = [
            [0, 0, 1.0, 0] if len(doc) % 2 or 'medication' in doc else [1.0, 0, 0, 0] 
            for doc in docs
        ]
        return np.array(proba)    
    
    te3 = TextExplainer().fit(doc, predict_proba_len)
    te3.show_prediction(target_names=twenty_train.target_names)




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
    
        
        (probability <b>0.998</b>, score <b>6.380</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
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
            +6.445
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.20%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.065
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as </span><span style="background-color: hsl(120, 100.00%, 97.39%); opacity: 0.80" title="0.120">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.90%); opacity: 0.81" title="0.154">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.62%); opacity: 0.80" title="0.105">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.29%); opacity: 0.80" title="0.066">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.013">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.117">with</span><span style="opacity: 0.80"> kidney </span><span style="background-color: hsl(0, 100.00%, 97.52%); opacity: 0.80" title="-0.112">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.031">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.95%); opacity: 0.80" title="0.085">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 95.21%); opacity: 0.81" title="-0.286">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.74%); opacity: 0.84" title="1.095">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="5.929">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.26%); opacity: 0.84" title="1.157">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.34%); opacity: 0.80" title="-0.123">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.57%); opacity: 0.80" title="-0.109">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.48%); opacity: 0.80" title="-0.055">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.59%); opacity: 0.81" title="-0.254">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.74%); opacity: 0.82" title="-0.518">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.35%); opacity: 0.80" title="-0.062">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.91%); opacity: 0.80" title="-0.035">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.60%); opacity: 0.81" title="0.175">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.14%); opacity: 0.80" title="-0.074">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.063">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.92%); opacity: 0.80" title="-0.034">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.29%); opacity: 0.80" title="-0.127">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 99.33%); opacity: 0.80" title="-0.017">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.11%); opacity: 0.80" title="0.075">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.77%); opacity: 0.80" title="0.096">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.82%); opacity: 0.80" title="0.093">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.94%); opacity: 0.81" title="-0.151">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.315">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.88%); opacity: 0.80" title="-0.002">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.84%); opacity: 0.81" title="0.158">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.28%); opacity: 0.81" title="0.280">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 97.36%); opacity: 0.80" title="0.122">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.11%); opacity: 0.80" title="0.075">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.77%); opacity: 0.80" title="0.096">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 97.82%); opacity: 0.80" title="0.093">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.64%); opacity: 0.81" title="0.172">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.63%); opacity: 0.80" title="-0.048">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.315">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 96.54%); opacity: 0.81" title="-0.180">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.93%); opacity: 0.80" title="-0.086">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.30%); opacity: 0.81" title="0.278">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.29%); opacity: 0.81" title="0.198">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.94%); opacity: 0.80" title="-0.085">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.11%); opacity: 0.80" title="-0.076">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.009">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.47%); opacity: 0.80" title="0.115">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.57%); opacity: 0.80" title="-0.051">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.165">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.26%); opacity: 0.81" title="0.370">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.22%); opacity: 0.80" title="-0.070">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.12%); opacity: 0.80" title="0.075">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 95.05%); opacity: 0.81" title="0.299">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.43%); opacity: 0.81" title="0.267">had</span><span style="opacity: 0.80"> kidney
    </span><span style="background-color: hsl(120, 100.00%, 97.54%); opacity: 0.80" title="0.110">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.17%); opacity: 0.80" title="-0.072">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.67%); opacity: 0.80" title="-0.006">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.27%); opacity: 0.80" title="-0.128">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.21%); opacity: 0.80" title="0.132">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.030">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.120">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.48%); opacity: 0.80" title="-0.012">less</span><span style="opacity: 0.80">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




:class:`~.TextExplainer` correctly figured out that 'medication' is important,
but failed to account for "len(doc) % 2" condition, so the explanation
is incomplete. We can detect this failure by looking at metrics - they
are low:

.. code:: ipython3

    te3.metrics_




.. parsed-literal::

    {'mean_KL_divergence': 0.29813769123006623, 'score': 0.80148602213214504}



If (a big if...) we suspect that the fact document length is even or odd
is important, it is possible to customize :class:`~.TextExplainer` to check
this hypothesis.

To do that, we need to create a vectorizer which returns both "is odd"
feature and bag-of-words features, and pass this vectorizer to
:class:`~.TextExplainer`. This vectorizer should follow scikit-learn API. The
easiest way is to use ``FeatureUnion`` - just make sure all transformers
joined by ``FeatureUnion`` have ``get_feature_names()`` methods.

.. code:: ipython3

    from sklearn.pipeline import make_union
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.base import TransformerMixin
    
    class DocLength(TransformerMixin):
        def fit(self, X, y=None):  # some boilerplate
            return self
        
        def transform(self, X):
            return [
                # note that we needed both positive and negative 
                # feature - otherwise for linear model there won't 
                # be a feature to show in a half of the cases
                [len(doc) % 2, not len(doc) % 2] 
                for doc in X
            ]
        
        def get_feature_names(self):
            return ['is_odd', 'is_even']
    
    vec = make_union(DocLength(), CountVectorizer(ngram_range=(1,2)))
    te4 = TextExplainer(vec=vec).fit(doc[:-1], predict_proba_len)
    
    print(te4.metrics_)
    te4.explain_prediction(target_names=twenty_train.target_names)


.. parsed-literal::

    {'score': 1.0, 'mean_KL_divergence': 0.0247695693408547}




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
    
        
        (probability <b>0.997</b>, score <b>5.654</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
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
            +8.864
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            countvectorizer: Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.083
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 90.35%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.128
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            doclength__is_even
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <b>countvectorizer:</b> <span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.063">as</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.063">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.82%); opacity: 0.80" title="0.050">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.007">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.013">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.71%); opacity: 0.80" title="0.130">with</span><span style="opacity: 0.80"> kidney </span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.162">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.162">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.056">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(120, 100.00%, 88.88%); opacity: 0.83" title="1.241">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="7.725">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.95%); opacity: 0.83" title="1.230">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.96%); opacity: 0.80" title="-0.042">can</span><span style="opacity: 0.80"> do </span><span style="background-color: hsl(120, 100.00%, 98.41%); opacity: 0.80" title="0.077">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.031">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.93%); opacity: 0.80" title="0.112">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.023">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.91%); opacity: 0.80" title="0.114">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.064">the</span><span style="opacity: 0.80"> pain.
    
    </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.001">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.001">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.051">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.060">or</span><span style="opacity: 0.80"> they </span><span style="background-color: hsl(120, 100.00%, 98.59%); opacity: 0.80" title="0.065">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.063">to</span><span style="opacity: 0.80"> be broken </span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.018">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.117">with</span><span style="opacity: 0.80"> sound, </span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.060">or</span><span style="opacity: 0.80"> they </span><span style="background-color: hsl(120, 100.00%, 98.59%); opacity: 0.80" title="0.065">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.063">to</span><span style="opacity: 0.80"> be </span><span style="background-color: hsl(0, 100.00%, 99.29%); opacity: 0.80" title="-0.024">extracted</span><span style="opacity: 0.80"> surgically.
    
    when i </span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.026">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.19%); opacity: 0.80" title="-0.029">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.044">the</span><span style="opacity: 0.80"> x-ray tech happened </span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.031">to</span><span style="opacity: 0.80"> mention </span><span style="background-color: hsl(0, 100.00%, 97.96%); opacity: 0.80" title="-0.110">that</span><span style="opacity: 0.80"> she'd had kidney
    stones </span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.030">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.30%); opacity: 0.80" title="0.085">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.030">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.044">the</span><span style="opacity: 0.80"> childbirth hurt less</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Much better! It was a toy example, but the idea stands - if you think
something could be important, add it to the mix as a feature for
:class:`~.TextExplainer`.

Let's make it fail, again
-------------------------

Another possible issue is the dataset generation method. Not only
feature extraction should be powerful enough, but auto-generated texts
also should be diverse enough.

:class:`~.TextExplainer` removes random words by default, so by default it
can't e.g. provide a good explanation for a black-box classifier which
works on character level. Let's try to use :class:`~.TextExplainer` to explain
a classifier which uses char ngrams as features:

.. code:: ipython3

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    
    vec_char = HashingVectorizer(analyzer='char_wb', ngram_range=(4,5))
    clf_char = SGDClassifier(loss='log')
    
    pipe_char = make_pipeline(vec_char, clf_char)
    pipe_char.fit(twenty_train.data, twenty_train.target)
    pipe_char.score(twenty_test.data, twenty_test.target)




.. parsed-literal::

    0.87017310252996005



This pipeline is supported by eli5 directly, so in practice there is no
need to use :class:`~.TextExplainer` for it. We're using this pipeline as an
example - it is possible check the "true" explanation first, without
using :class:`~.TextExplainer`, and then compare the results with
:class:`~.TextExplainer` results.

.. code:: ipython3

    eli5.show_prediction(clf_char, doc, vec=vec_char,
                        targets=['sci.med'], target_names=twenty_train.target_names)




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
    
        
        (probability <b>0.572</b>, score <b>-0.116</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                        Contribution<sup>?</sup>
                    </th>
                
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 81.66%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.880
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.995
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 97.62%); opacity: 0.80" title="-0.002">as</span><span style="background-color: hsl(0, 100.00%, 95.41%); opacity: 0.81" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 96.76%); opacity: 0.81" title="-0.003">i</span><span style="background-color: hsl(120, 100.00%, 91.48%); opacity: 0.82" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 88.29%); opacity: 0.83" title="0.021">r</span><span style="background-color: hsl(120, 100.00%, 84.36%); opacity: 0.85" title="0.032">e</span><span style="background-color: hsl(120, 100.00%, 82.15%); opacity: 0.86" title="0.039">c</span><span style="background-color: hsl(120, 100.00%, 86.94%); opacity: 0.84" title="0.025">a</span><span style="background-color: hsl(120, 100.00%, 89.94%); opacity: 0.83" title="0.017">l</span><span style="background-color: hsl(120, 100.00%, 93.26%); opacity: 0.82" title="0.010">l</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(0, 100.00%, 96.61%); opacity: 0.81" title="-0.004">f</span><span style="background-color: hsl(0, 100.00%, 95.91%); opacity: 0.81" title="-0.005">ro</span><span style="background-color: hsl(0, 100.00%, 95.57%); opacity: 0.81" title="-0.005">m</span><span style="background-color: hsl(120, 100.00%, 87.29%); opacity: 0.84" title="0.024"> </span><span style="background-color: hsl(120, 100.00%, 86.71%); opacity: 0.84" title="0.026">my</span><span style="background-color: hsl(120, 100.00%, 87.81%); opacity: 0.84" title="0.023"> </span><span style="background-color: hsl(120, 100.00%, 97.53%); opacity: 0.80" title="0.002">b</span><span style="background-color: hsl(120, 100.00%, 96.96%); opacity: 0.81" title="0.003">ou</span><span style="background-color: hsl(120, 100.00%, 95.16%); opacity: 0.81" title="0.006">t</span><span style="background-color: hsl(120, 100.00%, 94.26%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 94.20%); opacity: 0.81" title="0.008">w</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">it</span><span style="background-color: hsl(0, 100.00%, 99.05%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 96.13%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 95.11%); opacity: 0.81" title="0.006">k</span><span style="background-color: hsl(120, 100.00%, 93.18%); opacity: 0.82" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 91.69%); opacity: 0.82" title="0.013">d</span><span style="background-color: hsl(120, 100.00%, 91.49%); opacity: 0.82" title="0.014">n</span><span style="background-color: hsl(120, 100.00%, 92.97%); opacity: 0.82" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 94.86%); opacity: 0.81" title="0.007">y</span><span style="background-color: hsl(120, 100.00%, 96.31%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(0, 100.00%, 98.87%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(0, 100.00%, 98.62%); opacity: 0.80" title="-0.001">o</span><span style="background-color: hsl(0, 100.00%, 98.57%); opacity: 0.80" title="-0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 97.25%); opacity: 0.80" title="0.003">s</span><span style="background-color: hsl(120, 100.00%, 97.71%); opacity: 0.80" title="0.002">,</span><span style="background-color: hsl(120, 100.00%, 97.51%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(0, 100.00%, 92.67%); opacity: 0.82" title="-0.011">t</span><span style="background-color: hsl(0, 100.00%, 86.89%); opacity: 0.84" title="-0.025">h</span><span style="background-color: hsl(0, 100.00%, 85.19%); opacity: 0.85" title="-0.030">e</span><span style="background-color: hsl(0, 100.00%, 86.82%); opacity: 0.84" title="-0.025">r</span><span style="background-color: hsl(0, 100.00%, 88.98%); opacity: 0.83" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 92.97%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 94.83%); opacity: 0.81" title="-0.007">i</span><span style="background-color: hsl(0, 100.00%, 93.61%); opacity: 0.81" title="-0.009">s</span><span style="background-color: hsl(0, 100.00%, 92.39%); opacity: 0.82" title="-0.012">n</span><span style="background-color: hsl(0, 100.00%, 92.94%); opacity: 0.82" title="-0.010">'</span><span style="background-color: hsl(0, 100.00%, 94.77%); opacity: 0.81" title="-0.007">t</span><span style="background-color: hsl(0, 100.00%, 92.91%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 96.70%); opacity: 0.81" title="-0.004">any</span><span style="background-color: hsl(120, 100.00%, 81.24%); opacity: 0.87" title="0.042">
    </span><span style="background-color: hsl(120, 100.00%, 72.08%); opacity: 0.92" title="0.074">m</span><span style="background-color: hsl(120, 100.00%, 64.41%); opacity: 0.97" title="0.105">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.124">d</span><span style="background-color: hsl(120, 100.00%, 64.06%); opacity: 0.97" title="0.107">i</span><span style="background-color: hsl(120, 100.00%, 72.40%); opacity: 0.92" title="0.073">c</span><span style="background-color: hsl(120, 100.00%, 84.70%); opacity: 0.85" title="0.031">a</span><span style="background-color: hsl(120, 100.00%, 92.58%); opacity: 0.82" title="0.011">t</span><span style="background-color: hsl(0, 100.00%, 94.37%); opacity: 0.81" title="-0.008">i</span><span style="background-color: hsl(0, 100.00%, 93.14%); opacity: 0.82" title="-0.010">o</span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.008">n</span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 92.40%); opacity: 0.82" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 90.90%); opacity: 0.82" title="-0.015">ha</span><span style="background-color: hsl(0, 100.00%, 93.33%); opacity: 0.82" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 98.15%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.48%); opacity: 0.81" title="0.007">can</span><span style="background-color: hsl(0, 100.00%, 93.32%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.009">do</span><span style="background-color: hsl(0, 100.00%, 89.50%); opacity: 0.83" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 91.59%); opacity: 0.82" title="-0.013">a</span><span style="background-color: hsl(0, 100.00%, 89.83%); opacity: 0.83" title="-0.018">n</span><span style="background-color: hsl(0, 100.00%, 87.70%); opacity: 0.84" title="-0.023">y</span><span style="background-color: hsl(0, 100.00%, 92.59%); opacity: 0.82" title="-0.011">t</span><span style="background-color: hsl(0, 100.00%, 92.94%); opacity: 0.82" title="-0.010">h</span><span style="background-color: hsl(0, 100.00%, 91.80%); opacity: 0.82" title="-0.013">i</span><span style="background-color: hsl(0, 100.00%, 94.14%); opacity: 0.81" title="-0.008">n</span><span style="background-color: hsl(0, 100.00%, 92.83%); opacity: 0.82" title="-0.011">g</span><span style="background-color: hsl(0, 100.00%, 97.88%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 93.00%); opacity: 0.82" title="0.010">a</span><span style="background-color: hsl(120, 100.00%, 90.62%); opacity: 0.83" title="0.016">b</span><span style="background-color: hsl(120, 100.00%, 90.29%); opacity: 0.83" title="0.016">o</span><span style="background-color: hsl(120, 100.00%, 91.31%); opacity: 0.82" title="0.014">u</span><span style="background-color: hsl(120, 100.00%, 93.88%); opacity: 0.81" title="0.009">t</span><span style="background-color: hsl(0, 100.00%, 96.40%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 90.89%); opacity: 0.82" title="-0.015">t</span><span style="background-color: hsl(0, 100.00%, 89.57%); opacity: 0.83" title="-0.018">he</span><span style="background-color: hsl(0, 100.00%, 91.49%); opacity: 0.82" title="-0.014">m</span><span style="background-color: hsl(0, 100.00%, 97.27%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 94.71%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 93.53%); opacity: 0.81" title="0.009">x</span><span style="background-color: hsl(120, 100.00%, 95.09%); opacity: 0.81" title="0.006">c</span><span style="background-color: hsl(120, 100.00%, 94.89%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.002">p</span><span style="background-color: hsl(0, 100.00%, 99.45%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(0, 100.00%, 90.03%); opacity: 0.83" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 86.68%); opacity: 0.84" title="-0.026">r</span><span style="background-color: hsl(0, 100.00%, 83.90%); opacity: 0.85" title="-0.034">e</span><span style="background-color: hsl(0, 100.00%, 83.15%); opacity: 0.86" title="-0.036">l</span><span style="background-color: hsl(0, 100.00%, 86.56%); opacity: 0.84" title="-0.026">i</span><span style="background-color: hsl(0, 100.00%, 96.06%); opacity: 0.81" title="-0.005">e</span><span style="background-color: hsl(0, 100.00%, 98.95%); opacity: 0.80" title="-0.001">v</span><span style="background-color: hsl(120, 100.00%, 96.86%); opacity: 0.81" title="0.003">e</span><span style="background-color: hsl(0, 100.00%, 94.19%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 87.45%); opacity: 0.84" title="-0.024">the</span><span style="background-color: hsl(0, 100.00%, 96.67%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 86.61%); opacity: 0.84" title="0.026">p</span><span style="background-color: hsl(120, 100.00%, 84.93%); opacity: 0.85" title="0.031">a</span><span style="background-color: hsl(120, 100.00%, 83.08%); opacity: 0.86" title="0.036">i</span><span style="background-color: hsl(120, 100.00%, 85.87%); opacity: 0.85" title="0.028">n</span><span style="background-color: hsl(120, 100.00%, 92.48%); opacity: 0.82" title="0.011">.</span><span style="background-color: hsl(120, 100.00%, 96.77%); opacity: 0.81" title="0.003"> </span><span style="background-color: hsl(0, 100.00%, 91.48%); opacity: 0.82" title="-0.014">e</span><span style="background-color: hsl(0, 100.00%, 88.86%); opacity: 0.83" title="-0.020">i</span><span style="background-color: hsl(0, 100.00%, 83.32%); opacity: 0.86" title="-0.036">t</span><span style="background-color: hsl(0, 100.00%, 82.64%); opacity: 0.86" title="-0.038">h</span><span style="background-color: hsl(0, 100.00%, 85.73%); opacity: 0.85" title="-0.028">e</span><span style="background-color: hsl(0, 100.00%, 87.77%); opacity: 0.84" title="-0.023">r</span><span style="background-color: hsl(0, 100.00%, 92.45%); opacity: 0.82" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 97.10%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 95.42%); opacity: 0.81" title="0.006">he</span><span style="background-color: hsl(120, 100.00%, 93.05%); opacity: 0.82" title="0.010">y</span><span style="background-color: hsl(0, 100.00%, 98.87%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 93.00%); opacity: 0.82" title="-0.010">p</span><span style="background-color: hsl(0, 100.00%, 92.32%); opacity: 0.82" title="-0.012">a</span><span style="background-color: hsl(0, 100.00%, 92.53%); opacity: 0.82" title="-0.011">s</span><span style="background-color: hsl(0, 100.00%, 94.24%); opacity: 0.81" title="-0.008">s</span><span style="background-color: hsl(0, 100.00%, 98.77%); opacity: 0.80" title="-0.001">,</span><span style="background-color: hsl(0, 100.00%, 96.39%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 96.52%); opacity: 0.81" title="-0.004">or</span><span style="background-color: hsl(0, 100.00%, 95.09%); opacity: 0.81" title="-0.006"> </span><span style="background-color: hsl(120, 100.00%, 97.10%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 95.42%); opacity: 0.81" title="0.006">he</span><span style="background-color: hsl(120, 100.00%, 93.05%); opacity: 0.82" title="0.010">y</span><span style="background-color: hsl(0, 100.00%, 97.62%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(0, 100.00%, 91.28%); opacity: 0.82" title="-0.014">h</span><span style="background-color: hsl(0, 100.00%, 88.84%); opacity: 0.83" title="-0.020">av</span><span style="background-color: hsl(0, 100.00%, 89.91%); opacity: 0.83" title="-0.017">e</span><span style="background-color: hsl(0, 100.00%, 98.65%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(120, 100.00%, 94.60%); opacity: 0.81" title="0.007">to</span><span style="background-color: hsl(120, 100.00%, 97.76%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(0, 100.00%, 95.73%); opacity: 0.81" title="-0.005">be</span><span style="background-color: hsl(0, 100.00%, 95.46%); opacity: 0.81" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 98.73%); opacity: 0.80" title="-0.001">b</span><span style="background-color: hsl(0, 100.00%, 98.10%); opacity: 0.80" title="-0.002">r</span><span style="background-color: hsl(0, 100.00%, 97.70%); opacity: 0.80" title="-0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">k</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.00%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(120, 100.00%, 95.46%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 97.04%); opacity: 0.80" title="0.003">up</span><span style="background-color: hsl(120, 100.00%, 94.36%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 94.20%); opacity: 0.81" title="0.008">w</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">it</span><span style="background-color: hsl(0, 100.00%, 99.05%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 97.27%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 90.04%); opacity: 0.83" title="0.017">s</span><span style="background-color: hsl(120, 100.00%, 88.21%); opacity: 0.83" title="0.022">o</span><span style="background-color: hsl(120, 100.00%, 88.92%); opacity: 0.83" title="0.020">u</span><span style="background-color: hsl(120, 100.00%, 91.62%); opacity: 0.82" title="0.013">n</span><span style="background-color: hsl(120, 100.00%, 97.08%); opacity: 0.80" title="0.003">d</span><span style="background-color: hsl(0, 100.00%, 94.01%); opacity: 0.81" title="-0.008">,</span><span style="background-color: hsl(0, 100.00%, 93.04%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 96.52%); opacity: 0.81" title="-0.004">or</span><span style="background-color: hsl(0, 100.00%, 95.09%); opacity: 0.81" title="-0.006"> </span><span style="background-color: hsl(120, 100.00%, 97.10%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 95.42%); opacity: 0.81" title="0.006">he</span><span style="background-color: hsl(120, 100.00%, 93.05%); opacity: 0.82" title="0.010">y</span><span style="background-color: hsl(0, 100.00%, 97.62%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(0, 100.00%, 91.28%); opacity: 0.82" title="-0.014">h</span><span style="background-color: hsl(0, 100.00%, 88.84%); opacity: 0.83" title="-0.020">av</span><span style="background-color: hsl(0, 100.00%, 89.91%); opacity: 0.83" title="-0.017">e</span><span style="background-color: hsl(0, 100.00%, 98.65%); opacity: 0.80" title="-0.001">
    </span><span style="background-color: hsl(120, 100.00%, 94.60%); opacity: 0.81" title="0.007">to</span><span style="background-color: hsl(120, 100.00%, 97.76%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(0, 100.00%, 95.73%); opacity: 0.81" title="-0.005">be</span><span style="background-color: hsl(0, 100.00%, 99.45%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.015">e</span><span style="background-color: hsl(120, 100.00%, 88.78%); opacity: 0.83" title="0.020">x</span><span style="background-color: hsl(120, 100.00%, 88.33%); opacity: 0.83" title="0.021">t</span><span style="background-color: hsl(120, 100.00%, 88.14%); opacity: 0.84" title="0.022">r</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.018">a</span><span style="background-color: hsl(120, 100.00%, 90.05%); opacity: 0.83" title="0.017">c</span><span style="background-color: hsl(120, 100.00%, 86.71%); opacity: 0.84" title="0.026">t</span><span style="background-color: hsl(120, 100.00%, 88.15%); opacity: 0.84" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 89.57%); opacity: 0.83" title="0.018">d</span><span style="background-color: hsl(120, 100.00%, 84.58%); opacity: 0.85" title="0.032"> </span><span style="background-color: hsl(120, 100.00%, 85.18%); opacity: 0.85" title="0.030">s</span><span style="background-color: hsl(120, 100.00%, 84.25%); opacity: 0.85" title="0.033">u</span><span style="background-color: hsl(120, 100.00%, 82.51%); opacity: 0.86" title="0.038">r</span><span style="background-color: hsl(120, 100.00%, 88.12%); opacity: 0.84" title="0.022">g</span><span style="background-color: hsl(120, 100.00%, 88.51%); opacity: 0.83" title="0.021">i</span><span style="background-color: hsl(120, 100.00%, 87.44%); opacity: 0.84" title="0.024">c</span><span style="background-color: hsl(120, 100.00%, 86.73%); opacity: 0.84" title="0.026">a</span><span style="background-color: hsl(120, 100.00%, 87.23%); opacity: 0.84" title="0.024">l</span><span style="background-color: hsl(120, 100.00%, 92.11%); opacity: 0.82" title="0.012">l</span><span style="background-color: hsl(120, 100.00%, 95.00%); opacity: 0.81" title="0.006">y</span><span style="background-color: hsl(0, 100.00%, 97.45%); opacity: 0.80" title="-0.002">.</span><span style="background-color: hsl(120, 100.00%, 94.51%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 89.63%); opacity: 0.83" title="0.018">w</span><span style="background-color: hsl(120, 100.00%, 88.51%); opacity: 0.83" title="0.021">he</span><span style="background-color: hsl(120, 100.00%, 89.83%); opacity: 0.83" title="0.018">n</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 96.76%); opacity: 0.81" title="-0.003">i</span><span style="background-color: hsl(0, 100.00%, 95.26%); opacity: 0.81" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 94.02%); opacity: 0.81" title="-0.008">w</span><span style="background-color: hsl(0, 100.00%, 95.91%); opacity: 0.81" title="-0.005">as</span><span style="background-color: hsl(0, 100.00%, 96.85%); opacity: 0.81" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 97.92%); opacity: 0.80" title="0.002">in,</span><span style="background-color: hsl(0, 100.00%, 93.29%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 87.45%); opacity: 0.84" title="-0.024">the</span><span style="background-color: hsl(0, 100.00%, 89.41%); opacity: 0.83" title="-0.019"> </span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.001">x</span><span style="background-color: hsl(120, 100.00%, 98.32%); opacity: 0.80" title="0.001">-</span><span style="background-color: hsl(0, 100.00%, 96.75%); opacity: 0.81" title="-0.003">r</span><span style="background-color: hsl(0, 100.00%, 96.56%); opacity: 0.81" title="-0.004">a</span><span style="background-color: hsl(0, 100.00%, 96.10%); opacity: 0.81" title="-0.004">y</span><span style="background-color: hsl(0, 100.00%, 94.51%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 96.11%); opacity: 0.81" title="-0.004">t</span><span style="background-color: hsl(0, 100.00%, 96.15%); opacity: 0.81" title="-0.004">ec</span><span style="background-color: hsl(0, 100.00%, 96.96%); opacity: 0.81" title="-0.003">h</span><span style="background-color: hsl(120, 100.00%, 97.50%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.007">h</span><span style="background-color: hsl(120, 100.00%, 91.58%); opacity: 0.82" title="0.013">a</span><span style="background-color: hsl(120, 100.00%, 90.31%); opacity: 0.83" title="0.016">p</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.012">p</span><span style="background-color: hsl(120, 100.00%, 95.51%); opacity: 0.81" title="0.005">e</span><span style="background-color: hsl(0, 100.00%, 96.49%); opacity: 0.81" title="-0.004">n</span><span style="background-color: hsl(0, 100.00%, 92.83%); opacity: 0.82" title="-0.011">e</span><span style="background-color: hsl(0, 100.00%, 93.99%); opacity: 0.81" title="-0.008">d</span><span style="background-color: hsl(120, 100.00%, 97.04%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 94.60%); opacity: 0.81" title="0.007">to</span><span style="background-color: hsl(120, 100.00%, 94.16%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 89.69%); opacity: 0.83" title="0.018">e</span><span style="background-color: hsl(120, 100.00%, 89.63%); opacity: 0.83" title="0.018">n</span><span style="background-color: hsl(120, 100.00%, 87.58%); opacity: 0.84" title="0.023">t</span><span style="background-color: hsl(120, 100.00%, 92.74%); opacity: 0.82" title="0.011">i</span><span style="background-color: hsl(0, 100.00%, 95.74%); opacity: 0.81" title="-0.005">o</span><span style="background-color: hsl(0, 100.00%, 95.80%); opacity: 0.81" title="-0.005">n</span><span style="background-color: hsl(0, 100.00%, 90.77%); opacity: 0.82" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 92.40%); opacity: 0.82" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 90.90%); opacity: 0.82" title="-0.015">ha</span><span style="background-color: hsl(0, 100.00%, 93.33%); opacity: 0.82" title="-0.010">t</span><span style="background-color: hsl(120, 100.00%, 95.02%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 90.81%); opacity: 0.82" title="0.015">s</span><span style="background-color: hsl(120, 100.00%, 90.71%); opacity: 0.82" title="0.015">h</span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.001">'</span><span style="background-color: hsl(0, 100.00%, 98.89%); opacity: 0.80" title="-0.001">d</span><span style="background-color: hsl(120, 100.00%, 91.43%); opacity: 0.82" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 88.08%); opacity: 0.84" title="0.022">had</span><span style="background-color: hsl(120, 100.00%, 90.04%); opacity: 0.83" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 95.11%); opacity: 0.81" title="0.006">k</span><span style="background-color: hsl(120, 100.00%, 93.18%); opacity: 0.82" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 91.69%); opacity: 0.82" title="0.013">d</span><span style="background-color: hsl(120, 100.00%, 91.49%); opacity: 0.82" title="0.014">n</span><span style="background-color: hsl(120, 100.00%, 92.97%); opacity: 0.82" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 94.86%); opacity: 0.81" title="0.007">y</span><span style="background-color: hsl(120, 100.00%, 96.31%); opacity: 0.81" title="0.004">
    </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(0, 100.00%, 98.87%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.45%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(120, 100.00%, 97.28%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.005">s</span><span style="background-color: hsl(120, 100.00%, 93.87%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 92.97%); opacity: 0.82" title="0.010">and</span><span style="background-color: hsl(120, 100.00%, 91.03%); opacity: 0.82" title="0.015"> </span><span style="background-color: hsl(120, 100.00%, 96.62%); opacity: 0.81" title="0.004">c</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(0, 100.00%, 98.81%); opacity: 0.80" title="-0.001">i</span><span style="background-color: hsl(0, 100.00%, 94.65%); opacity: 0.81" title="-0.007">l</span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.005">d</span><span style="background-color: hsl(0, 100.00%, 95.94%); opacity: 0.81" title="-0.005">r</span><span style="background-color: hsl(0, 100.00%, 94.97%); opacity: 0.81" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 95.88%); opacity: 0.81" title="-0.005">n</span><span style="background-color: hsl(0, 100.00%, 96.25%); opacity: 0.81" title="-0.004">,</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 92.97%); opacity: 0.82" title="0.010">and</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(0, 100.00%, 87.45%); opacity: 0.84" title="-0.024">the</span><span style="background-color: hsl(0, 100.00%, 90.50%); opacity: 0.83" title="-0.016"> </span><span style="background-color: hsl(120, 100.00%, 96.62%); opacity: 0.81" title="0.004">c</span><span style="background-color: hsl(120, 100.00%, 97.66%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 97.38%); opacity: 0.80" title="0.003">i</span><span style="background-color: hsl(0, 100.00%, 97.82%); opacity: 0.80" title="-0.002">l</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.003">b</span><span style="background-color: hsl(120, 100.00%, 96.31%); opacity: 0.81" title="0.004">i</span><span style="background-color: hsl(120, 100.00%, 96.65%); opacity: 0.81" title="0.004">r</span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.19%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 98.19%); opacity: 0.80" title="0.001">ur</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 95.40%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 93.85%); opacity: 0.81" title="0.009">l</span><span style="background-color: hsl(120, 100.00%, 92.48%); opacity: 0.82" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 92.73%); opacity: 0.82" title="0.011">s</span><span style="background-color: hsl(120, 100.00%, 94.33%); opacity: 0.81" title="0.008">s</span><span style="background-color: hsl(120, 100.00%, 97.20%); opacity: 0.80" title="0.003">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




:class:`~.TextExplainer` produces a different result:

.. code:: ipython3

    te = TextExplainer(random_state=42).fit(doc, pipe_char.predict_proba)
    print(te.metrics_)
    te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)


.. parsed-literal::

    {'score': 0.93454054240068041, 'mean_KL_divergence': 0.014021429806131684}




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
    
        
        (probability <b>0.564</b>, score <b>0.602</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
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
            +0.982
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 89.72%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.380
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 97.29%); opacity: 0.80" title="0.016">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.33%); opacity: 0.80" title="-0.008">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.08%); opacity: 0.84" title="0.169">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.21%); opacity: 0.80" title="-0.009">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.14%); opacity: 0.81" title="0.027">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.75%); opacity: 0.80" title="-0.005">bout</span><span style="opacity: 0.80"> with </span><span style="background-color: hsl(120, 100.00%, 89.81%); opacity: 0.83" title="0.108">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.59%); opacity: 0.81" title="0.023">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 88.41%); opacity: 0.83" title="-0.130">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.43%); opacity: 0.83" title="0.114">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 89.43%); opacity: 0.83" title="0.114">t</span><span style="opacity: 0.80"> any
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.763">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.88%); opacity: 0.85" title="-0.172">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.49%); opacity: 0.81" title="0.057">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.13%); opacity: 0.81" title="0.049">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.54%); opacity: 0.81" title="-0.056">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.71%); opacity: 0.82" title="0.081">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.59%); opacity: 0.84" title="-0.160">them</span><span style="opacity: 0.80"> except </span><span style="background-color: hsl(0, 100.00%, 83.77%); opacity: 0.86" title="-0.210">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.43%); opacity: 0.84" title="-0.163">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.17%); opacity: 0.87" title="0.280">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 89.87%); opacity: 0.83" title="-0.107">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.94%); opacity: 0.80" title="-0.011">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.49%); opacity: 0.81" title="-0.034">pass</span><span style="opacity: 0.80">, or </span><span style="background-color: hsl(0, 100.00%, 97.94%); opacity: 0.80" title="-0.011">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.93%); opacity: 0.82" title="-0.092">have</span><span style="opacity: 0.80"> to be </span><span style="background-color: hsl(0, 100.00%, 91.03%); opacity: 0.82" title="-0.090">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.90%); opacity: 0.81" title="-0.052">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.36%); opacity: 0.82" title="-0.085">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.70%); opacity: 0.81" title="-0.054">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.031">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.94%); opacity: 0.80" title="-0.011">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.93%); opacity: 0.82" title="-0.092">have</span><span style="opacity: 0.80">
    to </span><span style="background-color: hsl(0, 100.00%, 94.84%); opacity: 0.81" title="-0.041">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.65%); opacity: 0.86" title="0.212">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.57%); opacity: 0.90" title="0.399">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 88.91%); opacity: 0.83" title="0.122">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.40%); opacity: 0.81" title="-0.024">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.02%); opacity: 0.80" title="-0.010">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.95%); opacity: 0.81" title="0.029">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 88.10%); opacity: 0.84" title="-0.135">the</span><span style="opacity: 0.80"> x-ray </span><span style="background-color: hsl(120, 100.00%, 95.85%); opacity: 0.81" title="0.030">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.018">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.67%); opacity: 0.81" title="-0.022">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.06%); opacity: 0.85" title="0.187">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.49%); opacity: 0.86" title="-0.234">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.004">she</span><span style="opacity: 0.80">'d </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.169">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.34%); opacity: 0.83" title="0.100">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 96.59%); opacity: 0.81" title="0.023">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.95%); opacity: 0.82" title="0.091">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.58%); opacity: 0.82" title="-0.082">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 90.95%); opacity: 0.82" title="0.091">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.64%); opacity: 0.81" title="-0.055">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.34%); opacity: 0.82" title="0.072">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.09%); opacity: 0.81" title="0.038">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Scores look OK but not great; the explanation kind of makes sense on a
first sight, but we know that the classifier works in a different way.

To explain such black-box classifiers we need to change both dataset
generation method (change/remove individual characters, not only words)
and feature extraction method (e.g. use char ngrams instead of words and
word ngrams).

:class:`~.TextExplainer` has an option (``char_based=True``) to use char-based
sampling and char-based classifier. If this makes a more powerful
explanation engine why not always use it?

.. code:: ipython3

    te = TextExplainer(char_based=True, random_state=42)
    te.fit(doc, pipe_char.predict_proba)
    print(te.metrics_)
    te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)


.. parsed-literal::

    {'score': 0.52104555439486744, 'mean_KL_divergence': 0.19554815684055157}




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
    
        
        (probability <b>0.360</b>, score <b>0.043</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
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
            +0.241
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 82.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.198
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 85.48%); opacity: 0.85" title="0.058">a</span><span style="background-color: hsl(120, 100.00%, 88.34%); opacity: 0.83" title="0.043">s</span><span style="background-color: hsl(0, 100.00%, 92.51%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 99.29%); opacity: 0.80" title="-0.001">i</span><span style="background-color: hsl(0, 100.00%, 90.90%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(120, 100.00%, 96.18%); opacity: 0.81" title="0.009">r</span><span style="background-color: hsl(120, 100.00%, 89.05%); opacity: 0.83" title="0.039">e</span><span style="background-color: hsl(120, 100.00%, 84.95%); opacity: 0.85" title="0.061">c</span><span style="background-color: hsl(120, 100.00%, 82.77%); opacity: 0.86" title="0.074">a</span><span style="background-color: hsl(120, 100.00%, 82.91%); opacity: 0.86" title="0.074">l</span><span style="background-color: hsl(0, 100.00%, 94.68%); opacity: 0.81" title="-0.014">l</span><span style="background-color: hsl(0, 100.00%, 90.50%); opacity: 0.83" title="-0.032"> </span><span style="background-color: hsl(0, 100.00%, 97.28%); opacity: 0.80" title="-0.005">f</span><span style="background-color: hsl(120, 100.00%, 92.74%); opacity: 0.82" title="0.022">r</span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.005">o</span><span style="background-color: hsl(0, 100.00%, 89.26%); opacity: 0.83" title="-0.038">m</span><span style="background-color: hsl(0, 100.00%, 93.83%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(120, 100.00%, 94.36%); opacity: 0.81" title="0.015">m</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(0, 100.00%, 96.12%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(120, 100.00%, 83.06%); opacity: 0.86" title="0.073">b</span><span style="background-color: hsl(120, 100.00%, 85.27%); opacity: 0.85" title="0.060">o</span><span style="background-color: hsl(120, 100.00%, 92.78%); opacity: 0.82" title="0.022">u</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.005">t</span><span style="background-color: hsl(0, 100.00%, 96.07%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 94.06%); opacity: 0.81" title="-0.016">w</span><span style="background-color: hsl(120, 100.00%, 87.63%); opacity: 0.84" title="0.046">i</span><span style="background-color: hsl(120, 100.00%, 80.27%); opacity: 0.87" title="0.090">t</span><span style="background-color: hsl(120, 100.00%, 90.71%); opacity: 0.82" title="0.031">h </span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.021">k</span><span style="background-color: hsl(120, 100.00%, 93.36%); opacity: 0.82" title="0.019">i</span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.006">d</span><span style="background-color: hsl(120, 100.00%, 93.27%); opacity: 0.82" title="0.019">n</span><span style="background-color: hsl(120, 100.00%, 88.20%); opacity: 0.83" title="0.043">e</span><span style="background-color: hsl(120, 100.00%, 94.38%); opacity: 0.81" title="0.015">y</span><span style="background-color: hsl(0, 100.00%, 96.11%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(0, 100.00%, 88.25%); opacity: 0.83" title="-0.043">to</span><span style="background-color: hsl(0, 100.00%, 81.83%); opacity: 0.86" title="-0.080">n</span><span style="background-color: hsl(0, 100.00%, 89.31%); opacity: 0.83" title="-0.038">e</span><span style="background-color: hsl(0, 100.00%, 93.83%); opacity: 0.81" title="-0.017">s</span><span style="background-color: hsl(0, 100.00%, 90.78%); opacity: 0.82" title="-0.030">,</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 94.04%); opacity: 0.81" title="0.016">t</span><span style="background-color: hsl(0, 100.00%, 84.63%); opacity: 0.85" title="-0.063">h</span><span style="background-color: hsl(0, 100.00%, 80.98%); opacity: 0.87" title="-0.086">e</span><span style="background-color: hsl(0, 100.00%, 95.71%); opacity: 0.81" title="-0.010">r</span><span style="background-color: hsl(0, 100.00%, 93.47%); opacity: 0.82" title="-0.019">e</span><span style="background-color: hsl(0, 100.00%, 89.26%); opacity: 0.83" title="-0.038"> </span><span style="background-color: hsl(0, 100.00%, 89.92%); opacity: 0.83" title="-0.035">i</span><span style="opacity: 0.80">s</span><span style="background-color: hsl(0, 100.00%, 83.57%); opacity: 0.86" title="-0.070">n</span><span style="background-color: hsl(0, 100.00%, 86.18%); opacity: 0.84" title="-0.054">'</span><span style="background-color: hsl(120, 100.00%, 94.34%); opacity: 0.81" title="0.015">t</span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 91.61%); opacity: 0.82" title="0.027">a</span><span style="background-color: hsl(120, 100.00%, 93.05%); opacity: 0.82" title="0.020">n</span><span style="background-color: hsl(120, 100.00%, 92.35%); opacity: 0.82" title="0.023">y</span><span style="background-color: hsl(120, 100.00%, 91.59%); opacity: 0.82" title="0.027">
    </span><span style="background-color: hsl(120, 100.00%, 65.30%); opacity: 0.96" title="0.203">m</span><span style="background-color: hsl(120, 100.00%, 65.48%); opacity: 0.96" title="0.201">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.248">d</span><span style="background-color: hsl(120, 100.00%, 66.10%); opacity: 0.96" title="0.196">i</span><span style="background-color: hsl(120, 100.00%, 76.59%); opacity: 0.89" title="0.115">c</span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(0, 100.00%, 99.67%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(0, 100.00%, 78.26%); opacity: 0.88" title="-0.104">io</span><span style="background-color: hsl(0, 100.00%, 82.70%); opacity: 0.86" title="-0.075">n</span><span style="background-color: hsl(0, 100.00%, 90.80%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 95.89%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 91.76%); opacity: 0.82" title="-0.026">h</span><span style="background-color: hsl(0, 100.00%, 93.76%); opacity: 0.81" title="-0.017">a</span><span style="background-color: hsl(0, 100.00%, 84.13%); opacity: 0.85" title="-0.066">t</span><span style="background-color: hsl(0, 100.00%, 79.38%); opacity: 0.88" title="-0.096"> </span><span style="background-color: hsl(0, 100.00%, 78.57%); opacity: 0.88" title="-0.102">c</span><span style="background-color: hsl(0, 100.00%, 99.12%); opacity: 0.80" title="-0.001">a</span><span style="background-color: hsl(120, 100.00%, 89.26%); opacity: 0.83" title="0.038">n</span><span style="background-color: hsl(0, 100.00%, 85.82%); opacity: 0.85" title="-0.056"> d</span><span style="opacity: 0.80">o</span><span style="background-color: hsl(120, 100.00%, 96.95%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 88.05%); opacity: 0.84" title="0.044">a</span><span style="background-color: hsl(0, 100.00%, 98.59%); opacity: 0.80" title="-0.002">n</span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.014">y</span><span style="background-color: hsl(0, 100.00%, 95.27%); opacity: 0.81" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 89.38%); opacity: 0.83" title="-0.037">h</span><span style="background-color: hsl(0, 100.00%, 91.23%); opacity: 0.82" title="-0.028">i</span><span style="background-color: hsl(120, 100.00%, 93.59%); opacity: 0.81" title="0.018">n</span><span style="background-color: hsl(0, 100.00%, 86.46%); opacity: 0.84" title="-0.053">g</span><span style="background-color: hsl(0, 100.00%, 93.90%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(120, 100.00%, 97.28%); opacity: 0.80" title="0.005">a</span><span style="background-color: hsl(120, 100.00%, 82.02%); opacity: 0.86" title="0.079">bo</span><span style="background-color: hsl(120, 100.00%, 91.16%); opacity: 0.82" title="0.029">u</span><span style="background-color: hsl(0, 100.00%, 97.22%); opacity: 0.80" title="-0.005">t</span><span style="background-color: hsl(0, 100.00%, 96.19%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 92.25%); opacity: 0.82" title="-0.024">t</span><span style="background-color: hsl(0, 100.00%, 83.64%); opacity: 0.86" title="-0.069">h</span><span style="background-color: hsl(0, 100.00%, 74.40%); opacity: 0.91" title="-0.131">e</span><span style="background-color: hsl(0, 100.00%, 87.45%); opacity: 0.84" title="-0.047">m</span><span style="background-color: hsl(0, 100.00%, 86.17%); opacity: 0.84" title="-0.054"> </span><span style="background-color: hsl(0, 100.00%, 88.84%); opacity: 0.83" title="-0.040">e</span><span style="background-color: hsl(0, 100.00%, 87.61%); opacity: 0.84" title="-0.046">x</span><span style="background-color: hsl(0, 100.00%, 96.44%); opacity: 0.81" title="-0.008">c</span><span style="background-color: hsl(0, 100.00%, 94.44%); opacity: 0.81" title="-0.015">e</span><span style="background-color: hsl(0, 100.00%, 99.35%); opacity: 0.80" title="-0.001">p</span><span style="background-color: hsl(120, 100.00%, 89.90%); opacity: 0.83" title="0.035">t</span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 83.05%); opacity: 0.86" title="-0.073">r</span><span style="background-color: hsl(0, 100.00%, 93.77%); opacity: 0.81" title="-0.017">e</span><span style="background-color: hsl(120, 100.00%, 90.15%); opacity: 0.83" title="0.034">l</span><span style="background-color: hsl(120, 100.00%, 84.17%); opacity: 0.85" title="0.066">i</span><span style="background-color: hsl(120, 100.00%, 95.40%); opacity: 0.81" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 92.69%); opacity: 0.82" title="0.022">v</span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.005">e</span><span style="background-color: hsl(120, 100.00%, 95.09%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 92.30%); opacity: 0.82" title="0.024">t</span><span style="background-color: hsl(0, 100.00%, 85.04%); opacity: 0.85" title="-0.061">h</span><span style="background-color: hsl(0, 100.00%, 87.19%); opacity: 0.84" title="-0.049">e</span><span style="background-color: hsl(0, 100.00%, 92.04%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 99.11%); opacity: 0.80" title="-0.001">p</span><span style="background-color: hsl(120, 100.00%, 90.32%); opacity: 0.83" title="0.033">a</span><span style="background-color: hsl(120, 100.00%, 79.80%); opacity: 0.88" title="0.093">i</span><span style="background-color: hsl(120, 100.00%, 80.48%); opacity: 0.87" title="0.089">n</span><span style="background-color: hsl(120, 100.00%, 97.76%); opacity: 0.80" title="0.004">.</span><span style="background-color: hsl(0, 100.00%, 91.77%); opacity: 0.82" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 88.50%); opacity: 0.83" title="-0.042">e</span><span style="background-color: hsl(120, 100.00%, 94.87%); opacity: 0.81" title="0.013">i</span><span style="background-color: hsl(0, 100.00%, 97.55%); opacity: 0.80" title="-0.005">t</span><span style="background-color: hsl(0, 100.00%, 76.47%); opacity: 0.89" title="-0.116">he</span><span style="background-color: hsl(0, 100.00%, 85.94%); opacity: 0.84" title="-0.056">r</span><span style="background-color: hsl(0, 100.00%, 88.97%); opacity: 0.83" title="-0.039"> </span><span style="background-color: hsl(120, 100.00%, 90.56%); opacity: 0.83" title="0.032">t</span><span style="background-color: hsl(0, 100.00%, 84.91%); opacity: 0.85" title="-0.062">h</span><span style="background-color: hsl(0, 100.00%, 86.84%); opacity: 0.84" title="-0.051">e</span><span style="background-color: hsl(120, 100.00%, 95.67%); opacity: 0.81" title="0.010">y</span><span style="background-color: hsl(0, 100.00%, 83.82%); opacity: 0.85" title="-0.068"> </span><span style="background-color: hsl(0, 100.00%, 82.24%); opacity: 0.86" title="-0.078">p</span><span style="background-color: hsl(120, 100.00%, 91.38%); opacity: 0.82" title="0.028">as</span><span style="background-color: hsl(0, 100.00%, 93.75%); opacity: 0.81" title="-0.018">s</span><span style="opacity: 0.80">,</span><span style="background-color: hsl(0, 100.00%, 97.08%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 94.85%); opacity: 0.81" title="-0.013">o</span><span style="background-color: hsl(0, 100.00%, 90.49%); opacity: 0.83" title="-0.032">r</span><span style="background-color: hsl(0, 100.00%, 94.25%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(120, 100.00%, 93.55%); opacity: 0.81" title="0.018">t</span><span style="background-color: hsl(0, 100.00%, 83.41%); opacity: 0.86" title="-0.071">h</span><span style="background-color: hsl(0, 100.00%, 85.26%); opacity: 0.85" title="-0.060">e</span><span style="background-color: hsl(0, 100.00%, 97.02%); opacity: 0.80" title="-0.006">y</span><span style="background-color: hsl(0, 100.00%, 91.77%); opacity: 0.82" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 89.91%); opacity: 0.83" title="-0.035">h</span><span style="background-color: hsl(0, 100.00%, 94.19%); opacity: 0.81" title="-0.016">a</span><span style="background-color: hsl(0, 100.00%, 95.68%); opacity: 0.81" title="-0.010">ve</span><span style="background-color: hsl(120, 100.00%, 90.46%); opacity: 0.83" title="0.032"> </span><span style="background-color: hsl(120, 100.00%, 87.23%); opacity: 0.84" title="0.049">t</span><span style="background-color: hsl(120, 100.00%, 88.54%); opacity: 0.83" title="0.042">o</span><span style="background-color: hsl(120, 100.00%, 88.04%); opacity: 0.84" title="0.044"> </span><span style="background-color: hsl(120, 100.00%, 90.55%); opacity: 0.83" title="0.032">b</span><span style="background-color: hsl(0, 100.00%, 97.58%); opacity: 0.80" title="-0.005">e</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.002"> b</span><span style="background-color: hsl(0, 100.00%, 92.42%); opacity: 0.82" title="-0.023">rok</span><span style="background-color: hsl(120, 100.00%, 97.82%); opacity: 0.80" title="0.004">en u</span><span style="opacity: 0.80">p</span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 94.69%); opacity: 0.81" title="0.014">w</span><span style="background-color: hsl(120, 100.00%, 87.44%); opacity: 0.84" title="0.047">i</span><span style="background-color: hsl(120, 100.00%, 73.80%); opacity: 0.91" title="0.136">t</span><span style="background-color: hsl(0, 100.00%, 98.42%); opacity: 0.80" title="-0.002">h</span><span style="background-color: hsl(0, 100.00%, 98.99%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 92.07%); opacity: 0.82" title="0.025">o</span><span style="background-color: hsl(0, 100.00%, 98.15%); opacity: 0.80" title="-0.003">un</span><span style="background-color: hsl(0, 100.00%, 97.60%); opacity: 0.80" title="-0.004">d</span><span style="background-color: hsl(0, 100.00%, 94.11%); opacity: 0.81" title="-0.016">,</span><span style="background-color: hsl(0, 100.00%, 94.91%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 94.85%); opacity: 0.81" title="-0.013">o</span><span style="background-color: hsl(0, 100.00%, 90.49%); opacity: 0.83" title="-0.032">r</span><span style="background-color: hsl(0, 100.00%, 94.25%); opacity: 0.81" title="-0.016"> </span><span style="background-color: hsl(120, 100.00%, 93.55%); opacity: 0.81" title="0.018">t</span><span style="background-color: hsl(0, 100.00%, 83.41%); opacity: 0.86" title="-0.071">h</span><span style="background-color: hsl(0, 100.00%, 85.26%); opacity: 0.85" title="-0.060">e</span><span style="background-color: hsl(0, 100.00%, 97.02%); opacity: 0.80" title="-0.006">y</span><span style="background-color: hsl(0, 100.00%, 91.77%); opacity: 0.82" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 89.91%); opacity: 0.83" title="-0.035">h</span><span style="background-color: hsl(0, 100.00%, 88.43%); opacity: 0.83" title="-0.042">a</span><span style="background-color: hsl(0, 100.00%, 86.80%); opacity: 0.84" title="-0.051">ve</span><span style="background-color: hsl(0, 100.00%, 89.71%); opacity: 0.83" title="-0.036">
    </span><span style="background-color: hsl(0, 100.00%, 93.34%); opacity: 0.82" title="-0.019">t</span><span style="background-color: hsl(120, 100.00%, 92.78%); opacity: 0.82" title="0.022">o</span><span style="background-color: hsl(120, 100.00%, 90.17%); opacity: 0.83" title="0.033"> </span><span style="background-color: hsl(0, 100.00%, 94.75%); opacity: 0.81" title="-0.014">b</span><span style="background-color: hsl(0, 100.00%, 90.92%); opacity: 0.82" title="-0.030">e</span><span style="background-color: hsl(0, 100.00%, 89.01%); opacity: 0.83" title="-0.039"> </span><span style="background-color: hsl(0, 100.00%, 91.20%); opacity: 0.82" title="-0.029">e</span><span style="background-color: hsl(120, 100.00%, 92.65%); opacity: 0.82" title="0.022">x</span><span style="background-color: hsl(120, 100.00%, 81.82%); opacity: 0.86" title="0.080">t</span><span style="background-color: hsl(120, 100.00%, 84.01%); opacity: 0.85" title="0.067">r</span><span style="background-color: hsl(120, 100.00%, 96.19%); opacity: 0.81" title="0.009">a</span><span style="background-color: hsl(120, 100.00%, 86.43%); opacity: 0.84" title="0.053">c</span><span style="background-color: hsl(120, 100.00%, 85.00%); opacity: 0.85" title="0.061">t</span><span style="background-color: hsl(120, 100.00%, 85.27%); opacity: 0.85" title="0.060">e</span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.016">d</span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 94.49%); opacity: 0.81" title="0.015">s</span><span style="background-color: hsl(120, 100.00%, 89.86%); opacity: 0.83" title="0.035">u</span><span style="background-color: hsl(0, 100.00%, 99.06%); opacity: 0.80" title="-0.001">r</span><span style="background-color: hsl(0, 100.00%, 95.07%); opacity: 0.81" title="-0.012">g</span><span style="background-color: hsl(120, 100.00%, 95.35%); opacity: 0.81" title="0.011">i</span><span style="background-color: hsl(120, 100.00%, 89.20%); opacity: 0.83" title="0.038">c</span><span style="background-color: hsl(120, 100.00%, 82.99%); opacity: 0.86" title="0.073">a</span><span style="background-color: hsl(120, 100.00%, 88.66%); opacity: 0.83" title="0.041">l</span><span style="background-color: hsl(120, 100.00%, 95.77%); opacity: 0.81" title="0.010">l</span><span style="background-color: hsl(0, 100.00%, 96.90%); opacity: 0.81" title="-0.006">y</span><span style="background-color: hsl(0, 100.00%, 85.59%); opacity: 0.85" title="-0.058">.</span><span style="background-color: hsl(0, 100.00%, 88.88%); opacity: 0.83" title="-0.040"> </span><span style="background-color: hsl(0, 100.00%, 91.85%); opacity: 0.82" title="-0.026">w</span><span style="background-color: hsl(0, 100.00%, 80.52%); opacity: 0.87" title="-0.089">h</span><span style="background-color: hsl(0, 100.00%, 84.63%); opacity: 0.85" title="-0.063">e</span><span style="background-color: hsl(120, 100.00%, 86.46%); opacity: 0.84" title="0.053">n</span><span style="background-color: hsl(120, 100.00%, 95.16%); opacity: 0.81" title="0.012"> i</span><span style="background-color: hsl(120, 100.00%, 89.17%); opacity: 0.83" title="0.038"> </span><span style="background-color: hsl(120, 100.00%, 92.90%); opacity: 0.82" title="0.021">w</span><span style="background-color: hsl(120, 100.00%, 82.17%); opacity: 0.86" title="0.078">as</span><span style="background-color: hsl(120, 100.00%, 93.46%); opacity: 0.82" title="0.019"> </span><span style="background-color: hsl(120, 100.00%, 81.97%); opacity: 0.86" title="0.079">i</span><span style="background-color: hsl(120, 100.00%, 81.04%); opacity: 0.87" title="0.085">n</span><span style="background-color: hsl(120, 100.00%, 95.46%); opacity: 0.81" title="0.011">,</span><span style="background-color: hsl(120, 100.00%, 93.26%); opacity: 0.82" title="0.020"> </span><span style="background-color: hsl(120, 100.00%, 93.08%); opacity: 0.82" title="0.020">t</span><span style="background-color: hsl(0, 100.00%, 82.10%); opacity: 0.86" title="-0.079">h</span><span style="background-color: hsl(0, 100.00%, 83.16%); opacity: 0.86" title="-0.072">e</span><span style="background-color: hsl(120, 100.00%, 91.90%); opacity: 0.82" title="0.025"> </span><span style="background-color: hsl(120, 100.00%, 80.86%); opacity: 0.87" title="0.087">x-</span><span style="background-color: hsl(0, 100.00%, 96.67%); opacity: 0.81" title="-0.007">ra</span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.003">y</span><span style="background-color: hsl(120, 100.00%, 93.69%); opacity: 0.81" title="0.018"> t</span><span style="background-color: hsl(120, 100.00%, 95.89%); opacity: 0.81" title="0.010">ec</span><span style="background-color: hsl(0, 100.00%, 85.39%); opacity: 0.85" title="-0.059">h</span><span style="background-color: hsl(0, 100.00%, 90.53%); opacity: 0.83" title="-0.032"> </span><span style="background-color: hsl(120, 100.00%, 91.09%); opacity: 0.82" title="0.029">h</span><span style="background-color: hsl(120, 100.00%, 89.33%); opacity: 0.83" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 86.78%); opacity: 0.84" title="0.051">p</span><span style="background-color: hsl(120, 100.00%, 88.11%); opacity: 0.84" title="0.044">p</span><span style="background-color: hsl(120, 100.00%, 92.27%); opacity: 0.82" title="0.024">e</span><span style="background-color: hsl(120, 100.00%, 93.40%); opacity: 0.82" title="0.019">n</span><span style="background-color: hsl(0, 100.00%, 95.89%); opacity: 0.81" title="-0.010">ed</span><span style="background-color: hsl(0, 100.00%, 99.61%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.47%); opacity: 0.80" title="0.002">t</span><span style="background-color: hsl(120, 100.00%, 95.69%); opacity: 0.81" title="0.010">o</span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 94.45%); opacity: 0.81" title="-0.015">m</span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 96.95%); opacity: 0.81" title="0.006">n</span><span style="background-color: hsl(120, 100.00%, 93.20%); opacity: 0.82" title="0.020">t</span><span style="background-color: hsl(0, 100.00%, 81.50%); opacity: 0.87" title="-0.082">i</span><span style="background-color: hsl(0, 100.00%, 78.15%); opacity: 0.88" title="-0.105">o</span><span style="background-color: hsl(0, 100.00%, 78.88%); opacity: 0.88" title="-0.100">n</span><span style="background-color: hsl(0, 100.00%, 90.80%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 95.89%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 91.76%); opacity: 0.82" title="-0.026">h</span><span style="background-color: hsl(0, 100.00%, 87.62%); opacity: 0.84" title="-0.046">a</span><span style="background-color: hsl(0, 100.00%, 93.02%); opacity: 0.82" title="-0.020">t</span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 90.75%); opacity: 0.82" title="0.031">s</span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.014">h</span><span style="background-color: hsl(0, 100.00%, 96.82%); opacity: 0.81" title="-0.007">e</span><span style="background-color: hsl(0, 100.00%, 95.30%); opacity: 0.81" title="-0.012">'</span><span style="background-color: hsl(0, 100.00%, 94.00%); opacity: 0.81" title="-0.016">d</span><span style="background-color: hsl(120, 100.00%, 88.86%); opacity: 0.83" title="0.040"> </span><span style="background-color: hsl(120, 100.00%, 77.81%); opacity: 0.89" title="0.107">h</span><span style="background-color: hsl(120, 100.00%, 80.44%); opacity: 0.87" title="0.089">a</span><span style="background-color: hsl(120, 100.00%, 85.07%); opacity: 0.85" title="0.061">d</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(0, 100.00%, 88.17%); opacity: 0.84" title="-0.044">k</span><span style="background-color: hsl(0, 100.00%, 92.90%); opacity: 0.82" title="-0.021">id</span><span style="background-color: hsl(120, 100.00%, 91.97%); opacity: 0.82" title="0.025">n</span><span style="background-color: hsl(120, 100.00%, 97.39%); opacity: 0.80" title="0.005">e</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.023">
    </span><span style="background-color: hsl(0, 100.00%, 87.86%); opacity: 0.84" title="-0.045">s</span><span style="background-color: hsl(0, 100.00%, 88.90%); opacity: 0.83" title="-0.040">t</span><span style="background-color: hsl(0, 100.00%, 92.73%); opacity: 0.82" title="-0.022">o</span><span style="background-color: hsl(0, 100.00%, 86.99%); opacity: 0.84" title="-0.050">n</span><span style="background-color: hsl(120, 100.00%, 94.32%); opacity: 0.81" title="0.015">e</span><span style="background-color: hsl(120, 100.00%, 89.69%); opacity: 0.83" title="0.036">s</span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.029"> </span><span style="background-color: hsl(120, 100.00%, 87.80%); opacity: 0.84" title="0.045">a</span><span style="background-color: hsl(120, 100.00%, 93.94%); opacity: 0.81" title="0.017">n</span><span style="background-color: hsl(0, 100.00%, 92.59%); opacity: 0.82" title="-0.022">d</span><span style="background-color: hsl(0, 100.00%, 95.82%); opacity: 0.81" title="-0.010"> c</span><span style="background-color: hsl(120, 100.00%, 88.58%); opacity: 0.83" title="0.041">h</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(0, 100.00%, 91.67%); opacity: 0.82" title="-0.026">l</span><span style="background-color: hsl(0, 100.00%, 89.61%); opacity: 0.83" title="-0.036">d</span><span style="background-color: hsl(0, 100.00%, 92.89%); opacity: 0.82" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 91.08%); opacity: 0.82" title="-0.029">e</span><span style="background-color: hsl(120, 100.00%, 95.49%); opacity: 0.81" title="0.011">n</span><span style="background-color: hsl(120, 100.00%, 97.15%); opacity: 0.80" title="0.006">,</span><span style="background-color: hsl(120, 100.00%, 95.21%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 86.99%); opacity: 0.84" title="0.050">a</span><span style="background-color: hsl(120, 100.00%, 90.32%); opacity: 0.83" title="0.033">n</span><span style="opacity: 0.80">d</span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.008"> t</span><span style="background-color: hsl(0, 100.00%, 83.16%); opacity: 0.86" title="-0.072">he</span><span style="background-color: hsl(120, 100.00%, 97.92%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 95.07%); opacity: 0.81" title="0.012">c</span><span style="background-color: hsl(120, 100.00%, 88.36%); opacity: 0.83" title="0.043">h</span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.008">ild</span><span style="background-color: hsl(120, 100.00%, 95.94%); opacity: 0.81" title="0.009">bi</span><span style="background-color: hsl(120, 100.00%, 98.61%); opacity: 0.80" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 84.15%); opacity: 0.85" title="0.066">t</span><span style="background-color: hsl(0, 100.00%, 97.45%); opacity: 0.80" title="-0.005">h</span><span style="background-color: hsl(0, 100.00%, 97.92%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 90.11%); opacity: 0.83" title="-0.034">h</span><span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.006">u</span><span style="background-color: hsl(120, 100.00%, 96.07%); opacity: 0.81" title="0.009">r</span><span style="background-color: hsl(0, 100.00%, 95.97%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(120, 100.00%, 95.71%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 82.56%); opacity: 0.86" title="0.076">l</span><span style="background-color: hsl(120, 100.00%, 77.32%); opacity: 0.89" title="0.110">e</span><span style="background-color: hsl(120, 100.00%, 89.94%); opacity: 0.83" title="0.035">s</span><span style="background-color: hsl(0, 100.00%, 96.35%); opacity: 0.81" title="-0.008">s.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Hm, the result look worse. :class:`~.TextExplainer` detected correctly that
only the first part of word "medication" is important, but the result is
noisy overall, and scores are bad. Let's try it with more samples:

.. code:: ipython3

    te = TextExplainer(char_based=True, n_samples=50000, random_state=42)
    te.fit(doc, pipe_char.predict_proba)
    print(te.metrics_)
    te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)


.. parsed-literal::

    {'score': 0.85575209964207921, 'mean_KL_divergence': 0.071035516578501337}




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
    
        
        (probability <b>0.648</b>, score <b>0.749</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
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
            +0.962
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 93.05%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.213
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">a</span><span style="background-color: hsl(0, 100.00%, 92.45%); opacity: 0.82" title="-0.018">s</span><span style="background-color: hsl(0, 100.00%, 87.22%); opacity: 0.84" title="-0.038"> </span><span style="background-color: hsl(0, 100.00%, 91.09%); opacity: 0.82" title="-0.023">i</span><span style="background-color: hsl(0, 100.00%, 92.03%); opacity: 0.82" title="-0.019"> </span><span style="background-color: hsl(120, 100.00%, 92.56%); opacity: 0.82" title="0.018">r</span><span style="background-color: hsl(120, 100.00%, 88.23%); opacity: 0.83" title="0.034">e</span><span style="background-color: hsl(120, 100.00%, 82.08%); opacity: 0.86" title="0.062">c</span><span style="background-color: hsl(120, 100.00%, 87.57%); opacity: 0.84" title="0.037">a</span><span style="background-color: hsl(120, 100.00%, 90.43%); opacity: 0.83" title="0.025">l</span><span style="background-color: hsl(0, 100.00%, 91.50%); opacity: 0.82" title="-0.021">l</span><span style="background-color: hsl(0, 100.00%, 90.96%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.012">f</span><span style="background-color: hsl(0, 100.00%, 97.14%); opacity: 0.80" title="-0.005">r</span><span style="background-color: hsl(0, 100.00%, 91.94%); opacity: 0.82" title="-0.020">o</span><span style="background-color: hsl(0, 100.00%, 92.95%); opacity: 0.82" title="-0.016">m</span><span style="background-color: hsl(120, 100.00%, 94.46%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 84.03%); opacity: 0.85" title="0.053">m</span><span style="background-color: hsl(120, 100.00%, 85.34%); opacity: 0.85" title="0.047">y</span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 93.90%); opacity: 0.81" title="0.013">b</span><span style="background-color: hsl(120, 100.00%, 85.98%); opacity: 0.84" title="0.044">o</span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.007">u</span><span style="background-color: hsl(0, 100.00%, 92.76%); opacity: 0.82" title="-0.017">t</span><span style="background-color: hsl(0, 100.00%, 92.70%); opacity: 0.82" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 95.46%); opacity: 0.81" title="-0.009">w</span><span style="background-color: hsl(0, 100.00%, 95.51%); opacity: 0.81" title="-0.009">i</span><span style="background-color: hsl(0, 100.00%, 97.24%); opacity: 0.80" title="-0.004">t</span><span style="background-color: hsl(0, 100.00%, 97.77%); opacity: 0.80" title="-0.003">h</span><span style="background-color: hsl(0, 100.00%, 95.74%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 95.90%); opacity: 0.81" title="-0.008">k</span><span style="background-color: hsl(120, 100.00%, 96.14%); opacity: 0.81" title="0.007">i</span><span style="background-color: hsl(120, 100.00%, 89.35%); opacity: 0.83" title="0.029">d</span><span style="background-color: hsl(120, 100.00%, 90.60%); opacity: 0.83" title="0.025">n</span><span style="background-color: hsl(120, 100.00%, 89.89%); opacity: 0.83" title="0.027">e</span><span style="background-color: hsl(120, 100.00%, 96.00%); opacity: 0.81" title="0.007">y</span><span style="background-color: hsl(0, 100.00%, 94.38%); opacity: 0.81" title="-0.012"> </span><span style="opacity: 0.80">s</span><span style="background-color: hsl(0, 100.00%, 95.40%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(0, 100.00%, 99.68%); opacity: 0.80" title="-0.000">o</span><span style="background-color: hsl(0, 100.00%, 99.13%); opacity: 0.80" title="-0.001">n</span><span style="background-color: hsl(120, 100.00%, 94.56%); opacity: 0.81" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 91.83%); opacity: 0.82" title="0.020">s</span><span style="background-color: hsl(0, 100.00%, 94.58%); opacity: 0.81" title="-0.011">,</span><span style="background-color: hsl(0, 100.00%, 96.70%); opacity: 0.81" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 98.02%); opacity: 0.80" title="-0.003">t</span><span style="background-color: hsl(0, 100.00%, 94.85%); opacity: 0.81" title="-0.010">h</span><span style="background-color: hsl(0, 100.00%, 89.07%); opacity: 0.83" title="-0.031">e</span><span style="background-color: hsl(0, 100.00%, 90.37%); opacity: 0.83" title="-0.026">r</span><span style="background-color: hsl(0, 100.00%, 90.82%); opacity: 0.82" title="-0.024">e</span><span style="background-color: hsl(0, 100.00%, 90.84%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 92.95%); opacity: 0.82" title="0.016">s</span><span style="background-color: hsl(120, 100.00%, 93.01%); opacity: 0.82" title="0.016">n</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 97.23%); opacity: 0.80" title="-0.004">t</span><span style="background-color: hsl(120, 100.00%, 93.90%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 91.81%); opacity: 0.82" title="0.020">a</span><span style="background-color: hsl(0, 100.00%, 98.13%); opacity: 0.80" title="-0.002">n</span><span style="background-color: hsl(0, 100.00%, 91.49%); opacity: 0.82" title="-0.021">y</span><span style="background-color: hsl(120, 100.00%, 90.64%); opacity: 0.83" title="0.025">
    </span><span style="background-color: hsl(120, 100.00%, 74.15%); opacity: 0.91" title="0.105">m</span><span style="background-color: hsl(120, 100.00%, 66.14%); opacity: 0.96" title="0.154">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.195">d</span><span style="background-color: hsl(120, 100.00%, 64.62%); opacity: 0.97" title="0.164">i</span><span style="background-color: hsl(120, 100.00%, 77.12%); opacity: 0.89" title="0.088">c</span><span style="background-color: hsl(120, 100.00%, 86.54%); opacity: 0.84" title="0.041">a</span><span style="background-color: hsl(120, 100.00%, 90.58%); opacity: 0.83" title="0.025">t</span><span style="background-color: hsl(0, 100.00%, 93.94%); opacity: 0.81" title="-0.013">i</span><span style="background-color: hsl(0, 100.00%, 90.20%); opacity: 0.83" title="-0.026">o</span><span style="background-color: hsl(0, 100.00%, 86.23%); opacity: 0.84" title="-0.043">n</span><span style="background-color: hsl(0, 100.00%, 89.58%); opacity: 0.83" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 99.78%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(120, 100.00%, 97.63%); opacity: 0.80" title="0.003">h</span><span style="background-color: hsl(120, 100.00%, 96.34%); opacity: 0.81" title="0.006">at</span><span style="background-color: hsl(0, 100.00%, 91.86%); opacity: 0.82" title="-0.020"> </span><span style="background-color: hsl(120, 100.00%, 96.57%); opacity: 0.81" title="0.006">c</span><span style="background-color: hsl(120, 100.00%, 87.53%); opacity: 0.84" title="0.037">a</span><span style="background-color: hsl(120, 100.00%, 96.65%); opacity: 0.81" title="0.006">n</span><span style="background-color: hsl(0, 100.00%, 89.06%); opacity: 0.83" title="-0.031"> </span><span style="background-color: hsl(0, 100.00%, 93.58%); opacity: 0.81" title="-0.014">d</span><span style="background-color: hsl(0, 100.00%, 92.86%); opacity: 0.82" title="-0.017">o</span><span style="background-color: hsl(0, 100.00%, 97.68%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 94.36%); opacity: 0.81" title="0.012">a</span><span style="background-color: hsl(0, 100.00%, 92.10%); opacity: 0.82" title="-0.019">n</span><span style="background-color: hsl(0, 100.00%, 86.93%); opacity: 0.84" title="-0.039">y</span><span style="background-color: hsl(0, 100.00%, 90.91%); opacity: 0.82" title="-0.023">t</span><span style="background-color: hsl(0, 100.00%, 90.33%); opacity: 0.83" title="-0.026">h</span><span style="background-color: hsl(120, 100.00%, 97.54%); opacity: 0.80" title="0.004">i</span><span style="background-color: hsl(120, 100.00%, 95.46%); opacity: 0.81" title="0.009">n</span><span style="background-color: hsl(0, 100.00%, 86.02%); opacity: 0.84" title="-0.043">g</span><span style="background-color: hsl(0, 100.00%, 95.41%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(120, 100.00%, 89.78%); opacity: 0.83" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 90.45%); opacity: 0.83" title="0.025">b</span><span style="background-color: hsl(120, 100.00%, 86.55%); opacity: 0.84" title="0.041">o</span><span style="background-color: hsl(120, 100.00%, 93.09%); opacity: 0.82" title="0.016">u</span><span style="background-color: hsl(0, 100.00%, 95.77%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(0, 100.00%, 98.09%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(0, 100.00%, 94.81%); opacity: 0.81" title="-0.011">t</span><span style="background-color: hsl(0, 100.00%, 98.81%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 93.90%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 95.25%); opacity: 0.81" title="-0.009">m</span><span style="background-color: hsl(0, 100.00%, 96.72%); opacity: 0.81" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(0, 100.00%, 99.66%); opacity: 0.80" title="-0.000">x</span><span style="background-color: hsl(120, 100.00%, 91.92%); opacity: 0.82" title="0.020">c</span><span style="background-color: hsl(120, 100.00%, 91.42%); opacity: 0.82" title="0.022">e</span><span style="background-color: hsl(0, 100.00%, 97.28%); opacity: 0.80" title="-0.004">p</span><span style="background-color: hsl(0, 100.00%, 95.52%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(0, 100.00%, 95.37%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 92.39%); opacity: 0.82" title="-0.018">r</span><span style="background-color: hsl(0, 100.00%, 89.90%); opacity: 0.83" title="-0.027">e</span><span style="background-color: hsl(0, 100.00%, 93.04%); opacity: 0.82" title="-0.016">l</span><span style="background-color: hsl(0, 100.00%, 96.28%); opacity: 0.81" title="-0.007">i</span><span style="background-color: hsl(0, 100.00%, 91.67%); opacity: 0.82" title="-0.021">e</span><span style="background-color: hsl(120, 100.00%, 95.37%); opacity: 0.81" title="0.009">v</span><span style="background-color: hsl(120, 100.00%, 97.42%); opacity: 0.80" title="0.004">e</span><span style="background-color: hsl(0, 100.00%, 90.67%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 91.44%); opacity: 0.82" title="-0.022">t</span><span style="background-color: hsl(0, 100.00%, 94.28%); opacity: 0.81" title="-0.012">h</span><span style="background-color: hsl(0, 100.00%, 88.32%); opacity: 0.83" title="-0.034">e</span><span style="background-color: hsl(0, 100.00%, 95.60%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(120, 100.00%, 85.43%); opacity: 0.85" title="0.046">p</span><span style="background-color: hsl(120, 100.00%, 85.99%); opacity: 0.84" title="0.044">a</span><span style="background-color: hsl(120, 100.00%, 79.92%); opacity: 0.87" title="0.073">i</span><span style="background-color: hsl(120, 100.00%, 83.19%); opacity: 0.86" title="0.057">n</span><span style="background-color: hsl(0, 100.00%, 91.07%); opacity: 0.82" title="-0.023">.</span><span style="background-color: hsl(0, 100.00%, 90.79%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 94.35%); opacity: 0.81" title="-0.012">e</span><span style="background-color: hsl(0, 100.00%, 87.12%); opacity: 0.84" title="-0.039">i</span><span style="background-color: hsl(0, 100.00%, 82.95%); opacity: 0.86" title="-0.058">t</span><span style="background-color: hsl(0, 100.00%, 81.96%); opacity: 0.86" title="-0.063">h</span><span style="background-color: hsl(0, 100.00%, 86.69%); opacity: 0.84" title="-0.041">e</span><span style="background-color: hsl(0, 100.00%, 89.99%); opacity: 0.83" title="-0.027">r</span><span style="background-color: hsl(0, 100.00%, 96.03%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 95.77%); opacity: 0.81" title="0.008">h</span><span style="background-color: hsl(0, 100.00%, 99.19%); opacity: 0.80" title="-0.001">e</span><span style="background-color: hsl(0, 100.00%, 95.91%); opacity: 0.81" title="-0.008">y</span><span style="background-color: hsl(0, 100.00%, 91.16%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 93.85%); opacity: 0.81" title="-0.013">p</span><span style="background-color: hsl(0, 100.00%, 92.89%); opacity: 0.82" title="-0.017">a</span><span style="background-color: hsl(0, 100.00%, 94.94%); opacity: 0.81" title="-0.010">s</span><span style="background-color: hsl(0, 100.00%, 95.48%); opacity: 0.81" title="-0.009">s</span><span style="background-color: hsl(0, 100.00%, 95.30%); opacity: 0.81" title="-0.009">,</span><span style="background-color: hsl(0, 100.00%, 90.40%); opacity: 0.83" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 93.42%); opacity: 0.82" title="-0.015">o</span><span style="background-color: hsl(0, 100.00%, 98.68%); opacity: 0.80" title="-0.001">r</span><span style="background-color: hsl(120, 100.00%, 97.44%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 95.77%); opacity: 0.81" title="0.008">h</span><span style="background-color: hsl(120, 100.00%, 95.76%); opacity: 0.81" title="0.008">e</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(0, 100.00%, 92.56%); opacity: 0.82" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 93.96%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 90.40%); opacity: 0.83" title="-0.025">a</span><span style="background-color: hsl(0, 100.00%, 93.84%); opacity: 0.81" title="-0.013">v</span><span style="background-color: hsl(0, 100.00%, 90.75%); opacity: 0.82" title="-0.024">e</span><span style="background-color: hsl(0, 100.00%, 90.86%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 94.91%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 91.78%); opacity: 0.82" title="-0.020">o</span><span style="background-color: hsl(0, 100.00%, 90.52%); opacity: 0.83" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 95.64%); opacity: 0.81" title="-0.008">b</span><span style="background-color: hsl(0, 100.00%, 96.35%); opacity: 0.81" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 88.84%); opacity: 0.83" title="-0.031"> </span><span style="background-color: hsl(0, 100.00%, 91.78%); opacity: 0.82" title="-0.020">b</span><span style="background-color: hsl(0, 100.00%, 98.07%); opacity: 0.80" title="-0.003">ro</span><span style="opacity: 0.80">k</span><span style="background-color: hsl(120, 100.00%, 98.04%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(0, 100.00%, 92.05%); opacity: 0.82" title="-0.019">n</span><span style="background-color: hsl(0, 100.00%, 93.31%); opacity: 0.82" title="-0.015"> </span><span style="background-color: hsl(120, 100.00%, 91.91%); opacity: 0.82" title="0.020">u</span><span style="background-color: hsl(120, 100.00%, 89.69%); opacity: 0.83" title="0.028">p</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 95.63%); opacity: 0.81" title="0.008">w</span><span style="background-color: hsl(0, 100.00%, 97.21%); opacity: 0.80" title="-0.004">i</span><span style="background-color: hsl(120, 100.00%, 96.92%); opacity: 0.81" title="0.005">t</span><span style="background-color: hsl(120, 100.00%, 98.52%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 95.05%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 91.24%); opacity: 0.82" title="0.022">s</span><span style="background-color: hsl(120, 100.00%, 85.29%); opacity: 0.85" title="0.047">o</span><span style="background-color: hsl(120, 100.00%, 86.68%); opacity: 0.84" title="0.041">u</span><span style="background-color: hsl(120, 100.00%, 91.63%); opacity: 0.82" title="0.021">nd</span><span style="background-color: hsl(0, 100.00%, 93.24%); opacity: 0.82" title="-0.015">,</span><span style="background-color: hsl(0, 100.00%, 89.68%); opacity: 0.83" title="-0.028"> </span><span style="background-color: hsl(0, 100.00%, 93.42%); opacity: 0.82" title="-0.015">o</span><span style="background-color: hsl(0, 100.00%, 98.68%); opacity: 0.80" title="-0.001">r</span><span style="background-color: hsl(120, 100.00%, 97.44%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 95.77%); opacity: 0.81" title="0.008">h</span><span style="background-color: hsl(120, 100.00%, 95.76%); opacity: 0.81" title="0.008">e</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(0, 100.00%, 92.56%); opacity: 0.82" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 93.96%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 90.06%); opacity: 0.83" title="-0.027">a</span><span style="background-color: hsl(0, 100.00%, 96.00%); opacity: 0.81" title="-0.007">v</span><span style="background-color: hsl(0, 100.00%, 96.81%); opacity: 0.81" title="-0.005">e</span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.001">
    </span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002">t</span><span style="background-color: hsl(0, 100.00%, 94.89%); opacity: 0.81" title="-0.010">o</span><span style="background-color: hsl(0, 100.00%, 91.61%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(0, 100.00%, 96.96%); opacity: 0.81" title="-0.005">b</span><span style="background-color: hsl(0, 100.00%, 97.82%); opacity: 0.80" title="-0.003">e</span><span style="background-color: hsl(0, 100.00%, 94.96%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(120, 100.00%, 96.66%); opacity: 0.81" title="0.006">ex</span><span style="background-color: hsl(120, 100.00%, 93.12%); opacity: 0.82" title="0.016">t</span><span style="background-color: hsl(120, 100.00%, 88.80%); opacity: 0.83" title="0.032">r</span><span style="background-color: hsl(120, 100.00%, 93.10%); opacity: 0.82" title="0.016">a</span><span style="background-color: hsl(120, 100.00%, 92.77%); opacity: 0.82" title="0.017">c</span><span style="background-color: hsl(120, 100.00%, 88.69%); opacity: 0.83" title="0.032">t</span><span style="background-color: hsl(120, 100.00%, 91.72%); opacity: 0.82" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 93.30%); opacity: 0.82" title="0.015">d</span><span style="background-color: hsl(120, 100.00%, 92.79%); opacity: 0.82" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 88.67%); opacity: 0.83" title="0.032">su</span><span style="background-color: hsl(120, 100.00%, 87.22%); opacity: 0.84" title="0.038">r</span><span style="background-color: hsl(120, 100.00%, 93.66%); opacity: 0.81" title="0.014">g</span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.017">i</span><span style="background-color: hsl(120, 100.00%, 88.94%); opacity: 0.83" title="0.031">c</span><span style="background-color: hsl(120, 100.00%, 86.54%); opacity: 0.84" title="0.041">a</span><span style="background-color: hsl(120, 100.00%, 87.49%); opacity: 0.84" title="0.037">l</span><span style="background-color: hsl(120, 100.00%, 90.13%); opacity: 0.83" title="0.026">ly</span><span style="background-color: hsl(0, 100.00%, 88.90%); opacity: 0.83" title="-0.031">.</span><span style="background-color: hsl(0, 100.00%, 99.84%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 82.48%); opacity: 0.86" title="0.060">w</span><span style="background-color: hsl(120, 100.00%, 81.09%); opacity: 0.87" title="0.067">h</span><span style="background-color: hsl(120, 100.00%, 85.14%); opacity: 0.85" title="0.047">e</span><span style="background-color: hsl(120, 100.00%, 95.26%); opacity: 0.81" title="0.009">n</span><span style="background-color: hsl(0, 100.00%, 91.80%); opacity: 0.82" title="-0.020"> </span><span style="background-color: hsl(0, 100.00%, 92.21%); opacity: 0.82" title="-0.019">i</span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(120, 100.00%, 96.56%); opacity: 0.81" title="0.006">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(0, 100.00%, 91.55%); opacity: 0.82" title="-0.021">s </span><span style="background-color: hsl(120, 100.00%, 91.26%); opacity: 0.82" title="0.022">in</span><span style="background-color: hsl(0, 100.00%, 92.54%); opacity: 0.82" title="-0.018">,</span><span style="background-color: hsl(0, 100.00%, 91.18%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 92.27%); opacity: 0.82" title="-0.019">t</span><span style="background-color: hsl(0, 100.00%, 94.28%); opacity: 0.81" title="-0.012">h</span><span style="background-color: hsl(0, 100.00%, 90.73%); opacity: 0.82" title="-0.024">e</span><span style="background-color: hsl(0, 100.00%, 91.55%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.005">x</span><span style="background-color: hsl(0, 100.00%, 97.04%); opacity: 0.80" title="-0.005">-</span><span style="background-color: hsl(120, 100.00%, 96.60%); opacity: 0.81" title="0.006">r</span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.005">a</span><span style="background-color: hsl(0, 100.00%, 91.63%); opacity: 0.82" title="-0.021">y</span><span style="background-color: hsl(0, 100.00%, 93.69%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 98.31%); opacity: 0.80" title="-0.002">t</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.018">c</span><span style="background-color: hsl(120, 100.00%, 97.01%); opacity: 0.80" title="0.005">h</span><span style="background-color: hsl(0, 100.00%, 94.27%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 96.86%); opacity: 0.81" title="-0.005">h</span><span style="background-color: hsl(120, 100.00%, 92.20%); opacity: 0.82" title="0.019">a</span><span style="background-color: hsl(120, 100.00%, 90.36%); opacity: 0.83" title="0.026">p</span><span style="background-color: hsl(120, 100.00%, 88.00%); opacity: 0.84" title="0.035">p</span><span style="background-color: hsl(120, 100.00%, 89.66%); opacity: 0.83" title="0.028">e</span><span style="background-color: hsl(0, 100.00%, 99.62%); opacity: 0.80" title="-0.000">n</span><span style="background-color: hsl(0, 100.00%, 97.65%); opacity: 0.80" title="-0.003">e</span><span style="background-color: hsl(0, 100.00%, 96.75%); opacity: 0.81" title="-0.005">d</span><span style="background-color: hsl(0, 100.00%, 95.46%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 95.75%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(0, 100.00%, 92.46%); opacity: 0.82" title="-0.018">o</span><span style="background-color: hsl(0, 100.00%, 94.50%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 93.15%); opacity: 0.82" title="0.016">m</span><span style="background-color: hsl(120, 100.00%, 88.28%); opacity: 0.83" title="0.034">e</span><span style="background-color: hsl(120, 100.00%, 91.14%); opacity: 0.82" title="0.023">n</span><span style="background-color: hsl(120, 100.00%, 86.11%); opacity: 0.84" title="0.043">t</span><span style="background-color: hsl(120, 100.00%, 96.07%); opacity: 0.81" title="0.007">i</span><span style="background-color: hsl(0, 100.00%, 94.30%); opacity: 0.81" title="-0.012">o</span><span style="background-color: hsl(0, 100.00%, 87.58%); opacity: 0.84" title="-0.037">n</span><span style="background-color: hsl(0, 100.00%, 89.58%); opacity: 0.83" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 99.78%); opacity: 0.80" title="-0.000">t</span><span style="background-color: hsl(120, 100.00%, 97.63%); opacity: 0.80" title="0.003">h</span><span style="background-color: hsl(120, 100.00%, 97.51%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(120, 100.00%, 95.87%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(120, 100.00%, 94.35%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 90.24%); opacity: 0.83" title="0.026">s</span><span style="background-color: hsl(120, 100.00%, 89.64%); opacity: 0.83" title="0.028">h</span><span style="background-color: hsl(120, 100.00%, 93.60%); opacity: 0.81" title="0.014">e</span><span style="background-color: hsl(0, 100.00%, 96.08%); opacity: 0.81" title="-0.007">'d</span><span style="background-color: hsl(120, 100.00%, 92.81%); opacity: 0.82" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 87.14%); opacity: 0.84" title="0.039">had</span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 96.34%); opacity: 0.81" title="-0.006">k</span><span style="background-color: hsl(120, 100.00%, 94.34%); opacity: 0.81" title="0.012">i</span><span style="background-color: hsl(120, 100.00%, 89.35%); opacity: 0.83" title="0.029">d</span><span style="background-color: hsl(120, 100.00%, 90.60%); opacity: 0.83" title="0.025">n</span><span style="background-color: hsl(120, 100.00%, 91.23%); opacity: 0.82" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 93.66%); opacity: 0.81" title="0.014">y</span><span style="background-color: hsl(120, 100.00%, 98.43%); opacity: 0.80" title="0.002">
    s</span><span style="background-color: hsl(0, 100.00%, 93.70%); opacity: 0.81" title="-0.014">t</span><span style="background-color: hsl(0, 100.00%, 99.68%); opacity: 0.80" title="-0.000">on</span><span style="background-color: hsl(120, 100.00%, 94.35%); opacity: 0.81" title="0.012">e</span><span style="background-color: hsl(120, 100.00%, 98.30%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 86.79%); opacity: 0.84" title="0.040">a</span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.024">n</span><span style="background-color: hsl(120, 100.00%, 95.27%); opacity: 0.81" title="0.009">d</span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 94.13%); opacity: 0.81" title="0.013">h</span><span style="background-color: hsl(0, 100.00%, 97.20%); opacity: 0.80" title="-0.004">i</span><span style="background-color: hsl(120, 100.00%, 92.97%); opacity: 0.82" title="0.016">l</span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.005">d</span><span style="background-color: hsl(0, 100.00%, 87.95%); opacity: 0.84" title="-0.035">r</span><span style="background-color: hsl(0, 100.00%, 85.22%); opacity: 0.85" title="-0.047">e</span><span style="background-color: hsl(0, 100.00%, 87.71%); opacity: 0.84" title="-0.036">n</span><span style="background-color: hsl(0, 100.00%, 86.84%); opacity: 0.84" title="-0.040">,</span><span style="background-color: hsl(0, 100.00%, 99.63%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 87.38%); opacity: 0.84" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.024">n</span><span style="background-color: hsl(120, 100.00%, 95.27%); opacity: 0.81" title="0.009">d</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(0, 100.00%, 92.27%); opacity: 0.82" title="-0.019">t</span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.016">h</span><span style="background-color: hsl(0, 100.00%, 87.32%); opacity: 0.84" title="-0.038">e</span><span style="background-color: hsl(0, 100.00%, 84.70%); opacity: 0.85" title="-0.049"> </span><span style="background-color: hsl(0, 100.00%, 97.41%); opacity: 0.80" title="-0.004">c</span><span style="background-color: hsl(120, 100.00%, 96.06%); opacity: 0.81" title="0.007">h</span><span style="background-color: hsl(0, 100.00%, 96.12%); opacity: 0.81" title="-0.007">i</span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.014">l</span><span style="background-color: hsl(120, 100.00%, 91.95%); opacity: 0.82" title="0.020">d</span><span style="background-color: hsl(120, 100.00%, 97.48%); opacity: 0.80" title="0.004">b</span><span style="background-color: hsl(120, 100.00%, 95.00%); opacity: 0.81" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 94.57%); opacity: 0.81" title="0.011">r</span><span style="background-color: hsl(120, 100.00%, 95.22%); opacity: 0.81" title="0.009">t</span><span style="background-color: hsl(120, 100.00%, 98.18%); opacity: 0.80" title="0.002">h </span><span style="opacity: 0.80">hurt</span><span style="background-color: hsl(0, 100.00%, 90.30%); opacity: 0.83" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 95.80%); opacity: 0.81" title="-0.008">l</span><span style="background-color: hsl(120, 100.00%, 87.42%); opacity: 0.84" title="0.037">e</span><span style="background-color: hsl(120, 100.00%, 94.50%); opacity: 0.81" title="0.011">s</span><span style="background-color: hsl(0, 100.00%, 99.21%); opacity: 0.80" title="-0.001">s.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




It is getting closer, but still not there yet. The problem is that it is
much more resource intensive - you need a lot more samples to get
non-noisy results. Here explaining a single example took more time than
training the original pipeline.

Generally speaking, to do an efficient explanation we should make some
assumptions about black-box classifier, such as:

1. it uses words as features and doesn't take word position in account;
2. it uses words as features and takes word positions in account;
3. it uses words ngrams as features;
4. it uses char ngrams as features, positions don't matter (i.e. an
   ngram means the same everywhere);
5. it uses arbitrary attention over the text characters, i.e. every part
   of text could be potentionally important for a classifier on its own;
6. it is important to have a particular token at a particular position,
   e.g. "third token is X", and if we delete 2nd token then prediction
   changes not because 2nd token changed, but because 3rd token is
   shifted.

Depending on assumptions we should choose both dataset generation method
and a white-box classifier. There is a tradeoff between generality and
speed.

Simple bag-of-words assumptions allow for fast sample generation, and
just a few hundreds of samples could be required to get an OK quality if
the assumption is correct. But such generation methods / models will
fail to explain a more complex classifier properly (they could still
provide an explanation which is useful in practice though).

On the other hand, allowing for each character to be important is a more
powerful method, but it can require a lot of samples (maybe hundreds
thousands) and a lot of CPU time to get non-noisy results.

What's bad about this kind of failure (wrong assumption about the
black-box pipeline) is that it could be impossible to detect the failure
by looking at the scores. Scores could be high because generated dataset
is not diverse enough, not because our approximation is good.

The takeaway is that it is important to understand the "lenses" you're
looking through when using LIME to explain a prediction.

Customizing TextExplainer: sampling
-----------------------------------

:class:`~.TextExplainer` uses :class:`~.MaskingTextSampler` or :class:`~.MaskingTextSamplers`
instances to generate texts to train on. :class:`~.MaskingTextSampler` is the
main text generation class; :class:`~.MaskingTextSamplers` provides a way to
combine multiple samplers in a single object with the same interface.

A custom sampler instance can be passed to :class:`~.TextExplainer` if we want
to experiment with sampling. For example, let's try a sampler which
replaces no more than 3 characters in the text (default is to replace a
random number of characters):

.. code:: ipython3

    from eli5.lime.samplers import MaskingTextSampler
    sampler = MaskingTextSampler(
        # Regex to split text into tokens.
        # "." means any single character is a token, i.e.
        # we work on chars.
        token_pattern='.',
    
        # replace no more than 3 tokens
        max_replace=3,
    
        # by default all tokens are replaced;
        # replace only a token at a given position.
        bow=False,
    )
    samples, similarity = sampler.sample_near(doc)
    print(samples[0])


.. parsed-literal::

    As I recall from my bout with kidney stones, there isn't any
    medication that can do anything about them except relieve the pain
    
    Either thy pass, or they have to be broken up with sound, or they have
    to be extracted surgically.
    
    When I was in, the X-ray tech happened to mention that she'd had kidney
    stones and children, and the childbirth hurt less.


.. code:: ipython3

    te = TextExplainer(char_based=True, sampler=sampler, random_state=42)
    te.fit(doc, pipe_char.predict_proba)
    print(te.metrics_)
    te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)


.. parsed-literal::

    {'score': 1.0, 'mean_KL_divergence': 1.0004596970275623}




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
    
        
        (probability <b>0.970</b>, score <b>4.522</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
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
            +4.512
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.72%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.010
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
            
    
            
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 94.40%); opacity: 0.81" title="0.057">as</span><span style="background-color: hsl(120, 100.00%, 83.37%); opacity: 0.86" title="0.271"> </span><span style="background-color: hsl(120, 100.00%, 69.42%); opacity: 0.94" title="0.647">i</span><span style="background-color: hsl(120, 100.00%, 72.43%); opacity: 0.92" title="0.558"> </span><span style="background-color: hsl(120, 100.00%, 79.82%); opacity: 0.88" title="0.357">r</span><span style="background-color: hsl(0, 100.00%, 89.00%); opacity: 0.83" title="-0.150">e</span><span style="background-color: hsl(0, 100.00%, 82.23%); opacity: 0.86" title="-0.298">c</span><span style="background-color: hsl(120, 100.00%, 73.17%); opacity: 0.91" title="0.536">a</span><span style="background-color: hsl(120, 100.00%, 69.67%); opacity: 0.93" title="0.639">l</span><span style="background-color: hsl(120, 100.00%, 74.88%); opacity: 0.90" title="0.488">l</span><span style="background-color: hsl(120, 100.00%, 88.32%); opacity: 0.83" title="0.163"> </span><span style="background-color: hsl(120, 100.00%, 95.57%); opacity: 0.81" title="0.041">f</span><span style="background-color: hsl(120, 100.00%, 90.65%); opacity: 0.83" title="0.119">r</span><span style="background-color: hsl(120, 100.00%, 84.94%); opacity: 0.85" title="0.235">o</span><span style="background-color: hsl(120, 100.00%, 82.92%); opacity: 0.86" title="0.282">m</span><span style="background-color: hsl(120, 100.00%, 77.96%); opacity: 0.89" title="0.405"> </span><span style="background-color: hsl(120, 100.00%, 84.41%); opacity: 0.85" title="0.247">m</span><span style="background-color: hsl(0, 100.00%, 87.58%); opacity: 0.84" title="-0.179">y</span><span style="background-color: hsl(0, 100.00%, 80.39%); opacity: 0.87" title="-0.343"> </span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.034">b</span><span style="background-color: hsl(120, 100.00%, 72.69%); opacity: 0.92" title="0.550">o</span><span style="background-color: hsl(120, 100.00%, 80.97%); opacity: 0.87" title="0.328">u</span><span style="background-color: hsl(0, 100.00%, 80.44%); opacity: 0.87" title="-0.342">t</span><span style="background-color: hsl(0, 100.00%, 77.44%); opacity: 0.89" title="-0.419"> </span><span style="background-color: hsl(0, 100.00%, 88.80%); opacity: 0.83" title="-0.154">w</span><span style="background-color: hsl(0, 100.00%, 95.19%); opacity: 0.81" title="-0.046">i</span><span style="background-color: hsl(120, 100.00%, 92.73%); opacity: 0.82" title="0.083">t</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.049">h</span><span style="background-color: hsl(120, 100.00%, 94.77%); opacity: 0.81" title="0.052"> </span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.028">k</span><span style="background-color: hsl(0, 100.00%, 98.12%); opacity: 0.80" title="-0.012">i</span><span style="background-color: hsl(0, 100.00%, 95.42%); opacity: 0.81" title="-0.043">d</span><span style="background-color: hsl(0, 100.00%, 95.68%); opacity: 0.81" title="-0.039">n</span><span style="background-color: hsl(120, 100.00%, 91.69%); opacity: 0.82" title="0.101">e</span><span style="background-color: hsl(120, 100.00%, 88.97%); opacity: 0.83" title="0.151">y</span><span style="background-color: hsl(120, 100.00%, 84.24%); opacity: 0.85" title="0.251"> </span><span style="background-color: hsl(120, 100.00%, 86.47%); opacity: 0.84" title="0.202">s</span><span style="background-color: hsl(0, 100.00%, 95.27%); opacity: 0.81" title="-0.045">t</span><span style="background-color: hsl(0, 100.00%, 93.41%); opacity: 0.82" title="-0.072">o</span><span style="background-color: hsl(0, 100.00%, 91.47%); opacity: 0.82" title="-0.104">n</span><span style="background-color: hsl(0, 100.00%, 88.42%); opacity: 0.83" title="-0.161">e</span><span style="background-color: hsl(0, 100.00%, 99.61%); opacity: 0.80" title="-0.001">s</span><span style="background-color: hsl(120, 100.00%, 81.04%); opacity: 0.87" title="0.327">,</span><span style="background-color: hsl(120, 100.00%, 76.67%); opacity: 0.89" title="0.439"> </span><span style="background-color: hsl(120, 100.00%, 90.78%); opacity: 0.82" title="0.117">t</span><span style="background-color: hsl(120, 100.00%, 91.55%); opacity: 0.82" title="0.103">h</span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 94.20%); opacity: 0.81" title="0.060">r</span><span style="background-color: hsl(0, 100.00%, 83.33%); opacity: 0.86" title="-0.272">e</span><span style="background-color: hsl(0, 100.00%, 87.18%); opacity: 0.84" title="-0.187"> </span><span style="background-color: hsl(120, 100.00%, 96.89%); opacity: 0.81" title="0.025">i</span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.019">s</span><span style="background-color: hsl(120, 100.00%, 86.55%); opacity: 0.84" title="0.200">n'</span><span style="background-color: hsl(0, 100.00%, 85.18%); opacity: 0.85" title="-0.230">t</span><span style="background-color: hsl(0, 100.00%, 84.41%); opacity: 0.85" title="-0.247"> </span><span style="background-color: hsl(0, 100.00%, 99.10%); opacity: 0.80" title="-0.004">a</span><span style="background-color: hsl(0, 100.00%, 98.24%); opacity: 0.80" title="-0.011">n</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(120, 100.00%, 88.21%); opacity: 0.83" title="0.166">
    </span><span style="background-color: hsl(120, 100.00%, 68.07%); opacity: 0.94" title="0.688">m</span><span style="background-color: hsl(120, 100.00%, 61.22%); opacity: 0.99" title="0.908">e</span><span style="background-color: hsl(120, 100.00%, 61.49%); opacity: 0.99" title="0.899">d</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.949">i</span><span style="background-color: hsl(120, 100.00%, 63.77%); opacity: 0.97" title="0.824">c</span><span style="background-color: hsl(120, 100.00%, 88.48%); opacity: 0.83" title="0.160">a</span><span style="background-color: hsl(0, 100.00%, 99.31%); opacity: 0.80" title="-0.003">t</span><span style="background-color: hsl(0, 100.00%, 82.93%); opacity: 0.86" title="-0.281">i</span><span style="background-color: hsl(0, 100.00%, 77.77%); opacity: 0.89" title="-0.410">o</span><span style="background-color: hsl(0, 100.00%, 86.05%); opacity: 0.84" title="-0.211">n</span><span style="background-color: hsl(0, 100.00%, 91.83%); opacity: 0.82" title="-0.098"> </span><span style="background-color: hsl(0, 100.00%, 82.91%); opacity: 0.86" title="-0.282">t</span><span style="background-color: hsl(0, 100.00%, 76.92%); opacity: 0.89" title="-0.433">h</span><span style="background-color: hsl(0, 100.00%, 81.50%); opacity: 0.87" title="-0.315">a</span><span style="background-color: hsl(0, 100.00%, 73.56%); opacity: 0.91" title="-0.525">t</span><span style="background-color: hsl(0, 100.00%, 77.67%); opacity: 0.89" title="-0.413"> </span><span style="background-color: hsl(0, 100.00%, 86.70%); opacity: 0.84" title="-0.197">c</span><span style="background-color: hsl(0, 100.00%, 94.75%); opacity: 0.81" title="-0.052">a</span><span style="background-color: hsl(120, 100.00%, 92.99%); opacity: 0.82" title="0.079">n</span><span style="background-color: hsl(120, 100.00%, 91.73%); opacity: 0.82" title="0.100"> </span><span style="background-color: hsl(0, 100.00%, 96.34%); opacity: 0.81" title="-0.031">d</span><span style="opacity: 0.80">o</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.007"> a</span><span style="opacity: 0.80">ny</span><span style="background-color: hsl(0, 100.00%, 91.69%); opacity: 0.82" title="-0.101">th</span><span style="background-color: hsl(120, 100.00%, 86.29%); opacity: 0.84" title="0.206">i</span><span style="background-color: hsl(120, 100.00%, 85.10%); opacity: 0.85" title="0.231">n</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.059">g</span><span style="background-color: hsl(120, 100.00%, 90.61%); opacity: 0.83" title="0.120"> </span><span style="background-color: hsl(120, 100.00%, 88.99%); opacity: 0.83" title="0.150">a</span><span style="background-color: hsl(120, 100.00%, 73.74%); opacity: 0.91" title="0.520">b</span><span style="background-color: hsl(120, 100.00%, 68.66%); opacity: 0.94" title="0.670">o</span><span style="background-color: hsl(120, 100.00%, 77.45%); opacity: 0.89" title="0.418">u</span><span style="background-color: hsl(0, 100.00%, 77.80%); opacity: 0.89" title="-0.409">t</span><span style="background-color: hsl(0, 100.00%, 79.31%); opacity: 0.88" title="-0.370"> </span><span style="background-color: hsl(0, 100.00%, 80.10%); opacity: 0.87" title="-0.350">t</span><span style="background-color: hsl(0, 100.00%, 76.83%); opacity: 0.89" title="-0.435">h</span><span style="background-color: hsl(0, 100.00%, 83.59%); opacity: 0.86" title="-0.266">e</span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.053">m</span><span style="background-color: hsl(120, 100.00%, 92.51%); opacity: 0.82" title="0.087"> </span><span style="background-color: hsl(120, 100.00%, 85.05%); opacity: 0.85" title="0.233">ex</span><span style="background-color: hsl(0, 100.00%, 85.82%); opacity: 0.85" title="-0.216">ce</span><span style="background-color: hsl(0, 100.00%, 92.43%); opacity: 0.82" title="-0.088">p</span><span style="background-color: hsl(0, 100.00%, 79.90%); opacity: 0.87" title="-0.355">t</span><span style="background-color: hsl(0, 100.00%, 74.45%); opacity: 0.91" title="-0.500"> </span><span style="background-color: hsl(0, 100.00%, 88.03%); opacity: 0.84" title="-0.169">r</span><span style="background-color: hsl(0, 100.00%, 80.27%); opacity: 0.87" title="-0.346">el</span><span style="background-color: hsl(0, 100.00%, 90.73%); opacity: 0.82" title="-0.118">i</span><span style="background-color: hsl(0, 100.00%, 94.69%); opacity: 0.81" title="-0.053">e</span><span style="background-color: hsl(120, 100.00%, 88.38%); opacity: 0.83" title="0.162">v</span><span style="background-color: hsl(120, 100.00%, 89.10%); opacity: 0.83" title="0.148">e</span><span style="background-color: hsl(120, 100.00%, 93.58%); opacity: 0.81" title="0.070"> </span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(0, 100.00%, 87.50%); opacity: 0.84" title="-0.180">h</span><span style="background-color: hsl(0, 100.00%, 78.71%); opacity: 0.88" title="-0.385">e</span><span style="background-color: hsl(0, 100.00%, 98.86%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(120, 100.00%, 73.68%); opacity: 0.91" title="0.522">p</span><span style="background-color: hsl(120, 100.00%, 70.24%); opacity: 0.93" title="0.622">a</span><span style="background-color: hsl(120, 100.00%, 70.25%); opacity: 0.93" title="0.622">i</span><span style="background-color: hsl(120, 100.00%, 77.55%); opacity: 0.89" title="0.416">n</span><span style="background-color: hsl(120, 100.00%, 97.30%); opacity: 0.80" title="0.020">.</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(0, 100.00%, 95.55%); opacity: 0.81" title="-0.041">e</span><span style="background-color: hsl(120, 100.00%, 97.41%); opacity: 0.80" title="0.019">i</span><span style="background-color: hsl(0, 100.00%, 85.37%); opacity: 0.85" title="-0.226">t</span><span style="background-color: hsl(0, 100.00%, 80.77%); opacity: 0.87" title="-0.333">h</span><span style="background-color: hsl(0, 100.00%, 84.32%); opacity: 0.85" title="-0.249">e</span><span style="background-color: hsl(120, 100.00%, 86.59%); opacity: 0.84" title="0.199">r</span><span style="background-color: hsl(120, 100.00%, 81.90%); opacity: 0.86" title="0.306"> </span><span style="background-color: hsl(0, 100.00%, 80.40%); opacity: 0.87" title="-0.343">t</span><span style="background-color: hsl(0, 100.00%, 81.04%); opacity: 0.87" title="-0.327">h</span><span style="background-color: hsl(0, 100.00%, 91.50%); opacity: 0.82" title="-0.104">e</span><span style="background-color: hsl(120, 100.00%, 96.28%); opacity: 0.81" title="0.032">y</span><span style="background-color: hsl(120, 100.00%, 84.94%); opacity: 0.85" title="0.235"> </span><span style="background-color: hsl(120, 100.00%, 82.28%); opacity: 0.86" title="0.297">p</span><span style="background-color: hsl(120, 100.00%, 82.64%); opacity: 0.86" title="0.288">a</span><span style="background-color: hsl(120, 100.00%, 97.39%); opacity: 0.80" title="0.019">s</span><span style="background-color: hsl(0, 100.00%, 91.86%); opacity: 0.82" title="-0.098">s</span><span style="background-color: hsl(0, 100.00%, 97.37%); opacity: 0.80" title="-0.019">,</span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.034"> </span><span style="background-color: hsl(0, 100.00%, 88.88%); opacity: 0.83" title="-0.152">o</span><span style="background-color: hsl(120, 100.00%, 89.05%); opacity: 0.83" title="0.149">r</span><span style="background-color: hsl(120, 100.00%, 84.04%); opacity: 0.85" title="0.255"> </span><span style="background-color: hsl(0, 100.00%, 78.72%); opacity: 0.88" title="-0.385">t</span><span style="background-color: hsl(0, 100.00%, 82.07%); opacity: 0.86" title="-0.302">h</span><span style="background-color: hsl(0, 100.00%, 90.02%); opacity: 0.83" title="-0.131">e</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.005">y</span><span style="background-color: hsl(0, 100.00%, 93.17%); opacity: 0.82" title="-0.076"> </span><span style="background-color: hsl(0, 100.00%, 77.49%); opacity: 0.89" title="-0.417">h</span><span style="background-color: hsl(0, 100.00%, 77.00%); opacity: 0.89" title="-0.431">a</span><span style="background-color: hsl(0, 100.00%, 87.70%); opacity: 0.84" title="-0.176">v</span><span style="background-color: hsl(0, 100.00%, 99.17%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(0, 100.00%, 94.04%); opacity: 0.81" title="-0.062"> </span><span style="background-color: hsl(120, 100.00%, 88.43%); opacity: 0.83" title="0.161">t</span><span style="background-color: hsl(0, 100.00%, 94.23%); opacity: 0.81" title="-0.060">o</span><span style="background-color: hsl(0, 100.00%, 86.47%); opacity: 0.84" title="-0.202"> </span><span style="background-color: hsl(0, 100.00%, 79.58%); opacity: 0.88" title="-0.363">b</span><span style="background-color: hsl(0, 100.00%, 82.78%); opacity: 0.86" title="-0.285">e</span><span style="background-color: hsl(0, 100.00%, 75.83%); opacity: 0.90" title="-0.462"> </span><span style="background-color: hsl(0, 100.00%, 90.86%); opacity: 0.82" title="-0.115">b</span><span style="background-color: hsl(0, 100.00%, 90.93%); opacity: 0.82" title="-0.114">r</span><span style="background-color: hsl(0, 100.00%, 92.83%); opacity: 0.82" title="-0.081">o</span><span style="background-color: hsl(0, 100.00%, 92.98%); opacity: 0.82" title="-0.079">k</span><span style="background-color: hsl(120, 100.00%, 86.72%); opacity: 0.84" title="0.197">e</span><span style="background-color: hsl(120, 100.00%, 81.91%); opacity: 0.86" title="0.306">n</span><span style="background-color: hsl(120, 100.00%, 83.29%); opacity: 0.86" title="0.273"> </span><span style="background-color: hsl(0, 100.00%, 89.63%); opacity: 0.83" title="-0.138">u</span><span style="background-color: hsl(0, 100.00%, 84.03%); opacity: 0.85" title="-0.256">p</span><span style="background-color: hsl(0, 100.00%, 80.79%); opacity: 0.87" title="-0.333"> </span><span style="background-color: hsl(0, 100.00%, 85.80%); opacity: 0.85" title="-0.216">w</span><span style="background-color: hsl(120, 100.00%, 95.85%); opacity: 0.81" title="0.037">i</span><span style="background-color: hsl(120, 100.00%, 89.26%); opacity: 0.83" title="0.145">t</span><span style="background-color: hsl(120, 100.00%, 97.98%); opacity: 0.80" title="0.013">h</span><span style="background-color: hsl(120, 100.00%, 90.94%); opacity: 0.82" title="0.114"> </span><span style="background-color: hsl(120, 100.00%, 90.28%); opacity: 0.83" title="0.126">s</span><span style="background-color: hsl(120, 100.00%, 75.00%); opacity: 0.90" title="0.485">o</span><span style="background-color: hsl(120, 100.00%, 79.73%); opacity: 0.88" title="0.359">u</span><span style="background-color: hsl(120, 100.00%, 87.56%); opacity: 0.84" title="0.179">n</span><span style="background-color: hsl(120, 100.00%, 92.05%); opacity: 0.82" title="0.094">d</span><span style="background-color: hsl(120, 100.00%, 88.48%); opacity: 0.83" title="0.160">,</span><span style="background-color: hsl(120, 100.00%, 95.98%); opacity: 0.81" title="0.036"> </span><span style="background-color: hsl(0, 100.00%, 92.73%); opacity: 0.82" title="-0.083">o</span><span style="background-color: hsl(120, 100.00%, 86.60%); opacity: 0.84" title="0.199">r</span><span style="background-color: hsl(120, 100.00%, 84.04%); opacity: 0.85" title="0.255"> </span><span style="background-color: hsl(0, 100.00%, 78.72%); opacity: 0.88" title="-0.385">t</span><span style="background-color: hsl(0, 100.00%, 82.07%); opacity: 0.86" title="-0.302">h</span><span style="background-color: hsl(0, 100.00%, 90.02%); opacity: 0.83" title="-0.131">e</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.005">y</span><span style="background-color: hsl(0, 100.00%, 93.17%); opacity: 0.82" title="-0.076"> </span><span style="background-color: hsl(0, 100.00%, 80.52%); opacity: 0.87" title="-0.340">h</span><span style="background-color: hsl(0, 100.00%, 79.99%); opacity: 0.87" title="-0.353">a</span><span style="background-color: hsl(0, 100.00%, 91.07%); opacity: 0.82" title="-0.111">v</span><span style="background-color: hsl(120, 100.00%, 85.81%); opacity: 0.85" title="0.216">e</span><span style="background-color: hsl(0, 100.00%, 90.11%); opacity: 0.83" title="-0.129">
    </span><span style="background-color: hsl(0, 100.00%, 82.73%); opacity: 0.86" title="-0.286">t</span><span style="background-color: hsl(0, 100.00%, 80.38%); opacity: 0.87" title="-0.343">o</span><span style="background-color: hsl(0, 100.00%, 80.07%); opacity: 0.87" title="-0.351"> </span><span style="background-color: hsl(0, 100.00%, 76.54%); opacity: 0.89" title="-0.443">b</span><span style="background-color: hsl(0, 100.00%, 83.92%); opacity: 0.85" title="-0.258">e</span><span style="background-color: hsl(0, 100.00%, 86.46%); opacity: 0.84" title="-0.202"> </span><span style="background-color: hsl(120, 100.00%, 82.35%); opacity: 0.86" title="0.295">e</span><span style="background-color: hsl(120, 100.00%, 83.55%); opacity: 0.86" title="0.267">x</span><span style="background-color: hsl(120, 100.00%, 83.42%); opacity: 0.86" title="0.270">t</span><span style="background-color: hsl(120, 100.00%, 81.85%); opacity: 0.86" title="0.307">r</span><span style="background-color: hsl(120, 100.00%, 93.28%); opacity: 0.82" title="0.074">a</span><span style="background-color: hsl(120, 100.00%, 70.28%); opacity: 0.93" title="0.621">c</span><span style="background-color: hsl(120, 100.00%, 62.50%); opacity: 0.98" title="0.866">t</span><span style="background-color: hsl(120, 100.00%, 66.83%); opacity: 0.95" title="0.726">e</span><span style="background-color: hsl(120, 100.00%, 74.67%); opacity: 0.90" title="0.494">d</span><span style="background-color: hsl(120, 100.00%, 71.67%); opacity: 0.92" title="0.580"> </span><span style="background-color: hsl(120, 100.00%, 84.79%); opacity: 0.85" title="0.238">s</span><span style="background-color: hsl(120, 100.00%, 84.37%); opacity: 0.85" title="0.248">u</span><span style="background-color: hsl(120, 100.00%, 86.27%); opacity: 0.84" title="0.206">r</span><span style="background-color: hsl(120, 100.00%, 83.66%); opacity: 0.86" title="0.264">g</span><span style="background-color: hsl(120, 100.00%, 76.62%); opacity: 0.89" title="0.441">i</span><span style="background-color: hsl(120, 100.00%, 79.04%); opacity: 0.88" title="0.377">c</span><span style="background-color: hsl(120, 100.00%, 74.12%); opacity: 0.91" title="0.510">a</span><span style="background-color: hsl(120, 100.00%, 78.63%); opacity: 0.88" title="0.388">l</span><span style="background-color: hsl(120, 100.00%, 88.57%); opacity: 0.83" title="0.159">l</span><span style="background-color: hsl(0, 100.00%, 85.51%); opacity: 0.85" title="-0.223">y</span><span style="background-color: hsl(0, 100.00%, 91.86%); opacity: 0.82" title="-0.098">.</span><span style="background-color: hsl(120, 100.00%, 87.10%); opacity: 0.84" title="0.188"> </span><span style="background-color: hsl(120, 100.00%, 72.78%); opacity: 0.92" title="0.548">w</span><span style="background-color: hsl(120, 100.00%, 69.85%); opacity: 0.93" title="0.634">h</span><span style="background-color: hsl(120, 100.00%, 74.15%); opacity: 0.91" title="0.509">e</span><span style="background-color: hsl(120, 100.00%, 75.92%); opacity: 0.90" title="0.460">n</span><span style="background-color: hsl(120, 100.00%, 76.70%); opacity: 0.89" title="0.438"> </span><span style="background-color: hsl(120, 100.00%, 81.27%); opacity: 0.87" title="0.321">i</span><span style="background-color: hsl(120, 100.00%, 85.92%); opacity: 0.85" title="0.214"> </span><span style="background-color: hsl(120, 100.00%, 98.03%); opacity: 0.80" title="0.013">w</span><span style="background-color: hsl(120, 100.00%, 93.55%); opacity: 0.81" title="0.070">a</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.068">s</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 86.09%); opacity: 0.84" title="0.210">i</span><span style="background-color: hsl(120, 100.00%, 91.42%); opacity: 0.82" title="0.105">n</span><span style="background-color: hsl(120, 100.00%, 87.26%); opacity: 0.84" title="0.185">,</span><span style="background-color: hsl(120, 100.00%, 83.78%); opacity: 0.86" title="0.261"> </span><span style="background-color: hsl(0, 100.00%, 96.77%); opacity: 0.81" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 92.20%); opacity: 0.82" title="-0.092">h</span><span style="background-color: hsl(0, 100.00%, 79.68%); opacity: 0.88" title="-0.361">e</span><span style="background-color: hsl(0, 100.00%, 82.77%); opacity: 0.86" title="-0.285"> </span><span style="background-color: hsl(120, 100.00%, 86.24%); opacity: 0.84" title="0.207">x</span><span style="background-color: hsl(120, 100.00%, 85.01%); opacity: 0.85" title="0.233">-</span><span style="background-color: hsl(120, 100.00%, 94.51%); opacity: 0.81" title="0.056">r</span><span style="background-color: hsl(0, 100.00%, 95.63%); opacity: 0.81" title="-0.040">a</span><span style="background-color: hsl(120, 100.00%, 97.85%); opacity: 0.80" title="0.015">y</span><span style="background-color: hsl(120, 100.00%, 92.88%); opacity: 0.82" title="0.081"> </span><span style="background-color: hsl(120, 100.00%, 86.31%); opacity: 0.84" title="0.205">t</span><span style="background-color: hsl(0, 100.00%, 88.90%); opacity: 0.83" title="-0.152">e</span><span style="background-color: hsl(0, 100.00%, 77.07%); opacity: 0.89" title="-0.429">c</span><span style="background-color: hsl(0, 100.00%, 90.17%); opacity: 0.83" title="-0.128">h</span><span style="background-color: hsl(0, 100.00%, 96.40%); opacity: 0.81" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 90.41%); opacity: 0.83" title="-0.123">h</span><span style="background-color: hsl(0, 100.00%, 89.10%); opacity: 0.83" title="-0.148">a</span><span style="opacity: 0.80">p</span><span style="background-color: hsl(0, 100.00%, 97.42%); opacity: 0.80" title="-0.019">p</span><span style="background-color: hsl(0, 100.00%, 98.39%); opacity: 0.80" title="-0.010">e</span><span style="background-color: hsl(0, 100.00%, 91.40%); opacity: 0.82" title="-0.106">n</span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.088">e</span><span style="background-color: hsl(120, 100.00%, 89.28%); opacity: 0.83" title="0.145">d</span><span style="background-color: hsl(120, 100.00%, 81.86%); opacity: 0.86" title="0.307"> </span><span style="background-color: hsl(120, 100.00%, 89.25%); opacity: 0.83" title="0.145">t</span><span style="background-color: hsl(0, 100.00%, 97.60%); opacity: 0.80" title="-0.017">o</span><span style="background-color: hsl(120, 100.00%, 90.87%); opacity: 0.82" title="0.115"> </span><span style="background-color: hsl(120, 100.00%, 74.30%); opacity: 0.91" title="0.505">m</span><span style="background-color: hsl(120, 100.00%, 74.53%); opacity: 0.90" title="0.498">e</span><span style="background-color: hsl(120, 100.00%, 84.25%); opacity: 0.85" title="0.251">n</span><span style="background-color: hsl(120, 100.00%, 91.59%); opacity: 0.82" title="0.102">t</span><span style="background-color: hsl(0, 100.00%, 79.47%); opacity: 0.88" title="-0.366">i</span><span style="background-color: hsl(0, 100.00%, 77.33%); opacity: 0.89" title="-0.422">o</span><span style="background-color: hsl(0, 100.00%, 86.05%); opacity: 0.84" title="-0.211">n</span><span style="background-color: hsl(0, 100.00%, 91.83%); opacity: 0.82" title="-0.098"> </span><span style="background-color: hsl(0, 100.00%, 82.91%); opacity: 0.86" title="-0.282">t</span><span style="background-color: hsl(0, 100.00%, 78.40%); opacity: 0.88" title="-0.394">h</span><span style="background-color: hsl(0, 100.00%, 87.02%); opacity: 0.84" title="-0.190">a</span><span style="background-color: hsl(0, 100.00%, 76.70%); opacity: 0.89" title="-0.439">t</span><span style="background-color: hsl(0, 100.00%, 81.07%); opacity: 0.87" title="-0.326"> </span><span style="background-color: hsl(0, 100.00%, 95.60%); opacity: 0.81" title="-0.040">s</span><span style="background-color: hsl(120, 100.00%, 90.15%); opacity: 0.83" title="0.128">h</span><span style="background-color: hsl(0, 100.00%, 95.81%); opacity: 0.81" title="-0.038">e</span><span style="background-color: hsl(0, 100.00%, 91.85%); opacity: 0.82" title="-0.098">'</span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.010">d</span><span style="background-color: hsl(120, 100.00%, 93.48%); opacity: 0.81" title="0.071"> </span><span style="background-color: hsl(120, 100.00%, 94.01%); opacity: 0.81" title="0.063">h</span><span style="background-color: hsl(120, 100.00%, 96.79%); opacity: 0.81" title="0.026">a</span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.254">d</span><span style="background-color: hsl(120, 100.00%, 84.42%); opacity: 0.85" title="0.247"> </span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.005">k</span><span style="background-color: hsl(0, 100.00%, 98.37%); opacity: 0.80" title="-0.010">i</span><span style="background-color: hsl(0, 100.00%, 92.55%); opacity: 0.82" title="-0.086">d</span><span style="background-color: hsl(0, 100.00%, 88.51%); opacity: 0.83" title="-0.160">ne</span><span style="background-color: hsl(0, 100.00%, 89.74%); opacity: 0.83" title="-0.136">y</span><span style="background-color: hsl(0, 100.00%, 93.31%); opacity: 0.82" title="-0.074">
    </span><span style="background-color: hsl(120, 100.00%, 96.70%); opacity: 0.81" title="0.027">s</span><span style="background-color: hsl(0, 100.00%, 90.50%); opacity: 0.83" title="-0.122">t</span><span style="background-color: hsl(0, 100.00%, 86.97%); opacity: 0.84" title="-0.191">o</span><span style="background-color: hsl(0, 100.00%, 85.86%); opacity: 0.85" title="-0.215">n</span><span style="background-color: hsl(0, 100.00%, 85.93%); opacity: 0.84" title="-0.213">e</span><span style="background-color: hsl(0, 100.00%, 98.29%); opacity: 0.80" title="-0.010">s</span><span style="background-color: hsl(120, 100.00%, 95.61%); opacity: 0.81" title="0.040"> </span><span style="background-color: hsl(120, 100.00%, 92.08%); opacity: 0.82" title="0.094">a</span><span style="background-color: hsl(120, 100.00%, 91.75%); opacity: 0.82" title="0.099">n</span><span style="background-color: hsl(120, 100.00%, 94.56%); opacity: 0.81" title="0.055">d</span><span style="background-color: hsl(120, 100.00%, 95.45%); opacity: 0.81" title="0.043"> </span><span style="background-color: hsl(0, 100.00%, 79.60%); opacity: 0.88" title="-0.363">c</span><span style="background-color: hsl(0, 100.00%, 81.06%); opacity: 0.87" title="-0.326">h</span><span style="background-color: hsl(0, 100.00%, 82.29%); opacity: 0.86" title="-0.296">i</span><span style="background-color: hsl(0, 100.00%, 80.14%); opacity: 0.87" title="-0.349">l</span><span style="background-color: hsl(0, 100.00%, 75.19%); opacity: 0.90" title="-0.480">d</span><span style="background-color: hsl(0, 100.00%, 77.20%); opacity: 0.89" title="-0.425">r</span><span style="background-color: hsl(0, 100.00%, 75.75%); opacity: 0.90" title="-0.464">e</span><span style="background-color: hsl(0, 100.00%, 83.13%); opacity: 0.86" title="-0.277">n</span><span style="background-color: hsl(0, 100.00%, 85.03%); opacity: 0.85" title="-0.233">,</span><span style="background-color: hsl(0, 100.00%, 90.84%); opacity: 0.82" title="-0.116"> </span><span style="background-color: hsl(0, 100.00%, 88.04%); opacity: 0.84" title="-0.169">a</span><span style="background-color: hsl(0, 100.00%, 90.50%); opacity: 0.83" title="-0.122">n</span><span style="background-color: hsl(120, 100.00%, 83.97%); opacity: 0.85" title="0.257">d</span><span style="background-color: hsl(120, 100.00%, 81.67%); opacity: 0.87" title="0.311"> </span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.041">t</span><span style="background-color: hsl(0, 100.00%, 85.21%); opacity: 0.85" title="-0.229">h</span><span style="background-color: hsl(0, 100.00%, 86.08%); opacity: 0.84" title="-0.210">e</span><span style="background-color: hsl(0, 100.00%, 87.84%); opacity: 0.84" title="-0.173"> </span><span style="background-color: hsl(0, 100.00%, 86.39%); opacity: 0.84" title="-0.203">c</span><span style="background-color: hsl(0, 100.00%, 82.35%); opacity: 0.86" title="-0.295">h</span><span style="background-color: hsl(0, 100.00%, 86.25%); opacity: 0.84" title="-0.206">i</span><span style="background-color: hsl(0, 100.00%, 94.38%); opacity: 0.81" title="-0.058">l</span><span style="background-color: hsl(0, 100.00%, 96.98%); opacity: 0.80" title="-0.024">d</span><span style="background-color: hsl(120, 100.00%, 93.01%); opacity: 0.82" title="0.078">b</span><span style="background-color: hsl(120, 100.00%, 98.29%); opacity: 0.80" title="0.010">i</span><span style="background-color: hsl(0, 100.00%, 68.69%); opacity: 0.94" title="-0.669">r</span><span style="background-color: hsl(0, 100.00%, 67.96%); opacity: 0.95" title="-0.691">t</span><span style="background-color: hsl(0, 100.00%, 84.05%); opacity: 0.85" title="-0.255">h</span><span style="background-color: hsl(120, 100.00%, 92.78%); opacity: 0.82" title="0.082"> </span><span style="background-color: hsl(120, 100.00%, 96.45%); opacity: 0.81" title="0.030">h</span><span style="background-color: hsl(120, 100.00%, 94.22%); opacity: 0.81" title="0.060">u</span><span style="background-color: hsl(0, 100.00%, 81.51%); opacity: 0.87" title="-0.315">r</span><span style="background-color: hsl(0, 100.00%, 70.47%); opacity: 0.93" title="-0.615">t</span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.049"> </span><span style="background-color: hsl(120, 100.00%, 81.30%); opacity: 0.87" title="0.320">l</span><span style="background-color: hsl(120, 100.00%, 85.34%); opacity: 0.85" title="0.226">e</span><span style="background-color: hsl(120, 100.00%, 91.69%); opacity: 0.82" title="0.101">s</span><span style="opacity: 0.80">s.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Note that accuracy score is perfect, but KL divergence is bad. It means
this sampler was not very useful: most generated texts were "easy" in
sense that most (or all?) of them should be still classified as
``sci.med``, so it was easy to get a good accuracy. But because
generated texts were not diverse enough classifier haven't learned
anything useful; it's having a hard time predicting the probability
output of the black-box pipeline on a held-out dataset.

By default :class:`~.TextExplainer` uses a mix of several sampling strategies
which seems to work OK for token-based explanations. But a good sampling
strategy which works for many real-world tasks could be a research topic
on itself. If you've got some experience with it we'd love to hear from
you - please share your findings in eli5 issue tracker (
https://github.com/TeamHG-Memex/eli5/issues )!

Customizing TextExplainer: classifier
-------------------------------------

In one of the previous examples we already changed the vectorizer
TextExplainer uses (to take additional features in account). It is also
possible to change the white-box classifier - for example, use a small
decision tree:

.. code:: ipython3

    from sklearn.tree import DecisionTreeClassifier
    
    te5 = TextExplainer(clf=DecisionTreeClassifier(max_depth=2), random_state=0)
    te5.fit(doc, pipe.predict_proba)
    print(te5.metrics_)
    te5.show_weights()


.. parsed-literal::

    {'score': 0.9838155527960798, 'mean_KL_divergence': 0.03812615869329402}




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
            <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em;">
                <thead>
                <tr style="border: none;">
                    <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                    <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
                </tr>
                </thead>
                <tbody>
                
                    <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
                        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                            0.5449
                            
                        </td>
                        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                            kidney
                        </td>
                    </tr>
                
                    <tr style="background-color: hsl(120, 100.00%, 82.37%); border: none;">
                        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                            0.4551
                            
                        </td>
                        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                            pain
                        </td>
                    </tr>
                
                
                </tbody>
            </table>
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
            
            <br>
            <pre><svg width="520pt" height="180pt"
     viewBox="0.00 0.00 788.00 280.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <g id="graph1" class="graph" transform="scale(1 1) rotate(0) translate(4 276)">
    <title>Tree</title>
    <polygon fill="white" stroke="white" points="-4,5 -4,-276 785,-276 785,5 -4,5"/>
    <!-- 0 -->
    <g id="node1" class="node"><title>0</title>
    <polygon fill="none" stroke="black" points="491,-272 301,-272 301,-200 491,-200 491,-272"/>
    <text text-anchor="middle" x="396" y="-255.4" font-family="Times,serif" font-size="14.00">kidney &lt;= 0.5</text>
    <text text-anchor="middle" x="396" y="-239.4" font-family="Times,serif" font-size="14.00">gini = 0.1595</text>
    <text text-anchor="middle" x="396" y="-223.4" font-family="Times,serif" font-size="14.00">samples = 100.0%</text>
    <text text-anchor="middle" x="396" y="-207.4" font-family="Times,serif" font-size="14.00">value = [0.01, 0.03, 0.92, 0.04]</text>
    </g>
    <!-- 1 -->
    <g id="node2" class="node"><title>1</title>
    <polygon fill="none" stroke="black" points="387.25,-164 204.75,-164 204.75,-92 387.25,-92 387.25,-164"/>
    <text text-anchor="middle" x="296" y="-147.4" font-family="Times,serif" font-size="14.00">pain &lt;= 0.5</text>
    <text text-anchor="middle" x="296" y="-131.4" font-family="Times,serif" font-size="14.00">gini = 0.3893</text>
    <text text-anchor="middle" x="296" y="-115.4" font-family="Times,serif" font-size="14.00">samples = 38.9%</text>
    <text text-anchor="middle" x="296" y="-99.4" font-family="Times,serif" font-size="14.00">value = [0.03, 0.1, 0.77, 0.11]</text>
    </g>
    <!-- 0&#45;&gt;1 -->
    <g id="edge2" class="edge"><title>0&#45;&gt;1</title>
    <path fill="none" stroke="black" d="M362.979,-199.998C354.45,-190.957 345.173,-181.123 336.317,-171.736"/>
    <polygon fill="black" stroke="black" points="338.736,-169.2 329.328,-164.328 333.644,-174.003 338.736,-169.2"/>
    <text text-anchor="middle" x="328.6" y="-184.717" font-family="Times,serif" font-size="14.00">True</text>
    </g>
    <!-- 4 -->
    <g id="node8" class="node"><title>4</title>
    <polygon fill="none" stroke="black" points="588.25,-164 405.75,-164 405.75,-92 588.25,-92 588.25,-164"/>
    <text text-anchor="middle" x="497" y="-147.4" font-family="Times,serif" font-size="14.00">pain &lt;= 0.5</text>
    <text text-anchor="middle" x="497" y="-131.4" font-family="Times,serif" font-size="14.00">gini = 0.0474</text>
    <text text-anchor="middle" x="497" y="-115.4" font-family="Times,serif" font-size="14.00">samples = 61.1%</text>
    <text text-anchor="middle" x="497" y="-99.4" font-family="Times,serif" font-size="14.00">value = [0.0, 0.01, 0.98, 0.01]</text>
    </g>
    <!-- 0&#45;&gt;4 -->
    <g id="edge8" class="edge"><title>0&#45;&gt;4</title>
    <path fill="none" stroke="black" d="M429.351,-199.998C437.965,-190.957 447.335,-181.123 456.28,-171.736"/>
    <polygon fill="black" stroke="black" points="458.974,-173.982 463.339,-164.328 453.907,-169.153 458.974,-173.982"/>
    <text text-anchor="middle" x="463.943" y="-184.72" font-family="Times,serif" font-size="14.00">False</text>
    </g>
    <!-- 2 -->
    <g id="node4" class="node"><title>2</title>
    <polygon fill="none" stroke="black" points="190,-56 0,-56 0,-0 190,-0 190,-56"/>
    <text text-anchor="middle" x="95" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.5253</text>
    <text text-anchor="middle" x="95" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 28.4%</text>
    <text text-anchor="middle" x="95" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.04, 0.14, 0.65, 0.16]</text>
    </g>
    <!-- 1&#45;&gt;2 -->
    <g id="edge4" class="edge"><title>1&#45;&gt;2</title>
    <path fill="none" stroke="black" d="M224.002,-91.8966C203,-81.6566 180.187,-70.534 159.667,-60.5294"/>
    <polygon fill="black" stroke="black" points="160.938,-57.2551 150.416,-56.0186 157.87,-63.5471 160.938,-57.2551"/>
    </g>
    <!-- 3 -->
    <g id="node6" class="node"><title>3</title>
    <polygon fill="none" stroke="black" points="384,-56 208,-56 208,-0 384,-0 384,-56"/>
    <text text-anchor="middle" x="296" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.0445</text>
    <text text-anchor="middle" x="296" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 10.6%</text>
    <text text-anchor="middle" x="296" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.0, 0.0, 0.98, 0.02]</text>
    </g>
    <!-- 1&#45;&gt;3 -->
    <g id="edge6" class="edge"><title>1&#45;&gt;3</title>
    <path fill="none" stroke="black" d="M296,-91.8966C296,-83.6325 296,-74.7936 296,-66.4314"/>
    <polygon fill="black" stroke="black" points="299.5,-66.1734 296,-56.1734 292.5,-66.1734 299.5,-66.1734"/>
    </g>
    <!-- 5 -->
    <g id="node10" class="node"><title>5</title>
    <polygon fill="none" stroke="black" points="592,-56 402,-56 402,-0 592,-0 592,-56"/>
    <text text-anchor="middle" x="497" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.1194</text>
    <text text-anchor="middle" x="497" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 22.8%</text>
    <text text-anchor="middle" x="497" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.01, 0.02, 0.94, 0.04]</text>
    </g>
    <!-- 4&#45;&gt;5 -->
    <g id="edge10" class="edge"><title>4&#45;&gt;5</title>
    <path fill="none" stroke="black" d="M497,-91.8966C497,-83.6325 497,-74.7936 497,-66.4314"/>
    <polygon fill="black" stroke="black" points="500.5,-66.1734 497,-56.1734 493.5,-66.1734 500.5,-66.1734"/>
    </g>
    <!-- 6 -->
    <g id="node12" class="node"><title>6</title>
    <polygon fill="none" stroke="black" points="780,-56 610,-56 610,-0 780,-0 780,-56"/>
    <text text-anchor="middle" x="695" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.0121</text>
    <text text-anchor="middle" x="695" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 38.2%</text>
    <text text-anchor="middle" x="695" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.0, 0.0, 0.99, 0.0]</text>
    </g>
    <!-- 4&#45;&gt;6 -->
    <g id="edge12" class="edge"><title>4&#45;&gt;6</title>
    <path fill="none" stroke="black" d="M567.923,-91.8966C588.612,-81.6566 611.084,-70.534 631.298,-60.5294"/>
    <polygon fill="black" stroke="black" points="633.002,-63.5913 640.411,-56.0186 629.897,-57.3177 633.002,-63.5913"/>
    </g>
    </g>
    </svg>
    </pre>
        
    
    
    




How to read it: "kidney <= 0.5" means "word 'kidney' is not in the
document" (we're explaining the orginal LDA+SVM pipeline again).

So according to this tree if "kidney" is not in the document and "pain"
is not in the document then the probability of a document belonging to
``sci.med`` drops to ``0.65``. If at least one of these words remain
``sci.med`` probability stays ``0.9+``.

.. code:: ipython3

    print("both words removed::")
    print_prediction(re.sub(r"(kidney|pain)", "", doc, flags=re.I))
    print("\nonly 'pain' removed:")
    print_prediction(re.sub(r"pain", "", doc, flags=re.I))


.. parsed-literal::

    both words removed::
    0.014 alt.atheism
    0.024 comp.graphics
    0.891 sci.med
    0.072 soc.religion.christian
    
    only 'pain' removed:
    0.002 alt.atheism
    0.004 comp.graphics
    0.978 sci.med
    0.015 soc.religion.christian


As expected, after removing both words probability of ``sci.med``
decreased, though not as much as our simple decision tree predicted (to
0.9 instead of 0.64). Removing ``pain`` provided exactly the same effect
as predicted - probability of ``sci.med`` became ``0.98``.
