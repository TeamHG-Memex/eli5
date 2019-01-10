
TextExplainer: debugging black-box text classifiers
===================================================

While eli5 supports many classifiers and preprocessing methods, it can’t
support them all.

If a library is not supported by eli5 directly, or the text processing
pipeline is too complex for eli5, eli5 can still help - it provides an
implementation of `LIME <http://arxiv.org/abs/1602.04938>`__ (Ribeiro et
al., 2016) algorithm which allows to explain predictions of arbitrary
classifiers, including text classifiers. ``eli5.lime`` can also help
when it is hard to get exact mapping between model coefficients and text
features, e.g. if there is dimension reduction involved.

Example problem: LSA+SVM for 20 Newsgroups dataset
--------------------------------------------------

Let’s load “20 Newsgroups” dataset and create a text processing pipeline
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
    
        
        (probability <b>0.000</b>, score <b>-9.663</b>)
    
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
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.360
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -9.303
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 96.66%); opacity: 0.81" title="-0.065">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.25%); opacity: 0.81" title="-0.076">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.69%); opacity: 0.81" title="-0.064">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.52%); opacity: 0.81" title="0.099">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.95%); opacity: 0.81" title="0.152">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.018">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.83%); opacity: 0.80" title="-0.015">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.66%); opacity: 0.87" title="-0.740">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.61%); opacity: 0.86" title="-0.630">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 96.66%); opacity: 0.81" title="0.065">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.81%); opacity: 0.88" title="0.909">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 90.99%); opacity: 0.82" title="0.268">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.139">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 85.98%); opacity: 0.84" title="-0.504">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.43%); opacity: 0.83" title="-0.292">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.119">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.16%); opacity: 0.83" title="0.304">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.16%); opacity: 0.83" title="0.304">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.29%); opacity: 0.82" title="-0.215">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.60%); opacity: 0.83" title="-0.285">them</span><span style="opacity: 0.80"> except </span><span style="background-color: hsl(0, 100.00%, 92.45%); opacity: 0.82" title="-0.208">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.69%); opacity: 0.81" title="-0.126">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.54%); opacity: 0.88" title="-0.865">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 98.50%); opacity: 0.80" title="-0.021">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.90%); opacity: 0.82" title="-0.191">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.61%); opacity: 0.82" title="-0.242">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.148">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.90%); opacity: 0.80" title="-0.033">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.003">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.16%); opacity: 0.83" title="0.349">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.231">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.06%); opacity: 0.82" title="0.224">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.080">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.62%); opacity: 0.80" title="0.040">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.08%); opacity: 0.82" title="-0.264">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.148">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.90%); opacity: 0.80" title="-0.033">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.003">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 89.16%); opacity: 0.83" title="0.349">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.007">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.13%); opacity: 0.80" title="-0.028">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.55%); opacity: 0.81" title="0.068">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 92.36%); opacity: 0.82" title="-0.212">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.46%); opacity: 0.82" title="-0.170">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.52%); opacity: 0.82" title="-0.205">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.86%); opacity: 0.81" title="-0.088">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 92.84%); opacity: 0.82" title="-0.193">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.04%); opacity: 0.80" title="-0.030">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 91.04%); opacity: 0.82" title="-0.266">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.91%); opacity: 0.82" title="-0.191">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.55%); opacity: 0.80" title="-0.042">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.086">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.32%); opacity: 0.83" title="0.388">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.73%); opacity: 0.81" title="-0.063">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.81%); opacity: 0.81" title="-0.157">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 93.37%); opacity: 0.82" title="-0.173">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.167">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.66%); opacity: 0.87" title="-0.740">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 83.61%); opacity: 0.86" title="-0.630">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.08%); opacity: 0.82" title="-0.184">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.95%); opacity: 0.82" title="-0.270">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 94.11%); opacity: 0.81" title="-0.146">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.49%); opacity: 0.80" title="0.043">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.83%); opacity: 0.81" title="0.060">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.09%); opacity: 0.82" title="-0.264">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.79%); opacity: 0.80" title="-0.036">less</span><span style="opacity: 0.80">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=comp.graphics
        
    </b>
    
        
        (probability <b>0.000</b>, score <b>-8.503</b>)
    
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
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.210
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 81.55%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -8.293
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as i </span><span style="background-color: hsl(0, 100.00%, 87.01%); opacity: 0.84" title="-0.452">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.117">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.192">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.42%); opacity: 0.81" title="0.135">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.79%); opacity: 0.81" title="-0.158">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.34%); opacity: 0.89" title="-1.001">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.69%); opacity: 0.88" title="-0.856">stones</span><span style="opacity: 0.80">, there </span><span style="background-color: hsl(0, 100.00%, 89.72%); opacity: 0.83" title="-0.324">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 94.56%); opacity: 0.81" title="0.130">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.25%); opacity: 0.80" title="0.049">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 81.51%); opacity: 0.87" title="-0.749">medication</span><span style="opacity: 0.80"> that </span><span style="background-color: hsl(0, 100.00%, 98.37%); opacity: 0.80" title="-0.023">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.83%); opacity: 0.80" title="-0.035">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.40%); opacity: 0.81" title="-0.072">anything</span><span style="opacity: 0.80"> about </span><span style="background-color: hsl(120, 100.00%, 93.52%); opacity: 0.81" title="0.168">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.11%); opacity: 0.81" title="0.146">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.65%); opacity: 0.81" title="-0.127">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.92%); opacity: 0.82" title="-0.271">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 62.18%); opacity: 0.98" title="-2.081">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 95.09%); opacity: 0.81" title="-0.113">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.22%); opacity: 0.81" title="0.142">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.93%); opacity: 0.81" title="0.058">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 95.42%); opacity: 0.81" title="-0.102">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.08%); opacity: 0.80" title="-0.054">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.27%); opacity: 0.80" title="-0.007">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.83%); opacity: 0.81" title="0.156">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.11%); opacity: 0.80" title="-0.010">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.27%); opacity: 0.80" title="0.049">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.04%); opacity: 0.81" title="0.149">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.79%); opacity: 0.81" title="-0.158">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.83%); opacity: 0.81" title="-0.060">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 95.42%); opacity: 0.81" title="-0.102">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.08%); opacity: 0.80" title="-0.054">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.27%); opacity: 0.80" title="-0.007">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 93.83%); opacity: 0.81" title="0.156">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.11%); opacity: 0.80" title="-0.010">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.60%); opacity: 0.81" title="0.096">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.89%); opacity: 0.81" title="0.059">surgically</span><span style="opacity: 0.80">.
    
    when i </span><span style="background-color: hsl(0, 100.00%, 94.27%); opacity: 0.81" title="-0.140">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.01%); opacity: 0.81" title="0.084">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.89%); opacity: 0.80" title="-0.034">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.10%); opacity: 0.81" title="0.081">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 72.58%); opacity: 0.92" title="1.314">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.75%); opacity: 0.80" title="-0.016">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.00%); opacity: 0.83" title="-0.357">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.166">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.52%); opacity: 0.82" title="0.206">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.78%); opacity: 0.81" title="0.123">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.42%); opacity: 0.80" title="-0.045">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 97.91%); opacity: 0.80" title="-0.033">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.60%); opacity: 0.80" title="-0.040">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.34%); opacity: 0.89" title="-1.001">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 79.69%); opacity: 0.88" title="-0.856">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.167">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.49%); opacity: 0.86" title="-0.637">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 99.04%); opacity: 0.80" title="-0.011">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.71%); opacity: 0.82" title="0.280">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.79%); opacity: 0.81" title="0.158">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.70%); opacity: 0.83" title="-0.325">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.996</b>, score <b>5.826</b>)
    
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
            
                <tr style="background-color: hsl(120, 100.00%, 85.41%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +5.929
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.15%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.103
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as i </span><span style="background-color: hsl(120, 100.00%, 85.84%); opacity: 0.85" title="0.511">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.069">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.094">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.79%); opacity: 0.81" title="-0.158">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.043">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.43%); opacity: 0.89" title="0.995">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.04%); opacity: 0.88" title="0.895">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 96.82%); opacity: 0.81" title="-0.060">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.01%); opacity: 0.81" title="-0.150">isn</span><span style="opacity: 0.80">'t any
    </span><span style="background-color: hsl(120, 100.00%, 79.89%); opacity: 0.87" title="0.844">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.094">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.73%); opacity: 0.82" title="-0.197">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.167">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.87%); opacity: 0.81" title="-0.120">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.19%); opacity: 0.80" title="0.051">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.83%); opacity: 0.81" title="-0.156">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.43%); opacity: 0.80" title="-0.045">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.94%); opacity: 0.81" title="0.152">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.07%); opacity: 0.82" title="0.184">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="2.254">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.057">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.16%); opacity: 0.81" title="-0.079">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.119">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 99.34%); opacity: 0.80" title="-0.006">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.05%); opacity: 0.80" title="-0.030">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.07%); opacity: 0.80" title="0.054">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.095">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.99%); opacity: 0.80" title="-0.012">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.83%); opacity: 0.83" title="-0.319">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.93%); opacity: 0.82" title="-0.229">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.84%); opacity: 0.81" title="0.089">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.46%); opacity: 0.81" title="0.071">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 99.34%); opacity: 0.80" title="-0.006">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.05%); opacity: 0.80" title="-0.030">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.07%); opacity: 0.80" title="0.054">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.095">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.60%); opacity: 0.81" title="-0.096">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.87%); opacity: 0.82" title="-0.192">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.080">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.015">when</span><span style="opacity: 0.80"> i </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.038">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.81%); opacity: 0.80" title="-0.001">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.45%); opacity: 0.80" title="-0.044">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.06%); opacity: 0.81" title="-0.082">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 82.67%); opacity: 0.86" title="-0.682">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.64%); opacity: 0.83" title="0.283">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.16%); opacity: 0.82" title="-0.181">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.37%); opacity: 0.81" title="-0.137">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.49%); opacity: 0.82" title="-0.207">mention</span><span style="opacity: 0.80"> that she'd had </span><span style="background-color: hsl(120, 100.00%, 77.43%); opacity: 0.89" title="0.995">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 79.04%); opacity: 0.88" title="0.895">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.33%); opacity: 0.81" title="0.074">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.57%); opacity: 0.81" title="-0.067">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 96.52%); opacity: 0.81" title="0.069">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.43%); opacity: 0.82" title="-0.209">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.43%); opacity: 0.82" title="-0.209">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.66%); opacity: 0.82" title="0.240">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=soc.religion.christian
        
    </b>
    
        
        (probability <b>0.004</b>, score <b>-5.504</b>)
    
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
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.02%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.342
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 86.76%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -5.162
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 96.03%); opacity: 0.81" title="-0.083">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.96%); opacity: 0.81" title="0.057">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.77%); opacity: 0.82" title="-0.236">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.29%); opacity: 0.81" title="0.140">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.33%); opacity: 0.80" title="-0.007">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.84%); opacity: 0.82" title="0.193">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.11%); opacity: 0.81" title="0.112">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.28%); opacity: 0.87" title="-0.821">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.92%); opacity: 0.86" title="-0.669">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 95.46%); opacity: 0.81" title="0.101">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.75%); opacity: 0.82" title="0.237">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 97.56%); opacity: 0.80" title="-0.042">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.23%); opacity: 0.80" title="0.050">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 84.40%); opacity: 0.85" title="-0.588">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.062">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.55%); opacity: 0.83" title="0.287">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.26%); opacity: 0.82" title="0.257">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.67%); opacity: 0.82" title="0.200">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.71%); opacity: 0.80" title="-0.017">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.82%); opacity: 0.81" title="0.156">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.37%); opacity: 0.80" title="-0.006">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.45%); opacity: 0.81" title="-0.134">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.84%); opacity: 0.82" title="-0.233">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 66.34%); opacity: 0.96" title="-1.762">pain</span><span style="opacity: 0.80">.
    
    either </span><span style="background-color: hsl(0, 100.00%, 96.54%); opacity: 0.81" title="-0.068">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.45%); opacity: 0.82" title="0.170">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.73%); opacity: 0.80" title="-0.037">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.58%); opacity: 0.81" title="0.130">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.05%); opacity: 0.80" title="0.054">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.128">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.24%); opacity: 0.80" title="-0.008">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.62%); opacity: 0.83" title="0.374">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.87%); opacity: 0.81" title="0.155">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.40%); opacity: 0.81" title="-0.136">with</span><span style="opacity: 0.80"> sound, </span><span style="background-color: hsl(0, 100.00%, 97.73%); opacity: 0.80" title="-0.037">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.58%); opacity: 0.81" title="0.130">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.05%); opacity: 0.80" title="0.054">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.128">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.48%); opacity: 0.82" title="0.207">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.03%); opacity: 0.82" title="0.266">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.04%); opacity: 0.81" title="-0.114">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 96.20%); opacity: 0.81" title="-0.078">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.57%); opacity: 0.81" title="-0.068">i</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(0, 100.00%, 95.13%); opacity: 0.81" title="-0.111">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 96.55%); opacity: 0.81" title="0.068">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.48%); opacity: 0.82" title="0.207">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 93.62%); opacity: 0.81" title="0.164">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.32%); opacity: 0.82" title="-0.175">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.439">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.59%); opacity: 0.82" title="0.203">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.20%); opacity: 0.83" title="0.347">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.56%); opacity: 0.81" title="0.098">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.13%); opacity: 0.80" title="-0.010">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.011">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.37%); opacity: 0.80" title="0.046">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.28%); opacity: 0.87" title="-0.821">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 82.92%); opacity: 0.86" title="-0.669">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.90%); opacity: 0.80" title="-0.034">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.57%); opacity: 0.84" title="0.474">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 94.16%); opacity: 0.81" title="-0.144">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.01%); opacity: 0.81" title="0.084">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.083">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.61%); opacity: 0.81" title="-0.129">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.08%); opacity: 0.81" title="-0.147">less</span><span style="opacity: 0.80">.</span>
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

    0.065 alt.atheism
    0.145 comp.graphics
    0.376 sci.med
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
            random_state=<mtrand.RandomState object at 0x10e1dcf78>,
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

    {'mean_KL_divergence': 0.020120624088861134, 'score': 0.98625304704899297}



-  ‘score’ is an accuracy score weighted by cosine distance between
   generated sample and the original document (i.e. texts which are
   closer to the example are more important). Accuracy shows how good
   are ‘top 1’ predictions.
-  ‘mean_KL_divergence’ is a mean `Kullback–Leibler
   divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__
   for all target classes; it is also weighted by distance. KL
   divergence shows how well are probabilities approximated; 0.0 means a
   perfect match.

In this example both accuracy and KL divergence are good; it means our
white-box classifier usually assigns the same labels as the black-box
classifier on the dataset we generated, and its predicted probabilities
are close to those predicted by our LSA+SVM pipeline. So it is likely
(though not guaranteed, we’ll discuss it later) that the explanation is
correct and can be trusted.

When working with LIME (e.g. via :class:`~.TextExplainer`) it is always a good
idea to check these scores. If they are not good then you can tell that
something is not right.

Let’s make it fail
------------------

By default :class:`~.TextExplainer` uses a very basic text processing pipeline:
Logistic Regression trained on bag-of-words and bag-of-bigrams features
(see ``te.clf_`` and ``te.vec_`` attributes). It limits a set of
black-box classifiers it can explain: because the text is seen as “bag
of words/ngrams”, the default white-box pipeline can’t distinguish
e.g. between the same word in the beginning of the document and in the
end of the document. Bigrams help to alleviate the problem in practice,
but not completely.

Black-box classifiers which use features like “text length” (not
directly related to tokens) can be also hard to approximate using the
default bag-of-words/ngrams model.

This kind of failure is usually detectable though - scores (accuracy and
KL divergence) will be low. Let’s check it on a completely synthetic
example - a black-box classifier which assigns a class based on oddity
of document length and on a presence of ‘medication’ word.

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
    
        
        (probability <b>0.989</b>, score <b>4.466</b>)
    
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
            +4.576
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.53%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.110
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.035">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.78%); opacity: 0.81" title="0.335">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.49%); opacity: 0.81" title="0.272">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.71%); opacity: 0.81" title="0.172">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.30%); opacity: 0.80" title="0.067">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.64%); opacity: 0.80" title="-0.049">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.67%); opacity: 0.80" title="0.105">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.56%); opacity: 0.81" title="-0.265">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.07%); opacity: 0.81" title="-0.308">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.97%); opacity: 0.80" title="-0.087">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.28%); opacity: 0.81" title="-0.381">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 97.88%); opacity: 0.80" title="-0.092">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.01%); opacity: 0.84" title="1.229">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="6.130">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.77%); opacity: 0.83" title="0.998">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.25%); opacity: 0.81" title="-0.292">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.124">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.204">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.029">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.17%); opacity: 0.80" title="-0.024">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.85%); opacity: 0.81" title="0.162">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.16%); opacity: 0.80" title="-0.025">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.44%); opacity: 0.81" title="-0.193">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.22%); opacity: 0.81" title="-0.211">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 95.69%); opacity: 0.81" title="-0.254">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.26%); opacity: 0.80" title="-0.021">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.64%); opacity: 0.81" title="-0.259">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.51%); opacity: 0.80" title="-0.116">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.35%); opacity: 0.81" title="-0.201">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.049">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.96%); opacity: 0.80" title="-0.000">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.64%); opacity: 0.81" title="-0.178">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.74%); opacity: 0.81" title="-0.338">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.07%); opacity: 0.80" title="-0.081">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.84%); opacity: 0.80" title="-0.095">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.76%); opacity: 0.80" title="-0.043">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.91%); opacity: 0.80" title="-0.036">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.35%); opacity: 0.81" title="-0.201">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.049">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 99.96%); opacity: 0.80" title="-0.000">to</span><span style="opacity: 0.80"> be </span><span style="background-color: hsl(0, 100.00%, 99.62%); opacity: 0.80" title="-0.008">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.023">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 96.16%); opacity: 0.81" title="0.215">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.55%); opacity: 0.81" title="0.266">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.88%); opacity: 0.81" title="0.160">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.42%); opacity: 0.81" title="-0.277">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.135">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.318">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 96.60%); opacity: 0.81" title="0.181">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.81%); opacity: 0.80" title="-0.040">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.105">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.98%); opacity: 0.81" title="-0.230">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.46%); opacity: 0.81" title="-0.274">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.92%); opacity: 0.81" title="0.157">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.29%); opacity: 0.80" title="0.131">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 97.66%); opacity: 0.80" title="0.106">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.90%); opacity: 0.80" title="-0.036">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.124">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 98.26%); opacity: 0.80" title="-0.069">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.47%); opacity: 0.80" title="0.119">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.043">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 94.18%); opacity: 0.81" title="-0.391">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.004">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.005">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.32%); opacity: 0.81" title="-0.286">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.67%); opacity: 0.80" title="0.106">less</span><span style="opacity: 0.80">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




:class:`~.TextExplainer` correctly figured out that ‘medication’ is important,
but failed to account for “len(doc) % 2” condition, so the explanation
is incomplete. We can detect this failure by looking at metrics - they
are low:

.. code:: ipython3

    te3.metrics_




.. parsed-literal::

    {'mean_KL_divergence': 0.3312922355257879, 'score': 0.79050673156810314}



If (a big if…) we suspect that the fact document length is even or odd
is important, it is possible to customize :class:`~.TextExplainer` to check
this hypothesis.

To do that, we need to create a vectorizer which returns both “is odd”
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

    {'mean_KL_divergence': 0.024826114773734968, 'score': 1.0}




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
    
        
        (probability <b>0.996</b>, score <b>5.511</b>)
    
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
            +8.590
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            countvectorizer: Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.51%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.043
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 90.34%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.037
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            doclength__is_even
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <b>countvectorizer:</b> <span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.070">as</span><span style="opacity: 0.80"> i recall from my </span><span style="background-color: hsl(120, 100.00%, 98.26%); opacity: 0.80" title="0.088">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.01%); opacity: 0.80" title="0.106">with</span><span style="opacity: 0.80"> kidney stones, there </span><span style="background-color: hsl(0, 100.00%, 98.57%); opacity: 0.80" title="-0.066">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(120, 100.00%, 89.59%); opacity: 0.83" title="1.130">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="7.730">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.85%); opacity: 0.83" title="1.246">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.026">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.00%); opacity: 0.80" title="0.107">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.057">anything</span><span style="opacity: 0.80"> about </span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.011">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.011">except</span><span style="opacity: 0.80"> relieve </span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.017">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.34%); opacity: 0.80" title="0.082">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.008">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.045">they</span><span style="opacity: 0.80"> pass, </span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.048">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.045">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.008">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.004">to</span><span style="opacity: 0.80"> be broken </span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.062">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.77%); opacity: 0.80" title="0.125">with</span><span style="opacity: 0.80"> sound, </span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.048">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.045">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.008">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.004">to</span><span style="opacity: 0.80"> be extracted </span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.041">surgically</span><span style="opacity: 0.80">.
    
    when i </span><span style="background-color: hsl(120, 100.00%, 98.02%); opacity: 0.80" title="0.105">was</span><span style="opacity: 0.80"> in, </span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.017">the</span><span style="opacity: 0.80"> x-ray tech </span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.041">happened</span><span style="opacity: 0.80"> to mention </span><span style="background-color: hsl(0, 100.00%, 97.83%); opacity: 0.80" title="-0.120">that</span><span style="opacity: 0.80"> she'd </span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.017">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.017">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.062">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.22%); opacity: 0.80" title="0.091">and</span><span style="opacity: 0.80"> children, </span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.029">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.100">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.96%); opacity: 0.81" title="0.194">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.116">hurt</span><span style="opacity: 0.80"> less</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Much better! It was a toy example, but the idea stands - if you think
something could be important, add it to the mix as a feature for
:class:`~.TextExplainer`.

Let’s make it fail, again
-------------------------

Another possible issue is the dataset generation method. Not only
feature extraction should be powerful enough, but auto-generated texts
also should be diverse enough.

:class:`~.TextExplainer` removes random words by default, so by default it
can’t e.g. provide a good explanation for a black-box classifier which
works on character level. Let’s try to use :class:`~.TextExplainer` to explain
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

    0.88082556591211714



This pipeline is supported by eli5 directly, so in practice there is no
need to use :class:`~.TextExplainer` for it. We’re using this pipeline as an
example - it is possible check the “true” explanation first, without
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
    
        
        (probability <b>0.565</b>, score <b>-0.037</b>)
    
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
            
                <tr style="background-color: hsl(120, 100.00%, 80.53%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.943
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.980
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 97.35%); opacity: 0.80" title="-0.003">as</span><span style="background-color: hsl(0, 100.00%, 94.76%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 96.24%); opacity: 0.81" title="-0.004">i</span><span style="background-color: hsl(120, 100.00%, 91.80%); opacity: 0.82" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 88.52%); opacity: 0.83" title="0.021">r</span><span style="background-color: hsl(120, 100.00%, 84.23%); opacity: 0.85" title="0.033">e</span><span style="background-color: hsl(120, 100.00%, 81.87%); opacity: 0.86" title="0.040">c</span><span style="background-color: hsl(120, 100.00%, 86.63%); opacity: 0.84" title="0.026">a</span><span style="background-color: hsl(120, 100.00%, 89.66%); opacity: 0.83" title="0.018">l</span><span style="background-color: hsl(120, 100.00%, 93.01%); opacity: 0.82" title="0.010">l</span><span style="background-color: hsl(0, 100.00%, 99.91%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(0, 100.00%, 96.53%); opacity: 0.81" title="-0.004">f</span><span style="background-color: hsl(0, 100.00%, 95.98%); opacity: 0.81" title="-0.005">ro</span><span style="background-color: hsl(0, 100.00%, 95.73%); opacity: 0.81" title="-0.005">m</span><span style="background-color: hsl(120, 100.00%, 87.08%); opacity: 0.84" title="0.025"> </span><span style="background-color: hsl(120, 100.00%, 86.68%); opacity: 0.84" title="0.026">my</span><span style="background-color: hsl(120, 100.00%, 87.81%); opacity: 0.84" title="0.023"> </span><span style="background-color: hsl(120, 100.00%, 96.69%); opacity: 0.81" title="0.004">b</span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.005">ou</span><span style="background-color: hsl(120, 100.00%, 94.15%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(120, 100.00%, 93.92%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 94.35%); opacity: 0.81" title="0.008">w</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">it</span><span style="background-color: hsl(0, 100.00%, 99.27%); opacity: 0.80" title="-0.000">h</span><span style="background-color: hsl(0, 100.00%, 96.27%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 95.35%); opacity: 0.81" title="0.006">k</span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.009">i</span><span style="background-color: hsl(120, 100.00%, 92.21%); opacity: 0.82" title="0.012">d</span><span style="background-color: hsl(120, 100.00%, 92.15%); opacity: 0.82" title="0.012">n</span><span style="background-color: hsl(120, 100.00%, 93.53%); opacity: 0.81" title="0.009">e</span><span style="background-color: hsl(120, 100.00%, 95.31%); opacity: 0.81" title="0.006">y</span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(0, 100.00%, 99.84%); opacity: 0.80" title="-0.000">s</span><span style="background-color: hsl(0, 100.00%, 98.13%); opacity: 0.80" title="-0.002">t</span><span style="background-color: hsl(0, 100.00%, 98.01%); opacity: 0.80" title="-0.002">o</span><span style="background-color: hsl(0, 100.00%, 98.71%); opacity: 0.80" title="-0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.65%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 96.77%); opacity: 0.81" title="0.003">s</span><span style="background-color: hsl(120, 100.00%, 97.27%); opacity: 0.80" title="0.003">,</span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 93.74%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(0, 100.00%, 88.37%); opacity: 0.83" title="-0.021">h</span><span style="background-color: hsl(0, 100.00%, 87.07%); opacity: 0.84" title="-0.025">e</span><span style="background-color: hsl(0, 100.00%, 88.62%); opacity: 0.83" title="-0.020">r</span><span style="background-color: hsl(0, 100.00%, 90.38%); opacity: 0.83" title="-0.016">e</span><span style="background-color: hsl(0, 100.00%, 93.97%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 94.44%); opacity: 0.81" title="-0.007">i</span><span style="background-color: hsl(0, 100.00%, 93.36%); opacity: 0.82" title="-0.009">s</span><span style="background-color: hsl(0, 100.00%, 92.04%); opacity: 0.82" title="-0.012">n</span><span style="background-color: hsl(0, 100.00%, 92.67%); opacity: 0.82" title="-0.011">'</span><span style="background-color: hsl(0, 100.00%, 94.67%); opacity: 0.81" title="-0.007">t</span><span style="background-color: hsl(0, 100.00%, 93.16%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 97.67%); opacity: 0.80" title="-0.002">any</span><span style="background-color: hsl(120, 100.00%, 80.94%); opacity: 0.87" title="0.043">
    </span><span style="background-color: hsl(120, 100.00%, 72.08%); opacity: 0.92" title="0.074">m</span><span style="background-color: hsl(120, 100.00%, 64.48%); opacity: 0.97" title="0.104">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.123">d</span><span style="background-color: hsl(120, 100.00%, 63.96%); opacity: 0.97" title="0.106">i</span><span style="background-color: hsl(120, 100.00%, 72.40%); opacity: 0.92" title="0.073">c</span><span style="background-color: hsl(120, 100.00%, 84.55%); opacity: 0.85" title="0.032">a</span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.013">t</span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.007">i</span><span style="background-color: hsl(0, 100.00%, 93.57%); opacity: 0.81" title="-0.009">o</span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.007">n</span><span style="background-color: hsl(0, 100.00%, 90.46%); opacity: 0.83" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 92.02%); opacity: 0.82" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 90.79%); opacity: 0.82" title="-0.015">ha</span><span style="background-color: hsl(0, 100.00%, 93.22%); opacity: 0.82" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 98.86%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(120, 100.00%, 94.31%); opacity: 0.81" title="0.008">can</span><span style="background-color: hsl(0, 100.00%, 93.68%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 93.74%); opacity: 0.81" title="-0.009">do</span><span style="background-color: hsl(0, 100.00%, 90.05%); opacity: 0.83" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 92.35%); opacity: 0.82" title="-0.012">a</span><span style="background-color: hsl(0, 100.00%, 90.86%); opacity: 0.82" title="-0.015">n</span><span style="background-color: hsl(0, 100.00%, 88.83%); opacity: 0.83" title="-0.020">y</span><span style="background-color: hsl(0, 100.00%, 95.16%); opacity: 0.81" title="-0.006">t</span><span style="background-color: hsl(0, 100.00%, 96.05%); opacity: 0.81" title="-0.005">h</span><span style="background-color: hsl(0, 100.00%, 94.61%); opacity: 0.81" title="-0.007">i</span><span style="background-color: hsl(0, 100.00%, 97.21%); opacity: 0.80" title="-0.003">n</span><span style="background-color: hsl(0, 100.00%, 94.41%); opacity: 0.81" title="-0.007">g</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 91.81%); opacity: 0.82" title="0.013">a</span><span style="background-color: hsl(120, 100.00%, 89.05%); opacity: 0.83" title="0.019">b</span><span style="background-color: hsl(120, 100.00%, 88.52%); opacity: 0.83" title="0.021">o</span><span style="background-color: hsl(120, 100.00%, 89.69%); opacity: 0.83" title="0.018">u</span><span style="background-color: hsl(120, 100.00%, 92.68%); opacity: 0.82" title="0.011">t</span><span style="background-color: hsl(0, 100.00%, 97.54%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(0, 100.00%, 91.26%); opacity: 0.82" title="-0.014">t</span><span style="background-color: hsl(0, 100.00%, 90.05%); opacity: 0.83" title="-0.017">he</span><span style="background-color: hsl(0, 100.00%, 91.81%); opacity: 0.82" title="-0.013">m</span><span style="background-color: hsl(0, 100.00%, 97.97%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.34%); opacity: 0.81" title="0.008">e</span><span style="background-color: hsl(120, 100.00%, 92.99%); opacity: 0.82" title="0.010">x</span><span style="background-color: hsl(120, 100.00%, 94.49%); opacity: 0.81" title="0.007">c</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.008">e</span><span style="background-color: hsl(120, 100.00%, 97.10%); opacity: 0.80" title="0.003">p</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(0, 100.00%, 90.41%); opacity: 0.83" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 86.96%); opacity: 0.84" title="-0.025">r</span><span style="background-color: hsl(0, 100.00%, 83.89%); opacity: 0.85" title="-0.034">e</span><span style="background-color: hsl(0, 100.00%, 83.05%); opacity: 0.86" title="-0.036">l</span><span style="background-color: hsl(0, 100.00%, 85.96%); opacity: 0.84" title="-0.028">i</span><span style="background-color: hsl(0, 100.00%, 94.81%); opacity: 0.81" title="-0.007">e</span><span style="background-color: hsl(0, 100.00%, 97.71%); opacity: 0.80" title="-0.002">v</span><span style="background-color: hsl(120, 100.00%, 97.71%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(0, 100.00%, 93.67%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 86.81%); opacity: 0.84" title="-0.025">the</span><span style="background-color: hsl(0, 100.00%, 94.68%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(120, 100.00%, 87.32%); opacity: 0.84" title="0.024">p</span><span style="background-color: hsl(120, 100.00%, 85.78%); opacity: 0.85" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 83.84%); opacity: 0.85" title="0.034">i</span><span style="background-color: hsl(120, 100.00%, 86.47%); opacity: 0.84" title="0.026">n</span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.011">.</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 92.45%); opacity: 0.82" title="-0.011">e</span><span style="background-color: hsl(0, 100.00%, 90.28%); opacity: 0.83" title="-0.016">i</span><span style="background-color: hsl(0, 100.00%, 84.86%); opacity: 0.85" title="-0.031">t</span><span style="background-color: hsl(0, 100.00%, 84.07%); opacity: 0.85" title="-0.033">h</span><span style="background-color: hsl(0, 100.00%, 86.87%); opacity: 0.84" title="-0.025">e</span><span style="background-color: hsl(0, 100.00%, 88.48%); opacity: 0.83" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 92.92%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(120, 100.00%, 96.52%); opacity: 0.81" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 94.75%); opacity: 0.81" title="0.007">he</span><span style="background-color: hsl(120, 100.00%, 92.68%); opacity: 0.82" title="0.011">y</span><span style="background-color: hsl(0, 100.00%, 98.81%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.011">p</span><span style="background-color: hsl(0, 100.00%, 91.72%); opacity: 0.82" title="-0.013">a</span><span style="background-color: hsl(0, 100.00%, 91.80%); opacity: 0.82" title="-0.013">s</span><span style="background-color: hsl(0, 100.00%, 93.62%); opacity: 0.81" title="-0.009">s</span><span style="background-color: hsl(0, 100.00%, 98.37%); opacity: 0.80" title="-0.001">,</span><span style="background-color: hsl(0, 100.00%, 96.52%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 96.89%); opacity: 0.81" title="-0.003">or</span><span style="background-color: hsl(0, 100.00%, 95.70%); opacity: 0.81" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 96.52%); opacity: 0.81" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 94.75%); opacity: 0.81" title="0.007">he</span><span style="background-color: hsl(120, 100.00%, 92.68%); opacity: 0.82" title="0.011">y</span><span style="background-color: hsl(0, 100.00%, 96.48%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 89.48%); opacity: 0.83" title="-0.018">h</span><span style="background-color: hsl(0, 100.00%, 86.54%); opacity: 0.84" title="-0.026">av</span><span style="background-color: hsl(0, 100.00%, 87.98%); opacity: 0.84" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 97.16%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 94.17%); opacity: 0.81" title="0.008">to</span><span style="background-color: hsl(120, 100.00%, 96.24%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 96.59%); opacity: 0.81" title="-0.004">be</span><span style="background-color: hsl(0, 100.00%, 96.50%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 99.50%); opacity: 0.80" title="-0.000">b</span><span style="background-color: hsl(0, 100.00%, 98.89%); opacity: 0.80" title="-0.001">r</span><span style="background-color: hsl(0, 100.00%, 98.42%); opacity: 0.80" title="-0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">k</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.32%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 95.69%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 97.04%); opacity: 0.80" title="0.003">up</span><span style="background-color: hsl(120, 100.00%, 94.59%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 94.35%); opacity: 0.81" title="0.008">w</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">it</span><span style="background-color: hsl(0, 100.00%, 99.27%); opacity: 0.80" title="-0.000">h</span><span style="background-color: hsl(0, 100.00%, 97.57%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 90.09%); opacity: 0.83" title="0.017">s</span><span style="background-color: hsl(120, 100.00%, 88.43%); opacity: 0.83" title="0.021">o</span><span style="background-color: hsl(120, 100.00%, 89.17%); opacity: 0.83" title="0.019">u</span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.013">n</span><span style="background-color: hsl(120, 100.00%, 97.32%); opacity: 0.80" title="0.003">d</span><span style="background-color: hsl(0, 100.00%, 94.11%); opacity: 0.81" title="-0.008">,</span><span style="background-color: hsl(0, 100.00%, 93.44%); opacity: 0.82" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 96.89%); opacity: 0.81" title="-0.003">or</span><span style="background-color: hsl(0, 100.00%, 95.70%); opacity: 0.81" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 96.52%); opacity: 0.81" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 94.75%); opacity: 0.81" title="0.007">he</span><span style="background-color: hsl(120, 100.00%, 92.68%); opacity: 0.82" title="0.011">y</span><span style="background-color: hsl(0, 100.00%, 96.48%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 89.48%); opacity: 0.83" title="-0.018">h</span><span style="background-color: hsl(0, 100.00%, 86.54%); opacity: 0.84" title="-0.026">av</span><span style="background-color: hsl(0, 100.00%, 87.98%); opacity: 0.84" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 97.16%); opacity: 0.80" title="-0.003">
    </span><span style="background-color: hsl(120, 100.00%, 94.17%); opacity: 0.81" title="0.008">to</span><span style="background-color: hsl(120, 100.00%, 96.24%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 96.59%); opacity: 0.81" title="-0.004">be</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 91.27%); opacity: 0.82" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 88.76%); opacity: 0.83" title="0.020">x</span><span style="background-color: hsl(120, 100.00%, 88.61%); opacity: 0.83" title="0.020">t</span><span style="background-color: hsl(120, 100.00%, 88.65%); opacity: 0.83" title="0.020">r</span><span style="background-color: hsl(120, 100.00%, 90.01%); opacity: 0.83" title="0.017">a</span><span style="background-color: hsl(120, 100.00%, 90.39%); opacity: 0.83" title="0.016">c</span><span style="background-color: hsl(120, 100.00%, 87.14%); opacity: 0.84" title="0.024">t</span><span style="background-color: hsl(120, 100.00%, 88.22%); opacity: 0.83" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 89.66%); opacity: 0.83" title="0.018">d</span><span style="background-color: hsl(120, 100.00%, 84.79%); opacity: 0.85" title="0.031"> </span><span style="background-color: hsl(120, 100.00%, 85.18%); opacity: 0.85" title="0.030">s</span><span style="background-color: hsl(120, 100.00%, 84.33%); opacity: 0.85" title="0.032">u</span><span style="background-color: hsl(120, 100.00%, 82.67%); opacity: 0.86" title="0.037">r</span><span style="background-color: hsl(120, 100.00%, 88.47%); opacity: 0.83" title="0.021">g</span><span style="background-color: hsl(120, 100.00%, 88.75%); opacity: 0.83" title="0.020">i</span><span style="background-color: hsl(120, 100.00%, 87.30%); opacity: 0.84" title="0.024">c</span><span style="background-color: hsl(120, 100.00%, 86.48%); opacity: 0.84" title="0.026">a</span><span style="background-color: hsl(120, 100.00%, 86.79%); opacity: 0.84" title="0.025">l</span><span style="background-color: hsl(120, 100.00%, 91.37%); opacity: 0.82" title="0.014">l</span><span style="background-color: hsl(120, 100.00%, 94.49%); opacity: 0.81" title="0.007">y</span><span style="background-color: hsl(0, 100.00%, 97.95%); opacity: 0.80" title="-0.002">.</span><span style="background-color: hsl(120, 100.00%, 93.50%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.021">w</span><span style="background-color: hsl(120, 100.00%, 87.32%); opacity: 0.84" title="0.024">he</span><span style="background-color: hsl(120, 100.00%, 89.01%); opacity: 0.83" title="0.019">n</span><span style="background-color: hsl(120, 100.00%, 96.27%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 96.24%); opacity: 0.81" title="-0.004">i</span><span style="background-color: hsl(0, 100.00%, 95.12%); opacity: 0.81" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 94.03%); opacity: 0.81" title="-0.008">w</span><span style="background-color: hsl(0, 100.00%, 96.42%); opacity: 0.81" title="-0.004">as</span><span style="background-color: hsl(0, 100.00%, 97.28%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 97.98%); opacity: 0.80" title="0.002">in,</span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 86.81%); opacity: 0.84" title="-0.025">the</span><span style="background-color: hsl(0, 100.00%, 88.54%); opacity: 0.83" title="-0.021"> </span><span style="background-color: hsl(120, 100.00%, 98.38%); opacity: 0.80" title="0.001">x</span><span style="background-color: hsl(120, 100.00%, 98.32%); opacity: 0.80" title="0.001">-</span><span style="background-color: hsl(0, 100.00%, 96.70%); opacity: 0.81" title="-0.003">r</span><span style="background-color: hsl(0, 100.00%, 96.52%); opacity: 0.81" title="-0.004">a</span><span style="background-color: hsl(0, 100.00%, 96.06%); opacity: 0.81" title="-0.004">y</span><span style="background-color: hsl(0, 100.00%, 94.40%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.005">t</span><span style="background-color: hsl(0, 100.00%, 96.00%); opacity: 0.81" title="-0.005">ec</span><span style="background-color: hsl(0, 100.00%, 96.88%); opacity: 0.81" title="-0.003">h</span><span style="background-color: hsl(120, 100.00%, 97.57%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.73%); opacity: 0.81" title="0.007">h</span><span style="background-color: hsl(120, 100.00%, 92.06%); opacity: 0.82" title="0.012">a</span><span style="background-color: hsl(120, 100.00%, 90.78%); opacity: 0.82" title="0.015">p</span><span style="background-color: hsl(120, 100.00%, 92.56%); opacity: 0.82" title="0.011">p</span><span style="background-color: hsl(120, 100.00%, 95.73%); opacity: 0.81" title="0.005">e</span><span style="background-color: hsl(0, 100.00%, 97.09%); opacity: 0.80" title="-0.003">n</span><span style="background-color: hsl(0, 100.00%, 93.40%); opacity: 0.82" title="-0.009">e</span><span style="background-color: hsl(0, 100.00%, 94.55%); opacity: 0.81" title="-0.007">d</span><span style="background-color: hsl(120, 100.00%, 96.02%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 94.17%); opacity: 0.81" title="0.008">to</span><span style="background-color: hsl(120, 100.00%, 94.08%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(0, 100.00%, 99.47%); opacity: 0.80" title="-0.000">m</span><span style="background-color: hsl(120, 100.00%, 90.01%); opacity: 0.83" title="0.017">e</span><span style="background-color: hsl(120, 100.00%, 90.01%); opacity: 0.83" title="0.017">n</span><span style="background-color: hsl(120, 100.00%, 87.44%); opacity: 0.84" title="0.024">t</span><span style="background-color: hsl(120, 100.00%, 92.34%); opacity: 0.82" title="0.012">i</span><span style="background-color: hsl(0, 100.00%, 96.05%); opacity: 0.81" title="-0.005">o</span><span style="background-color: hsl(0, 100.00%, 96.21%); opacity: 0.81" title="-0.004">n</span><span style="background-color: hsl(0, 100.00%, 90.46%); opacity: 0.83" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 92.02%); opacity: 0.82" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 90.79%); opacity: 0.82" title="-0.015">ha</span><span style="background-color: hsl(0, 100.00%, 93.22%); opacity: 0.82" title="-0.010">t</span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 90.83%); opacity: 0.82" title="0.015">s</span><span style="background-color: hsl(120, 100.00%, 90.72%); opacity: 0.82" title="0.015">h</span><span style="background-color: hsl(120, 100.00%, 91.17%); opacity: 0.82" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.001">'</span><span style="background-color: hsl(0, 100.00%, 98.85%); opacity: 0.80" title="-0.001">d</span><span style="background-color: hsl(120, 100.00%, 91.11%); opacity: 0.82" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 87.62%); opacity: 0.84" title="0.023">had</span><span style="background-color: hsl(120, 100.00%, 89.76%); opacity: 0.83" title="0.018"> </span><span style="background-color: hsl(120, 100.00%, 95.35%); opacity: 0.81" title="0.006">k</span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.009">i</span><span style="background-color: hsl(120, 100.00%, 92.21%); opacity: 0.82" title="0.012">d</span><span style="background-color: hsl(120, 100.00%, 92.15%); opacity: 0.82" title="0.012">n</span><span style="background-color: hsl(120, 100.00%, 93.53%); opacity: 0.81" title="0.009">e</span><span style="background-color: hsl(120, 100.00%, 95.31%); opacity: 0.81" title="0.006">y</span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.003">
    </span><span style="background-color: hsl(0, 100.00%, 99.84%); opacity: 0.80" title="-0.000">s</span><span style="background-color: hsl(0, 100.00%, 98.13%); opacity: 0.80" title="-0.002">t</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 97.31%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(120, 100.00%, 97.08%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 95.55%); opacity: 0.81" title="0.005">s</span><span style="background-color: hsl(120, 100.00%, 93.65%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 92.80%); opacity: 0.82" title="0.011">and</span><span style="background-color: hsl(120, 100.00%, 90.87%); opacity: 0.82" title="0.015"> </span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.003">c</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(0, 100.00%, 97.72%); opacity: 0.80" title="-0.002">i</span><span style="background-color: hsl(0, 100.00%, 93.76%); opacity: 0.81" title="-0.009">l</span><span style="background-color: hsl(0, 100.00%, 94.95%); opacity: 0.81" title="-0.006">d</span><span style="background-color: hsl(0, 100.00%, 95.23%); opacity: 0.81" title="-0.006">r</span><span style="background-color: hsl(0, 100.00%, 94.38%); opacity: 0.81" title="-0.007">e</span><span style="background-color: hsl(0, 100.00%, 95.47%); opacity: 0.81" title="-0.005">n</span><span style="background-color: hsl(0, 100.00%, 96.11%); opacity: 0.81" title="-0.004">,</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 92.80%); opacity: 0.82" title="0.011">and</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(0, 100.00%, 86.81%); opacity: 0.84" title="-0.025">the</span><span style="background-color: hsl(0, 100.00%, 89.55%); opacity: 0.83" title="-0.018"> </span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.003">c</span><span style="background-color: hsl(120, 100.00%, 98.21%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 97.79%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(0, 100.00%, 97.42%); opacity: 0.80" title="-0.002">l</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 96.46%); opacity: 0.81" title="0.004">b</span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.006">i</span><span style="background-color: hsl(120, 100.00%, 95.53%); opacity: 0.81" title="0.005">r</span><span style="background-color: hsl(120, 100.00%, 95.69%); opacity: 0.81" title="0.005">t</span><span style="background-color: hsl(120, 100.00%, 96.63%); opacity: 0.81" title="0.004">h</span><span style="background-color: hsl(120, 100.00%, 97.95%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 98.00%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 98.25%); opacity: 0.80" title="0.001">ur</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 95.08%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 93.31%); opacity: 0.82" title="0.010">l</span><span style="background-color: hsl(120, 100.00%, 91.71%); opacity: 0.82" title="0.013">e</span><span style="background-color: hsl(120, 100.00%, 91.76%); opacity: 0.82" title="0.013">s</span><span style="background-color: hsl(120, 100.00%, 93.47%); opacity: 0.82" title="0.009">s</span><span style="background-color: hsl(120, 100.00%, 96.45%); opacity: 0.81" title="0.004">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




:class:`~.TextExplainer` produces a different result:

.. code:: ipython3

    te = TextExplainer(random_state=42).fit(doc, pipe_char.predict_proba)
    print(te.metrics_)
    te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)


.. parsed-literal::

    {'mean_KL_divergence': 0.020247299052285436, 'score': 0.92434669226497945}




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
    
        
        (probability <b>0.576</b>, score <b>0.621</b>)
    
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
            +0.972
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 90.20%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.351
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as </span><span style="background-color: hsl(0, 100.00%, 96.66%); opacity: 0.81" title="-0.021">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.34%); opacity: 0.84" title="0.144">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.49%); opacity: 0.80" title="-0.014">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.18%); opacity: 0.81" title="0.026">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.90%); opacity: 0.81" title="-0.019">bout</span><span style="opacity: 0.80"> with </span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.074">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.56%); opacity: 0.81" title="0.022">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 91.09%); opacity: 0.82" title="-0.087">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.34%); opacity: 0.82" title="0.070">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 92.34%); opacity: 0.82" title="0.070">t</span><span style="opacity: 0.80"> any
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.746">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.07%); opacity: 0.84" title="-0.149">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.90%); opacity: 0.84" title="0.151">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.14%); opacity: 0.84" title="0.131">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.05%); opacity: 0.80" title="-0.010">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.44%); opacity: 0.82" title="0.056">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.42%); opacity: 0.85" title="-0.176">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.005">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.13%); opacity: 0.85" title="-0.199">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.00%); opacity: 0.84" title="-0.166">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.55%); opacity: 0.87" title="0.247">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 89.10%); opacity: 0.83" title="-0.116">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.95%); opacity: 0.80" title="-0.011">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.95%); opacity: 0.82" title="-0.062">pass</span><span style="opacity: 0.80">, or </span><span style="background-color: hsl(0, 100.00%, 97.95%); opacity: 0.80" title="-0.011">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.43%); opacity: 0.83" title="-0.097">have</span><span style="opacity: 0.80"> to be </span><span style="background-color: hsl(0, 100.00%, 88.78%); opacity: 0.83" title="-0.121">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.01%); opacity: 0.82" title="-0.075">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.14%); opacity: 0.82" title="-0.073">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.81%); opacity: 0.81" title="-0.020">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 93.73%); opacity: 0.81" title="0.053">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.95%); opacity: 0.80" title="-0.011">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.43%); opacity: 0.83" title="-0.097">have</span><span style="opacity: 0.80">
    to be </span><span style="background-color: hsl(120, 100.00%, 81.73%); opacity: 0.87" title="0.243">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.81%); opacity: 0.90" title="0.385">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 88.05%); opacity: 0.84" title="0.133">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.66%); opacity: 0.81" title="-0.021">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.78%); opacity: 0.81" title="-0.030">was</span><span style="opacity: 0.80"> in, </span><span style="background-color: hsl(0, 100.00%, 86.00%); opacity: 0.84" title="-0.166">the</span><span style="opacity: 0.80"> x-ray </span><span style="background-color: hsl(120, 100.00%, 93.90%); opacity: 0.81" title="0.051">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.90%); opacity: 0.81" title="0.051">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.98%); opacity: 0.82" title="0.062">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.79%); opacity: 0.86" title="0.205">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.98%); opacity: 0.85" title="-0.202">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.84%); opacity: 0.81" title="0.020">she</span><span style="opacity: 0.80">'d </span><span style="background-color: hsl(120, 100.00%, 85.76%); opacity: 0.85" title="0.171">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.074">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 96.56%); opacity: 0.81" title="0.022">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.41%); opacity: 0.82" title="0.083">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.58%); opacity: 0.83" title="-0.094">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 91.41%); opacity: 0.82" title="0.083">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.86%); opacity: 0.83" title="-0.105">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.075">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.59%); opacity: 0.81" title="0.054">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.57%); opacity: 0.82" title="0.067">less</span><span style="opacity: 0.80">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Scores look OK but not great; the explanation kind of makes sense on a
first sight, but we know that the classifier works in a different way.

To explain such black-box classifiers we need to change both dataset
generation method (change/remove individual characters, not only words)
and feature extraction method (e.g. use char ngrams instead of words and
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

    {'mean_KL_divergence': 0.22136004391576117, 'score': 0.55669450678688481}




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
    
        
        (probability <b>0.366</b>, score <b>-0.003</b>)
    
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
            
                <tr style="background-color: hsl(120, 100.00%, 80.20%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.199
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.202
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 92.53%); opacity: 0.82" title="0.023">a</span><span style="background-color: hsl(120, 100.00%, 96.10%); opacity: 0.81" title="0.009">s</span><span style="background-color: hsl(0, 100.00%, 93.77%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 99.92%); opacity: 0.80" title="-0.000">i</span><span style="background-color: hsl(0, 100.00%, 84.50%); opacity: 0.85" title="-0.066"> </span><span style="background-color: hsl(120, 100.00%, 93.85%); opacity: 0.81" title="0.018">r</span><span style="background-color: hsl(120, 100.00%, 85.47%); opacity: 0.85" title="0.060">e</span><span style="background-color: hsl(120, 100.00%, 84.30%); opacity: 0.85" title="0.067">c</span><span style="background-color: hsl(120, 100.00%, 82.89%); opacity: 0.86" title="0.075">a</span><span style="background-color: hsl(120, 100.00%, 82.24%); opacity: 0.86" title="0.080">l</span><span style="background-color: hsl(0, 100.00%, 99.07%); opacity: 0.80" title="-0.001">l</span><span style="background-color: hsl(0, 100.00%, 96.53%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.025">f</span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.012">r</span><span style="background-color: hsl(0, 100.00%, 99.31%); opacity: 0.80" title="-0.001">o</span><span style="background-color: hsl(0, 100.00%, 97.45%); opacity: 0.80" title="-0.005">m </span><span style="background-color: hsl(0, 100.00%, 94.82%); opacity: 0.81" title="-0.014">m</span><span style="background-color: hsl(0, 100.00%, 92.29%); opacity: 0.82" title="-0.024">y </span><span style="background-color: hsl(120, 100.00%, 88.44%); opacity: 0.83" title="0.043">b</span><span style="background-color: hsl(120, 100.00%, 90.59%); opacity: 0.83" title="0.032">o</span><span style="background-color: hsl(120, 100.00%, 91.42%); opacity: 0.82" title="0.028">u</span><span style="background-color: hsl(120, 100.00%, 94.69%); opacity: 0.81" title="0.014">t</span><span style="background-color: hsl(0, 100.00%, 98.03%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(0, 100.00%, 94.86%); opacity: 0.81" title="-0.014">w</span><span style="background-color: hsl(120, 100.00%, 93.59%); opacity: 0.81" title="0.019">it</span><span style="background-color: hsl(120, 100.00%, 92.88%); opacity: 0.82" title="0.022">h</span><span style="background-color: hsl(120, 100.00%, 94.91%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 90.17%); opacity: 0.83" title="0.034">k</span><span style="background-color: hsl(120, 100.00%, 88.50%); opacity: 0.83" title="0.043">i</span><span style="background-color: hsl(0, 100.00%, 91.74%); opacity: 0.82" title="-0.027">d</span><span style="background-color: hsl(120, 100.00%, 96.68%); opacity: 0.81" title="0.007">n</span><span style="background-color: hsl(120, 100.00%, 91.01%); opacity: 0.82" title="0.030">e</span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.006">y</span><span style="background-color: hsl(0, 100.00%, 95.73%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 95.17%); opacity: 0.81" title="-0.012">s</span><span style="background-color: hsl(0, 100.00%, 88.19%); opacity: 0.83" title="-0.044">t</span><span style="background-color: hsl(0, 100.00%, 88.26%); opacity: 0.83" title="-0.044">o</span><span style="background-color: hsl(0, 100.00%, 84.48%); opacity: 0.85" title="-0.066">n</span><span style="background-color: hsl(0, 100.00%, 88.55%); opacity: 0.83" title="-0.042">e</span><span style="background-color: hsl(0, 100.00%, 93.59%); opacity: 0.81" title="-0.019">s</span><span style="background-color: hsl(0, 100.00%, 90.55%); opacity: 0.83" title="-0.032">,</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.13%); opacity: 0.81" title="0.016">t</span><span style="background-color: hsl(0, 100.00%, 89.55%); opacity: 0.83" title="-0.037">h</span><span style="background-color: hsl(0, 100.00%, 88.82%); opacity: 0.83" title="-0.041">e</span><span style="background-color: hsl(120, 100.00%, 88.19%); opacity: 0.83" title="0.044">r</span><span style="background-color: hsl(120, 100.00%, 97.96%); opacity: 0.80" title="0.004">e</span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.038"> </span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.014">i</span><span style="opacity: 0.80">s</span><span style="background-color: hsl(0, 100.00%, 85.14%); opacity: 0.85" title="-0.062">n</span><span style="background-color: hsl(0, 100.00%, 89.69%); opacity: 0.83" title="-0.037">'</span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.025">t</span><span style="background-color: hsl(120, 100.00%, 95.86%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 94.09%); opacity: 0.81" title="0.017">a</span><span style="background-color: hsl(120, 100.00%, 93.15%); opacity: 0.82" title="0.020">n</span><span style="background-color: hsl(120, 100.00%, 94.60%); opacity: 0.81" title="0.015">y</span><span style="background-color: hsl(120, 100.00%, 84.19%); opacity: 0.85" title="0.067">
    </span><span style="background-color: hsl(120, 100.00%, 64.45%); opacity: 0.97" title="0.214">m</span><span style="background-color: hsl(120, 100.00%, 65.52%); opacity: 0.96" title="0.205">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.254">d</span><span style="background-color: hsl(120, 100.00%, 67.98%); opacity: 0.95" title="0.185">i</span><span style="background-color: hsl(120, 100.00%, 77.33%); opacity: 0.89" title="0.113">c</span><span style="background-color: hsl(0, 100.00%, 98.33%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(0, 100.00%, 95.79%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 82.55%); opacity: 0.86" title="-0.078">i</span><span style="background-color: hsl(0, 100.00%, 84.27%); opacity: 0.85" title="-0.067">o</span><span style="background-color: hsl(0, 100.00%, 91.49%); opacity: 0.82" title="-0.028">n</span><span style="background-color: hsl(0, 100.00%, 95.42%); opacity: 0.81" title="-0.011"> t</span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.029">h</span><span style="background-color: hsl(0, 100.00%, 95.10%); opacity: 0.81" title="-0.013">a</span><span style="background-color: hsl(0, 100.00%, 82.83%); opacity: 0.86" title="-0.076">t</span><span style="background-color: hsl(0, 100.00%, 79.81%); opacity: 0.88" title="-0.096"> </span><span style="background-color: hsl(0, 100.00%, 79.83%); opacity: 0.88" title="-0.095">c</span><span style="background-color: hsl(0, 100.00%, 93.19%); opacity: 0.82" title="-0.020">a</span><span style="background-color: hsl(120, 100.00%, 91.94%); opacity: 0.82" title="0.026">n</span><span style="background-color: hsl(0, 100.00%, 90.34%); opacity: 0.83" title="-0.033"> </span><span style="background-color: hsl(0, 100.00%, 90.67%); opacity: 0.82" title="-0.032">d</span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.002">o </span><span style="background-color: hsl(120, 100.00%, 91.63%); opacity: 0.82" title="0.027">a</span><span style="background-color: hsl(0, 100.00%, 95.44%); opacity: 0.81" title="-0.011">n</span><span style="background-color: hsl(0, 100.00%, 91.09%); opacity: 0.82" title="-0.030">y</span><span style="background-color: hsl(0, 100.00%, 91.58%); opacity: 0.82" title="-0.027">t</span><span style="background-color: hsl(0, 100.00%, 89.75%); opacity: 0.83" title="-0.036">h</span><span style="background-color: hsl(0, 100.00%, 88.01%); opacity: 0.84" title="-0.045">i</span><span style="background-color: hsl(0, 100.00%, 97.21%); opacity: 0.80" title="-0.006">n</span><span style="background-color: hsl(0, 100.00%, 87.30%); opacity: 0.84" title="-0.049">g</span><span style="background-color: hsl(0, 100.00%, 92.35%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 98.18%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(120, 100.00%, 84.21%); opacity: 0.85" title="0.067">bo</span><span style="background-color: hsl(120, 100.00%, 91.48%); opacity: 0.82" title="0.028">u</span><span style="background-color: hsl(120, 100.00%, 97.03%); opacity: 0.80" title="0.006">t</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(0, 100.00%, 94.07%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(0, 100.00%, 87.43%); opacity: 0.84" title="-0.049">h</span><span style="background-color: hsl(0, 100.00%, 78.06%); opacity: 0.88" title="-0.108">e</span><span style="background-color: hsl(0, 100.00%, 87.51%); opacity: 0.84" title="-0.048">m</span><span style="background-color: hsl(0, 100.00%, 89.70%); opacity: 0.83" title="-0.037"> </span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.010">e</span><span style="background-color: hsl(0, 100.00%, 90.81%); opacity: 0.82" title="-0.031">x</span><span style="background-color: hsl(0, 100.00%, 99.28%); opacity: 0.80" title="-0.001">c</span><span style="background-color: hsl(0, 100.00%, 93.54%); opacity: 0.81" title="-0.019">e</span><span style="background-color: hsl(120, 100.00%, 94.79%); opacity: 0.81" title="0.014">p</span><span style="background-color: hsl(120, 100.00%, 86.96%); opacity: 0.84" title="0.051">t</span><span style="background-color: hsl(0, 100.00%, 91.96%); opacity: 0.82" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 87.02%); opacity: 0.84" title="-0.051">r</span><span style="background-color: hsl(120, 100.00%, 94.82%); opacity: 0.81" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 95.64%); opacity: 0.81" title="0.011">l</span><span style="background-color: hsl(120, 100.00%, 97.90%); opacity: 0.80" title="0.004">i</span><span style="background-color: hsl(0, 100.00%, 99.11%); opacity: 0.80" title="-0.001">e</span><span style="background-color: hsl(120, 100.00%, 92.08%); opacity: 0.82" title="0.025">v</span><span style="background-color: hsl(0, 100.00%, 95.67%); opacity: 0.81" title="-0.011">e</span><span style="background-color: hsl(0, 100.00%, 91.69%); opacity: 0.82" title="-0.027"> </span><span style="background-color: hsl(0, 100.00%, 96.94%); opacity: 0.81" title="-0.006">t</span><span style="background-color: hsl(0, 100.00%, 84.41%); opacity: 0.85" title="-0.066">h</span><span style="background-color: hsl(0, 100.00%, 82.80%); opacity: 0.86" title="-0.076">e</span><span style="background-color: hsl(0, 100.00%, 85.58%); opacity: 0.85" title="-0.059"> </span><span style="background-color: hsl(0, 100.00%, 98.15%); opacity: 0.80" title="-0.003">p</span><span style="background-color: hsl(120, 100.00%, 90.93%); opacity: 0.82" title="0.030">a</span><span style="background-color: hsl(120, 100.00%, 83.79%); opacity: 0.86" title="0.070">in</span><span style="background-color: hsl(120, 100.00%, 98.30%); opacity: 0.80" title="0.003">.</span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 90.49%); opacity: 0.83" title="-0.033">e</span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.016">i</span><span style="background-color: hsl(0, 100.00%, 89.65%); opacity: 0.83" title="-0.037">t</span><span style="background-color: hsl(0, 100.00%, 83.11%); opacity: 0.86" title="-0.074">h</span><span style="background-color: hsl(0, 100.00%, 83.03%); opacity: 0.86" title="-0.075">e</span><span style="background-color: hsl(0, 100.00%, 96.05%); opacity: 0.81" title="-0.009">r</span><span style="background-color: hsl(120, 100.00%, 96.58%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 91.31%); opacity: 0.82" title="0.029">t</span><span style="background-color: hsl(0, 100.00%, 90.11%); opacity: 0.83" title="-0.034">h</span><span style="background-color: hsl(0, 100.00%, 91.20%); opacity: 0.82" title="-0.029">e</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.008">y</span><span style="background-color: hsl(0, 100.00%, 92.43%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 86.95%); opacity: 0.84" title="-0.051">p</span><span style="background-color: hsl(0, 100.00%, 93.75%); opacity: 0.81" title="-0.018">a</span><span style="background-color: hsl(0, 100.00%, 94.70%); opacity: 0.81" title="-0.014">s</span><span style="background-color: hsl(0, 100.00%, 94.60%); opacity: 0.81" title="-0.015">s</span><span style="opacity: 0.80">,</span><span style="background-color: hsl(0, 100.00%, 96.57%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.017">o</span><span style="background-color: hsl(0, 100.00%, 97.52%); opacity: 0.80" title="-0.005">r</span><span style="background-color: hsl(120, 100.00%, 95.39%); opacity: 0.81" title="0.012"> t</span><span style="background-color: hsl(0, 100.00%, 85.53%); opacity: 0.85" title="-0.059">h</span><span style="background-color: hsl(0, 100.00%, 86.44%); opacity: 0.84" title="-0.054">e</span><span style="background-color: hsl(0, 100.00%, 95.21%); opacity: 0.81" title="-0.012">y</span><span style="background-color: hsl(0, 100.00%, 92.52%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 86.40%); opacity: 0.84" title="-0.054">h</span><span style="background-color: hsl(0, 100.00%, 93.26%); opacity: 0.82" title="-0.020">a</span><span style="background-color: hsl(0, 100.00%, 95.81%); opacity: 0.81" title="-0.010">v</span><span style="background-color: hsl(0, 100.00%, 89.98%); opacity: 0.83" title="-0.035">e</span><span style="background-color: hsl(120, 100.00%, 96.46%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 88.74%); opacity: 0.83" title="0.041">t</span><span style="background-color: hsl(120, 100.00%, 92.47%); opacity: 0.82" title="0.023">o</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.034"> </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.029">b</span><span style="background-color: hsl(0, 100.00%, 95.35%); opacity: 0.81" title="-0.012">e</span><span style="background-color: hsl(0, 100.00%, 95.99%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(120, 100.00%, 95.57%); opacity: 0.81" title="0.011">b</span><span style="background-color: hsl(0, 100.00%, 96.26%); opacity: 0.81" title="-0.009">r</span><span style="background-color: hsl(0, 100.00%, 93.35%); opacity: 0.82" title="-0.020">o</span><span style="opacity: 0.80">ken u</span><span style="background-color: hsl(0, 100.00%, 90.07%); opacity: 0.83" title="-0.035">p</span><span style="background-color: hsl(0, 100.00%, 91.07%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.011">wi</span><span style="background-color: hsl(120, 100.00%, 84.50%); opacity: 0.85" title="0.066">t</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(0, 100.00%, 94.82%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 95.50%); opacity: 0.81" title="-0.011">s</span><span style="background-color: hsl(120, 100.00%, 95.68%); opacity: 0.81" title="0.011">o</span><span style="background-color: hsl(0, 100.00%, 95.40%); opacity: 0.81" title="-0.012">un</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">d,</span><span style="background-color: hsl(0, 100.00%, 96.64%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 94.01%); opacity: 0.81" title="-0.017">o</span><span style="background-color: hsl(0, 100.00%, 97.61%); opacity: 0.80" title="-0.005">r</span><span style="background-color: hsl(120, 100.00%, 95.39%); opacity: 0.81" title="0.012"> t</span><span style="background-color: hsl(0, 100.00%, 85.53%); opacity: 0.85" title="-0.059">h</span><span style="background-color: hsl(0, 100.00%, 86.44%); opacity: 0.84" title="-0.054">e</span><span style="background-color: hsl(0, 100.00%, 95.21%); opacity: 0.81" title="-0.012">y</span><span style="background-color: hsl(0, 100.00%, 92.52%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 86.40%); opacity: 0.84" title="-0.054">h</span><span style="background-color: hsl(0, 100.00%, 86.94%); opacity: 0.84" title="-0.051">a</span><span style="background-color: hsl(0, 100.00%, 85.93%); opacity: 0.84" title="-0.057">v</span><span style="background-color: hsl(0, 100.00%, 85.15%); opacity: 0.85" title="-0.062">e</span><span style="background-color: hsl(0, 100.00%, 88.83%); opacity: 0.83" title="-0.041">
    </span><span style="background-color: hsl(0, 100.00%, 91.47%); opacity: 0.82" title="-0.028">t</span><span style="background-color: hsl(120, 100.00%, 96.55%); opacity: 0.81" title="0.008">o</span><span style="background-color: hsl(120, 100.00%, 91.45%); opacity: 0.82" title="0.028"> </span><span style="background-color: hsl(0, 100.00%, 97.43%); opacity: 0.80" title="-0.005">b</span><span style="background-color: hsl(0, 100.00%, 88.51%); opacity: 0.83" title="-0.043">e</span><span style="background-color: hsl(0, 100.00%, 88.12%); opacity: 0.84" title="-0.045"> </span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.008">e</span><span style="background-color: hsl(120, 100.00%, 92.79%); opacity: 0.82" title="0.022">x</span><span style="background-color: hsl(120, 100.00%, 82.53%); opacity: 0.86" title="0.078">t</span><span style="background-color: hsl(120, 100.00%, 90.22%); opacity: 0.83" title="0.034">r</span><span style="background-color: hsl(0, 100.00%, 97.29%); opacity: 0.80" title="-0.005">a</span><span style="background-color: hsl(120, 100.00%, 85.63%); opacity: 0.85" title="0.059">c</span><span style="background-color: hsl(120, 100.00%, 80.39%); opacity: 0.87" title="0.092">t</span><span style="background-color: hsl(120, 100.00%, 81.37%); opacity: 0.87" title="0.085">e</span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.022">d</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.23%); opacity: 0.80" title="0.003">s</span><span style="background-color: hsl(120, 100.00%, 91.55%); opacity: 0.82" title="0.028">u</span><span style="background-color: hsl(0, 100.00%, 91.50%); opacity: 0.82" title="-0.028">r</span><span style="background-color: hsl(0, 100.00%, 92.77%); opacity: 0.82" title="-0.022">g</span><span style="background-color: hsl(120, 100.00%, 96.38%); opacity: 0.81" title="0.008">i</span><span style="background-color: hsl(120, 100.00%, 86.19%); opacity: 0.84" title="0.056">c</span><span style="background-color: hsl(120, 100.00%, 79.90%); opacity: 0.87" title="0.095">a</span><span style="background-color: hsl(120, 100.00%, 82.94%); opacity: 0.86" title="0.075">l</span><span style="background-color: hsl(120, 100.00%, 89.49%); opacity: 0.83" title="0.038">l</span><span style="background-color: hsl(120, 100.00%, 94.17%); opacity: 0.81" title="0.016">y</span><span style="background-color: hsl(0, 100.00%, 88.03%); opacity: 0.84" title="-0.045">.</span><span style="background-color: hsl(0, 100.00%, 91.01%); opacity: 0.82" title="-0.030"> </span><span style="background-color: hsl(0, 100.00%, 95.32%); opacity: 0.81" title="-0.012">w</span><span style="background-color: hsl(0, 100.00%, 92.77%); opacity: 0.82" title="-0.022">h</span><span style="background-color: hsl(0, 100.00%, 93.31%); opacity: 0.82" title="-0.020">e</span><span style="background-color: hsl(120, 100.00%, 83.45%); opacity: 0.86" title="0.072">n</span><span style="background-color: hsl(120, 100.00%, 93.77%); opacity: 0.81" title="0.018"> </span><span style="background-color: hsl(120, 100.00%, 94.83%); opacity: 0.81" title="0.014">i</span><span style="background-color: hsl(120, 100.00%, 94.33%); opacity: 0.81" title="0.016"> </span><span style="background-color: hsl(120, 100.00%, 93.86%); opacity: 0.81" title="0.017">w</span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.029">as</span><span style="background-color: hsl(120, 100.00%, 93.02%); opacity: 0.82" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 85.36%); opacity: 0.85" title="0.060">i</span><span style="background-color: hsl(120, 100.00%, 89.19%); opacity: 0.83" title="0.039">n</span><span style="background-color: hsl(0, 100.00%, 99.55%); opacity: 0.80" title="-0.000">,</span><span style="background-color: hsl(0, 100.00%, 95.06%); opacity: 0.81" title="-0.013"> t</span><span style="background-color: hsl(0, 100.00%, 83.46%); opacity: 0.86" title="-0.072">h</span><span style="background-color: hsl(0, 100.00%, 86.57%); opacity: 0.84" title="-0.053">e</span><span style="background-color: hsl(120, 100.00%, 93.07%); opacity: 0.82" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 75.62%); opacity: 0.90" title="0.125">x</span><span style="background-color: hsl(120, 100.00%, 79.38%); opacity: 0.88" title="0.098">-</span><span style="background-color: hsl(120, 100.00%, 98.00%); opacity: 0.80" title="0.004">r</span><span style="background-color: hsl(0, 100.00%, 98.24%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(0, 100.00%, 98.70%); opacity: 0.80" title="-0.002">y</span><span style="background-color: hsl(120, 100.00%, 94.61%); opacity: 0.81" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 93.87%); opacity: 0.81" title="0.017">t</span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">ec</span><span style="background-color: hsl(0, 100.00%, 95.71%); opacity: 0.81" title="-0.010">h</span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 93.78%); opacity: 0.81" title="0.018">h</span><span style="background-color: hsl(120, 100.00%, 91.40%); opacity: 0.82" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 87.78%); opacity: 0.84" title="0.047">p</span><span style="background-color: hsl(120, 100.00%, 89.89%); opacity: 0.83" title="0.036">p</span><span style="background-color: hsl(120, 100.00%, 93.53%); opacity: 0.81" title="0.019">en</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(0, 100.00%, 96.72%); opacity: 0.81" title="-0.007">d</span><span style="background-color: hsl(120, 100.00%, 98.21%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 95.37%); opacity: 0.81" title="0.012">t</span><span style="background-color: hsl(120, 100.00%, 94.20%); opacity: 0.81" title="0.016">o </span><span style="background-color: hsl(0, 100.00%, 94.67%); opacity: 0.81" title="-0.014">m</span><span style="background-color: hsl(120, 100.00%, 93.38%); opacity: 0.82" title="0.019">e</span><span style="background-color: hsl(120, 100.00%, 89.80%); opacity: 0.83" title="0.036">nt</span><span style="background-color: hsl(0, 100.00%, 82.92%); opacity: 0.86" title="-0.075">i</span><span style="background-color: hsl(0, 100.00%, 82.34%); opacity: 0.86" title="-0.079">o</span><span style="background-color: hsl(0, 100.00%, 87.16%); opacity: 0.84" title="-0.050">n</span><span style="background-color: hsl(0, 100.00%, 95.42%); opacity: 0.81" title="-0.011"> t</span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.029">h</span><span style="background-color: hsl(0, 100.00%, 89.20%); opacity: 0.83" title="-0.039">a</span><span style="background-color: hsl(0, 100.00%, 93.93%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 93.39%); opacity: 0.82" title="0.019">s</span><span style="background-color: hsl(120, 100.00%, 96.84%); opacity: 0.81" title="0.007">h</span><span style="background-color: hsl(120, 100.00%, 94.42%); opacity: 0.81" title="0.015">e</span><span style="background-color: hsl(0, 100.00%, 94.40%); opacity: 0.81" title="-0.015">'</span><span style="background-color: hsl(120, 100.00%, 96.63%); opacity: 0.81" title="0.007">d</span><span style="background-color: hsl(120, 100.00%, 87.45%); opacity: 0.84" title="0.048"> </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.102">h</span><span style="background-color: hsl(120, 100.00%, 81.86%); opacity: 0.86" title="0.082">a</span><span style="background-color: hsl(120, 100.00%, 86.21%); opacity: 0.84" title="0.055">d</span><span style="background-color: hsl(0, 100.00%, 97.10%); opacity: 0.80" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 91.84%); opacity: 0.82" title="-0.026">k</span><span style="background-color: hsl(120, 100.00%, 95.34%); opacity: 0.81" title="0.012">i</span><span style="background-color: hsl(0, 100.00%, 90.78%); opacity: 0.82" title="-0.031">d</span><span style="background-color: hsl(120, 100.00%, 94.67%); opacity: 0.81" title="0.014">n</span><span style="background-color: hsl(0, 100.00%, 92.43%); opacity: 0.82" title="-0.024">e</span><span style="background-color: hsl(0, 100.00%, 95.88%); opacity: 0.81" title="-0.010">y
    </span><span style="background-color: hsl(0, 100.00%, 88.45%); opacity: 0.83" title="-0.043">s</span><span style="background-color: hsl(0, 100.00%, 92.34%); opacity: 0.82" title="-0.024">to</span><span style="background-color: hsl(0, 100.00%, 90.35%); opacity: 0.83" title="-0.033">n</span><span style="background-color: hsl(120, 100.00%, 95.72%); opacity: 0.81" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 89.30%); opacity: 0.83" title="0.039">s</span><span style="background-color: hsl(120, 100.00%, 92.14%); opacity: 0.82" title="0.025"> </span><span style="background-color: hsl(120, 100.00%, 90.26%); opacity: 0.83" title="0.034">a</span><span style="background-color: hsl(120, 100.00%, 94.98%); opacity: 0.81" title="0.013">n</span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.012">d</span><span style="background-color: hsl(0, 100.00%, 97.54%); opacity: 0.80" title="-0.005"> c</span><span style="background-color: hsl(120, 100.00%, 90.68%); opacity: 0.82" title="0.032">h</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(0, 100.00%, 93.61%); opacity: 0.81" title="-0.018">l</span><span style="background-color: hsl(0, 100.00%, 90.72%); opacity: 0.82" title="-0.032">d</span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(0, 100.00%, 98.65%); opacity: 0.80" title="-0.002">e</span><span style="background-color: hsl(120, 100.00%, 98.23%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(0, 100.00%, 99.65%); opacity: 0.80" title="-0.000">,</span><span style="background-color: hsl(120, 100.00%, 95.07%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 89.35%); opacity: 0.83" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 94.07%); opacity: 0.81" title="0.017">n</span><span style="background-color: hsl(0, 100.00%, 98.11%); opacity: 0.80" title="-0.003">d</span><span style="background-color: hsl(0, 100.00%, 95.17%); opacity: 0.81" title="-0.012"> t</span><span style="background-color: hsl(0, 100.00%, 84.41%); opacity: 0.85" title="-0.066">h</span><span style="background-color: hsl(0, 100.00%, 81.33%); opacity: 0.87" title="-0.085">e</span><span style="background-color: hsl(0, 100.00%, 89.76%); opacity: 0.83" title="-0.036"> </span><span style="background-color: hsl(120, 100.00%, 95.02%); opacity: 0.81" title="0.013">c</span><span style="background-color: hsl(120, 100.00%, 90.47%); opacity: 0.83" title="0.033">h</span><span style="background-color: hsl(0, 100.00%, 94.59%); opacity: 0.81" title="-0.015">ild</span><span style="background-color: hsl(0, 100.00%, 99.44%); opacity: 0.80" title="-0.001">bi</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 96.04%); opacity: 0.81" title="0.009">t</span><span style="background-color: hsl(0, 100.00%, 98.96%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 96.16%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 91.89%); opacity: 0.82" title="-0.026">h</span><span style="background-color: hsl(0, 100.00%, 94.35%); opacity: 0.81" title="-0.015">u</span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.003">rt</span><span style="background-color: hsl(120, 100.00%, 93.32%); opacity: 0.82" title="0.020"> </span><span style="background-color: hsl(120, 100.00%, 84.50%); opacity: 0.85" title="0.066">l</span><span style="background-color: hsl(120, 100.00%, 82.25%); opacity: 0.86" title="0.080">e</span><span style="background-color: hsl(120, 100.00%, 93.79%); opacity: 0.81" title="0.018">s</span><span style="background-color: hsl(0, 100.00%, 97.31%); opacity: 0.80" title="-0.005">s</span><span style="background-color: hsl(0, 100.00%, 96.10%); opacity: 0.81" title="-0.009">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Hm, the result look worse. :class:`~.TextExplainer` detected correctly that
only the first part of word “medication” is important, but the result is
noisy overall, and scores are bad. Let’s try it with more samples:

.. code:: ipython3

    te = TextExplainer(char_based=True, n_samples=50000, random_state=42)
    te.fit(doc, pipe_char.predict_proba)
    print(te.metrics_)
    te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)


.. parsed-literal::

    {'mean_KL_divergence': 0.060019833958355841, 'score': 0.86048000626542609}




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
    
        
        (probability <b>0.630</b>, score <b>0.800</b>)
    
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
            +1.018
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 93.19%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.219
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">a</span><span style="background-color: hsl(0, 100.00%, 91.90%); opacity: 0.82" title="-0.020">s</span><span style="background-color: hsl(0, 100.00%, 86.77%); opacity: 0.84" title="-0.041"> </span><span style="background-color: hsl(0, 100.00%, 90.60%); opacity: 0.83" title="-0.025">i</span><span style="background-color: hsl(0, 100.00%, 92.89%); opacity: 0.82" title="-0.017"> </span><span style="background-color: hsl(120, 100.00%, 92.74%); opacity: 0.82" title="0.017">r</span><span style="background-color: hsl(120, 100.00%, 88.73%); opacity: 0.83" title="0.032">e</span><span style="background-color: hsl(120, 100.00%, 85.07%); opacity: 0.85" title="0.048">c</span><span style="background-color: hsl(120, 100.00%, 89.77%); opacity: 0.83" title="0.028">al</span><span style="background-color: hsl(0, 100.00%, 92.05%); opacity: 0.82" title="-0.020">l</span><span style="background-color: hsl(0, 100.00%, 89.13%); opacity: 0.83" title="-0.031"> </span><span style="background-color: hsl(0, 100.00%, 92.36%); opacity: 0.82" title="-0.019">f</span><span style="background-color: hsl(0, 100.00%, 96.60%); opacity: 0.81" title="-0.006">r</span><span style="background-color: hsl(0, 100.00%, 92.36%); opacity: 0.82" title="-0.019">o</span><span style="background-color: hsl(0, 100.00%, 94.14%); opacity: 0.81" title="-0.013">m</span><span style="background-color: hsl(120, 100.00%, 92.88%); opacity: 0.82" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 82.55%); opacity: 0.86" title="0.061">m</span><span style="background-color: hsl(120, 100.00%, 84.20%); opacity: 0.85" title="0.053">y</span><span style="background-color: hsl(120, 100.00%, 95.13%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 93.75%); opacity: 0.81" title="0.014">b</span><span style="background-color: hsl(120, 100.00%, 88.19%); opacity: 0.84" title="0.035">o</span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.002">u</span><span style="background-color: hsl(0, 100.00%, 94.45%); opacity: 0.81" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 94.20%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 96.78%); opacity: 0.81" title="-0.005">w</span><span style="background-color: hsl(0, 100.00%, 97.07%); opacity: 0.80" title="-0.005">i</span><span style="background-color: hsl(0, 100.00%, 98.81%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(0, 100.00%, 97.61%); opacity: 0.80" title="-0.004">h</span><span style="background-color: hsl(0, 100.00%, 95.68%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 95.89%); opacity: 0.81" title="-0.008">k</span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 87.94%); opacity: 0.84" title="0.036">d</span><span style="background-color: hsl(120, 100.00%, 90.10%); opacity: 0.83" title="0.027">n</span><span style="background-color: hsl(120, 100.00%, 89.59%); opacity: 0.83" title="0.029">e</span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.008">y</span><span style="background-color: hsl(0, 100.00%, 95.61%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 99.40%); opacity: 0.80" title="-0.000">s</span><span style="background-color: hsl(0, 100.00%, 95.48%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(120, 100.00%, 98.06%); opacity: 0.80" title="0.003">o</span><span style="background-color: hsl(0, 100.00%, 99.47%); opacity: 0.80" title="-0.000">n</span><span style="background-color: hsl(120, 100.00%, 94.98%); opacity: 0.81" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 92.25%); opacity: 0.82" title="0.019">s</span><span style="background-color: hsl(0, 100.00%, 94.86%); opacity: 0.81" title="-0.011">,</span><span style="background-color: hsl(0, 100.00%, 95.88%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.007">t</span><span style="background-color: hsl(0, 100.00%, 92.18%); opacity: 0.82" title="-0.019">h</span><span style="background-color: hsl(0, 100.00%, 89.31%); opacity: 0.83" title="-0.030">e</span><span style="background-color: hsl(0, 100.00%, 91.59%); opacity: 0.82" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 92.92%); opacity: 0.82" title="-0.017">e</span><span style="background-color: hsl(0, 100.00%, 92.64%); opacity: 0.82" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 99.05%); opacity: 0.80" title="-0.001">i</span><span style="background-color: hsl(120, 100.00%, 93.70%); opacity: 0.81" title="0.014">sn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 97.02%); opacity: 0.80" title="-0.005">t</span><span style="background-color: hsl(120, 100.00%, 94.39%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 92.86%); opacity: 0.82" title="0.017">a</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(0, 100.00%, 94.27%); opacity: 0.81" title="-0.012">y</span><span style="background-color: hsl(120, 100.00%, 90.05%); opacity: 0.83" title="0.027">
    </span><span style="background-color: hsl(120, 100.00%, 75.17%); opacity: 0.90" title="0.100">m</span><span style="background-color: hsl(120, 100.00%, 65.69%); opacity: 0.96" title="0.159">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.198">d</span><span style="background-color: hsl(120, 100.00%, 64.32%); opacity: 0.97" title="0.168">i</span><span style="background-color: hsl(120, 100.00%, 77.13%); opacity: 0.89" title="0.089">c</span><span style="background-color: hsl(120, 100.00%, 87.77%); opacity: 0.84" title="0.036">a</span><span style="background-color: hsl(120, 100.00%, 89.46%); opacity: 0.83" title="0.029">t</span><span style="background-color: hsl(0, 100.00%, 95.16%); opacity: 0.81" title="-0.010">i</span><span style="background-color: hsl(0, 100.00%, 92.93%); opacity: 0.82" title="-0.017">o</span><span style="background-color: hsl(0, 100.00%, 88.73%); opacity: 0.83" title="-0.032">n</span><span style="background-color: hsl(0, 100.00%, 91.13%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">th</span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.005">a</span><span style="background-color: hsl(120, 100.00%, 95.79%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.009"> </span><span style="background-color: hsl(0, 100.00%, 97.49%); opacity: 0.80" title="-0.004">c</span><span style="background-color: hsl(120, 100.00%, 90.46%); opacity: 0.83" title="0.026">a</span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(0, 100.00%, 90.10%); opacity: 0.83" title="-0.027"> </span><span style="background-color: hsl(0, 100.00%, 92.91%); opacity: 0.82" title="-0.017">d</span><span style="background-color: hsl(0, 100.00%, 91.34%); opacity: 0.82" title="-0.022">o</span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(120, 100.00%, 94.77%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(0, 100.00%, 93.67%); opacity: 0.81" title="-0.014">n</span><span style="background-color: hsl(0, 100.00%, 89.59%); opacity: 0.83" title="-0.029">y</span><span style="background-color: hsl(0, 100.00%, 92.62%); opacity: 0.82" title="-0.018">t</span><span style="background-color: hsl(0, 100.00%, 92.78%); opacity: 0.82" title="-0.017">h</span><span style="background-color: hsl(120, 100.00%, 96.21%); opacity: 0.81" title="0.007">i</span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.006">n</span><span style="background-color: hsl(0, 100.00%, 84.64%); opacity: 0.85" title="-0.050">g</span><span style="background-color: hsl(0, 100.00%, 91.76%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(120, 100.00%, 91.26%); opacity: 0.82" title="0.023">a</span><span style="background-color: hsl(120, 100.00%, 89.68%); opacity: 0.83" title="0.029">b</span><span style="background-color: hsl(120, 100.00%, 87.52%); opacity: 0.84" title="0.037">o</span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.009">u</span><span style="background-color: hsl(0, 100.00%, 95.85%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(0, 100.00%, 96.99%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 94.96%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 96.69%); opacity: 0.81" title="-0.006">h</span><span style="background-color: hsl(0, 100.00%, 94.28%); opacity: 0.81" title="-0.012">e</span><span style="background-color: hsl(0, 100.00%, 96.15%); opacity: 0.81" title="-0.007">m</span><span style="background-color: hsl(0, 100.00%, 96.59%); opacity: 0.81" title="-0.006"> </span><span style="background-color: hsl(0, 100.00%, 98.48%); opacity: 0.80" title="-0.002">e</span><span style="background-color: hsl(0, 100.00%, 99.62%); opacity: 0.80" title="-0.000">x</span><span style="background-color: hsl(120, 100.00%, 91.88%); opacity: 0.82" title="0.020">ce</span><span style="background-color: hsl(0, 100.00%, 98.12%); opacity: 0.80" title="-0.003">p</span><span style="background-color: hsl(0, 100.00%, 95.07%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 97.12%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 94.20%); opacity: 0.81" title="-0.013">r</span><span style="background-color: hsl(0, 100.00%, 92.76%); opacity: 0.82" title="-0.017">e</span><span style="background-color: hsl(0, 100.00%, 95.80%); opacity: 0.81" title="-0.008">l</span><span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.004">i</span><span style="background-color: hsl(0, 100.00%, 94.02%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(120, 100.00%, 93.20%); opacity: 0.82" title="0.016">v</span><span style="background-color: hsl(120, 100.00%, 97.56%); opacity: 0.80" title="0.004">e</span><span style="background-color: hsl(0, 100.00%, 89.69%); opacity: 0.83" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 90.43%); opacity: 0.83" title="-0.026">t</span><span style="background-color: hsl(0, 100.00%, 90.88%); opacity: 0.82" title="-0.024">h</span><span style="background-color: hsl(0, 100.00%, 87.19%); opacity: 0.84" title="-0.039">e</span><span style="background-color: hsl(0, 100.00%, 93.56%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(120, 100.00%, 86.18%); opacity: 0.84" title="0.043">p</span><span style="background-color: hsl(120, 100.00%, 87.29%); opacity: 0.84" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 82.16%); opacity: 0.86" title="0.062">i</span><span style="background-color: hsl(120, 100.00%, 85.31%); opacity: 0.85" title="0.047">n</span><span style="background-color: hsl(0, 100.00%, 91.14%); opacity: 0.82" title="-0.023">.</span><span style="background-color: hsl(0, 100.00%, 90.74%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 93.40%); opacity: 0.82" title="-0.015">e</span><span style="background-color: hsl(0, 100.00%, 86.89%); opacity: 0.84" title="-0.040">i</span><span style="background-color: hsl(0, 100.00%, 84.71%); opacity: 0.85" title="-0.050">t</span><span style="background-color: hsl(0, 100.00%, 83.87%); opacity: 0.85" title="-0.054">h</span><span style="background-color: hsl(0, 100.00%, 88.00%); opacity: 0.84" title="-0.035">e</span><span style="background-color: hsl(0, 100.00%, 90.72%); opacity: 0.82" title="-0.025">r</span><span style="background-color: hsl(0, 100.00%, 98.86%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 99.24%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 97.84%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(0, 100.00%, 93.00%); opacity: 0.82" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 94.40%); opacity: 0.81" title="-0.012">p</span><span style="background-color: hsl(0, 100.00%, 95.73%); opacity: 0.81" title="-0.008">as</span><span style="background-color: hsl(0, 100.00%, 96.90%); opacity: 0.81" title="-0.005">s</span><span style="background-color: hsl(0, 100.00%, 95.42%); opacity: 0.81" title="-0.009">,</span><span style="background-color: hsl(0, 100.00%, 90.31%); opacity: 0.83" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 93.25%); opacity: 0.82" title="-0.016">o</span><span style="background-color: hsl(0, 100.00%, 97.36%); opacity: 0.80" title="-0.004">r</span><span style="background-color: hsl(0, 100.00%, 98.86%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 99.24%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 96.18%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 96.81%); opacity: 0.81" title="0.005">y</span><span style="background-color: hsl(0, 100.00%, 92.88%); opacity: 0.82" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 92.29%); opacity: 0.82" title="-0.019">h</span><span style="background-color: hsl(0, 100.00%, 88.32%); opacity: 0.83" title="-0.034">a</span><span style="background-color: hsl(0, 100.00%, 94.65%); opacity: 0.81" title="-0.011">v</span><span style="background-color: hsl(0, 100.00%, 93.14%); opacity: 0.82" title="-0.016">e</span><span style="background-color: hsl(0, 100.00%, 93.80%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 98.56%); opacity: 0.80" title="-0.002">t</span><span style="background-color: hsl(0, 100.00%, 92.59%); opacity: 0.82" title="-0.018">o</span><span style="background-color: hsl(0, 100.00%, 90.45%); opacity: 0.83" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 94.59%); opacity: 0.81" title="-0.011">b</span><span style="background-color: hsl(0, 100.00%, 94.48%); opacity: 0.81" title="-0.012">e</span><span style="background-color: hsl(0, 100.00%, 89.70%); opacity: 0.83" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 91.67%); opacity: 0.82" title="-0.021">b</span><span style="background-color: hsl(0, 100.00%, 95.31%); opacity: 0.81" title="-0.009">r</span><span style="opacity: 0.80">ok</span><span style="background-color: hsl(120, 100.00%, 95.95%); opacity: 0.81" title="0.008">e</span><span style="background-color: hsl(0, 100.00%, 95.67%); opacity: 0.81" title="-0.008">n</span><span style="background-color: hsl(120, 100.00%, 97.88%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 88.17%); opacity: 0.84" title="0.035">u</span><span style="background-color: hsl(120, 100.00%, 87.90%); opacity: 0.84" title="0.036">p</span><span style="background-color: hsl(120, 100.00%, 97.08%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.006">w</span><span style="background-color: hsl(0, 100.00%, 97.59%); opacity: 0.80" title="-0.004">i</span><span style="background-color: hsl(120, 100.00%, 97.26%); opacity: 0.80" title="0.004">t</span><span style="background-color: hsl(0, 100.00%, 97.91%); opacity: 0.80" title="-0.003">h</span><span style="background-color: hsl(120, 100.00%, 95.21%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 91.08%); opacity: 0.82" title="0.023">s</span><span style="background-color: hsl(120, 100.00%, 86.42%); opacity: 0.84" title="0.042">o</span><span style="background-color: hsl(120, 100.00%, 88.09%); opacity: 0.84" title="0.035">u</span><span style="background-color: hsl(120, 100.00%, 86.43%); opacity: 0.84" title="0.042">n</span><span style="background-color: hsl(120, 100.00%, 89.76%); opacity: 0.83" title="0.028">d</span><span style="background-color: hsl(0, 100.00%, 93.43%); opacity: 0.82" title="-0.015">,</span><span style="background-color: hsl(0, 100.00%, 89.62%); opacity: 0.83" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 93.25%); opacity: 0.82" title="-0.016">o</span><span style="background-color: hsl(0, 100.00%, 97.36%); opacity: 0.80" title="-0.004">r</span><span style="background-color: hsl(0, 100.00%, 98.86%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 99.24%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 96.18%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 96.81%); opacity: 0.81" title="0.005">y</span><span style="background-color: hsl(0, 100.00%, 92.88%); opacity: 0.82" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 92.29%); opacity: 0.82" title="-0.019">h</span><span style="background-color: hsl(0, 100.00%, 87.33%); opacity: 0.84" title="-0.038">a</span><span style="background-color: hsl(0, 100.00%, 93.84%); opacity: 0.81" title="-0.014">v</span><span style="background-color: hsl(0, 100.00%, 96.41%); opacity: 0.81" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 99.19%); opacity: 0.80" title="-0.001">
    t</span><span style="background-color: hsl(0, 100.00%, 93.63%); opacity: 0.81" title="-0.014">o</span><span style="background-color: hsl(0, 100.00%, 91.37%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(0, 100.00%, 98.33%); opacity: 0.80" title="-0.002">b</span><span style="background-color: hsl(0, 100.00%, 98.15%); opacity: 0.80" title="-0.002">e</span><span style="background-color: hsl(0, 100.00%, 96.17%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(120, 100.00%, 97.47%); opacity: 0.80" title="0.004">e</span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.013">xt</span><span style="background-color: hsl(120, 100.00%, 88.99%); opacity: 0.83" title="0.031">r</span><span style="background-color: hsl(120, 100.00%, 92.44%); opacity: 0.82" title="0.018">a</span><span style="background-color: hsl(120, 100.00%, 92.34%); opacity: 0.82" title="0.019">c</span><span style="background-color: hsl(120, 100.00%, 90.64%); opacity: 0.83" title="0.025">t</span><span style="background-color: hsl(120, 100.00%, 93.35%); opacity: 0.82" title="0.015">e</span><span style="background-color: hsl(120, 100.00%, 95.07%); opacity: 0.81" title="0.010">d</span><span style="background-color: hsl(120, 100.00%, 93.29%); opacity: 0.82" title="0.015"> </span><span style="background-color: hsl(120, 100.00%, 88.35%); opacity: 0.83" title="0.034">s</span><span style="background-color: hsl(120, 100.00%, 88.44%); opacity: 0.83" title="0.034">u</span><span style="background-color: hsl(120, 100.00%, 87.21%); opacity: 0.84" title="0.039">r</span><span style="background-color: hsl(120, 100.00%, 94.79%); opacity: 0.81" title="0.011">g</span><span style="background-color: hsl(120, 100.00%, 94.12%); opacity: 0.81" title="0.013">i</span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.023">c</span><span style="background-color: hsl(120, 100.00%, 88.37%); opacity: 0.83" title="0.034">al</span><span style="background-color: hsl(120, 100.00%, 91.39%); opacity: 0.82" title="0.022">ly</span><span style="background-color: hsl(0, 100.00%, 92.51%); opacity: 0.82" title="-0.018">.</span><span style="background-color: hsl(120, 100.00%, 98.14%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 82.37%); opacity: 0.86" title="0.061">w</span><span style="background-color: hsl(120, 100.00%, 81.58%); opacity: 0.87" title="0.065">h</span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.051">e</span><span style="background-color: hsl(120, 100.00%, 91.26%); opacity: 0.82" title="0.023">n</span><span style="background-color: hsl(0, 100.00%, 94.35%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(0, 100.00%, 90.60%); opacity: 0.83" title="-0.025">i</span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.028"> </span><span style="background-color: hsl(120, 100.00%, 98.41%); opacity: 0.80" title="0.002">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(0, 100.00%, 91.63%); opacity: 0.82" title="-0.021">s</span><span style="background-color: hsl(0, 100.00%, 89.86%); opacity: 0.83" title="-0.028"> </span><span style="background-color: hsl(120, 100.00%, 92.74%); opacity: 0.82" title="0.017">i</span><span style="background-color: hsl(120, 100.00%, 92.46%); opacity: 0.82" title="0.018">n</span><span style="background-color: hsl(0, 100.00%, 92.46%); opacity: 0.82" title="-0.018">,</span><span style="background-color: hsl(0, 100.00%, 90.47%); opacity: 0.83" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 90.88%); opacity: 0.82" title="-0.024">th</span><span style="background-color: hsl(0, 100.00%, 89.74%); opacity: 0.83" title="-0.028">e</span><span style="background-color: hsl(0, 100.00%, 91.13%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(120, 100.00%, 96.17%); opacity: 0.81" title="0.007">x</span><span style="background-color: hsl(0, 100.00%, 97.52%); opacity: 0.80" title="-0.004">-</span><span style="background-color: hsl(120, 100.00%, 95.30%); opacity: 0.81" title="0.009">r</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(0, 100.00%, 91.64%); opacity: 0.82" title="-0.021">y</span><span style="background-color: hsl(0, 100.00%, 95.38%); opacity: 0.81" title="-0.009"> </span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.09%); opacity: 0.80" title="0.005">ec</span><span style="background-color: hsl(0, 100.00%, 96.06%); opacity: 0.81" title="-0.007">h </span><span style="opacity: 0.80">h</span><span style="background-color: hsl(120, 100.00%, 92.47%); opacity: 0.82" title="0.018">a</span><span style="background-color: hsl(120, 100.00%, 90.53%); opacity: 0.83" title="0.025">p</span><span style="background-color: hsl(120, 100.00%, 87.73%); opacity: 0.84" title="0.037">p</span><span style="background-color: hsl(120, 100.00%, 89.44%); opacity: 0.83" title="0.030">e</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(0, 100.00%, 97.74%); opacity: 0.80" title="-0.003">d</span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.007"> t</span><span style="background-color: hsl(0, 100.00%, 90.65%); opacity: 0.83" title="-0.025">o</span><span style="background-color: hsl(0, 100.00%, 91.34%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(120, 100.00%, 95.24%); opacity: 0.81" title="0.009">m</span><span style="background-color: hsl(120, 100.00%, 89.97%); opacity: 0.83" title="0.027">e</span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.017">n</span><span style="background-color: hsl(120, 100.00%, 88.04%); opacity: 0.84" title="0.035">t</span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.003">i</span><span style="background-color: hsl(0, 100.00%, 96.53%); opacity: 0.81" title="-0.006">o</span><span style="background-color: hsl(0, 100.00%, 90.50%); opacity: 0.83" title="-0.025">n</span><span style="background-color: hsl(0, 100.00%, 91.13%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">th</span><span style="background-color: hsl(120, 100.00%, 96.26%); opacity: 0.81" title="0.007">a</span><span style="background-color: hsl(120, 100.00%, 94.38%); opacity: 0.81" title="0.012">t</span><span style="background-color: hsl(120, 100.00%, 92.83%); opacity: 0.82" title="0.017"> </span><span style="background-color: hsl(120, 100.00%, 86.94%); opacity: 0.84" title="0.040">s</span><span style="background-color: hsl(120, 100.00%, 87.47%); opacity: 0.84" title="0.038">h</span><span style="background-color: hsl(120, 100.00%, 92.50%); opacity: 0.82" title="0.018">e</span><span style="background-color: hsl(0, 100.00%, 97.32%); opacity: 0.80" title="-0.004">'d</span><span style="background-color: hsl(120, 100.00%, 92.43%); opacity: 0.82" title="0.018"> </span><span style="background-color: hsl(120, 100.00%, 86.45%); opacity: 0.84" title="0.042">had</span><span style="background-color: hsl(120, 100.00%, 96.18%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(0, 100.00%, 96.12%); opacity: 0.81" title="-0.007">k</span><span style="background-color: hsl(120, 100.00%, 93.52%); opacity: 0.81" title="0.015">i</span><span style="background-color: hsl(120, 100.00%, 87.94%); opacity: 0.84" title="0.036">d</span><span style="background-color: hsl(120, 100.00%, 90.81%); opacity: 0.82" title="0.024">n</span><span style="background-color: hsl(120, 100.00%, 90.59%); opacity: 0.83" title="0.025">e</span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.013">y</span><span style="background-color: hsl(0, 100.00%, 98.00%); opacity: 0.80" title="-0.003">
    s</span><span style="background-color: hsl(0, 100.00%, 95.48%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(120, 100.00%, 98.06%); opacity: 0.80" title="0.003">o</span><span style="background-color: hsl(0, 100.00%, 99.47%); opacity: 0.80" title="-0.000">n</span><span style="background-color: hsl(120, 100.00%, 94.98%); opacity: 0.81" title="0.010">e</span><span style="background-color: hsl(0, 100.00%, 98.30%); opacity: 0.80" title="-0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.033">a</span><span style="background-color: hsl(120, 100.00%, 89.54%); opacity: 0.83" title="0.029">n</span><span style="background-color: hsl(120, 100.00%, 91.86%); opacity: 0.82" title="0.020">d</span><span style="background-color: hsl(0, 100.00%, 96.76%); opacity: 0.81" title="-0.005"> </span><span style="background-color: hsl(0, 100.00%, 96.02%); opacity: 0.81" title="-0.007">c</span><span style="background-color: hsl(120, 100.00%, 95.49%); opacity: 0.81" title="0.009">h</span><span style="background-color: hsl(0, 100.00%, 99.05%); opacity: 0.80" title="-0.001">i</span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.019">l</span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.006">d</span><span style="background-color: hsl(0, 100.00%, 87.75%); opacity: 0.84" title="-0.037">r</span><span style="background-color: hsl(0, 100.00%, 86.31%); opacity: 0.84" title="-0.043">e</span><span style="background-color: hsl(0, 100.00%, 89.13%); opacity: 0.83" title="-0.031">n</span><span style="background-color: hsl(0, 100.00%, 88.76%); opacity: 0.83" title="-0.032">,</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.033">a</span><span style="background-color: hsl(120, 100.00%, 89.54%); opacity: 0.83" title="0.029">n</span><span style="background-color: hsl(120, 100.00%, 91.86%); opacity: 0.82" title="0.020">d</span><span style="background-color: hsl(0, 100.00%, 95.17%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 90.88%); opacity: 0.82" title="-0.024">t</span><span style="background-color: hsl(0, 100.00%, 90.27%); opacity: 0.83" title="-0.026">h</span><span style="background-color: hsl(0, 100.00%, 86.66%); opacity: 0.84" title="-0.041">e</span><span style="background-color: hsl(0, 100.00%, 86.21%); opacity: 0.84" title="-0.043"> </span><span style="background-color: hsl(0, 100.00%, 95.17%); opacity: 0.81" title="-0.010">c</span><span style="background-color: hsl(120, 100.00%, 97.09%); opacity: 0.80" title="0.005">h</span><span style="background-color: hsl(0, 100.00%, 97.31%); opacity: 0.80" title="-0.004">i</span><span style="background-color: hsl(120, 100.00%, 93.23%); opacity: 0.82" title="0.016">l</span><span style="background-color: hsl(120, 100.00%, 91.04%); opacity: 0.82" title="0.023">d</span><span style="background-color: hsl(0, 100.00%, 97.76%); opacity: 0.80" title="-0.003">b</span><span style="background-color: hsl(120, 100.00%, 95.78%); opacity: 0.81" title="0.008">i</span><span style="background-color: hsl(120, 100.00%, 95.25%); opacity: 0.81" title="0.009">r</span><span style="background-color: hsl(120, 100.00%, 95.82%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">h </span><span style="opacity: 0.80">h</span><span style="background-color: hsl(120, 100.00%, 96.81%); opacity: 0.81" title="0.005">ur</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(0, 100.00%, 88.33%); opacity: 0.83" title="-0.034"> </span><span style="background-color: hsl(0, 100.00%, 94.58%); opacity: 0.81" title="-0.011">l</span><span style="background-color: hsl(120, 100.00%, 87.53%); opacity: 0.84" title="0.037">e</span><span style="background-color: hsl(120, 100.00%, 93.49%); opacity: 0.81" title="0.015">s</span><span style="background-color: hsl(0, 100.00%, 96.05%); opacity: 0.81" title="-0.007">s.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




It is getting closer, but still not there yet. The problem is that it is
much more resource intensive - you need a lot more samples to get
non-noisy results. Here explaining a single example took more time than
training the original pipeline.

Generally speaking, to do an efficient explanation we should make some
assumptions about black-box classifier, such as:

1. it uses words as features and doesn’t take word position in account;
2. it uses words as features and takes word positions in account;
3. it uses words ngrams as features;
4. it uses char ngrams as features, positions don’t matter (i.e. an
   ngram means the same everywhere);
5. it uses arbitrary attention over the text characters, i.e. every part
   of text could be potentionally important for a classifier on its own;
6. it is important to have a particular token at a particular position,
   e.g. “third token is X”, and if we delete 2nd token then prediction
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

What’s bad about this kind of failure (wrong assumption about the
black-box pipeline) is that it could be impossible to detect the failure
by looking at the scores. Scores could be high because generated dataset
is not diverse enough, not because our approximation is good.

The takeaway is that it is important to understand the “lenses” you’re
looking through when using LIME to explain a prediction.

Customizing TextExplainer: sampling
-----------------------------------

:class:`~.TextExplainer` uses :class:`~.MaskingTextSampler` or :class:`~.MaskingTextSamplers`
instances to generate texts to train on. :class:`~.MaskingTextSampler` is the
main text generation class; :class:`~.MaskingTextSamplers` provides a way to
combine multiple samplers in a single object with the same interface.

A custom sampler instance can be passed to :class:`~.TextExplainer` if we want
to experiment with sampling. For example, let’s try a sampler which
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

    As I recal from my bout with kidney stones, there isn't any
    medication that can do anything about them except relieve the ain.
    
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

    {'mean_KL_divergence': 0.71042368337755823, 'score': 0.99933430578588944}




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
    
        
        (probability <b>0.958</b>, score <b>2.434</b>)
    
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
            +2.430
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
        
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.75%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.005
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
        
    </tr>
            
            
    
            
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 81.63%); opacity: 0.87" title="0.080">s </span><span style="background-color: hsl(120, 100.00%, 80.76%); opacity: 0.87" title="0.086">i</span><span style="background-color: hsl(0, 100.00%, 72.63%); opacity: 0.92" title="-0.142"> </span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 60.35%); opacity: 1.00" title="0.241">e</span><span style="background-color: hsl(120, 100.00%, 68.33%); opacity: 0.94" title="0.175">c</span><span style="background-color: hsl(120, 100.00%, 70.83%); opacity: 0.93" title="0.155">a</span><span style="background-color: hsl(0, 100.00%, 86.23%); opacity: 0.84" title="-0.053">l</span><span style="background-color: hsl(0, 100.00%, 74.27%); opacity: 0.91" title="-0.130">l</span><span style="background-color: hsl(0, 100.00%, 75.87%); opacity: 0.90" title="-0.119"> </span><span style="background-color: hsl(0, 100.00%, 91.41%); opacity: 0.82" title="-0.027">f</span><span style="opacity: 0.80">ro</span><span style="background-color: hsl(120, 100.00%, 96.19%); opacity: 0.81" title="0.008">m my </span><span style="opacity: 0.80">bo</span><span style="background-color: hsl(120, 100.00%, 88.29%); opacity: 0.83" title="0.042">u</span><span style="background-color: hsl(120, 100.00%, 87.92%); opacity: 0.84" title="0.044">t</span><span style="background-color: hsl(120, 100.00%, 86.70%); opacity: 0.84" title="0.051"> w</span><span style="background-color: hsl(0, 100.00%, 98.77%); opacity: 0.80" title="-0.002">it</span><span style="background-color: hsl(0, 100.00%, 96.28%); opacity: 0.81" title="-0.008">h</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.39%); opacity: 0.83" title="-0.032">ki</span><span style="opacity: 0.80">dney</span><span style="background-color: hsl(120, 100.00%, 89.85%); opacity: 0.83" title="0.034"> s</span><span style="background-color: hsl(120, 100.00%, 91.39%); opacity: 0.82" title="0.027">t</span><span style="background-color: hsl(0, 100.00%, 96.57%); opacity: 0.81" title="-0.007">on</span><span style="background-color: hsl(0, 100.00%, 97.98%); opacity: 0.80" title="-0.003">es</span><span style="background-color: hsl(120, 100.00%, 72.15%); opacity: 0.92" title="0.146">,</span><span style="background-color: hsl(120, 100.00%, 73.26%); opacity: 0.91" title="0.137"> </span><span style="background-color: hsl(120, 100.00%, 94.41%); opacity: 0.81" title="0.015">th</span><span style="background-color: hsl(0, 100.00%, 90.86%); opacity: 0.82" title="-0.030">e</span><span style="background-color: hsl(120, 100.00%, 90.06%); opacity: 0.83" title="0.033">r</span><span style="background-color: hsl(120, 100.00%, 85.94%); opacity: 0.84" title="0.055">e</span><span style="background-color: hsl(120, 100.00%, 93.26%); opacity: 0.82" title="0.019"> i</span><span style="background-color: hsl(120, 100.00%, 92.40%); opacity: 0.82" title="0.023">s</span><span style="background-color: hsl(120, 100.00%, 89.56%); opacity: 0.83" title="0.036">n</span><span style="background-color: hsl(120, 100.00%, 78.84%); opacity: 0.88" title="0.098">'</span><span style="background-color: hsl(120, 100.00%, 83.55%); opacity: 0.86" title="0.069">t </span><span style="background-color: hsl(120, 100.00%, 93.43%); opacity: 0.82" title="0.018">a</span><span style="background-color: hsl(120, 100.00%, 89.49%); opacity: 0.83" title="0.036">n</span><span style="background-color: hsl(120, 100.00%, 78.03%); opacity: 0.88" title="0.104">y</span><span style="background-color: hsl(120, 100.00%, 77.23%); opacity: 0.89" title="0.109">
    </span><span style="background-color: hsl(120, 100.00%, 63.42%); opacity: 0.98" title="0.215">m</span><span style="background-color: hsl(120, 100.00%, 64.42%); opacity: 0.97" title="0.207">e</span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.081">d</span><span style="background-color: hsl(120, 100.00%, 77.50%); opacity: 0.89" title="0.107">i</span><span style="background-color: hsl(120, 100.00%, 72.80%); opacity: 0.92" title="0.141">c</span><span style="background-color: hsl(120, 100.00%, 94.12%); opacity: 0.81" title="0.016">at</span><span style="background-color: hsl(0, 100.00%, 84.16%); opacity: 0.85" title="-0.065">i</span><span style="background-color: hsl(0, 100.00%, 79.18%); opacity: 0.88" title="-0.096">o</span><span style="background-color: hsl(0, 100.00%, 80.98%); opacity: 0.87" title="-0.084">n</span><span style="background-color: hsl(0, 100.00%, 93.83%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 93.93%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(120, 100.00%, 93.77%); opacity: 0.81" title="0.017">h</span><span style="background-color: hsl(0, 100.00%, 87.17%); opacity: 0.84" title="-0.048">a</span><span style="background-color: hsl(0, 100.00%, 85.64%); opacity: 0.85" title="-0.057">t</span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 96.12%); opacity: 0.81" title="0.009">c</span><span style="background-color: hsl(0, 100.00%, 85.62%); opacity: 0.85" title="-0.057">a</span><span style="background-color: hsl(0, 100.00%, 87.15%); opacity: 0.84" title="-0.048">n </span><span style="background-color: hsl(120, 100.00%, 91.31%); opacity: 0.82" title="0.028">do</span><span style="opacity: 0.80"> an</span><span style="background-color: hsl(0, 100.00%, 98.03%); opacity: 0.80" title="-0.003">yt</span><span style="background-color: hsl(0, 100.00%, 89.57%); opacity: 0.83" title="-0.036">h</span><span style="background-color: hsl(0, 100.00%, 82.92%); opacity: 0.86" title="-0.072">i</span><span style="background-color: hsl(0, 100.00%, 89.41%); opacity: 0.83" title="-0.037">n</span><span style="opacity: 0.80">g abo</span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.010">ut</span><span style="background-color: hsl(0, 100.00%, 93.44%); opacity: 0.82" title="-0.018"> t</span><span style="background-color: hsl(120, 100.00%, 90.45%); opacity: 0.83" title="0.032">he</span><span style="background-color: hsl(120, 100.00%, 88.77%); opacity: 0.83" title="0.040">m</span><span style="opacity: 0.80"> ex</span><span style="background-color: hsl(120, 100.00%, 86.50%); opacity: 0.84" title="0.052">c</span><span style="background-color: hsl(120, 100.00%, 81.04%); opacity: 0.87" title="0.084">ep</span><span style="background-color: hsl(120, 100.00%, 90.27%); opacity: 0.83" title="0.032">t</span><span style="background-color: hsl(0, 100.00%, 75.98%); opacity: 0.90" title="-0.118"> </span><span style="background-color: hsl(0, 100.00%, 73.70%); opacity: 0.91" title="-0.134">r</span><span style="background-color: hsl(120, 100.00%, 87.68%); opacity: 0.84" title="0.045">e</span><span style="background-color: hsl(0, 100.00%, 83.34%); opacity: 0.86" title="-0.070">li</span><span style="background-color: hsl(0, 100.00%, 82.72%); opacity: 0.86" title="-0.074">e</span><span style="background-color: hsl(120, 100.00%, 90.51%); opacity: 0.83" title="0.031">v</span><span style="background-color: hsl(0, 100.00%, 92.60%); opacity: 0.82" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 82.08%); opacity: 0.86" title="-0.078"> </span><span style="background-color: hsl(0, 100.00%, 75.86%); opacity: 0.90" title="-0.119">t</span><span style="background-color: hsl(0, 100.00%, 79.61%); opacity: 0.88" title="-0.093">h</span><span style="background-color: hsl(0, 100.00%, 84.62%); opacity: 0.85" title="-0.062">e</span><span style="background-color: hsl(0, 100.00%, 86.17%); opacity: 0.84" title="-0.054"> </span><span style="background-color: hsl(0, 100.00%, 96.96%); opacity: 0.81" title="-0.006">p</span><span style="background-color: hsl(120, 100.00%, 88.06%); opacity: 0.84" title="0.043">a</span><span style="background-color: hsl(0, 100.00%, 91.83%); opacity: 0.82" title="-0.025">in</span><span style="background-color: hsl(0, 100.00%, 96.07%); opacity: 0.81" title="-0.009">. e</span><span style="background-color: hsl(0, 100.00%, 92.22%); opacity: 0.82" title="-0.024">i</span><span style="background-color: hsl(0, 100.00%, 85.12%); opacity: 0.85" title="-0.059">t</span><span style="background-color: hsl(0, 100.00%, 86.71%); opacity: 0.84" title="-0.051">h</span><span style="background-color: hsl(0, 100.00%, 85.77%); opacity: 0.85" title="-0.056">e</span><span style="background-color: hsl(0, 100.00%, 86.95%); opacity: 0.84" title="-0.049">r</span><span style="background-color: hsl(0, 100.00%, 89.50%); opacity: 0.83" title="-0.036"> </span><span style="background-color: hsl(0, 100.00%, 96.27%); opacity: 0.81" title="-0.008">the</span><span style="opacity: 0.80">y </span><span style="background-color: hsl(0, 100.00%, 91.54%); opacity: 0.82" title="-0.027">p</span><span style="background-color: hsl(0, 100.00%, 86.03%); opacity: 0.84" title="-0.054">a</span><span style="background-color: hsl(0, 100.00%, 83.77%); opacity: 0.86" title="-0.067">ss</span><span style="background-color: hsl(120, 100.00%, 93.01%); opacity: 0.82" title="0.020">,</span><span style="background-color: hsl(120, 100.00%, 92.59%); opacity: 0.82" title="0.022"> </span><span style="background-color: hsl(0, 100.00%, 73.89%); opacity: 0.91" title="-0.133">o</span><span style="background-color: hsl(0, 100.00%, 81.44%); opacity: 0.87" title="-0.082">r</span><span style="background-color: hsl(0, 100.00%, 96.27%); opacity: 0.81" title="-0.008"> t</span><span style="background-color: hsl(0, 100.00%, 92.14%); opacity: 0.82" title="-0.024">he</span><span style="background-color: hsl(120, 100.00%, 89.76%); opacity: 0.83" title="0.035">y </span><span style="background-color: hsl(0, 100.00%, 97.25%); opacity: 0.80" title="-0.005">h</span><span style="background-color: hsl(0, 100.00%, 93.80%); opacity: 0.81" title="-0.017">a</span><span style="background-color: hsl(0, 100.00%, 84.22%); opacity: 0.85" title="-0.065">v</span><span style="background-color: hsl(120, 100.00%, 96.66%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 96.22%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 97.18%); opacity: 0.80" title="0.006">t</span><span style="background-color: hsl(0, 100.00%, 89.94%); opacity: 0.83" title="-0.034">o </span><span style="background-color: hsl(120, 100.00%, 77.53%); opacity: 0.89" title="0.107">b</span><span style="background-color: hsl(120, 100.00%, 79.16%); opacity: 0.88" title="0.096">e</span><span style="background-color: hsl(0, 100.00%, 87.53%); opacity: 0.84" title="-0.046"> </span><span style="background-color: hsl(0, 100.00%, 95.09%); opacity: 0.81" title="-0.012">b</span><span style="background-color: hsl(0, 100.00%, 88.84%); opacity: 0.83" title="-0.039">ro</span><span style="background-color: hsl(0, 100.00%, 91.39%); opacity: 0.82" title="-0.027">k</span><span style="background-color: hsl(0, 100.00%, 93.28%); opacity: 0.82" title="-0.019">e</span><span style="opacity: 0.80">n up</span><span style="background-color: hsl(120, 100.00%, 96.83%); opacity: 0.81" title="0.007"> w</span><span style="background-color: hsl(0, 100.00%, 98.77%); opacity: 0.80" title="-0.002">i</span><span style="background-color: hsl(0, 100.00%, 96.12%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(0, 100.00%, 86.10%); opacity: 0.84" title="-0.054">h</span><span style="background-color: hsl(0, 100.00%, 95.36%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 92.50%); opacity: 0.82" title="-0.022">s</span><span style="background-color: hsl(0, 100.00%, 85.59%); opacity: 0.85" title="-0.057">o</span><span style="background-color: hsl(120, 100.00%, 83.91%); opacity: 0.85" title="0.066">un</span><span style="opacity: 0.80">d</span><span style="background-color: hsl(120, 100.00%, 80.49%); opacity: 0.87" title="0.088">,</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.035"> </span><span style="background-color: hsl(0, 100.00%, 75.71%); opacity: 0.90" title="-0.120">o</span><span style="background-color: hsl(0, 100.00%, 81.44%); opacity: 0.87" title="-0.082">r</span><span style="background-color: hsl(0, 100.00%, 96.27%); opacity: 0.81" title="-0.008"> t</span><span style="background-color: hsl(0, 100.00%, 92.14%); opacity: 0.82" title="-0.024">he</span><span style="background-color: hsl(120, 100.00%, 89.76%); opacity: 0.83" title="0.035">y </span><span style="background-color: hsl(0, 100.00%, 96.75%); opacity: 0.81" title="-0.007">h</span><span style="background-color: hsl(0, 100.00%, 83.10%); opacity: 0.86" title="-0.071">a</span><span style="background-color: hsl(0, 100.00%, 67.92%); opacity: 0.95" title="-0.178">v</span><span style="background-color: hsl(0, 100.00%, 77.62%); opacity: 0.89" title="-0.107">e</span><span style="background-color: hsl(0, 100.00%, 87.15%); opacity: 0.84" title="-0.048">
    </span><span style="background-color: hsl(120, 100.00%, 91.34%); opacity: 0.82" title="0.027">t</span><span style="background-color: hsl(0, 100.00%, 87.25%); opacity: 0.84" title="-0.048">o</span><span style="background-color: hsl(0, 100.00%, 91.53%); opacity: 0.82" title="-0.027"> </span><span style="background-color: hsl(120, 100.00%, 74.91%); opacity: 0.90" title="0.125">b</span><span style="background-color: hsl(120, 100.00%, 75.96%); opacity: 0.90" title="0.118">e</span><span style="background-color: hsl(0, 100.00%, 92.31%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(120, 100.00%, 91.01%); opacity: 0.82" title="0.029">e</span><span style="background-color: hsl(120, 100.00%, 80.23%); opacity: 0.87" title="0.089">x</span><span style="background-color: hsl(120, 100.00%, 71.38%); opacity: 0.92" title="0.151">t</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.244">r</span><span style="background-color: hsl(120, 100.00%, 66.08%); opacity: 0.96" title="0.193">a</span><span style="background-color: hsl(120, 100.00%, 79.69%); opacity: 0.88" title="0.093">c</span><span style="background-color: hsl(120, 100.00%, 83.55%); opacity: 0.86" title="0.069">t</span><span style="background-color: hsl(120, 100.00%, 86.02%); opacity: 0.84" title="0.054">e</span><span style="background-color: hsl(120, 100.00%, 90.46%); opacity: 0.83" title="0.032">d</span><span style="background-color: hsl(120, 100.00%, 84.00%); opacity: 0.85" title="0.066"> </span><span style="background-color: hsl(120, 100.00%, 89.80%); opacity: 0.83" title="0.035">s</span><span style="background-color: hsl(120, 100.00%, 90.58%); opacity: 0.83" title="0.031">u</span><span style="background-color: hsl(120, 100.00%, 89.62%); opacity: 0.83" title="0.036">r</span><span style="background-color: hsl(120, 100.00%, 85.15%); opacity: 0.85" title="0.059">g</span><span style="background-color: hsl(120, 100.00%, 82.95%); opacity: 0.86" title="0.072">i</span><span style="background-color: hsl(0, 100.00%, 99.61%); opacity: 0.80" title="-0.000">c</span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.013">a</span><span style="background-color: hsl(120, 100.00%, 91.42%); opacity: 0.82" title="0.027">ll</span><span style="background-color: hsl(120, 100.00%, 89.03%); opacity: 0.83" title="0.038">y</span><span style="opacity: 0.80">. w</span><span style="background-color: hsl(120, 100.00%, 94.41%); opacity: 0.81" title="0.015">h</span><span style="background-color: hsl(120, 100.00%, 90.00%); opacity: 0.83" title="0.034">e</span><span style="background-color: hsl(120, 100.00%, 87.74%); opacity: 0.84" title="0.045">n </span><span style="background-color: hsl(120, 100.00%, 83.54%); opacity: 0.86" title="0.069">i</span><span style="background-color: hsl(0, 100.00%, 97.93%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 92.48%); opacity: 0.82" title="0.022">wa</span><span style="background-color: hsl(120, 100.00%, 74.28%); opacity: 0.91" title="0.130">s</span><span style="background-color: hsl(120, 100.00%, 87.80%); opacity: 0.84" title="0.045"> </span><span style="background-color: hsl(120, 100.00%, 96.29%); opacity: 0.81" title="0.008">i</span><span style="background-color: hsl(0, 100.00%, 76.84%); opacity: 0.89" title="-0.112">n</span><span style="background-color: hsl(120, 100.00%, 83.26%); opacity: 0.86" title="0.070">,</span><span style="background-color: hsl(120, 100.00%, 86.38%); opacity: 0.84" title="0.052"> </span><span style="background-color: hsl(0, 100.00%, 85.09%); opacity: 0.85" title="-0.060">t</span><span style="background-color: hsl(0, 100.00%, 91.67%); opacity: 0.82" title="-0.026">h</span><span style="background-color: hsl(0, 100.00%, 85.80%); opacity: 0.85" title="-0.056">e</span><span style="background-color: hsl(0, 100.00%, 80.93%); opacity: 0.87" title="-0.085"> </span><span style="background-color: hsl(0, 100.00%, 89.26%); opacity: 0.83" title="-0.037">x</span><span style="background-color: hsl(0, 100.00%, 93.32%); opacity: 0.82" title="-0.019">-</span><span style="background-color: hsl(120, 100.00%, 79.62%); opacity: 0.88" title="0.093">ra</span><span style="opacity: 0.80">y</span><span style="background-color: hsl(120, 100.00%, 98.10%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 90.66%); opacity: 0.83" title="0.031">t</span><span style="background-color: hsl(120, 100.00%, 90.37%); opacity: 0.83" title="0.032">ech</span><span style="background-color: hsl(120, 100.00%, 93.66%); opacity: 0.81" title="0.018"> </span><span style="background-color: hsl(0, 100.00%, 92.66%); opacity: 0.82" title="-0.022">ha</span><span style="background-color: hsl(0, 100.00%, 90.29%); opacity: 0.83" title="-0.032">p</span><span style="background-color: hsl(0, 100.00%, 94.05%); opacity: 0.81" title="-0.016">pe</span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.011">n</span><span style="opacity: 0.80">ed to </span><span style="background-color: hsl(120, 100.00%, 83.68%); opacity: 0.86" title="0.068">m</span><span style="background-color: hsl(120, 100.00%, 77.38%); opacity: 0.89" title="0.108">e</span><span style="background-color: hsl(120, 100.00%, 76.63%); opacity: 0.89" title="0.113">nt</span><span style="background-color: hsl(120, 100.00%, 90.43%); opacity: 0.83" title="0.032">i</span><span style="background-color: hsl(0, 100.00%, 88.97%); opacity: 0.83" title="-0.039">o</span><span style="background-color: hsl(0, 100.00%, 80.98%); opacity: 0.87" title="-0.084">n</span><span style="background-color: hsl(0, 100.00%, 93.83%); opacity: 0.81" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 93.93%); opacity: 0.81" title="-0.017">t</span><span style="opacity: 0.80">h</span><span style="background-color: hsl(0, 100.00%, 82.49%); opacity: 0.86" title="-0.075">a</span><span style="background-color: hsl(0, 100.00%, 93.52%); opacity: 0.81" title="-0.018">t</span><span style="background-color: hsl(120, 100.00%, 79.38%); opacity: 0.88" title="0.095"> </span><span style="background-color: hsl(120, 100.00%, 78.44%); opacity: 0.88" title="0.101">s</span><span style="background-color: hsl(120, 100.00%, 95.84%); opacity: 0.81" title="0.010">h</span><span style="background-color: hsl(120, 100.00%, 93.20%); opacity: 0.82" title="0.019">e</span><span style="opacity: 0.80">'d</span><span style="background-color: hsl(120, 100.00%, 89.86%); opacity: 0.83" title="0.034"> </span><span style="background-color: hsl(120, 100.00%, 87.51%); opacity: 0.84" title="0.046">h</span><span style="background-color: hsl(120, 100.00%, 80.51%); opacity: 0.87" title="0.087">a</span><span style="background-color: hsl(120, 100.00%, 78.76%); opacity: 0.88" title="0.099">d </span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.020">ki</span><span style="background-color: hsl(120, 100.00%, 95.31%); opacity: 0.81" title="0.011">d</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 94.42%); opacity: 0.81" title="0.015">ey</span><span style="background-color: hsl(120, 100.00%, 90.20%); opacity: 0.83" title="0.033">
    s</span><span style="background-color: hsl(120, 100.00%, 85.03%); opacity: 0.85" title="0.060">t</span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.004">on</span><span style="background-color: hsl(0, 100.00%, 92.66%); opacity: 0.82" title="-0.022">es</span><span style="background-color: hsl(0, 100.00%, 93.50%); opacity: 0.81" title="-0.018"> </span><span style="background-color: hsl(120, 100.00%, 90.16%); opacity: 0.83" title="0.033">a</span><span style="background-color: hsl(120, 100.00%, 94.34%); opacity: 0.81" title="0.015">nd</span><span style="background-color: hsl(120, 100.00%, 92.62%); opacity: 0.82" title="0.022"> </span><span style="background-color: hsl(120, 100.00%, 91.67%); opacity: 0.82" title="0.026">c</span><span style="background-color: hsl(0, 100.00%, 92.16%); opacity: 0.82" title="-0.024">h</span><span style="background-color: hsl(0, 100.00%, 83.52%); opacity: 0.86" title="-0.069">i</span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.022">ld</span><span style="background-color: hsl(120, 100.00%, 80.59%); opacity: 0.87" title="0.087">re</span><span style="background-color: hsl(120, 100.00%, 94.22%); opacity: 0.81" title="0.015">n</span><span style="background-color: hsl(120, 100.00%, 74.26%); opacity: 0.91" title="0.130">,</span><span style="background-color: hsl(120, 100.00%, 73.62%); opacity: 0.91" title="0.135"> </span><span style="background-color: hsl(120, 100.00%, 97.88%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(0, 100.00%, 93.35%); opacity: 0.82" title="-0.019">n</span><span style="background-color: hsl(0, 100.00%, 96.73%); opacity: 0.81" title="-0.007">d</span><span style="background-color: hsl(0, 100.00%, 81.99%); opacity: 0.86" title="-0.078"> </span><span style="background-color: hsl(0, 100.00%, 85.00%); opacity: 0.85" title="-0.060">t</span><span style="background-color: hsl(0, 100.00%, 85.80%); opacity: 0.85" title="-0.056">he</span><span style="background-color: hsl(0, 100.00%, 91.93%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.009">c</span><span style="background-color: hsl(0, 100.00%, 92.16%); opacity: 0.82" title="-0.024">h</span><span style="background-color: hsl(0, 100.00%, 83.52%); opacity: 0.86" title="-0.069">i</span><span style="background-color: hsl(120, 100.00%, 93.61%); opacity: 0.81" title="0.018">ld</span><span style="background-color: hsl(120, 100.00%, 88.68%); opacity: 0.83" title="0.040">b</span><span style="background-color: hsl(120, 100.00%, 93.49%); opacity: 0.81" title="0.018">i</span><span style="background-color: hsl(0, 100.00%, 89.10%); opacity: 0.83" title="-0.038">rt</span><span style="background-color: hsl(0, 100.00%, 93.67%); opacity: 0.81" title="-0.018">h</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.11%); opacity: 0.86" title="-0.077">hu</span><span style="background-color: hsl(0, 100.00%, 91.45%); opacity: 0.82" title="-0.027">rt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Note that accuracy score is perfect, but KL divergence is bad. It means
this sampler was not very useful: most generated texts were “easy” in
sense that most (or all?) of them should be still classified as
``sci.med``, so it was easy to get a good accuracy. But because
generated texts were not diverse enough classifier haven’t learned
anything useful; it’s having a hard time predicting the probability
output of the black-box pipeline on a held-out dataset.

By default :class:`~.TextExplainer` uses a mix of several sampling strategies
which seems to work OK for token-based explanations. But a good sampling
strategy which works for many real-world tasks could be a research topic
on itself. If you’ve got some experience with it we’d love to hear from
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

    {'mean_KL_divergence': 0.037836554598348969, 'score': 0.9838155527960798}




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
                            0.5461
                            
                        </td>
                        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                            kidney
                        </td>
                    </tr>
                
                    <tr style="background-color: hsl(120, 100.00%, 82.43%); border: none;">
                        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                            0.4539
                            
                        </td>
                        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                            pain
                        </td>
                    </tr>
                
                
                </tbody>
            </table>
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
            
            <br>
            <pre><svg width="520pt" height="180pt"
     viewBox="0.00 0.00 790.00 280.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <g id="graph1" class="graph" transform="scale(1 1) rotate(0) translate(4 276)">
    <title>Tree</title>
    <polygon fill="white" stroke="white" points="-4,5 -4,-276 787,-276 787,5 -4,5"/>
    <!-- 0 -->
    <g id="node1" class="node"><title>0</title>
    <polygon fill="none" stroke="black" points="492,-272 302,-272 302,-200 492,-200 492,-272"/>
    <text text-anchor="middle" x="397" y="-255.4" font-family="Times,serif" font-size="14.00">kidney &lt;= 0.5</text>
    <text text-anchor="middle" x="397" y="-239.4" font-family="Times,serif" font-size="14.00">gini = 0.1561</text>
    <text text-anchor="middle" x="397" y="-223.4" font-family="Times,serif" font-size="14.00">samples = 100.0%</text>
    <text text-anchor="middle" x="397" y="-207.4" font-family="Times,serif" font-size="14.00">value = [0.01, 0.03, 0.92, 0.04]</text>
    </g>
    <!-- 1 -->
    <g id="node2" class="node"><title>1</title>
    <polygon fill="none" stroke="black" points="390,-164 200,-164 200,-92 390,-92 390,-164"/>
    <text text-anchor="middle" x="295" y="-147.4" font-family="Times,serif" font-size="14.00">pain &lt;= 0.5</text>
    <text text-anchor="middle" x="295" y="-131.4" font-family="Times,serif" font-size="14.00">gini = 0.3834</text>
    <text text-anchor="middle" x="295" y="-115.4" font-family="Times,serif" font-size="14.00">samples = 38.9%</text>
    <text text-anchor="middle" x="295" y="-99.4" font-family="Times,serif" font-size="14.00">value = [0.03, 0.09, 0.77, 0.11]</text>
    </g>
    <!-- 0&#45;&gt;1 -->
    <g id="edge2" class="edge"><title>0&#45;&gt;1</title>
    <path fill="none" stroke="black" d="M363.319,-199.998C354.619,-190.957 345.156,-181.123 336.124,-171.736"/>
    <polygon fill="black" stroke="black" points="338.45,-169.106 328.994,-164.328 333.406,-173.96 338.45,-169.106"/>
    <text text-anchor="middle" x="328.514" y="-184.723" font-family="Times,serif" font-size="14.00">True</text>
    </g>
    <!-- 4 -->
    <g id="node8" class="node"><title>4</title>
    <polygon fill="none" stroke="black" points="591.25,-164 408.75,-164 408.75,-92 591.25,-92 591.25,-164"/>
    <text text-anchor="middle" x="500" y="-147.4" font-family="Times,serif" font-size="14.00">pain &lt;= 0.5</text>
    <text text-anchor="middle" x="500" y="-131.4" font-family="Times,serif" font-size="14.00">gini = 0.0456</text>
    <text text-anchor="middle" x="500" y="-115.4" font-family="Times,serif" font-size="14.00">samples = 61.1%</text>
    <text text-anchor="middle" x="500" y="-99.4" font-family="Times,serif" font-size="14.00">value = [0.0, 0.01, 0.98, 0.01]</text>
    </g>
    <!-- 0&#45;&gt;4 -->
    <g id="edge8" class="edge"><title>0&#45;&gt;4</title>
    <path fill="none" stroke="black" d="M431.012,-199.998C439.796,-190.957 449.352,-181.123 458.473,-171.736"/>
    <polygon fill="black" stroke="black" points="461.214,-173.939 465.672,-164.328 456.193,-169.06 461.214,-173.939"/>
    <text text-anchor="middle" x="466.031" y="-184.725" font-family="Times,serif" font-size="14.00">False</text>
    </g>
    <!-- 2 -->
    <g id="node4" class="node"><title>2</title>
    <polygon fill="none" stroke="black" points="190,-56 0,-56 0,-0 190,-0 190,-56"/>
    <text text-anchor="middle" x="95" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.5185</text>
    <text text-anchor="middle" x="95" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 28.4%</text>
    <text text-anchor="middle" x="95" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.04, 0.14, 0.66, 0.16]</text>
    </g>
    <!-- 1&#45;&gt;2 -->
    <g id="edge4" class="edge"><title>1&#45;&gt;2</title>
    <path fill="none" stroke="black" d="M223.36,-91.8966C202.462,-81.6566 179.763,-70.534 159.346,-60.5294"/>
    <polygon fill="black" stroke="black" points="160.66,-57.2758 150.14,-56.0186 157.58,-63.5617 160.66,-57.2758"/>
    </g>
    <!-- 3 -->
    <g id="node6" class="node"><title>3</title>
    <polygon fill="none" stroke="black" points="384,-56 208,-56 208,-0 384,-0 384,-56"/>
    <text text-anchor="middle" x="296" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.0434</text>
    <text text-anchor="middle" x="296" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 10.6%</text>
    <text text-anchor="middle" x="296" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.0, 0.0, 0.98, 0.02]</text>
    </g>
    <!-- 1&#45;&gt;3 -->
    <g id="edge6" class="edge"><title>1&#45;&gt;3</title>
    <path fill="none" stroke="black" d="M295.358,-91.8966C295.443,-83.6325 295.533,-74.7936 295.618,-66.4314"/>
    <polygon fill="black" stroke="black" points="299.12,-66.2086 295.723,-56.1734 292.121,-66.1371 299.12,-66.2086"/>
    </g>
    <!-- 5 -->
    <g id="node10" class="node"><title>5</title>
    <polygon fill="none" stroke="black" points="594,-56 404,-56 404,-0 594,-0 594,-56"/>
    <text text-anchor="middle" x="499" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.1153</text>
    <text text-anchor="middle" x="499" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 22.8%</text>
    <text text-anchor="middle" x="499" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.01, 0.02, 0.94, 0.04]</text>
    </g>
    <!-- 4&#45;&gt;5 -->
    <g id="edge10" class="edge"><title>4&#45;&gt;5</title>
    <path fill="none" stroke="black" d="M499.642,-91.8966C499.557,-83.6325 499.467,-74.7936 499.382,-66.4314"/>
    <polygon fill="black" stroke="black" points="502.879,-66.1371 499.277,-56.1734 495.88,-66.2086 502.879,-66.1371"/>
    </g>
    <!-- 6 -->
    <g id="node12" class="node"><title>6</title>
    <polygon fill="none" stroke="black" points="782,-56 612,-56 612,-0 782,-0 782,-56"/>
    <text text-anchor="middle" x="697" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.0114</text>
    <text text-anchor="middle" x="697" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 38.2%</text>
    <text text-anchor="middle" x="697" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.0, 0.0, 0.99, 0.0]</text>
    </g>
    <!-- 4&#45;&gt;6 -->
    <g id="edge12" class="edge"><title>4&#45;&gt;6</title>
    <path fill="none" stroke="black" d="M570.565,-91.8966C591.149,-81.6566 613.508,-70.534 633.619,-60.5294"/>
    <polygon fill="black" stroke="black" points="635.293,-63.6062 642.687,-56.0186 632.175,-57.3389 635.293,-63.6062"/>
    </g>
    </g>
    </svg>
    </pre>
        
    
    
    




How to read it: “kidney <= 0.5” means “word ‘kidney’ is not in the
document” (we’re explaining the orginal LDA+SVM pipeline again).

So according to this tree if “kidney” is not in the document and “pain”
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
    0.013 alt.atheism
    0.022 comp.graphics
    0.894 sci.med
    0.072 soc.religion.christian
    
    only 'pain' removed:
    0.002 alt.atheism
    0.004 comp.graphics
    0.979 sci.med
    0.015 soc.religion.christian


As expected, after removing both words probability of ``sci.med``
decreased, though not as much as our simple decision tree predicted (to
0.9 instead of 0.64). Removing ``pain`` provided exactly the same effect
as predicted - probability of ``sci.med`` became ``0.98``.
