
TextExplainer: debugging complex text processing pipelines using LIME
=====================================================================

While eli5 supports many classifiers and preprocessing methods, it can't
support them all.

If a library is not supported by eli5 directly, or the text processing
pipeline is too complex for eli5, eli5 can still help - it provides an
implementation of LIME (Ribeiro et al., 2016) algorithm which allows to
explain predictions of arbitrary classifiers (including text
classifiers). ``eli5.lime`` can also help when it is hard to get exact
mapping between model coefficients and text features, e.g. if there is
dimension reduction involved.

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

    0.000 alt.atheism
    0.000 comp.graphics
    0.996 sci.med
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
    
    te = TextExplainer()
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
    
        
        (probability <b>0.000</b>, score <b>-8.040</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 97.46%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.400
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -7.640
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.060">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.34%); opacity: 0.82" title="-0.185">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.94%); opacity: 0.82" title="-0.287">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.24%); opacity: 0.82" title="-0.230">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.29%); opacity: 0.81" title="-0.080">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.005">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.26%); opacity: 0.82" title="-0.188">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.53%); opacity: 0.86" title="-0.673">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.38%); opacity: 0.85" title="-0.567">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.29%); opacity: 0.80" title="-0.026">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.91%); opacity: 0.86" title="0.710">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 94.69%); opacity: 0.81" title="0.134">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.14%); opacity: 0.81" title="-0.154">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 89.46%); opacity: 0.83" title="-0.356">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.48%); opacity: 0.81" title="0.106">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.05%); opacity: 0.82" title="0.196">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.081">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.004">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.07%); opacity: 0.82" title="0.195">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.70%); opacity: 0.80" title="0.040">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.041">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.23%); opacity: 0.81" title="0.150">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.22%); opacity: 0.82" title="-0.190">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.99%); opacity: 0.90" title="-1.153">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 96.76%); opacity: 0.81" title="-0.066">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.83%); opacity: 0.81" title="-0.095">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.69%); opacity: 0.81" title="0.068">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 95.24%); opacity: 0.81" title="0.114">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.17%); opacity: 0.81" title="0.084">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.066">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.135">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.104">be</span><span style="opacity: 0.80"> broken </span><span style="background-color: hsl(0, 100.00%, 95.65%); opacity: 0.81" title="-0.100">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.16%); opacity: 0.81" title="-0.117">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.83%); opacity: 0.80" title="-0.037">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 96.32%); opacity: 0.81" title="0.079">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.17%); opacity: 0.81" title="0.084">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.066">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.135">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.104">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.01%); opacity: 0.80" title="-0.033">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.86%); opacity: 0.82" title="0.290">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 95.51%); opacity: 0.81" title="0.105">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.34%); opacity: 0.82" title="-0.185">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.45%); opacity: 0.80" title="-0.047">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.31%); opacity: 0.81" title="0.079">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.30%); opacity: 0.80" title="-0.026">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.67%); opacity: 0.82" title="-0.299">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 94.60%); opacity: 0.81" title="-0.137">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.70%); opacity: 0.82" title="-0.253">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.93%); opacity: 0.81" title="0.125">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.16%); opacity: 0.81" title="0.117">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.58%); opacity: 0.84" title="0.450">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.48%); opacity: 0.81" title="0.179">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.008">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 94.56%); opacity: 0.81" title="0.138">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.35%); opacity: 0.81" title="0.110">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.53%); opacity: 0.86" title="-0.673">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 85.38%); opacity: 0.85" title="-0.567">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.95%); opacity: 0.80" title="-0.034">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.067">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.022">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.64%); opacity: 0.81" title="0.069">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.39%); opacity: 0.80" title="0.024">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.44%); opacity: 0.82" title="-0.221">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.02%); opacity: 0.80" title="-0.059">less</span><span style="opacity: 0.80">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=comp.graphics
        
    </b>
    
        
        (probability <b>0.000</b>, score <b>-7.854</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.28%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.230
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 80.03%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -7.623
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 95.05%); opacity: 0.81" title="0.121">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.015">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.36%); opacity: 0.84" title="-0.461">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.03%); opacity: 0.80" title="-0.012">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.98%); opacity: 0.81" title="0.123">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.97%); opacity: 0.82" title="0.285">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.25%); opacity: 0.81" title="0.114">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.82%); opacity: 0.88" title="-0.900">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.33%); opacity: 0.86" title="-0.685">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.017">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.68%); opacity: 0.80" title="-0.018">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 99.49%); opacity: 0.80" title="-0.005">t</span><span style="opacity: 0.80"> any
    </span><span style="background-color: hsl(0, 100.00%, 82.60%); opacity: 0.86" title="-0.728">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.53%); opacity: 0.81" title="-0.073">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.008">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.51%); opacity: 0.81" title="0.140">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.29%); opacity: 0.80" title="0.026">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.53%); opacity: 0.80" title="-0.021">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.077">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.77%); opacity: 0.81" title="0.168">except</span><span style="opacity: 0.80"> relieve </span><span style="background-color: hsl(0, 100.00%, 89.93%); opacity: 0.83" title="-0.333">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.89%); opacity: 0.96" title="-1.904">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 91.51%); opacity: 0.82" title="-0.261">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.31%); opacity: 0.80" title="-0.051">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.69%); opacity: 0.81" title="0.171">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 96.04%); opacity: 0.81" title="0.088">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.02%); opacity: 0.80" title="-0.012">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.66%); opacity: 0.82" title="0.212">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.02%); opacity: 0.81" title="0.089">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.008">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.46%); opacity: 0.81" title="-0.075">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.76%); opacity: 0.81" title="-0.066">up</span><span style="opacity: 0.80"> with sound, </span><span style="background-color: hsl(0, 100.00%, 95.04%); opacity: 0.81" title="-0.121">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.02%); opacity: 0.80" title="-0.012">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.66%); opacity: 0.82" title="0.212">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 96.02%); opacity: 0.81" title="0.089">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.008">be</span><span style="opacity: 0.80"> extracted surgically.
    
    </span><span style="background-color: hsl(0, 100.00%, 95.17%); opacity: 0.81" title="-0.117">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.008">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.33%); opacity: 0.81" title="0.147">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.39%); opacity: 0.80" title="0.049">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 97.41%); opacity: 0.80" title="0.048">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.80%); opacity: 0.81" title="0.065">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(120, 100.00%, 76.75%); opacity: 0.89" title="1.101">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.160">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.34%); opacity: 0.81" title="-0.146">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.25%); opacity: 0.81" title="-0.150">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.36%); opacity: 0.80" title="0.049">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.43%); opacity: 0.80" title="0.023">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.76%); opacity: 0.81" title="-0.168">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 98.12%); opacity: 0.80" title="0.030">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.12%); opacity: 0.80" title="0.030">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.82%); opacity: 0.88" title="-0.900">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 82.83%); opacity: 0.86" title="-0.714">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.10%); opacity: 0.80" title="-0.011">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.16%); opacity: 0.84" title="-0.525">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.071">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.68%); opacity: 0.81" title="-0.068">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.32%); opacity: 0.81" title="-0.079">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.50%); opacity: 0.83" title="-0.354">hurt</span><span style="opacity: 0.80"> less.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=sci.med
        
    </b>
    
        
        (probability <b>0.998</b>, score <b>7.030</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 81.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +7.102
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.24%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.072
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.065">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.44%); opacity: 0.80" title="0.047">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.21%); opacity: 0.85" title="0.577">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.90%); opacity: 0.80" title="0.036">from</span><span style="opacity: 0.80"> my </span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.100">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.41%); opacity: 0.80" title="-0.048">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.93%); opacity: 0.89" title="1.022">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.41%); opacity: 0.87" title="0.800">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 97.10%); opacity: 0.80" title="-0.056">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.02%); opacity: 0.81" title="-0.088">isn</span><span style="opacity: 0.80">'t any
    </span><span style="background-color: hsl(120, 100.00%, 80.28%); opacity: 0.87" title="0.871">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.42%); opacity: 0.81" title="-0.076">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.24%); opacity: 0.82" title="-0.188">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.133">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.32%); opacity: 0.81" title="-0.111">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.047">about</span><span style="opacity: 0.80"> them except </span><span style="background-color: hsl(120, 100.00%, 93.81%); opacity: 0.81" title="0.166">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.20%); opacity: 0.82" title="0.190">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="2.391">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 91.53%); opacity: 0.82" title="0.260">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.88%); opacity: 0.80" title="-0.036">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.08%); opacity: 0.82" title="-0.195">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 94.82%); opacity: 0.81" title="-0.129">or</span><span style="opacity: 0.80"> they </span><span style="background-color: hsl(0, 100.00%, 96.98%); opacity: 0.80" title="-0.060">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.07%); opacity: 0.80" title="-0.011">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.009">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.32%); opacity: 0.81" title="-0.147">broken</span><span style="opacity: 0.80"> up </span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.037">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.43%); opacity: 0.81" title="-0.076">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 98.13%); opacity: 0.80" title="0.030">or</span><span style="opacity: 0.80"> they </span><span style="background-color: hsl(0, 100.00%, 96.98%); opacity: 0.80" title="-0.060">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 99.07%); opacity: 0.80" title="-0.011">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.009">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.30%); opacity: 0.81" title="-0.080">extracted</span><span style="opacity: 0.80"> surgically.
    
    </span><span style="background-color: hsl(120, 100.00%, 97.84%); opacity: 0.80" title="0.037">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.52%); opacity: 0.81" title="0.105">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.73%); opacity: 0.80" title="-0.040">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.50%); opacity: 0.81" title="-0.140">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 95.64%); opacity: 0.81" title="-0.101">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.00%); opacity: 0.80" title="0.059">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 86.99%); opacity: 0.84" title="-0.480">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.47%); opacity: 0.83" title="0.355">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.04%); opacity: 0.82" title="-0.197">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.47%); opacity: 0.80" title="-0.023">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.68%); opacity: 0.82" title="-0.211">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.57%); opacity: 0.80" title="0.044">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.02%); opacity: 0.80" title="0.033">she</span><span style="opacity: 0.80">'d had </span><span style="background-color: hsl(120, 100.00%, 77.72%); opacity: 0.89" title="1.036">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 78.94%); opacity: 0.88" title="0.956">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.59%); opacity: 0.81" title="0.102">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.07%); opacity: 0.82" title="-0.195">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.005">and</span><span style="opacity: 0.80"> the childbirth </span><span style="background-color: hsl(120, 100.00%, 94.19%); opacity: 0.81" title="0.152">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.57%); opacity: 0.81" title="-0.103">less</span><span style="opacity: 0.80">.</span>
        </p>
    
        
            
        
            
            
        
            <p style="margin-bottom: 0.5em; margin-top: 0em">
                <b>
        
            y=soc.religion.christian
        
    </b>
    
        
        (probability <b>0.001</b>, score <b>-6.890</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 98.18%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.249
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 81.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -6.640
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 98.34%); opacity: 0.80" title="-0.025">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.09%); opacity: 0.81" title="0.156">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.73%); opacity: 0.83" title="-0.391">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.64%); opacity: 0.81" title="-0.069">from</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.91%); opacity: 0.81" title="0.062">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.24%); opacity: 0.81" title="0.114">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.71%); opacity: 0.81" title="0.067">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.21%); opacity: 0.87" title="-0.875">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.91%); opacity: 0.86" title="-0.770">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 95.94%); opacity: 0.81" title="0.091">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.94%); opacity: 0.81" title="0.091">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 98.15%); opacity: 0.80" title="0.030">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.49%); opacity: 0.81" title="0.074">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 84.07%); opacity: 0.85" title="-0.642">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.52%); opacity: 0.81" title="0.105">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.39%); opacity: 0.82" title="0.266">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.84%); opacity: 0.82" title="0.205">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.58%); opacity: 0.82" title="0.216">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.38%); opacity: 0.81" title="0.077">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.30%); opacity: 0.80" title="0.051">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.36%); opacity: 0.81" title="-0.110">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.19%); opacity: 0.82" title="-0.232">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.20%); opacity: 0.82" title="-0.231">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.89%); opacity: 0.96" title="-1.905">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 90.17%); opacity: 0.83" title="-0.322">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.45%); opacity: 0.80" title="-0.047">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.18%); opacity: 0.82" title="0.232">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 93.42%); opacity: 0.82" title="0.181">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.23%); opacity: 0.80" title="-0.028">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.49%); opacity: 0.81" title="0.106">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.99%); opacity: 0.80" title="0.059">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.060">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.94%); opacity: 0.82" title="0.242">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.56%); opacity: 0.80" title="-0.004">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.84%); opacity: 0.81" title="0.064">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.57%); opacity: 0.81" title="0.175">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 95.00%); opacity: 0.81" title="0.123">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.23%); opacity: 0.80" title="-0.028">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.49%); opacity: 0.81" title="0.106">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 96.99%); opacity: 0.80" title="0.059">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.22%); opacity: 0.81" title="-0.082">be</span><span style="opacity: 0.80"> extracted </span><span style="background-color: hsl(0, 100.00%, 95.56%); opacity: 0.81" title="-0.103">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 97.40%); opacity: 0.80" title="-0.048">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.70%); opacity: 0.80" title="-0.018">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.015">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.43%); opacity: 0.82" title="0.222">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 94.75%); opacity: 0.81" title="0.131">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.066">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 98.48%); opacity: 0.80" title="-0.022">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.88%); opacity: 0.83" title="-0.384">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.321">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.006">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.77%); opacity: 0.82" title="0.207">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.08%); opacity: 0.81" title="-0.087">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.98%); opacity: 0.80" title="0.034">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 97.28%); opacity: 0.80" title="0.051">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.24%); opacity: 0.81" title="-0.082">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.21%); opacity: 0.87" title="-0.875">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 81.91%); opacity: 0.86" title="-0.770">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.31%); opacity: 0.81" title="-0.079">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.14%); opacity: 0.86" title="0.696">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 95.74%); opacity: 0.81" title="0.098">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.04%); opacity: 0.80" title="0.032">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.003">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.021">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.93%); opacity: 0.81" title="0.162">less</span><span style="opacity: 0.80">.</span>
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

    0.067 alt.atheism
    0.145 comp.graphics
    0.359 sci.med
    0.428 soc.religion.christian


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

      recall from     ,  ' 
        anything    relieve the pain.
    
      pass,       up  sound,   
     be  surgically.
    
       in, the -      'd  
     and children, and the childbirth  .


By default :class:`~.TextExplainer` generated 5000 distorted texts (use
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
            random_state=<mtrand.RandomState object at 0x111b36678>,
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

    {'mean_KL_divergence': 0.022572840108760588, 'score': 0.9859107408450869}



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
    
        
        (probability <b>0.997</b>, score <b>5.810</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +5.776
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.45%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.034
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
            
    
            
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.041">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.18%); opacity: 0.80" title="0.074">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.29%); opacity: 0.80" title="-0.068">recall</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.56%); opacity: 0.80" title="0.113">from</span><span style="opacity: 0.80"> my </span><span style="background-color: hsl(0, 100.00%, 97.87%); opacity: 0.80" title="-0.093">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.26%); opacity: 0.80" title="-0.133">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.02%); opacity: 0.80" title="-0.083">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.93%); opacity: 0.80" title="-0.035">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.104">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.99%); opacity: 0.81" title="0.314">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 97.73%); opacity: 0.80" title="0.101">t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.72%); opacity: 0.84" title="1.263">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="6.105">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.11%); opacity: 0.84" title="1.078">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.47%); opacity: 0.80" title="-0.013">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.236">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.13%); opacity: 0.81" title="-0.302">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.05%); opacity: 0.80" title="-0.147">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.38%); opacity: 0.81" title="-0.280">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.96%); opacity: 0.81" title="0.153">except</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.28%); opacity: 0.81" title="0.289">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.28%); opacity: 0.81" title="-0.288">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.266">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 98.50%); opacity: 0.80" title="-0.056">either</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.004">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.51%); opacity: 0.81" title="0.268">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 99.10%); opacity: 0.80" title="-0.027">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.13%); opacity: 0.80" title="0.142">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.75%); opacity: 0.80" title="0.100">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.25%); opacity: 0.80" title="-0.021">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.011">be</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.89%); opacity: 0.80" title="-0.036">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.58%); opacity: 0.80" title="-0.052">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.91%); opacity: 0.80" title="-0.036">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.61%); opacity: 0.81" title="0.180">sound</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 96.04%); opacity: 0.81" title="-0.224">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.13%); opacity: 0.80" title="0.142">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.75%); opacity: 0.80" title="0.100">have</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 99.25%); opacity: 0.80" title="-0.021">to</span><span style="opacity: 0.80"> be </span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.019">extracted</span><span style="opacity: 0.80"> surgically.
    
    </span><span style="background-color: hsl(120, 100.00%, 97.34%); opacity: 0.80" title="0.127">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.85%); opacity: 0.80" title="0.094">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.61%); opacity: 0.81" title="-0.260">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.005">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 96.56%); opacity: 0.81" title="0.183">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.02%); opacity: 0.81" title="0.311">x</span><span style="opacity: 0.80">-</span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.149">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.29%); opacity: 0.80" title="0.068">tech</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.46%); opacity: 0.81" title="-0.191">happened</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.56%); opacity: 0.80" title="-0.113">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.20%); opacity: 0.81" title="0.386">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.22%); opacity: 0.81" title="0.294">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.05%); opacity: 0.80" title="0.081">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 97.39%); opacity: 0.80" title="-0.123">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.65%); opacity: 0.82" title="-0.542">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.78%); opacity: 0.81" title="-0.332">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(0, 100.00%, 98.93%); opacity: 0.80" title="-0.035">stones</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(0, 100.00%, 95.52%); opacity: 0.81" title="-0.267">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 98.50%); opacity: 0.80" title="-0.056">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.61%); opacity: 0.80" title="-0.050">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.07%); opacity: 0.80" title="0.146">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.51%); opacity: 0.80" title="-0.011">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.06%); opacity: 0.80" title="-0.081">less</span><span style="opacity: 0.80">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




:class:`~.TextExplainer` correctly figured out that 'medication' is important,
but failed to account for "len(doc) % 2" condition, so the explanation
is incomplete. We can detect this failure by looking at metrics - they
are low:

.. code:: ipython3

    te3.metrics_




.. parsed-literal::

    {'mean_KL_divergence': 0.29423004212730852, 'score': 0.80249195929598161}



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
                # be a feature to show in 50% cases
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

    {'score': 1.0, 'mean_KL_divergence': 0.023676903493283757}




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
    
        
        (probability <b>0.995</b>, score <b>5.398</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +8.585
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            countvectorizer: Highlighted in text (sum)
        </td>
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 99.53%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.040
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
                <tr style="background-color: hsl(0, 100.00%, 90.09%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -3.146
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            doclength__is_even
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <b>countvectorizer:</b> <span style="opacity: 0.80">as i recall from </span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.042">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.084">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.050">with</span><span style="opacity: 0.80"> kidney </span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.025">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.050">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.68%); opacity: 0.80" title="-0.008">isn</span><span style="opacity: 0.80">'t </span><span style="background-color: hsl(120, 100.00%, 89.45%); opacity: 0.83" title="1.168">any</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="7.833">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.73%); opacity: 0.83" title="1.283">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.043">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.012">do</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.64%); opacity: 0.80" title="0.137">anything</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.10%); opacity: 0.80" title="0.100">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.048">them</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.048">except</span><span style="opacity: 0.80"> relieve the pain.
    
    either </span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.052">they</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.03%); opacity: 0.80" title="-0.038">pass</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 99.28%); opacity: 0.80" title="-0.025">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.052">they</span><span style="opacity: 0.80"> have </span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.041">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.071">be</span><span style="opacity: 0.80"> broken </span><span style="background-color: hsl(120, 100.00%, 98.24%); opacity: 0.80" title="0.090">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.05%); opacity: 0.80" title="0.104">with</span><span style="opacity: 0.80"> sound, </span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.013">or</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.052">they</span><span style="opacity: 0.80"> have
    </span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.041">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.071">be</span><span style="opacity: 0.80"> extracted surgically.
    
    when i was in, the x-</span><span style="background-color: hsl(120, 100.00%, 98.66%); opacity: 0.80" title="0.061">ray</span><span style="opacity: 0.80"> tech happened </span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.029">to</span><span style="opacity: 0.80"> mention </span><span style="background-color: hsl(0, 100.00%, 98.29%); opacity: 0.80" title="-0.087">that</span><span style="opacity: 0.80"> she'd </span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.015">had</span><span style="opacity: 0.80"> kidney
    </span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.025">stones</span><span style="opacity: 0.80"> and children, and the childbirth hurt </span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.072">less</span>
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

    0.87816245006657789



This pipeline is supported by eli5 directly, so in practice there is no
need to use :class:`~.TextExplainer` for it. We're using it as an example - it
is possible check the "true" explanation first, without using
:class:`~.TextExplainer`, and then compare it with :class:`~.TextExplainer` results.

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
    
        
        (probability <b>0.628</b>, score <b>0.068</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.053
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.91%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.985
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">as</span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.005"> i</span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 89.28%); opacity: 0.83" title="0.019">r</span><span style="background-color: hsl(120, 100.00%, 84.83%); opacity: 0.85" title="0.031">e</span><span style="background-color: hsl(120, 100.00%, 82.60%); opacity: 0.86" title="0.038">c</span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.024">a</span><span style="background-color: hsl(120, 100.00%, 90.21%); opacity: 0.83" title="0.017">l</span><span style="background-color: hsl(120, 100.00%, 93.46%); opacity: 0.82" title="0.009">l</span><span style="background-color: hsl(0, 100.00%, 99.64%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(0, 100.00%, 96.69%); opacity: 0.81" title="-0.004">f</span><span style="background-color: hsl(0, 100.00%, 96.20%); opacity: 0.81" title="-0.004">ro</span><span style="background-color: hsl(0, 100.00%, 95.80%); opacity: 0.81" title="-0.005">m</span><span style="background-color: hsl(120, 100.00%, 87.24%); opacity: 0.84" title="0.024"> </span><span style="background-color: hsl(120, 100.00%, 86.91%); opacity: 0.84" title="0.025">my</span><span style="background-color: hsl(120, 100.00%, 88.14%); opacity: 0.84" title="0.022"> </span><span style="background-color: hsl(120, 100.00%, 96.37%); opacity: 0.81" title="0.004">b</span><span style="background-color: hsl(120, 100.00%, 95.34%); opacity: 0.81" title="0.006">ou</span><span style="background-color: hsl(120, 100.00%, 93.66%); opacity: 0.81" title="0.009">t</span><span style="background-color: hsl(120, 100.00%, 93.86%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 94.76%); opacity: 0.81" title="0.007">w</span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">it</span><span style="background-color: hsl(0, 100.00%, 98.51%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 96.16%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 95.07%); opacity: 0.81" title="0.006">k</span><span style="background-color: hsl(120, 100.00%, 93.15%); opacity: 0.82" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 91.67%); opacity: 0.82" title="0.013">d</span><span style="background-color: hsl(120, 100.00%, 91.49%); opacity: 0.82" title="0.014">n</span><span style="background-color: hsl(120, 100.00%, 93.00%); opacity: 0.82" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 94.89%); opacity: 0.81" title="0.007">y</span><span style="background-color: hsl(120, 100.00%, 96.52%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(0, 100.00%, 99.03%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(0, 100.00%, 98.79%); opacity: 0.80" title="-0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 97.86%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(120, 100.00%, 96.44%); opacity: 0.81" title="0.004">s</span><span style="background-color: hsl(120, 100.00%, 96.71%); opacity: 0.81" title="0.003">,</span><span style="background-color: hsl(120, 100.00%, 94.97%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(0, 100.00%, 96.17%); opacity: 0.81" title="-0.004">t</span><span style="background-color: hsl(0, 100.00%, 90.33%); opacity: 0.83" title="-0.016">h</span><span style="background-color: hsl(0, 100.00%, 88.90%); opacity: 0.83" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 90.04%); opacity: 0.83" title="-0.017">r</span><span style="background-color: hsl(0, 100.00%, 90.95%); opacity: 0.82" title="-0.015">e</span><span style="background-color: hsl(0, 100.00%, 94.10%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 94.94%); opacity: 0.81" title="-0.006">i</span><span style="background-color: hsl(0, 100.00%, 93.80%); opacity: 0.81" title="-0.009">s</span><span style="background-color: hsl(0, 100.00%, 92.14%); opacity: 0.82" title="-0.012">n</span><span style="background-color: hsl(0, 100.00%, 92.65%); opacity: 0.82" title="-0.011">'</span><span style="background-color: hsl(0, 100.00%, 94.40%); opacity: 0.81" title="-0.007">t</span><span style="background-color: hsl(0, 100.00%, 92.92%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 97.93%); opacity: 0.80" title="-0.002">any</span><span style="background-color: hsl(120, 100.00%, 80.81%); opacity: 0.87" title="0.043">
    </span><span style="background-color: hsl(120, 100.00%, 72.01%); opacity: 0.92" title="0.074">m</span><span style="background-color: hsl(120, 100.00%, 64.40%); opacity: 0.97" title="0.105">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.123">d</span><span style="background-color: hsl(120, 100.00%, 64.08%); opacity: 0.97" title="0.106">i</span><span style="background-color: hsl(120, 100.00%, 72.54%); opacity: 0.92" title="0.072">c</span><span style="background-color: hsl(120, 100.00%, 84.81%); opacity: 0.85" title="0.031">a</span><span style="background-color: hsl(120, 100.00%, 92.26%); opacity: 0.82" title="0.012">t</span><span style="background-color: hsl(0, 100.00%, 95.31%); opacity: 0.81" title="-0.006">i</span><span style="background-color: hsl(0, 100.00%, 94.12%); opacity: 0.81" title="-0.008">o</span><span style="background-color: hsl(0, 100.00%, 95.02%); opacity: 0.81" title="-0.006">n</span><span style="background-color: hsl(0, 100.00%, 91.70%); opacity: 0.82" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 92.80%); opacity: 0.82" title="-0.011">t</span><span style="background-color: hsl(0, 100.00%, 91.68%); opacity: 0.82" title="-0.013">ha</span><span style="background-color: hsl(0, 100.00%, 93.99%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 93.73%); opacity: 0.81" title="0.009">can</span><span style="background-color: hsl(0, 100.00%, 94.41%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 94.06%); opacity: 0.81" title="-0.008">do</span><span style="background-color: hsl(0, 100.00%, 90.35%); opacity: 0.83" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 92.50%); opacity: 0.82" title="-0.011">a</span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.014">n</span><span style="background-color: hsl(0, 100.00%, 89.00%); opacity: 0.83" title="-0.020">y</span><span style="background-color: hsl(0, 100.00%, 95.12%); opacity: 0.81" title="-0.006">t</span><span style="background-color: hsl(0, 100.00%, 95.67%); opacity: 0.81" title="-0.005">h</span><span style="background-color: hsl(0, 100.00%, 94.24%); opacity: 0.81" title="-0.008">i</span><span style="background-color: hsl(0, 100.00%, 96.71%); opacity: 0.81" title="-0.003">n</span><span style="background-color: hsl(0, 100.00%, 94.23%); opacity: 0.81" title="-0.008">g</span><span style="background-color: hsl(120, 100.00%, 97.94%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 91.01%); opacity: 0.82" title="0.015">a</span><span style="background-color: hsl(120, 100.00%, 88.08%); opacity: 0.84" title="0.022">b</span><span style="background-color: hsl(120, 100.00%, 87.43%); opacity: 0.84" title="0.024">o</span><span style="background-color: hsl(120, 100.00%, 88.81%); opacity: 0.83" title="0.020">u</span><span style="background-color: hsl(120, 100.00%, 92.02%); opacity: 0.82" title="0.012">t</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.012">t</span><span style="background-color: hsl(0, 100.00%, 90.90%); opacity: 0.82" title="-0.015">he</span><span style="background-color: hsl(0, 100.00%, 92.15%); opacity: 0.82" title="-0.012">m</span><span style="background-color: hsl(0, 100.00%, 97.73%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 93.34%); opacity: 0.82" title="0.010">x</span><span style="background-color: hsl(120, 100.00%, 94.71%); opacity: 0.81" title="0.007">c</span><span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 96.89%); opacity: 0.81" title="0.003">p</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(0, 100.00%, 90.34%); opacity: 0.83" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 86.91%); opacity: 0.84" title="-0.025">r</span><span style="background-color: hsl(0, 100.00%, 84.05%); opacity: 0.85" title="-0.033">e</span><span style="background-color: hsl(0, 100.00%, 83.29%); opacity: 0.86" title="-0.035">l</span><span style="background-color: hsl(0, 100.00%, 86.33%); opacity: 0.84" title="-0.027">i</span><span style="background-color: hsl(0, 100.00%, 95.38%); opacity: 0.81" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 98.15%); opacity: 0.80" title="-0.002">v</span><span style="background-color: hsl(120, 100.00%, 97.52%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(0, 100.00%, 94.84%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 88.18%); opacity: 0.84" title="-0.022">the</span><span style="background-color: hsl(0, 100.00%, 96.45%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 86.94%); opacity: 0.84" title="0.025">p</span><span style="background-color: hsl(120, 100.00%, 85.30%); opacity: 0.85" title="0.030">a</span><span style="background-color: hsl(120, 100.00%, 83.41%); opacity: 0.86" title="0.035">i</span><span style="background-color: hsl(120, 100.00%, 86.12%); opacity: 0.84" title="0.027">n</span><span style="background-color: hsl(120, 100.00%, 92.61%); opacity: 0.82" title="0.011">.</span><span style="background-color: hsl(120, 100.00%, 96.62%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(0, 100.00%, 91.80%); opacity: 0.82" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 89.36%); opacity: 0.83" title="-0.019">i</span><span style="background-color: hsl(0, 100.00%, 84.31%); opacity: 0.85" title="-0.032">t</span><span style="background-color: hsl(0, 100.00%, 83.45%); opacity: 0.86" title="-0.035">h</span><span style="background-color: hsl(0, 100.00%, 86.46%); opacity: 0.84" title="-0.026">e</span><span style="background-color: hsl(0, 100.00%, 88.36%); opacity: 0.83" title="-0.021">r</span><span style="background-color: hsl(0, 100.00%, 92.86%); opacity: 0.82" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 96.20%); opacity: 0.81" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 94.65%); opacity: 0.81" title="0.007">he</span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.010">y</span><span style="background-color: hsl(0, 100.00%, 98.20%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 92.53%); opacity: 0.82" title="-0.011">p</span><span style="background-color: hsl(0, 100.00%, 91.90%); opacity: 0.82" title="-0.013">a</span><span style="background-color: hsl(0, 100.00%, 92.16%); opacity: 0.82" title="-0.012">s</span><span style="background-color: hsl(0, 100.00%, 94.08%); opacity: 0.81" title="-0.008">s</span><span style="background-color: hsl(0, 100.00%, 99.01%); opacity: 0.80" title="-0.001">,</span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 96.53%); opacity: 0.81" title="-0.004">or</span><span style="background-color: hsl(0, 100.00%, 95.98%); opacity: 0.81" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 96.20%); opacity: 0.81" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 94.65%); opacity: 0.81" title="0.007">he</span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.010">y</span><span style="background-color: hsl(0, 100.00%, 96.29%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 90.00%); opacity: 0.83" title="-0.017">h</span><span style="background-color: hsl(0, 100.00%, 87.24%); opacity: 0.84" title="-0.024">av</span><span style="background-color: hsl(0, 100.00%, 88.71%); opacity: 0.83" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 97.88%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.008">to</span><span style="background-color: hsl(120, 100.00%, 96.06%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(0, 100.00%, 96.91%); opacity: 0.81" title="-0.003">be</span><span style="background-color: hsl(0, 100.00%, 96.93%); opacity: 0.81" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">b</span><span style="background-color: hsl(0, 100.00%, 99.38%); opacity: 0.80" title="-0.000">r</span><span style="background-color: hsl(0, 100.00%, 98.96%); opacity: 0.80" title="-0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.001">k</span><span style="background-color: hsl(120, 100.00%, 98.38%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.21%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 95.13%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 96.37%); opacity: 0.81" title="0.004">up</span><span style="background-color: hsl(120, 100.00%, 94.37%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 94.76%); opacity: 0.81" title="0.007">w</span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">it</span><span style="background-color: hsl(0, 100.00%, 98.51%); opacity: 0.80" title="-0.001">h</span><span style="background-color: hsl(0, 100.00%, 96.61%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 90.78%); opacity: 0.82" title="0.015">s</span><span style="background-color: hsl(120, 100.00%, 89.21%); opacity: 0.83" title="0.019">o</span><span style="background-color: hsl(120, 100.00%, 90.01%); opacity: 0.83" title="0.017">u</span><span style="background-color: hsl(120, 100.00%, 92.56%); opacity: 0.82" title="0.011">n</span><span style="background-color: hsl(120, 100.00%, 97.95%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.008">,</span><span style="background-color: hsl(0, 100.00%, 93.09%); opacity: 0.82" title="-0.010"> </span><span style="background-color: hsl(0, 100.00%, 96.53%); opacity: 0.81" title="-0.004">or</span><span style="background-color: hsl(0, 100.00%, 95.98%); opacity: 0.81" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 96.20%); opacity: 0.81" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 94.65%); opacity: 0.81" title="0.007">he</span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.010">y</span><span style="background-color: hsl(0, 100.00%, 96.29%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 90.00%); opacity: 0.83" title="-0.017">h</span><span style="background-color: hsl(0, 100.00%, 87.24%); opacity: 0.84" title="-0.024">av</span><span style="background-color: hsl(0, 100.00%, 88.71%); opacity: 0.83" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 97.88%); opacity: 0.80" title="-0.002">
    </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.008">to</span><span style="background-color: hsl(120, 100.00%, 96.06%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(0, 100.00%, 96.91%); opacity: 0.81" title="-0.003">be</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 91.17%); opacity: 0.82" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 88.67%); opacity: 0.83" title="0.020">x</span><span style="background-color: hsl(120, 100.00%, 88.16%); opacity: 0.84" title="0.022">t</span><span style="background-color: hsl(120, 100.00%, 88.03%); opacity: 0.84" title="0.022">r</span><span style="background-color: hsl(120, 100.00%, 89.46%); opacity: 0.83" title="0.018">a</span><span style="background-color: hsl(120, 100.00%, 89.88%); opacity: 0.83" title="0.017">c</span><span style="background-color: hsl(120, 100.00%, 86.74%); opacity: 0.84" title="0.026">t</span><span style="background-color: hsl(120, 100.00%, 88.14%); opacity: 0.84" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 89.51%); opacity: 0.83" title="0.018">d</span><span style="background-color: hsl(120, 100.00%, 84.54%); opacity: 0.85" title="0.032"> </span><span style="background-color: hsl(120, 100.00%, 85.07%); opacity: 0.85" title="0.030">s</span><span style="background-color: hsl(120, 100.00%, 84.14%); opacity: 0.85" title="0.033">u</span><span style="background-color: hsl(120, 100.00%, 82.39%); opacity: 0.86" title="0.038">r</span><span style="background-color: hsl(120, 100.00%, 87.99%); opacity: 0.84" title="0.022">g</span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.020">i</span><span style="background-color: hsl(120, 100.00%, 87.58%); opacity: 0.84" title="0.023">c</span><span style="background-color: hsl(120, 100.00%, 87.11%); opacity: 0.84" title="0.024">a</span><span style="background-color: hsl(120, 100.00%, 87.71%); opacity: 0.84" title="0.023">l</span><span style="background-color: hsl(120, 100.00%, 92.46%); opacity: 0.82" title="0.011">l</span><span style="background-color: hsl(120, 100.00%, 95.40%); opacity: 0.81" title="0.006">y</span><span style="background-color: hsl(0, 100.00%, 97.35%); opacity: 0.80" title="-0.003">.</span><span style="background-color: hsl(120, 100.00%, 94.13%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 89.02%); opacity: 0.83" title="0.019">w</span><span style="background-color: hsl(120, 100.00%, 87.96%); opacity: 0.84" title="0.022">he</span><span style="background-color: hsl(120, 100.00%, 89.42%); opacity: 0.83" title="0.018">n</span><span style="background-color: hsl(120, 100.00%, 97.16%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.005">i</span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 93.10%); opacity: 0.82" title="-0.010">w</span><span style="background-color: hsl(0, 100.00%, 95.63%); opacity: 0.81" title="-0.005">as</span><span style="background-color: hsl(0, 100.00%, 96.68%); opacity: 0.81" title="-0.004"> </span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.002">in,</span><span style="background-color: hsl(0, 100.00%, 94.12%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 88.18%); opacity: 0.84" title="-0.022">the</span><span style="background-color: hsl(0, 100.00%, 89.39%); opacity: 0.83" title="-0.019"> </span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.001">x</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">-</span><span style="background-color: hsl(0, 100.00%, 96.19%); opacity: 0.81" title="-0.004">r</span><span style="background-color: hsl(0, 100.00%, 96.10%); opacity: 0.81" title="-0.004">a</span><span style="background-color: hsl(0, 100.00%, 95.78%); opacity: 0.81" title="-0.005">y</span><span style="background-color: hsl(0, 100.00%, 94.07%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 95.60%); opacity: 0.81" title="-0.005">t</span><span style="background-color: hsl(0, 100.00%, 95.54%); opacity: 0.81" title="-0.005">ec</span><span style="background-color: hsl(0, 100.00%, 96.49%); opacity: 0.81" title="-0.004">h</span><span style="background-color: hsl(120, 100.00%, 98.06%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.97%); opacity: 0.81" title="0.006">h</span><span style="background-color: hsl(120, 100.00%, 91.97%); opacity: 0.82" title="0.012">a</span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.015">p</span><span style="background-color: hsl(120, 100.00%, 92.58%); opacity: 0.82" title="0.011">p</span><span style="background-color: hsl(120, 100.00%, 95.99%); opacity: 0.81" title="0.005">e</span><span style="background-color: hsl(0, 100.00%, 96.36%); opacity: 0.81" title="-0.004">n</span><span style="background-color: hsl(0, 100.00%, 92.76%); opacity: 0.82" title="-0.011">e</span><span style="background-color: hsl(0, 100.00%, 94.04%); opacity: 0.81" title="-0.008">d</span><span style="background-color: hsl(120, 100.00%, 96.39%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.008">to</span><span style="background-color: hsl(120, 100.00%, 94.08%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 89.63%); opacity: 0.83" title="0.018">e</span><span style="background-color: hsl(120, 100.00%, 89.55%); opacity: 0.83" title="0.018">n</span><span style="background-color: hsl(120, 100.00%, 87.14%); opacity: 0.84" title="0.024">t</span><span style="background-color: hsl(120, 100.00%, 91.67%); opacity: 0.82" title="0.013">i</span><span style="background-color: hsl(0, 100.00%, 96.88%); opacity: 0.81" title="-0.003">o</span><span style="background-color: hsl(0, 100.00%, 96.94%); opacity: 0.81" title="-0.003">n</span><span style="background-color: hsl(0, 100.00%, 91.70%); opacity: 0.82" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 92.80%); opacity: 0.82" title="-0.011">t</span><span style="background-color: hsl(0, 100.00%, 91.68%); opacity: 0.82" title="-0.013">ha</span><span style="background-color: hsl(0, 100.00%, 93.99%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(120, 100.00%, 94.73%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 91.14%); opacity: 0.82" title="0.014">s</span><span style="background-color: hsl(120, 100.00%, 90.98%); opacity: 0.82" title="0.015">h</span><span style="background-color: hsl(120, 100.00%, 91.41%); opacity: 0.82" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 98.65%); opacity: 0.80" title="0.001">'</span><span style="background-color: hsl(0, 100.00%, 99.02%); opacity: 0.80" title="-0.001">d</span><span style="background-color: hsl(120, 100.00%, 91.28%); opacity: 0.82" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 87.86%); opacity: 0.84" title="0.022">had</span><span style="background-color: hsl(120, 100.00%, 89.79%); opacity: 0.83" title="0.018"> </span><span style="background-color: hsl(120, 100.00%, 95.07%); opacity: 0.81" title="0.006">k</span><span style="background-color: hsl(120, 100.00%, 93.15%); opacity: 0.82" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 91.67%); opacity: 0.82" title="0.013">d</span><span style="background-color: hsl(120, 100.00%, 91.49%); opacity: 0.82" title="0.014">n</span><span style="background-color: hsl(120, 100.00%, 93.00%); opacity: 0.82" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 94.89%); opacity: 0.81" title="0.007">y</span><span style="background-color: hsl(120, 100.00%, 96.52%); opacity: 0.81" title="0.004">
    </span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(0, 100.00%, 99.03%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.15%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(120, 100.00%, 97.22%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.005">s</span><span style="background-color: hsl(120, 100.00%, 94.26%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 93.32%); opacity: 0.82" title="0.010">and</span><span style="background-color: hsl(120, 100.00%, 91.20%); opacity: 0.82" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 97.04%); opacity: 0.80" title="0.003">c</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(0, 100.00%, 98.16%); opacity: 0.80" title="-0.002">i</span><span style="background-color: hsl(0, 100.00%, 94.41%); opacity: 0.81" title="-0.007">l</span><span style="background-color: hsl(0, 100.00%, 95.96%); opacity: 0.81" title="-0.005">d</span><span style="background-color: hsl(0, 100.00%, 96.30%); opacity: 0.81" title="-0.004">r</span><span style="background-color: hsl(0, 100.00%, 95.25%); opacity: 0.81" title="-0.006">e</span><span style="background-color: hsl(0, 100.00%, 96.13%); opacity: 0.81" title="-0.004">n</span><span style="background-color: hsl(0, 100.00%, 96.34%); opacity: 0.81" title="-0.004">,</span><span style="background-color: hsl(0, 100.00%, 99.26%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(120, 100.00%, 93.32%); opacity: 0.82" title="0.010">and</span><span style="background-color: hsl(120, 100.00%, 97.73%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(0, 100.00%, 88.18%); opacity: 0.84" title="-0.022">the</span><span style="background-color: hsl(0, 100.00%, 90.47%); opacity: 0.83" title="-0.016"> </span><span style="background-color: hsl(120, 100.00%, 97.04%); opacity: 0.80" title="0.003">c</span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 98.00%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(0, 100.00%, 97.30%); opacity: 0.80" title="-0.003">l</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 97.12%); opacity: 0.80" title="0.003">b</span><span style="background-color: hsl(120, 100.00%, 96.12%); opacity: 0.81" title="0.004">i</span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 96.88%); opacity: 0.81" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 97.66%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 98.44%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 97.76%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 97.80%); opacity: 0.80" title="0.002">ur</span><span style="background-color: hsl(120, 100.00%, 98.23%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 95.02%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 93.46%); opacity: 0.82" title="0.009">l</span><span style="background-color: hsl(120, 100.00%, 91.73%); opacity: 0.82" title="0.013">e</span><span style="background-color: hsl(120, 100.00%, 91.64%); opacity: 0.82" title="0.013">s</span><span style="background-color: hsl(120, 100.00%, 93.31%); opacity: 0.82" title="0.010">s</span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.004">.</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




:class:`~.TextExplainer` produces a different result:

.. code:: ipython3

    te = TextExplainer(random_state=42).fit(doc, pipe_char.predict_proba)
    print(te.metrics_)
    te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)


.. parsed-literal::

    {'score': 0.94184989091126792, 'mean_KL_divergence': 0.022047503918099929}




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
    
        
        (probability <b>0.653</b>, score <b>0.694</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.051
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 90.61%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.357
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 97.85%); opacity: 0.80" title="0.011">as</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.55%); opacity: 0.80" title="-0.013">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.16%); opacity: 0.84" title="0.128">recall</span><span style="opacity: 0.80"> from </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.012">my</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.47%); opacity: 0.81" title="-0.033">bout</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.001">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.71%); opacity: 0.82" title="0.077">kidney</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.022">stones</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 89.74%); opacity: 0.83" title="-0.105">there</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.86%); opacity: 0.83" title="0.118">isn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(120, 100.00%, 88.86%); opacity: 0.83" title="0.118">t</span><span style="opacity: 0.80"> any
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.731">medication</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.58%); opacity: 0.84" title="-0.137">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.06%); opacity: 0.84" title="0.130">can</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.61%); opacity: 0.83" title="0.092">do</span><span style="opacity: 0.80"> anything </span><span style="background-color: hsl(120, 100.00%, 89.56%); opacity: 0.83" title="0.107">about</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.85%); opacity: 0.85" title="-0.200">them</span><span style="opacity: 0.80"> except </span><span style="background-color: hsl(0, 100.00%, 84.69%); opacity: 0.85" title="-0.185">relieve</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.75%); opacity: 0.84" title="-0.151">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.22%); opacity: 0.87" title="0.267">pain</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(0, 100.00%, 90.55%); opacity: 0.83" title="-0.093">either</span><span style="opacity: 0.80"> they </span><span style="background-color: hsl(0, 100.00%, 91.31%); opacity: 0.82" title="-0.083">pass</span><span style="opacity: 0.80">, or they </span><span style="background-color: hsl(0, 100.00%, 91.31%); opacity: 0.82" title="-0.082">have</span><span style="opacity: 0.80"> to be </span><span style="background-color: hsl(0, 100.00%, 89.90%); opacity: 0.83" title="-0.102">broken</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.87%); opacity: 0.81" title="-0.039">up</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.52%); opacity: 0.82" title="-0.067">with</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.067">sound</span><span style="opacity: 0.80">, or they </span><span style="background-color: hsl(0, 100.00%, 91.31%); opacity: 0.82" title="-0.082">have</span><span style="opacity: 0.80">
    to be </span><span style="background-color: hsl(120, 100.00%, 81.89%); opacity: 0.86" title="0.236">extracted</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.42%); opacity: 0.91" title="0.386">surgically</span><span style="opacity: 0.80">.
    
    </span><span style="background-color: hsl(120, 100.00%, 89.40%); opacity: 0.83" title="0.110">when</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.07%); opacity: 0.81" title="-0.048">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.62%); opacity: 0.82" title="-0.065">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.49%); opacity: 0.80" title="0.014">in</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(0, 100.00%, 88.10%); opacity: 0.84" title="-0.129">the</span><span style="opacity: 0.80"> x-</span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.016">ray</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.016">tech</span><span style="opacity: 0.80"> happened to </span><span style="background-color: hsl(120, 100.00%, 86.37%); opacity: 0.84" title="0.157">mention</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.45%); opacity: 0.86" title="-0.225">that</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.09%); opacity: 0.81" title="-0.037">she</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 96.30%); opacity: 0.81" title="-0.024">d</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.66%); opacity: 0.84" title="0.152">had</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.13%); opacity: 0.82" title="0.059">kidney</span><span style="opacity: 0.80">
    </span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.022">stones</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.61%); opacity: 0.83" title="0.121">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.29%); opacity: 0.81" title="-0.045">children</span><span style="opacity: 0.80">, </span><span style="background-color: hsl(120, 100.00%, 88.61%); opacity: 0.83" title="0.121">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.067">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.65%); opacity: 0.80" title="0.013">childbirth</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.06%); opacity: 0.82" title="0.073">hurt</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.22%); opacity: 0.83" title="0.098">less</span><span style="opacity: 0.80">.</span>
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

    {'score': 0.56624531491786811, 'mean_KL_divergence': 0.20342321890185364}




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
    
        
        (probability <b>0.377</b>, score <b>-0.086</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 85.62%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.143
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.229
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 88.25%); opacity: 0.83" title="0.039">a</span><span style="background-color: hsl(120, 100.00%, 91.06%); opacity: 0.82" title="0.027">s</span><span style="background-color: hsl(0, 100.00%, 95.99%); opacity: 0.81" title="-0.008"> </span><span style="background-color: hsl(0, 100.00%, 96.69%); opacity: 0.81" title="-0.006">i</span><span style="background-color: hsl(0, 100.00%, 89.22%); opacity: 0.83" title="-0.035"> </span><span style="background-color: hsl(0, 100.00%, 96.23%); opacity: 0.81" title="-0.008">r</span><span style="background-color: hsl(120, 100.00%, 94.51%); opacity: 0.81" title="0.013">e</span><span style="background-color: hsl(120, 100.00%, 87.46%); opacity: 0.84" title="0.043">ca</span><span style="background-color: hsl(120, 100.00%, 86.97%); opacity: 0.84" title="0.046">l</span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(0, 100.00%, 91.44%); opacity: 0.82" title="-0.025"> </span><span style="background-color: hsl(0, 100.00%, 99.00%); opacity: 0.80" title="-0.001">f</span><span style="background-color: hsl(120, 100.00%, 91.14%); opacity: 0.82" title="0.026">r</span><span style="opacity: 0.80">o</span><span style="background-color: hsl(0, 100.00%, 97.34%); opacity: 0.80" title="-0.005">m</span><span style="background-color: hsl(0, 100.00%, 97.52%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 95.03%); opacity: 0.81" title="-0.012">m</span><span style="background-color: hsl(0, 100.00%, 92.46%); opacity: 0.82" title="-0.021">y</span><span style="background-color: hsl(0, 100.00%, 92.29%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(120, 100.00%, 87.52%); opacity: 0.84" title="0.043">b</span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.019">o</span><span style="background-color: hsl(120, 100.00%, 86.46%); opacity: 0.84" title="0.048">u</span><span style="background-color: hsl(0, 100.00%, 96.60%); opacity: 0.81" title="-0.007">t</span><span style="background-color: hsl(0, 100.00%, 94.62%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 95.49%); opacity: 0.81" title="-0.010">w</span><span style="background-color: hsl(120, 100.00%, 85.28%); opacity: 0.85" title="0.054">i</span><span style="background-color: hsl(120, 100.00%, 78.69%); opacity: 0.88" title="0.092">t</span><span style="background-color: hsl(120, 100.00%, 91.49%); opacity: 0.82" title="0.025">h</span><span style="background-color: hsl(0, 100.00%, 94.59%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(120, 100.00%, 93.95%); opacity: 0.81" title="0.015">k</span><span style="background-color: hsl(120, 100.00%, 95.81%); opacity: 0.81" title="0.009">i</span><span style="background-color: hsl(0, 100.00%, 94.53%); opacity: 0.81" title="-0.013">d</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 89.52%); opacity: 0.83" title="0.033">e</span><span style="background-color: hsl(120, 100.00%, 94.40%); opacity: 0.81" title="0.014">y</span><span style="background-color: hsl(0, 100.00%, 97.87%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(0, 100.00%, 93.36%); opacity: 0.82" title="-0.017">s</span><span style="background-color: hsl(0, 100.00%, 86.54%); opacity: 0.84" title="-0.048">t</span><span style="background-color: hsl(0, 100.00%, 83.43%); opacity: 0.86" title="-0.064">o</span><span style="background-color: hsl(0, 100.00%, 77.47%); opacity: 0.89" title="-0.100">n</span><span style="background-color: hsl(0, 100.00%, 86.72%); opacity: 0.84" title="-0.047">e</span><span style="background-color: hsl(0, 100.00%, 98.41%); opacity: 0.80" title="-0.002">s</span><span style="background-color: hsl(0, 100.00%, 83.81%); opacity: 0.85" title="-0.062">,</span><span style="background-color: hsl(0, 100.00%, 89.26%); opacity: 0.83" title="-0.035"> </span><span style="background-color: hsl(120, 100.00%, 93.87%); opacity: 0.81" title="0.016">t</span><span style="background-color: hsl(0, 100.00%, 86.71%); opacity: 0.84" title="-0.047">h</span><span style="background-color: hsl(0, 100.00%, 75.18%); opacity: 0.90" title="-0.115">e</span><span style="background-color: hsl(0, 100.00%, 91.57%); opacity: 0.82" title="-0.024">r</span><span style="background-color: hsl(0, 100.00%, 89.85%); opacity: 0.83" title="-0.032">e</span><span style="background-color: hsl(0, 100.00%, 89.55%); opacity: 0.83" title="-0.033"> </span><span style="background-color: hsl(0, 100.00%, 94.44%); opacity: 0.81" title="-0.014">i</span><span style="opacity: 0.80">s</span><span style="background-color: hsl(0, 100.00%, 83.94%); opacity: 0.85" title="-0.062">n</span><span style="background-color: hsl(0, 100.00%, 88.93%); opacity: 0.83" title="-0.036">'</span><span style="background-color: hsl(120, 100.00%, 91.36%); opacity: 0.82" title="0.025">t</span><span style="background-color: hsl(120, 100.00%, 92.45%); opacity: 0.82" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 86.18%); opacity: 0.84" title="0.050">a</span><span style="background-color: hsl(120, 100.00%, 86.84%); opacity: 0.84" title="0.046">n</span><span style="background-color: hsl(120, 100.00%, 91.53%); opacity: 0.82" title="0.025">y</span><span style="background-color: hsl(120, 100.00%, 89.67%); opacity: 0.83" title="0.033">
    </span><span style="background-color: hsl(120, 100.00%, 67.17%); opacity: 0.95" title="0.171">m</span><span style="background-color: hsl(120, 100.00%, 66.33%); opacity: 0.96" title="0.177">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.227">d</span><span style="background-color: hsl(120, 100.00%, 65.75%); opacity: 0.96" title="0.182">i</span><span style="background-color: hsl(120, 100.00%, 77.35%); opacity: 0.89" title="0.101">c</span><span style="background-color: hsl(120, 100.00%, 95.85%); opacity: 0.81" title="0.009">a</span><span style="background-color: hsl(0, 100.00%, 96.19%); opacity: 0.81" title="-0.008">t</span><span style="background-color: hsl(0, 100.00%, 91.91%); opacity: 0.82" title="-0.023">i</span><span style="background-color: hsl(0, 100.00%, 92.55%); opacity: 0.82" title="-0.021">o</span><span style="background-color: hsl(0, 100.00%, 91.71%); opacity: 0.82" title="-0.024">n</span><span style="background-color: hsl(0, 100.00%, 89.43%); opacity: 0.83" title="-0.034"> </span><span style="background-color: hsl(120, 100.00%, 97.64%); opacity: 0.80" title="0.004">t</span><span style="background-color: hsl(0, 100.00%, 97.96%); opacity: 0.80" title="-0.003">h</span><span style="background-color: hsl(120, 100.00%, 96.50%); opacity: 0.81" title="0.007">a</span><span style="background-color: hsl(0, 100.00%, 87.67%); opacity: 0.84" title="-0.042">t</span><span style="background-color: hsl(0, 100.00%, 81.96%); opacity: 0.86" title="-0.073"> </span><span style="background-color: hsl(0, 100.00%, 80.42%); opacity: 0.87" title="-0.082">c</span><span style="background-color: hsl(120, 100.00%, 92.95%); opacity: 0.82" title="0.019">a</span><span style="background-color: hsl(120, 100.00%, 86.16%); opacity: 0.84" title="0.050">n</span><span style="background-color: hsl(0, 100.00%, 83.54%); opacity: 0.86" title="-0.064"> </span><span style="background-color: hsl(0, 100.00%, 86.52%); opacity: 0.84" title="-0.048">d</span><span style="background-color: hsl(0, 100.00%, 99.84%); opacity: 0.80" title="-0.000">o</span><span style="background-color: hsl(120, 100.00%, 95.99%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 85.43%); opacity: 0.85" title="0.054">a</span><span style="background-color: hsl(120, 100.00%, 91.23%); opacity: 0.82" title="0.026">n</span><span style="background-color: hsl(0, 100.00%, 90.91%); opacity: 0.82" title="-0.027">y</span><span style="background-color: hsl(120, 100.00%, 91.22%); opacity: 0.82" title="0.026">t</span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.015">h</span><span style="background-color: hsl(0, 100.00%, 86.57%); opacity: 0.84" title="-0.048">i</span><span style="background-color: hsl(0, 100.00%, 97.95%); opacity: 0.80" title="-0.003">n</span><span style="background-color: hsl(0, 100.00%, 80.94%); opacity: 0.87" title="-0.079">g</span><span style="background-color: hsl(0, 100.00%, 89.64%); opacity: 0.83" title="-0.033"> </span><span style="background-color: hsl(0, 100.00%, 98.05%); opacity: 0.80" title="-0.003">a</span><span style="background-color: hsl(120, 100.00%, 81.92%); opacity: 0.86" title="0.073">bo</span><span style="background-color: hsl(120, 100.00%, 87.20%); opacity: 0.84" title="0.044">u</span><span style="background-color: hsl(0, 100.00%, 92.21%); opacity: 0.82" title="-0.022">t</span><span style="background-color: hsl(0, 100.00%, 87.74%); opacity: 0.84" title="-0.042"> </span><span style="background-color: hsl(0, 100.00%, 92.83%); opacity: 0.82" title="-0.019">t</span><span style="background-color: hsl(0, 100.00%, 87.18%); opacity: 0.84" title="-0.045">h</span><span style="background-color: hsl(0, 100.00%, 71.54%); opacity: 0.92" title="-0.139">e</span><span style="background-color: hsl(0, 100.00%, 87.68%); opacity: 0.84" title="-0.042">m</span><span style="background-color: hsl(0, 100.00%, 88.54%); opacity: 0.83" title="-0.038"> </span><span style="background-color: hsl(0, 100.00%, 93.40%); opacity: 0.82" title="-0.017">e</span><span style="background-color: hsl(0, 100.00%, 90.03%); opacity: 0.83" title="-0.031">x</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(0, 100.00%, 92.26%); opacity: 0.82" title="-0.022">e</span><span style="background-color: hsl(0, 100.00%, 96.56%); opacity: 0.81" title="-0.007">p</span><span style="background-color: hsl(120, 100.00%, 94.30%); opacity: 0.81" title="0.014">t</span><span style="background-color: hsl(0, 100.00%, 87.60%); opacity: 0.84" title="-0.043"> </span><span style="background-color: hsl(0, 100.00%, 76.46%); opacity: 0.89" title="-0.106">r</span><span style="background-color: hsl(0, 100.00%, 85.23%); opacity: 0.85" title="-0.055">e</span><span style="background-color: hsl(0, 100.00%, 97.64%); opacity: 0.80" title="-0.004">l</span><span style="background-color: hsl(120, 100.00%, 87.43%); opacity: 0.84" title="0.043">i</span><span style="background-color: hsl(0, 100.00%, 96.40%); opacity: 0.81" title="-0.007">e</span><span style="background-color: hsl(120, 100.00%, 93.91%); opacity: 0.81" title="0.015">v</span><span style="background-color: hsl(0, 100.00%, 94.68%); opacity: 0.81" title="-0.013">e</span><span style="background-color: hsl(0, 100.00%, 87.43%); opacity: 0.84" title="-0.043"> </span><span style="background-color: hsl(120, 100.00%, 96.11%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(0, 100.00%, 83.73%); opacity: 0.86" title="-0.063">h</span><span style="background-color: hsl(0, 100.00%, 77.36%); opacity: 0.89" title="-0.101">e</span><span style="background-color: hsl(0, 100.00%, 89.95%); opacity: 0.83" title="-0.031"> </span><span style="background-color: hsl(120, 100.00%, 96.01%); opacity: 0.81" title="0.008">p</span><span style="background-color: hsl(120, 100.00%, 89.32%); opacity: 0.83" title="0.034">a</span><span style="background-color: hsl(120, 100.00%, 82.15%); opacity: 0.86" title="0.072">in</span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.002">.</span><span style="background-color: hsl(0, 100.00%, 93.15%); opacity: 0.82" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 89.47%); opacity: 0.83" title="-0.034">e</span><span style="background-color: hsl(120, 100.00%, 95.54%); opacity: 0.81" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 98.59%); opacity: 0.80" title="0.002">t</span><span style="background-color: hsl(0, 100.00%, 78.77%); opacity: 0.88" title="-0.092">h</span><span style="background-color: hsl(0, 100.00%, 72.96%); opacity: 0.91" title="-0.130">e</span><span style="background-color: hsl(0, 100.00%, 88.82%); opacity: 0.83" title="-0.037">r</span><span style="background-color: hsl(0, 100.00%, 91.00%); opacity: 0.82" title="-0.027"> </span><span style="background-color: hsl(120, 100.00%, 86.71%); opacity: 0.84" title="0.047">t</span><span style="background-color: hsl(0, 100.00%, 89.05%); opacity: 0.83" title="-0.036">h</span><span style="background-color: hsl(0, 100.00%, 86.27%); opacity: 0.84" title="-0.049">e</span><span style="background-color: hsl(120, 100.00%, 92.93%); opacity: 0.82" title="0.019">y</span><span style="background-color: hsl(0, 100.00%, 92.47%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(0, 100.00%, 87.80%); opacity: 0.84" title="-0.042">p</span><span style="background-color: hsl(120, 100.00%, 92.19%); opacity: 0.82" title="0.022">a</span><span style="background-color: hsl(120, 100.00%, 92.63%); opacity: 0.82" title="0.020">s</span><span style="background-color: hsl(0, 100.00%, 95.61%); opacity: 0.81" title="-0.010">s</span><span style="background-color: hsl(0, 100.00%, 93.16%); opacity: 0.82" title="-0.018">,</span><span style="background-color: hsl(0, 100.00%, 94.47%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 94.68%); opacity: 0.81" title="-0.013">o</span><span style="background-color: hsl(0, 100.00%, 97.82%); opacity: 0.80" title="-0.004">r </span><span style="background-color: hsl(120, 100.00%, 89.34%); opacity: 0.83" title="0.034">t</span><span style="background-color: hsl(0, 100.00%, 86.78%); opacity: 0.84" title="-0.047">h</span><span style="background-color: hsl(0, 100.00%, 84.19%); opacity: 0.85" title="-0.060">e</span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.008">y</span><span style="background-color: hsl(0, 100.00%, 94.45%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 89.02%); opacity: 0.83" title="-0.036">h</span><span style="background-color: hsl(0, 100.00%, 95.37%); opacity: 0.81" title="-0.010">a</span><span style="background-color: hsl(0, 100.00%, 96.54%); opacity: 0.81" title="-0.007">v</span><span style="background-color: hsl(0, 100.00%, 96.36%); opacity: 0.81" title="-0.007">e</span><span style="background-color: hsl(120, 100.00%, 92.50%); opacity: 0.82" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 84.70%); opacity: 0.85" title="0.057">t</span><span style="background-color: hsl(120, 100.00%, 84.63%); opacity: 0.85" title="0.058">o</span><span style="background-color: hsl(120, 100.00%, 84.68%); opacity: 0.85" title="0.058"> </span><span style="background-color: hsl(120, 100.00%, 89.02%); opacity: 0.83" title="0.036">b</span><span style="background-color: hsl(0, 100.00%, 92.86%); opacity: 0.82" title="-0.019">e</span><span style="background-color: hsl(120, 100.00%, 95.28%); opacity: 0.81" title="0.011"> b</span><span style="background-color: hsl(0, 100.00%, 89.67%); opacity: 0.83" title="-0.033">r</span><span style="background-color: hsl(0, 100.00%, 87.41%); opacity: 0.84" title="-0.043">ok</span><span style="opacity: 0.80">en u</span><span style="background-color: hsl(0, 100.00%, 87.87%); opacity: 0.84" title="-0.041">p</span><span style="background-color: hsl(0, 100.00%, 90.87%); opacity: 0.82" title="-0.027"> </span><span style="background-color: hsl(120, 100.00%, 89.82%); opacity: 0.83" title="0.032">w</span><span style="background-color: hsl(120, 100.00%, 84.71%); opacity: 0.85" title="0.057">i</span><span style="background-color: hsl(120, 100.00%, 72.37%); opacity: 0.92" title="0.134">t</span><span style="background-color: hsl(0, 100.00%, 97.96%); opacity: 0.80" title="-0.003">h</span><span style="background-color: hsl(0, 100.00%, 87.90%); opacity: 0.84" title="-0.041"> </span><span style="background-color: hsl(0, 100.00%, 98.06%); opacity: 0.80" title="-0.003">s</span><span style="background-color: hsl(120, 100.00%, 94.41%); opacity: 0.81" title="0.014">o</span><span style="background-color: hsl(0, 100.00%, 93.93%); opacity: 0.81" title="-0.015">un</span><span style="background-color: hsl(0, 100.00%, 95.57%); opacity: 0.81" title="-0.010">d</span><span style="background-color: hsl(0, 100.00%, 92.80%); opacity: 0.82" title="-0.020">,</span><span style="background-color: hsl(0, 100.00%, 94.46%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 94.68%); opacity: 0.81" title="-0.013">o</span><span style="background-color: hsl(0, 100.00%, 97.82%); opacity: 0.80" title="-0.004">r </span><span style="background-color: hsl(120, 100.00%, 89.34%); opacity: 0.83" title="0.034">t</span><span style="background-color: hsl(0, 100.00%, 86.78%); opacity: 0.84" title="-0.047">h</span><span style="background-color: hsl(0, 100.00%, 84.19%); opacity: 0.85" title="-0.060">e</span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.008">y</span><span style="background-color: hsl(0, 100.00%, 94.45%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 89.12%); opacity: 0.83" title="-0.035">h</span><span style="background-color: hsl(0, 100.00%, 91.74%); opacity: 0.82" title="-0.024">a</span><span style="background-color: hsl(0, 100.00%, 86.25%); opacity: 0.84" title="-0.049">ve</span><span style="background-color: hsl(0, 100.00%, 88.27%); opacity: 0.83" title="-0.039">
    </span><span style="background-color: hsl(0, 100.00%, 98.02%); opacity: 0.80" title="-0.003">t</span><span style="background-color: hsl(120, 100.00%, 92.52%); opacity: 0.82" title="0.021">o</span><span style="background-color: hsl(120, 100.00%, 88.15%); opacity: 0.84" title="0.040"> </span><span style="background-color: hsl(0, 100.00%, 96.20%); opacity: 0.81" title="-0.008">b</span><span style="background-color: hsl(0, 100.00%, 88.10%); opacity: 0.84" title="-0.040">e</span><span style="background-color: hsl(0, 100.00%, 90.61%); opacity: 0.83" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 94.44%); opacity: 0.81" title="-0.014">e</span><span style="background-color: hsl(120, 100.00%, 89.40%); opacity: 0.83" title="0.034">x</span><span style="background-color: hsl(120, 100.00%, 79.19%); opacity: 0.88" title="0.089">t</span><span style="background-color: hsl(120, 100.00%, 79.94%); opacity: 0.87" title="0.085">r</span><span style="background-color: hsl(120, 100.00%, 91.90%); opacity: 0.82" title="0.023">a</span><span style="background-color: hsl(120, 100.00%, 82.80%); opacity: 0.86" title="0.068">c</span><span style="background-color: hsl(120, 100.00%, 80.75%); opacity: 0.87" title="0.080">t</span><span style="background-color: hsl(120, 100.00%, 80.77%); opacity: 0.87" title="0.080">e</span><span style="background-color: hsl(120, 100.00%, 88.93%); opacity: 0.83" title="0.036">d</span><span style="background-color: hsl(120, 100.00%, 90.71%); opacity: 0.82" title="0.028"> </span><span style="background-color: hsl(120, 100.00%, 90.81%); opacity: 0.82" title="0.028">s</span><span style="background-color: hsl(120, 100.00%, 88.02%); opacity: 0.84" title="0.041">u</span><span style="background-color: hsl(120, 100.00%, 94.66%); opacity: 0.81" title="0.013">r</span><span style="opacity: 0.80">gi</span><span style="background-color: hsl(120, 100.00%, 89.66%); opacity: 0.83" title="0.033">c</span><span style="background-color: hsl(120, 100.00%, 86.05%); opacity: 0.84" title="0.050">a</span><span style="background-color: hsl(120, 100.00%, 92.48%); opacity: 0.82" title="0.021">l</span><span style="background-color: hsl(120, 100.00%, 85.97%); opacity: 0.84" title="0.051">l</span><span style="background-color: hsl(120, 100.00%, 88.29%); opacity: 0.83" title="0.039">y</span><span style="background-color: hsl(0, 100.00%, 84.93%); opacity: 0.85" title="-0.056">.</span><span style="background-color: hsl(0, 100.00%, 88.16%); opacity: 0.84" title="-0.040"> </span><span style="background-color: hsl(0, 100.00%, 91.05%); opacity: 0.82" title="-0.027">w</span><span style="background-color: hsl(0, 100.00%, 81.77%); opacity: 0.87" title="-0.074">h</span><span style="background-color: hsl(0, 100.00%, 82.73%); opacity: 0.86" title="-0.068">e</span><span style="background-color: hsl(120, 100.00%, 85.63%); opacity: 0.85" title="0.053">n</span><span style="background-color: hsl(120, 100.00%, 93.33%); opacity: 0.82" title="0.018"> i</span><span style="background-color: hsl(120, 100.00%, 90.39%); opacity: 0.83" title="0.030"> </span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 89.36%); opacity: 0.83" title="0.034">a</span><span style="background-color: hsl(120, 100.00%, 86.58%); opacity: 0.84" title="0.048">s</span><span style="background-color: hsl(120, 100.00%, 97.97%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 88.14%); opacity: 0.84" title="0.040">i</span><span style="background-color: hsl(120, 100.00%, 82.64%); opacity: 0.86" title="0.069">n</span><span style="background-color: hsl(120, 100.00%, 96.64%); opacity: 0.81" title="0.007">,</span><span style="background-color: hsl(0, 100.00%, 91.13%); opacity: 0.82" title="-0.026"> </span><span style="background-color: hsl(120, 100.00%, 95.42%); opacity: 0.81" title="0.010">t</span><span style="background-color: hsl(0, 100.00%, 81.79%); opacity: 0.86" title="-0.074">h</span><span style="background-color: hsl(0, 100.00%, 82.99%); opacity: 0.86" title="-0.067">e</span><span style="background-color: hsl(120, 100.00%, 88.94%); opacity: 0.83" title="0.036"> </span><span style="background-color: hsl(120, 100.00%, 74.54%); opacity: 0.90" title="0.119">x</span><span style="background-color: hsl(120, 100.00%, 79.58%); opacity: 0.88" title="0.087">-</span><span style="opacity: 0.80">ra</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(0, 100.00%, 98.84%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(0, 100.00%, 96.99%); opacity: 0.80" title="-0.006">t</span><span style="background-color: hsl(0, 100.00%, 97.70%); opacity: 0.80" title="-0.004">ec</span><span style="background-color: hsl(0, 100.00%, 89.41%); opacity: 0.83" title="-0.034">h </span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 96.44%); opacity: 0.81" title="0.007">a</span><span style="background-color: hsl(120, 100.00%, 96.15%); opacity: 0.81" title="0.008">p</span><span style="background-color: hsl(120, 100.00%, 86.75%); opacity: 0.84" title="0.047">p</span><span style="background-color: hsl(120, 100.00%, 93.92%); opacity: 0.81" title="0.015">e</span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.008">n</span><span style="background-color: hsl(120, 100.00%, 98.20%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(0, 100.00%, 93.99%); opacity: 0.81" title="-0.015">d</span><span style="background-color: hsl(0, 100.00%, 92.88%); opacity: 0.82" title="-0.019"> </span><span style="background-color: hsl(0, 100.00%, 97.02%); opacity: 0.80" title="-0.006">t</span><span style="background-color: hsl(120, 100.00%, 91.62%); opacity: 0.82" title="0.024">o</span><span style="background-color: hsl(120, 100.00%, 96.42%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(0, 100.00%, 93.58%); opacity: 0.81" title="-0.017">m</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.009">n</span><span style="background-color: hsl(120, 100.00%, 93.10%); opacity: 0.82" title="0.018">t</span><span style="background-color: hsl(0, 100.00%, 93.24%); opacity: 0.82" title="-0.018">i</span><span style="background-color: hsl(0, 100.00%, 90.37%); opacity: 0.83" title="-0.030">o</span><span style="background-color: hsl(0, 100.00%, 85.77%); opacity: 0.85" title="-0.052">n</span><span style="background-color: hsl(0, 100.00%, 89.43%); opacity: 0.83" title="-0.034"> </span><span style="background-color: hsl(120, 100.00%, 97.64%); opacity: 0.80" title="0.004">t</span><span style="background-color: hsl(0, 100.00%, 97.96%); opacity: 0.80" title="-0.003">h</span><span style="background-color: hsl(0, 100.00%, 93.78%); opacity: 0.81" title="-0.016">a</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 93.10%); opacity: 0.82" title="0.018"> </span><span style="background-color: hsl(120, 100.00%, 91.14%); opacity: 0.82" title="0.026">s</span><span style="background-color: hsl(0, 100.00%, 89.90%); opacity: 0.83" title="-0.032">h</span><span style="background-color: hsl(0, 100.00%, 90.61%); opacity: 0.83" title="-0.029">e</span><span style="background-color: hsl(0, 100.00%, 91.39%); opacity: 0.82" title="-0.025">'</span><span style="background-color: hsl(0, 100.00%, 89.44%); opacity: 0.83" title="-0.034">d</span><span style="background-color: hsl(120, 100.00%, 95.51%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 80.30%); opacity: 0.87" title="0.082">h</span><span style="background-color: hsl(120, 100.00%, 82.51%); opacity: 0.86" title="0.070">a</span><span style="background-color: hsl(120, 100.00%, 85.15%); opacity: 0.85" title="0.055">d</span><span style="background-color: hsl(0, 100.00%, 98.68%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(0, 100.00%, 87.00%); opacity: 0.84" title="-0.046">k</span><span style="background-color: hsl(0, 100.00%, 89.43%); opacity: 0.83" title="-0.034">id</span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.009">n</span><span style="background-color: hsl(0, 100.00%, 94.91%); opacity: 0.81" title="-0.012">e</span><span style="background-color: hsl(0, 100.00%, 90.39%); opacity: 0.83" title="-0.030">y</span><span style="background-color: hsl(0, 100.00%, 86.77%); opacity: 0.84" title="-0.047">
    </span><span style="background-color: hsl(0, 100.00%, 85.53%); opacity: 0.85" title="-0.053">s</span><span style="background-color: hsl(0, 100.00%, 85.02%); opacity: 0.85" title="-0.056">t</span><span style="background-color: hsl(0, 100.00%, 83.82%); opacity: 0.85" title="-0.062">o</span><span style="background-color: hsl(0, 100.00%, 86.09%); opacity: 0.84" title="-0.050">n</span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.019">e</span><span style="background-color: hsl(120, 100.00%, 83.86%); opacity: 0.85" title="0.062">s</span><span style="background-color: hsl(120, 100.00%, 91.91%); opacity: 0.82" title="0.023"> </span><span style="background-color: hsl(120, 100.00%, 84.80%); opacity: 0.85" title="0.057">a</span><span style="background-color: hsl(120, 100.00%, 89.89%); opacity: 0.83" title="0.032">n</span><span style="background-color: hsl(0, 100.00%, 90.37%); opacity: 0.83" title="-0.030">d</span><span style="background-color: hsl(0, 100.00%, 99.19%); opacity: 0.80" title="-0.001"> c</span><span style="background-color: hsl(120, 100.00%, 86.32%); opacity: 0.84" title="0.049">h</span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.007">i</span><span style="background-color: hsl(0, 100.00%, 93.66%); opacity: 0.81" title="-0.016">l</span><span style="background-color: hsl(0, 100.00%, 90.41%); opacity: 0.83" title="-0.029">dr</span><span style="background-color: hsl(0, 100.00%, 87.57%); opacity: 0.84" title="-0.043">e</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">,</span><span style="background-color: hsl(120, 100.00%, 95.01%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 85.41%); opacity: 0.85" title="0.054">a</span><span style="background-color: hsl(120, 100.00%, 87.09%); opacity: 0.84" title="0.045">n</span><span style="opacity: 0.80">d</span><span style="background-color: hsl(0, 100.00%, 90.14%); opacity: 0.83" title="-0.031"> </span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.007">t</span><span style="background-color: hsl(0, 100.00%, 84.02%); opacity: 0.85" title="-0.061">h</span><span style="background-color: hsl(0, 100.00%, 77.61%); opacity: 0.89" title="-0.099">e</span><span style="background-color: hsl(0, 100.00%, 98.61%); opacity: 0.80" title="-0.002"> </span><span style="background-color: hsl(120, 100.00%, 90.57%); opacity: 0.83" title="0.029">c</span><span style="background-color: hsl(120, 100.00%, 83.26%); opacity: 0.86" title="0.065">h</span><span style="background-color: hsl(0, 100.00%, 94.84%); opacity: 0.81" title="-0.012">il</span><span style="background-color: hsl(0, 100.00%, 93.16%); opacity: 0.82" title="-0.018">d</span><span style="background-color: hsl(0, 100.00%, 95.14%); opacity: 0.81" title="-0.011">bi</span><span style="background-color: hsl(120, 100.00%, 86.52%); opacity: 0.84" title="0.048">r</span><span style="background-color: hsl(120, 100.00%, 79.75%); opacity: 0.88" title="0.086">t</span><span style="background-color: hsl(120, 100.00%, 86.12%); opacity: 0.84" title="0.050">h</span><span style="background-color: hsl(0, 100.00%, 91.76%); opacity: 0.82" title="-0.024"> </span><span style="background-color: hsl(0, 100.00%, 89.16%); opacity: 0.83" title="-0.035">h</span><span style="background-color: hsl(0, 100.00%, 94.07%); opacity: 0.81" title="-0.015">u</span><span style="background-color: hsl(120, 100.00%, 94.15%); opacity: 0.81" title="0.015">r</span><span style="background-color: hsl(120, 100.00%, 98.66%); opacity: 0.80" title="0.002">t</span><span style="background-color: hsl(120, 100.00%, 92.47%); opacity: 0.82" title="0.021"> </span><span style="background-color: hsl(120, 100.00%, 81.50%); opacity: 0.87" title="0.075">l</span><span style="background-color: hsl(120, 100.00%, 73.46%); opacity: 0.91" title="0.126">e</span><span style="background-color: hsl(120, 100.00%, 82.91%); opacity: 0.86" title="0.067">s</span><span style="background-color: hsl(120, 100.00%, 94.21%); opacity: 0.81" title="0.014">s</span><span style="background-color: hsl(0, 100.00%, 92.30%); opacity: 0.82" title="-0.022">.</span>
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

    {'score': 0.88798709737744663, 'mean_KL_divergence': 0.060635596001926591}




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
    
        
        (probability <b>0.679</b>, score <b>0.982</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.182
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
            
    
            
            
                <tr style="background-color: hsl(0, 100.00%, 94.23%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.200
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
            
    
            </tbody>
        </table>
    
        
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">a</span><span style="background-color: hsl(0, 100.00%, 92.81%); opacity: 0.82" title="-0.017">s</span><span style="background-color: hsl(0, 100.00%, 86.82%); opacity: 0.84" title="-0.040"> </span><span style="background-color: hsl(0, 100.00%, 90.02%); opacity: 0.83" title="-0.027">i</span><span style="background-color: hsl(0, 100.00%, 89.94%); opacity: 0.83" title="-0.027"> </span><span style="background-color: hsl(120, 100.00%, 94.19%); opacity: 0.81" title="0.013">r</span><span style="background-color: hsl(120, 100.00%, 86.96%); opacity: 0.84" title="0.040">e</span><span style="background-color: hsl(120, 100.00%, 83.20%); opacity: 0.86" title="0.057">c</span><span style="background-color: hsl(120, 100.00%, 88.19%); opacity: 0.83" title="0.034">a</span><span style="background-color: hsl(120, 100.00%, 88.96%); opacity: 0.83" title="0.031">l</span><span style="background-color: hsl(0, 100.00%, 93.13%); opacity: 0.82" title="-0.016">l</span><span style="background-color: hsl(0, 100.00%, 92.61%); opacity: 0.82" title="-0.018"> </span><span style="background-color: hsl(0, 100.00%, 94.96%); opacity: 0.81" title="-0.010">f</span><span style="background-color: hsl(0, 100.00%, 95.99%); opacity: 0.81" title="-0.007">r</span><span style="background-color: hsl(0, 100.00%, 90.69%); opacity: 0.82" title="-0.025">o</span><span style="background-color: hsl(0, 100.00%, 92.75%); opacity: 0.82" title="-0.017">m</span><span style="background-color: hsl(120, 100.00%, 94.28%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 83.78%); opacity: 0.86" title="0.054">m</span><span style="background-color: hsl(120, 100.00%, 85.04%); opacity: 0.85" title="0.048">y</span><span style="background-color: hsl(120, 100.00%, 96.99%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.018">b</span><span style="background-color: hsl(120, 100.00%, 85.23%); opacity: 0.85" title="0.047">o</span><span style="background-color: hsl(120, 100.00%, 95.23%); opacity: 0.81" title="0.009">u</span><span style="background-color: hsl(0, 100.00%, 94.79%); opacity: 0.81" title="-0.011">t</span><span style="background-color: hsl(0, 100.00%, 94.55%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(0, 100.00%, 95.87%); opacity: 0.81" title="-0.008">w</span><span style="background-color: hsl(0, 100.00%, 95.18%); opacity: 0.81" title="-0.010">i</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.010">t</span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.010">h</span><span style="background-color: hsl(0, 100.00%, 93.54%); opacity: 0.81" title="-0.015"> </span><span style="background-color: hsl(0, 100.00%, 95.82%); opacity: 0.81" title="-0.008">k</span><span style="background-color: hsl(120, 100.00%, 96.24%); opacity: 0.81" title="0.007">i</span><span style="background-color: hsl(120, 100.00%, 88.51%); opacity: 0.83" title="0.033">d</span><span style="background-color: hsl(120, 100.00%, 91.23%); opacity: 0.82" title="0.023">n</span><span style="background-color: hsl(120, 100.00%, 89.90%); opacity: 0.83" title="0.028">e</span><span style="background-color: hsl(120, 100.00%, 96.64%); opacity: 0.81" title="0.006">y</span><span style="background-color: hsl(0, 100.00%, 94.76%); opacity: 0.81" title="-0.011"> </span><span style="opacity: 0.80">s</span><span style="background-color: hsl(0, 100.00%, 96.15%); opacity: 0.81" title="-0.007">t</span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.004">o</span><span style="background-color: hsl(120, 100.00%, 97.65%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(120, 100.00%, 94.36%); opacity: 0.81" title="0.012">e</span><span style="background-color: hsl(120, 100.00%, 92.03%); opacity: 0.82" title="0.020">s</span><span style="background-color: hsl(0, 100.00%, 95.33%); opacity: 0.81" title="-0.009">,</span><span style="background-color: hsl(0, 100.00%, 98.98%); opacity: 0.80" title="-0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.82%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(0, 100.00%, 96.56%); opacity: 0.81" title="-0.006">h</span><span style="background-color: hsl(0, 100.00%, 90.67%); opacity: 0.83" title="-0.025">e</span><span style="background-color: hsl(0, 100.00%, 93.03%); opacity: 0.82" title="-0.016">r</span><span style="background-color: hsl(0, 100.00%, 92.22%); opacity: 0.82" title="-0.019">e </span><span style="opacity: 0.80">i</span><span style="background-color: hsl(120, 100.00%, 93.55%); opacity: 0.81" title="0.015">sn</span><span style="opacity: 0.80">'</span><span style="background-color: hsl(0, 100.00%, 96.98%); opacity: 0.80" title="-0.005">t</span><span style="background-color: hsl(120, 100.00%, 94.55%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 94.29%); opacity: 0.81" title="0.012">a</span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.007">n</span><span style="background-color: hsl(0, 100.00%, 91.98%); opacity: 0.82" title="-0.020">y</span><span style="background-color: hsl(120, 100.00%, 89.95%); opacity: 0.83" title="0.027">
    </span><span style="background-color: hsl(120, 100.00%, 74.82%); opacity: 0.90" title="0.102">m</span><span style="background-color: hsl(120, 100.00%, 66.42%); opacity: 0.96" title="0.153">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.197">d</span><span style="background-color: hsl(120, 100.00%, 65.80%); opacity: 0.96" title="0.158">i</span><span style="background-color: hsl(120, 100.00%, 78.55%); opacity: 0.88" title="0.081">c</span><span style="background-color: hsl(120, 100.00%, 91.62%); opacity: 0.82" title="0.021">a</span><span style="background-color: hsl(120, 100.00%, 95.87%); opacity: 0.81" title="0.008">t</span><span style="background-color: hsl(0, 100.00%, 94.34%); opacity: 0.81" title="-0.012">i</span><span style="background-color: hsl(0, 100.00%, 93.63%); opacity: 0.81" title="-0.014">o</span><span style="background-color: hsl(0, 100.00%, 87.77%); opacity: 0.84" title="-0.036">n</span><span style="background-color: hsl(0, 100.00%, 91.29%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(120, 100.00%, 95.71%); opacity: 0.81" title="0.008">th</span><span style="background-color: hsl(120, 100.00%, 98.03%); opacity: 0.80" title="0.003">at</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 95.69%); opacity: 0.81" title="0.008">c</span><span style="background-color: hsl(120, 100.00%, 88.72%); opacity: 0.83" title="0.032">a</span><span style="background-color: hsl(0, 100.00%, 99.18%); opacity: 0.80" title="-0.001">n</span><span style="background-color: hsl(0, 100.00%, 89.86%); opacity: 0.83" title="-0.028"> </span><span style="background-color: hsl(0, 100.00%, 94.59%); opacity: 0.81" title="-0.011">d</span><span style="background-color: hsl(0, 100.00%, 91.51%); opacity: 0.82" title="-0.022">o</span><span style="background-color: hsl(0, 100.00%, 96.86%); opacity: 0.81" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 93.69%); opacity: 0.81" title="0.014">a</span><span style="background-color: hsl(0, 100.00%, 90.31%); opacity: 0.83" title="-0.026">n</span><span style="background-color: hsl(0, 100.00%, 86.71%); opacity: 0.84" title="-0.041">y</span><span style="background-color: hsl(0, 100.00%, 90.76%); opacity: 0.82" title="-0.024">t</span><span style="background-color: hsl(0, 100.00%, 90.30%); opacity: 0.83" title="-0.026">h</span><span style="background-color: hsl(120, 100.00%, 96.26%); opacity: 0.81" title="0.007">i</span><span style="background-color: hsl(120, 100.00%, 93.30%); opacity: 0.82" title="0.015">n</span><span style="background-color: hsl(0, 100.00%, 85.93%); opacity: 0.84" title="-0.044">g</span><span style="background-color: hsl(0, 100.00%, 94.18%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(120, 100.00%, 90.02%); opacity: 0.83" title="0.027">a</span><span style="background-color: hsl(120, 100.00%, 87.59%); opacity: 0.84" title="0.037">b</span><span style="background-color: hsl(120, 100.00%, 83.66%); opacity: 0.86" title="0.055">o</span><span style="background-color: hsl(120, 100.00%, 92.30%); opacity: 0.82" title="0.019">u</span><span style="background-color: hsl(0, 100.00%, 96.91%); opacity: 0.81" title="-0.005">t</span><span style="background-color: hsl(0, 100.00%, 99.42%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(0, 100.00%, 95.43%); opacity: 0.81" title="-0.009">t</span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.004">h</span><span style="background-color: hsl(0, 100.00%, 95.05%); opacity: 0.81" title="-0.010">e</span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.008">m</span><span style="background-color: hsl(0, 100.00%, 96.17%); opacity: 0.81" title="-0.007"> </span><span style="background-color: hsl(0, 100.00%, 97.47%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(0, 100.00%, 99.63%); opacity: 0.80" title="-0.000">x</span><span style="background-color: hsl(120, 100.00%, 91.90%); opacity: 0.82" title="0.020">ce</span><span style="background-color: hsl(0, 100.00%, 97.19%); opacity: 0.80" title="-0.004">p</span><span style="background-color: hsl(0, 100.00%, 96.04%); opacity: 0.81" title="-0.007">t</span><span style="background-color: hsl(0, 100.00%, 94.16%); opacity: 0.81" title="-0.013"> </span><span style="background-color: hsl(0, 100.00%, 92.21%); opacity: 0.82" title="-0.019">r</span><span style="background-color: hsl(0, 100.00%, 91.81%); opacity: 0.82" title="-0.020">e</span><span style="background-color: hsl(0, 100.00%, 95.72%); opacity: 0.81" title="-0.008">l</span><span style="background-color: hsl(0, 100.00%, 98.63%); opacity: 0.80" title="-0.002">i</span><span style="background-color: hsl(0, 100.00%, 93.50%); opacity: 0.81" title="-0.015">e</span><span style="background-color: hsl(120, 100.00%, 95.28%); opacity: 0.81" title="0.009">v</span><span style="background-color: hsl(0, 100.00%, 98.94%); opacity: 0.80" title="-0.001">e</span><span style="background-color: hsl(0, 100.00%, 91.20%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 93.30%); opacity: 0.82" title="-0.015">t</span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.011">h</span><span style="background-color: hsl(0, 100.00%, 88.45%); opacity: 0.83" title="-0.033">e</span><span style="background-color: hsl(0, 100.00%, 94.23%); opacity: 0.81" title="-0.012"> </span><span style="background-color: hsl(120, 100.00%, 86.35%); opacity: 0.84" title="0.042">pa</span><span style="background-color: hsl(120, 100.00%, 80.92%); opacity: 0.87" title="0.068">i</span><span style="background-color: hsl(120, 100.00%, 84.25%); opacity: 0.85" title="0.052">n</span><span style="background-color: hsl(0, 100.00%, 90.98%); opacity: 0.82" title="-0.023">.</span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 93.57%); opacity: 0.81" title="-0.014">e</span><span style="background-color: hsl(0, 100.00%, 87.37%); opacity: 0.84" title="-0.038">i</span><span style="background-color: hsl(0, 100.00%, 84.00%); opacity: 0.85" title="-0.053">t</span><span style="background-color: hsl(0, 100.00%, 83.80%); opacity: 0.85" title="-0.054">h</span><span style="background-color: hsl(0, 100.00%, 88.43%); opacity: 0.83" title="-0.033">e</span><span style="background-color: hsl(0, 100.00%, 89.67%); opacity: 0.83" title="-0.028">r</span><span style="background-color: hsl(0, 100.00%, 96.98%); opacity: 0.80" title="-0.005"> </span><span style="background-color: hsl(120, 100.00%, 97.18%); opacity: 0.80" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.009">h</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.006">e</span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.002">y</span><span style="background-color: hsl(0, 100.00%, 92.94%); opacity: 0.82" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 96.64%); opacity: 0.81" title="-0.006">p</span><span style="background-color: hsl(0, 100.00%, 93.45%); opacity: 0.82" title="-0.015">a</span><span style="background-color: hsl(0, 100.00%, 95.35%); opacity: 0.81" title="-0.009">ss</span><span style="background-color: hsl(0, 100.00%, 95.33%); opacity: 0.81" title="-0.009">,</span><span style="background-color: hsl(0, 100.00%, 90.31%); opacity: 0.83" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 92.86%); opacity: 0.82" title="-0.017">o</span><span style="background-color: hsl(0, 100.00%, 97.11%); opacity: 0.80" title="-0.005">r</span><span style="background-color: hsl(120, 100.00%, 97.62%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 97.18%); opacity: 0.80" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.009">h</span><span style="background-color: hsl(120, 100.00%, 94.35%); opacity: 0.81" title="0.012">e</span><span style="background-color: hsl(120, 100.00%, 97.54%); opacity: 0.80" title="0.004">y</span><span style="background-color: hsl(0, 100.00%, 92.89%); opacity: 0.82" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 88.82%); opacity: 0.83" title="-0.032">a</span><span style="background-color: hsl(0, 100.00%, 94.57%); opacity: 0.81" title="-0.011">v</span><span style="background-color: hsl(0, 100.00%, 92.32%); opacity: 0.82" title="-0.019">e</span><span style="background-color: hsl(0, 100.00%, 91.06%); opacity: 0.82" title="-0.023"> </span><span style="background-color: hsl(0, 100.00%, 96.40%); opacity: 0.81" title="-0.006">t</span><span style="background-color: hsl(0, 100.00%, 91.89%); opacity: 0.82" title="-0.020">o</span><span style="background-color: hsl(0, 100.00%, 90.26%); opacity: 0.83" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 96.11%); opacity: 0.81" title="-0.007">b</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.010">e</span><span style="background-color: hsl(0, 100.00%, 91.54%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(0, 100.00%, 94.59%); opacity: 0.81" title="-0.011">b</span><span style="background-color: hsl(0, 100.00%, 97.70%); opacity: 0.80" title="-0.003">ro</span><span style="opacity: 0.80">k</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(0, 100.00%, 91.40%); opacity: 0.82" title="-0.022">n</span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.018"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.020">u</span><span style="background-color: hsl(120, 100.00%, 90.89%); opacity: 0.82" title="0.024">p</span><span style="background-color: hsl(120, 100.00%, 97.34%); opacity: 0.80" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 96.24%); opacity: 0.81" title="0.007">w</span><span style="background-color: hsl(0, 100.00%, 96.30%); opacity: 0.81" title="-0.007">i</span><span style="background-color: hsl(0, 100.00%, 97.76%); opacity: 0.80" title="-0.003">t</span><span style="background-color: hsl(0, 100.00%, 95.17%); opacity: 0.81" title="-0.010">h</span><span style="background-color: hsl(120, 100.00%, 96.80%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 90.46%); opacity: 0.83" title="0.025">s</span><span style="background-color: hsl(120, 100.00%, 85.11%); opacity: 0.85" title="0.048">o</span><span style="background-color: hsl(120, 100.00%, 86.65%); opacity: 0.84" title="0.041">u</span><span style="background-color: hsl(120, 100.00%, 85.94%); opacity: 0.84" title="0.044">n</span><span style="background-color: hsl(120, 100.00%, 89.11%); opacity: 0.83" title="0.031">d</span><span style="background-color: hsl(0, 100.00%, 93.08%); opacity: 0.82" title="-0.016">,</span><span style="background-color: hsl(0, 100.00%, 89.53%); opacity: 0.83" title="-0.029"> </span><span style="background-color: hsl(0, 100.00%, 92.86%); opacity: 0.82" title="-0.017">o</span><span style="background-color: hsl(0, 100.00%, 97.11%); opacity: 0.80" title="-0.005">r</span><span style="background-color: hsl(120, 100.00%, 97.62%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 97.18%); opacity: 0.80" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.009">h</span><span style="background-color: hsl(120, 100.00%, 94.35%); opacity: 0.81" title="0.012">e</span><span style="background-color: hsl(120, 100.00%, 97.54%); opacity: 0.80" title="0.004">y</span><span style="background-color: hsl(0, 100.00%, 92.89%); opacity: 0.82" title="-0.017"> </span><span style="background-color: hsl(0, 100.00%, 93.95%); opacity: 0.81" title="-0.013">h</span><span style="background-color: hsl(0, 100.00%, 87.91%); opacity: 0.84" title="-0.036">a</span><span style="background-color: hsl(0, 100.00%, 94.70%); opacity: 0.81" title="-0.011">v</span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.004">e</span><span style="background-color: hsl(0, 100.00%, 97.68%); opacity: 0.80" title="-0.003">
    </span><span style="background-color: hsl(0, 100.00%, 99.11%); opacity: 0.80" title="-0.001">t</span><span style="background-color: hsl(0, 100.00%, 94.72%); opacity: 0.81" title="-0.011">o</span><span style="background-color: hsl(0, 100.00%, 91.63%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(0, 100.00%, 96.11%); opacity: 0.81" title="-0.007">b</span><span style="background-color: hsl(0, 100.00%, 95.00%); opacity: 0.81" title="-0.010">e</span><span style="background-color: hsl(0, 100.00%, 94.96%); opacity: 0.81" title="-0.010"> </span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.004">x</span><span style="background-color: hsl(120, 100.00%, 92.58%); opacity: 0.82" title="0.018">t</span><span style="background-color: hsl(120, 100.00%, 86.68%); opacity: 0.84" title="0.041">r</span><span style="background-color: hsl(120, 100.00%, 91.05%); opacity: 0.82" title="0.023">a</span><span style="background-color: hsl(120, 100.00%, 94.67%); opacity: 0.81" title="0.011">c</span><span style="background-color: hsl(120, 100.00%, 90.15%); opacity: 0.83" title="0.027">t</span><span style="background-color: hsl(120, 100.00%, 91.69%); opacity: 0.82" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 93.23%); opacity: 0.82" title="0.016">d</span><span style="background-color: hsl(120, 100.00%, 92.16%); opacity: 0.82" title="0.019"> </span><span style="background-color: hsl(120, 100.00%, 87.99%); opacity: 0.84" title="0.035">s</span><span style="background-color: hsl(120, 100.00%, 87.04%); opacity: 0.84" title="0.039">u</span><span style="background-color: hsl(120, 100.00%, 85.84%); opacity: 0.85" title="0.045">r</span><span style="background-color: hsl(120, 100.00%, 93.58%); opacity: 0.81" title="0.014">g</span><span style="background-color: hsl(120, 100.00%, 94.10%); opacity: 0.81" title="0.013">i</span><span style="background-color: hsl(120, 100.00%, 92.32%); opacity: 0.82" title="0.019">c</span><span style="background-color: hsl(120, 100.00%, 87.35%); opacity: 0.84" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 88.10%); opacity: 0.84" title="0.035">l</span><span style="background-color: hsl(120, 100.00%, 89.48%); opacity: 0.83" title="0.029">l</span><span style="background-color: hsl(120, 100.00%, 94.50%); opacity: 0.81" title="0.012">y</span><span style="background-color: hsl(0, 100.00%, 85.71%); opacity: 0.85" title="-0.045">.</span><span style="background-color: hsl(0, 100.00%, 98.11%); opacity: 0.80" title="-0.003"> </span><span style="background-color: hsl(120, 100.00%, 82.82%); opacity: 0.86" title="0.059">w</span><span style="background-color: hsl(120, 100.00%, 81.75%); opacity: 0.87" title="0.064">h</span><span style="background-color: hsl(120, 100.00%, 86.03%); opacity: 0.84" title="0.044">e</span><span style="background-color: hsl(120, 100.00%, 95.45%); opacity: 0.81" title="0.009">n</span><span style="background-color: hsl(0, 100.00%, 90.24%); opacity: 0.83" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 90.74%); opacity: 0.82" title="-0.024">i</span><span style="background-color: hsl(0, 100.00%, 90.49%); opacity: 0.83" title="-0.025"> </span><span style="background-color: hsl(120, 100.00%, 96.08%); opacity: 0.81" title="0.007">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(0, 100.00%, 91.72%); opacity: 0.82" title="-0.021">s</span><span style="background-color: hsl(0, 100.00%, 89.96%); opacity: 0.83" title="-0.027"> </span><span style="background-color: hsl(120, 100.00%, 90.28%); opacity: 0.83" title="0.026">in</span><span style="background-color: hsl(0, 100.00%, 93.19%); opacity: 0.82" title="-0.016">,</span><span style="background-color: hsl(0, 100.00%, 93.21%); opacity: 0.82" title="-0.016"> </span><span style="background-color: hsl(0, 100.00%, 93.30%); opacity: 0.82" title="-0.015">t</span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.011">h</span><span style="background-color: hsl(0, 100.00%, 91.09%); opacity: 0.82" title="-0.023">e</span><span style="background-color: hsl(0, 100.00%, 91.59%); opacity: 0.82" title="-0.021"> </span><span style="background-color: hsl(120, 100.00%, 95.37%); opacity: 0.81" title="0.009">x</span><span style="background-color: hsl(0, 100.00%, 96.00%); opacity: 0.81" title="-0.007">-</span><span style="background-color: hsl(120, 100.00%, 94.83%); opacity: 0.81" title="0.011">r</span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">a</span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.019">y</span><span style="background-color: hsl(0, 100.00%, 94.76%); opacity: 0.81" title="-0.011"> </span><span style="opacity: 0.80">t</span><span style="background-color: hsl(0, 100.00%, 99.37%); opacity: 0.80" title="-0.001">e</span><span style="background-color: hsl(120, 100.00%, 91.28%); opacity: 0.82" title="0.022">c</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.005">h</span><span style="background-color: hsl(0, 100.00%, 93.66%); opacity: 0.81" title="-0.014"> </span><span style="background-color: hsl(0, 100.00%, 96.80%); opacity: 0.81" title="-0.005">h</span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.016">a</span><span style="background-color: hsl(120, 100.00%, 92.54%); opacity: 0.82" title="0.018">p</span><span style="background-color: hsl(120, 100.00%, 89.14%); opacity: 0.83" title="0.031">p</span><span style="background-color: hsl(120, 100.00%, 89.70%); opacity: 0.83" title="0.028">e</span><span style="background-color: hsl(0, 100.00%, 99.86%); opacity: 0.80" title="-0.000">n</span><span style="background-color: hsl(0, 100.00%, 97.26%); opacity: 0.80" title="-0.004">ed</span><span style="background-color: hsl(0, 100.00%, 97.27%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 98.55%); opacity: 0.80" title="-0.002">t</span><span style="background-color: hsl(0, 100.00%, 91.87%); opacity: 0.82" title="-0.020">o</span><span style="background-color: hsl(0, 100.00%, 94.58%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 93.98%); opacity: 0.81" title="0.013">m</span><span style="background-color: hsl(120, 100.00%, 88.15%); opacity: 0.84" title="0.035">e</span><span style="background-color: hsl(120, 100.00%, 91.52%); opacity: 0.82" title="0.021">n</span><span style="background-color: hsl(120, 100.00%, 88.12%); opacity: 0.84" title="0.035">t</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(0, 100.00%, 95.50%); opacity: 0.81" title="-0.009">o</span><span style="background-color: hsl(0, 100.00%, 89.13%); opacity: 0.83" title="-0.031">n</span><span style="background-color: hsl(0, 100.00%, 91.29%); opacity: 0.82" title="-0.022"> </span><span style="background-color: hsl(120, 100.00%, 95.71%); opacity: 0.81" title="0.008">th</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.005">t</span><span style="background-color: hsl(120, 100.00%, 92.33%); opacity: 0.82" title="0.019"> </span><span style="background-color: hsl(120, 100.00%, 90.75%); opacity: 0.82" title="0.024">s</span><span style="background-color: hsl(120, 100.00%, 89.60%); opacity: 0.83" title="0.029">h</span><span style="background-color: hsl(120, 100.00%, 93.32%); opacity: 0.82" title="0.015">e</span><span style="background-color: hsl(0, 100.00%, 97.07%); opacity: 0.80" title="-0.005">'d</span><span style="background-color: hsl(120, 100.00%, 92.47%); opacity: 0.82" title="0.018"> </span><span style="background-color: hsl(120, 100.00%, 86.53%); opacity: 0.84" title="0.042">had</span><span style="background-color: hsl(120, 100.00%, 96.91%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(0, 100.00%, 96.14%); opacity: 0.81" title="-0.007">k</span><span style="background-color: hsl(120, 100.00%, 94.82%); opacity: 0.81" title="0.011">i</span><span style="background-color: hsl(120, 100.00%, 88.51%); opacity: 0.83" title="0.033">d</span><span style="background-color: hsl(120, 100.00%, 91.23%); opacity: 0.82" title="0.023">n</span><span style="background-color: hsl(120, 100.00%, 91.36%); opacity: 0.82" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 94.69%); opacity: 0.81" title="0.011">y</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.002">
    s</span><span style="background-color: hsl(0, 100.00%, 94.21%); opacity: 0.81" title="-0.012">t</span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.004">o</span><span style="background-color: hsl(120, 100.00%, 97.65%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(120, 100.00%, 94.36%); opacity: 0.81" title="0.012">e</span><span style="background-color: hsl(0, 100.00%, 99.15%); opacity: 0.80" title="-0.001">s</span><span style="background-color: hsl(120, 100.00%, 95.54%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 87.39%); opacity: 0.84" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 88.62%); opacity: 0.83" title="0.033">n</span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.020">d</span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.011"> </span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.009">c</span><span style="background-color: hsl(120, 100.00%, 92.50%); opacity: 0.82" title="0.018">h</span><span style="background-color: hsl(0, 100.00%, 96.41%); opacity: 0.81" title="-0.006">i</span><span style="background-color: hsl(120, 100.00%, 95.96%); opacity: 0.81" title="0.007">l</span><span style="background-color: hsl(120, 100.00%, 96.86%); opacity: 0.81" title="0.005">d</span><span style="background-color: hsl(0, 100.00%, 89.04%); opacity: 0.83" title="-0.031">r</span><span style="background-color: hsl(0, 100.00%, 86.75%); opacity: 0.84" title="-0.041">e</span><span style="background-color: hsl(0, 100.00%, 90.26%); opacity: 0.83" title="-0.026">n</span><span style="background-color: hsl(0, 100.00%, 87.99%); opacity: 0.84" title="-0.035">,</span><span style="background-color: hsl(120, 100.00%, 95.23%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 87.99%); opacity: 0.84" title="0.035">a</span><span style="background-color: hsl(120, 100.00%, 89.17%); opacity: 0.83" title="0.030">n</span><span style="background-color: hsl(120, 100.00%, 90.79%); opacity: 0.82" title="0.024">d</span><span style="background-color: hsl(0, 100.00%, 99.40%); opacity: 0.80" title="-0.000"> </span><span style="background-color: hsl(0, 100.00%, 93.30%); opacity: 0.82" title="-0.015">t</span><span style="background-color: hsl(0, 100.00%, 92.80%); opacity: 0.82" title="-0.017">h</span><span style="background-color: hsl(0, 100.00%, 87.02%); opacity: 0.84" title="-0.039">e</span><span style="background-color: hsl(0, 100.00%, 85.32%); opacity: 0.85" title="-0.047"> </span><span style="background-color: hsl(120, 100.00%, 96.21%); opacity: 0.81" title="0.007">c</span><span style="background-color: hsl(120, 100.00%, 93.11%); opacity: 0.82" title="0.016">h</span><span style="background-color: hsl(0, 100.00%, 95.39%); opacity: 0.81" title="-0.009">i</span><span style="background-color: hsl(120, 100.00%, 92.50%); opacity: 0.82" title="0.018">l</span><span style="background-color: hsl(120, 100.00%, 90.75%); opacity: 0.82" title="0.024">d</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">b</span><span style="background-color: hsl(120, 100.00%, 95.45%); opacity: 0.81" title="0.009">i</span><span style="background-color: hsl(120, 100.00%, 94.52%); opacity: 0.81" title="0.012">r</span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.003">h </span><span style="opacity: 0.80">h</span><span style="background-color: hsl(120, 100.00%, 97.36%); opacity: 0.80" title="0.004">ur</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(0, 100.00%, 90.40%); opacity: 0.83" title="-0.026"> </span><span style="background-color: hsl(0, 100.00%, 94.71%); opacity: 0.81" title="-0.011">l</span><span style="background-color: hsl(120, 100.00%, 87.59%); opacity: 0.84" title="0.037">e</span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.010">s</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">s.</span>
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
and a white-box classifier. There is a tradeoff between genereality and
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
    medication that can do anything about them except relieve the pain.
    
    Either they pass, or they have to be broken up with sound, or they have
    to be extracted surgically.
    
    When I wasin, the X-ray tech happened to mention that she'd had kidney
    stones and children, and the childbirth hurt less.


.. code:: ipython3

    te = TextExplainer(char_based=True, sampler=sampler, random_state=42)
    te.fit(doc, pipe_char.predict_proba)
    print(te.metrics_)
    te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)


.. parsed-literal::

    {'score': 1.0, 'mean_KL_divergence': 0.68233124580173965}




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
    
        
        (probability <b>0.967</b>, score <b>3.663</b>)
    
    top features
            </p>
        
        <table class="eli5-weights"
               style="border-collapse: collapse; border: none; margin-top: 0em; margin-bottom: 2em;">
            <thead>
            <tr style="border: none;">
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
                <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            </tr>
            </thead>
            <tbody>
            
                <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +3.653
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
            
                <tr style="background-color: hsl(120, 100.00%, 99.67%); border: none;">
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
            <span style="background-color: hsl(0, 100.00%, 88.61%); opacity: 0.83" title="-0.059">a</span><span style="background-color: hsl(0, 100.00%, 88.77%); opacity: 0.83" title="-0.058">s</span><span style="background-color: hsl(0, 100.00%, 70.35%); opacity: 0.93" title="-0.233"> </span><span style="background-color: hsl(0, 100.00%, 60.25%); opacity: 1.00" title="-0.353">i</span><span style="background-color: hsl(0, 100.00%, 70.38%); opacity: 0.93" title="-0.232"> </span><span style="background-color: hsl(120, 100.00%, 66.71%); opacity: 0.95" title="0.274">rec</span><span style="background-color: hsl(120, 100.00%, 75.16%); opacity: 0.90" title="0.181">a</span><span style="background-color: hsl(0, 100.00%, 91.89%); opacity: 0.82" title="-0.036">l</span><span style="background-color: hsl(0, 100.00%, 84.19%); opacity: 0.85" title="-0.095">l</span><span style="background-color: hsl(0, 100.00%, 85.76%); opacity: 0.85" title="-0.082"> </span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.015">f</span><span style="background-color: hsl(120, 100.00%, 77.98%); opacity: 0.89" title="0.152">ro</span><span style="background-color: hsl(120, 100.00%, 78.24%); opacity: 0.88" title="0.149">m</span><span style="background-color: hsl(120, 100.00%, 89.34%); opacity: 0.83" title="0.054"> </span><span style="background-color: hsl(120, 100.00%, 78.30%); opacity: 0.88" title="0.149">m</span><span style="background-color: hsl(120, 100.00%, 72.20%); opacity: 0.92" title="0.212">y </span><span style="background-color: hsl(120, 100.00%, 81.74%); opacity: 0.87" title="0.116">b</span><span style="background-color: hsl(120, 100.00%, 76.28%); opacity: 0.89" title="0.169">o</span><span style="background-color: hsl(120, 100.00%, 76.65%); opacity: 0.89" title="0.165">u</span><span style="background-color: hsl(120, 100.00%, 70.24%); opacity: 0.93" title="0.234">t</span><span style="background-color: hsl(120, 100.00%, 65.18%); opacity: 0.96" title="0.292"> </span><span style="background-color: hsl(120, 100.00%, 82.24%); opacity: 0.86" title="0.112">w</span><span style="background-color: hsl(0, 100.00%, 84.73%); opacity: 0.85" title="-0.090">it</span><span style="background-color: hsl(0, 100.00%, 82.54%); opacity: 0.86" title="-0.109">h</span><span style="background-color: hsl(0, 100.00%, 84.63%); opacity: 0.85" title="-0.091"> </span><span style="background-color: hsl(0, 100.00%, 80.02%); opacity: 0.87" title="-0.132">k</span><span style="background-color: hsl(0, 100.00%, 77.63%); opacity: 0.89" title="-0.155">i</span><span style="background-color: hsl(0, 100.00%, 80.94%); opacity: 0.87" title="-0.124">d</span><span style="background-color: hsl(0, 100.00%, 93.67%); opacity: 0.81" title="-0.026">n</span><span style="background-color: hsl(120, 100.00%, 96.83%); opacity: 0.81" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 86.62%); opacity: 0.84" title="0.075">y</span><span style="background-color: hsl(120, 100.00%, 87.41%); opacity: 0.84" title="0.068"> </span><span style="opacity: 0.80">sto</span><span style="background-color: hsl(0, 100.00%, 90.35%); opacity: 0.83" title="-0.047">n</span><span style="background-color: hsl(0, 100.00%, 81.10%); opacity: 0.87" title="-0.122">e</span><span style="background-color: hsl(0, 100.00%, 76.69%); opacity: 0.89" title="-0.165">s,</span><span style="background-color: hsl(0, 100.00%, 89.59%); opacity: 0.83" title="-0.052"> </span><span style="background-color: hsl(0, 100.00%, 95.18%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(0, 100.00%, 93.77%); opacity: 0.81" title="-0.025">h</span><span style="background-color: hsl(0, 100.00%, 76.16%); opacity: 0.90" title="-0.170">e</span><span style="background-color: hsl(0, 100.00%, 70.37%); opacity: 0.93" title="-0.232">r</span><span style="background-color: hsl(0, 100.00%, 74.77%); opacity: 0.90" title="-0.185">e</span><span style="background-color: hsl(0, 100.00%, 81.25%); opacity: 0.87" title="-0.121"> i</span><span style="background-color: hsl(0, 100.00%, 97.47%); opacity: 0.80" title="-0.007">s</span><span style="opacity: 0.80">n'</span><span style="background-color: hsl(120, 100.00%, 83.21%); opacity: 0.86" title="0.103">t </span><span style="background-color: hsl(120, 100.00%, 93.20%); opacity: 0.82" title="0.028">a</span><span style="background-color: hsl(0, 100.00%, 78.71%); opacity: 0.88" title="-0.145">n</span><span style="background-color: hsl(0, 100.00%, 88.60%); opacity: 0.83" title="-0.059">y</span><span style="background-color: hsl(120, 100.00%, 72.95%); opacity: 0.91" title="0.204">
    </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.357">m</span><span style="background-color: hsl(120, 100.00%, 73.21%); opacity: 0.91" title="0.201">e</span><span style="background-color: hsl(120, 100.00%, 67.05%); opacity: 0.95" title="0.270">d</span><span style="background-color: hsl(120, 100.00%, 68.01%); opacity: 0.95" title="0.259">i</span><span style="background-color: hsl(120, 100.00%, 74.03%); opacity: 0.91" title="0.192">c</span><span style="background-color: hsl(120, 100.00%, 87.82%); opacity: 0.84" title="0.065">a</span><span style="background-color: hsl(120, 100.00%, 79.95%); opacity: 0.87" title="0.133">t</span><span style="background-color: hsl(120, 100.00%, 85.93%); opacity: 0.84" title="0.080">i</span><span style="background-color: hsl(120, 100.00%, 98.04%); opacity: 0.80" title="0.005">o</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(0, 100.00%, 95.40%); opacity: 0.81" title="-0.016"> t</span><span style="background-color: hsl(0, 100.00%, 95.64%); opacity: 0.81" title="-0.015">h</span><span style="background-color: hsl(0, 100.00%, 84.29%); opacity: 0.85" title="-0.094">a</span><span style="background-color: hsl(120, 100.00%, 96.86%); opacity: 0.81" title="0.009">t </span><span style="background-color: hsl(0, 100.00%, 84.29%); opacity: 0.85" title="-0.094">c</span><span style="background-color: hsl(0, 100.00%, 97.44%); opacity: 0.80" title="-0.007">a</span><span style="opacity: 0.80">n d</span><span style="background-color: hsl(120, 100.00%, 85.69%); opacity: 0.85" title="0.082">o</span><span style="background-color: hsl(120, 100.00%, 85.64%); opacity: 0.85" title="0.083"> a</span><span style="background-color: hsl(0, 100.00%, 83.74%); opacity: 0.86" title="-0.099">n</span><span style="background-color: hsl(0, 100.00%, 74.12%); opacity: 0.91" title="-0.191">y</span><span style="background-color: hsl(0, 100.00%, 90.13%); opacity: 0.83" title="-0.048">t</span><span style="background-color: hsl(0, 100.00%, 93.87%); opacity: 0.81" title="-0.024">h</span><span style="opacity: 0.80">ing a</span><span style="background-color: hsl(120, 100.00%, 97.27%); opacity: 0.80" title="0.008">b</span><span style="background-color: hsl(120, 100.00%, 83.83%); opacity: 0.85" title="0.098">o</span><span style="background-color: hsl(120, 100.00%, 78.10%); opacity: 0.88" title="0.151">u</span><span style="background-color: hsl(120, 100.00%, 75.15%); opacity: 0.90" title="0.181">t</span><span style="background-color: hsl(120, 100.00%, 76.84%); opacity: 0.89" title="0.163"> </span><span style="background-color: hsl(0, 100.00%, 95.18%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(120, 100.00%, 92.22%); opacity: 0.82" title="0.034">he</span><span style="background-color: hsl(120, 100.00%, 94.25%); opacity: 0.81" title="0.022">m</span><span style="background-color: hsl(120, 100.00%, 83.57%); opacity: 0.86" title="0.100"> </span><span style="background-color: hsl(120, 100.00%, 84.44%); opacity: 0.85" title="0.093">e</span><span style="background-color: hsl(120, 100.00%, 79.16%); opacity: 0.88" title="0.140">x</span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.099">c</span><span style="background-color: hsl(0, 100.00%, 97.28%); opacity: 0.80" title="-0.008">e</span><span style="background-color: hsl(0, 100.00%, 89.54%); opacity: 0.83" title="-0.053">p</span><span style="background-color: hsl(120, 100.00%, 89.79%); opacity: 0.83" title="0.051">t</span><span style="background-color: hsl(120, 100.00%, 91.36%); opacity: 0.82" title="0.040"> </span><span style="background-color: hsl(0, 100.00%, 90.83%); opacity: 0.82" title="-0.043">r</span><span style="background-color: hsl(0, 100.00%, 81.48%); opacity: 0.87" title="-0.119">e</span><span style="background-color: hsl(0, 100.00%, 83.95%); opacity: 0.85" title="-0.097">l</span><span style="background-color: hsl(0, 100.00%, 82.22%); opacity: 0.86" title="-0.112">i</span><span style="background-color: hsl(0, 100.00%, 80.56%); opacity: 0.87" title="-0.127">e</span><span style="background-color: hsl(0, 100.00%, 83.21%); opacity: 0.86" title="-0.103">v</span><span style="background-color: hsl(0, 100.00%, 86.40%); opacity: 0.84" title="-0.076">e</span><span style="background-color: hsl(0, 100.00%, 81.02%); opacity: 0.87" title="-0.123"> </span><span style="background-color: hsl(0, 100.00%, 76.57%); opacity: 0.89" title="-0.166">t</span><span style="background-color: hsl(0, 100.00%, 81.94%); opacity: 0.86" title="-0.114">h</span><span style="background-color: hsl(0, 100.00%, 79.43%); opacity: 0.88" title="-0.138">e</span><span style="background-color: hsl(0, 100.00%, 87.48%); opacity: 0.84" title="-0.068"> </span><span style="background-color: hsl(120, 100.00%, 92.88%); opacity: 0.82" title="0.030">p</span><span style="background-color: hsl(120, 100.00%, 66.00%); opacity: 0.96" title="0.283">a</span><span style="background-color: hsl(120, 100.00%, 69.63%); opacity: 0.93" title="0.241">in</span><span style="background-color: hsl(0, 100.00%, 80.53%); opacity: 0.87" title="-0.127">.</span><span style="background-color: hsl(0, 100.00%, 80.48%); opacity: 0.87" title="-0.128"> </span><span style="background-color: hsl(0, 100.00%, 86.05%); opacity: 0.84" title="-0.079">e</span><span style="background-color: hsl(0, 100.00%, 77.43%); opacity: 0.89" title="-0.157">i</span><span style="background-color: hsl(0, 100.00%, 74.02%); opacity: 0.91" title="-0.192">t</span><span style="background-color: hsl(0, 100.00%, 91.17%); opacity: 0.82" title="-0.041">h</span><span style="background-color: hsl(0, 100.00%, 87.01%); opacity: 0.84" title="-0.072">e</span><span style="background-color: hsl(0, 100.00%, 83.63%); opacity: 0.86" title="-0.100">r</span><span style="background-color: hsl(0, 100.00%, 89.45%); opacity: 0.83" title="-0.053"> </span><span style="background-color: hsl(0, 100.00%, 95.18%); opacity: 0.81" title="-0.017">t</span><span style="background-color: hsl(120, 100.00%, 93.45%); opacity: 0.82" title="0.027">h</span><span style="background-color: hsl(120, 100.00%, 91.90%); opacity: 0.82" title="0.036">e</span><span style="background-color: hsl(120, 100.00%, 86.62%); opacity: 0.84" title="0.075">y</span><span style="background-color: hsl(120, 100.00%, 82.39%); opacity: 0.86" title="0.110"> </span><span style="background-color: hsl(120, 100.00%, 92.18%); opacity: 0.82" title="0.035">pa</span><span style="background-color: hsl(0, 100.00%, 93.56%); opacity: 0.81" title="-0.026">s</span><span style="background-color: hsl(0, 100.00%, 87.33%); opacity: 0.84" title="-0.069">s,</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.40%); opacity: 0.82" title="-0.040">or</span><span style="background-color: hsl(0, 100.00%, 88.92%); opacity: 0.83" title="-0.057"> t</span><span style="background-color: hsl(0, 100.00%, 93.17%); opacity: 0.82" title="-0.029">h</span><span style="background-color: hsl(120, 100.00%, 94.55%); opacity: 0.81" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(0, 100.00%, 98.30%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 86.91%); opacity: 0.84" title="-0.072">h</span><span style="background-color: hsl(0, 100.00%, 88.98%); opacity: 0.83" title="-0.057">a</span><span style="background-color: hsl(120, 100.00%, 97.79%); opacity: 0.80" title="0.006">v</span><span style="background-color: hsl(120, 100.00%, 90.05%); opacity: 0.83" title="0.049">e</span><span style="background-color: hsl(120, 100.00%, 83.26%); opacity: 0.86" title="0.103"> t</span><span style="background-color: hsl(120, 100.00%, 79.43%); opacity: 0.88" title="0.138">o</span><span style="background-color: hsl(120, 100.00%, 93.39%); opacity: 0.82" title="0.027"> </span><span style="background-color: hsl(120, 100.00%, 91.72%); opacity: 0.82" title="0.038">b</span><span style="background-color: hsl(120, 100.00%, 96.65%); opacity: 0.81" title="0.010">e b</span><span style="background-color: hsl(0, 100.00%, 98.74%); opacity: 0.80" title="-0.003">r</span><span style="background-color: hsl(0, 100.00%, 88.24%); opacity: 0.83" title="-0.062">o</span><span style="background-color: hsl(0, 100.00%, 77.16%); opacity: 0.89" title="-0.160">k</span><span style="background-color: hsl(0, 100.00%, 80.71%); opacity: 0.87" title="-0.126">e</span><span style="background-color: hsl(0, 100.00%, 78.73%); opacity: 0.88" title="-0.145">n</span><span style="background-color: hsl(120, 100.00%, 83.96%); opacity: 0.85" title="0.097"> </span><span style="background-color: hsl(0, 100.00%, 96.99%); opacity: 0.80" title="-0.009">u</span><span style="background-color: hsl(0, 100.00%, 81.16%); opacity: 0.87" title="-0.122">p</span><span style="background-color: hsl(0, 100.00%, 81.80%); opacity: 0.86" title="-0.116"> </span><span style="background-color: hsl(0, 100.00%, 83.93%); opacity: 0.85" title="-0.097">w</span><span style="background-color: hsl(0, 100.00%, 78.80%); opacity: 0.88" title="-0.144">i</span><span style="background-color: hsl(0, 100.00%, 83.01%); opacity: 0.86" title="-0.105">t</span><span style="background-color: hsl(0, 100.00%, 91.47%); opacity: 0.82" title="-0.039">h</span><span style="background-color: hsl(0, 100.00%, 94.48%); opacity: 0.81" title="-0.021"> </span><span style="opacity: 0.80">s</span><span style="background-color: hsl(120, 100.00%, 86.78%); opacity: 0.84" title="0.073">o</span><span style="background-color: hsl(120, 100.00%, 94.10%); opacity: 0.81" title="0.023">u</span><span style="background-color: hsl(0, 100.00%, 82.24%); opacity: 0.86" title="-0.112">n</span><span style="background-color: hsl(0, 100.00%, 85.43%); opacity: 0.85" title="-0.084">d, </span><span style="background-color: hsl(0, 100.00%, 86.31%); opacity: 0.84" title="-0.077">o</span><span style="background-color: hsl(0, 100.00%, 91.40%); opacity: 0.82" title="-0.040">r</span><span style="background-color: hsl(0, 100.00%, 88.92%); opacity: 0.83" title="-0.057"> t</span><span style="background-color: hsl(0, 100.00%, 93.17%); opacity: 0.82" title="-0.029">h</span><span style="background-color: hsl(120, 100.00%, 94.55%); opacity: 0.81" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(0, 100.00%, 98.30%); opacity: 0.80" title="-0.004"> </span><span style="background-color: hsl(0, 100.00%, 84.69%); opacity: 0.85" title="-0.090">h</span><span style="background-color: hsl(0, 100.00%, 86.61%); opacity: 0.84" title="-0.075">a</span><span style="background-color: hsl(120, 100.00%, 90.84%); opacity: 0.82" title="0.043">v</span><span style="background-color: hsl(120, 100.00%, 87.43%); opacity: 0.84" title="0.068">e
    </span><span style="background-color: hsl(120, 100.00%, 85.18%); opacity: 0.85" title="0.086">t</span><span style="opacity: 0.80">o be</span><span style="background-color: hsl(120, 100.00%, 91.16%); opacity: 0.82" title="0.041"> </span><span style="background-color: hsl(120, 100.00%, 95.34%); opacity: 0.81" title="0.017">e</span><span style="background-color: hsl(0, 100.00%, 97.93%); opacity: 0.80" title="-0.005">x</span><span style="background-color: hsl(0, 100.00%, 90.32%); opacity: 0.83" title="-0.047">t</span><span style="background-color: hsl(120, 100.00%, 81.08%); opacity: 0.87" title="0.122">r</span><span style="background-color: hsl(120, 100.00%, 80.51%); opacity: 0.87" title="0.128">a</span><span style="background-color: hsl(120, 100.00%, 68.13%); opacity: 0.94" title="0.258">ct</span><span style="background-color: hsl(120, 100.00%, 89.30%); opacity: 0.83" title="0.054">e</span><span style="background-color: hsl(0, 100.00%, 80.76%); opacity: 0.87" title="-0.125">d</span><span style="background-color: hsl(0, 100.00%, 79.49%); opacity: 0.88" title="-0.137"> </span><span style="background-color: hsl(0, 100.00%, 85.93%); opacity: 0.84" title="-0.080">s</span><span style="background-color: hsl(0, 100.00%, 88.32%); opacity: 0.83" title="-0.061">u</span><span style="background-color: hsl(0, 100.00%, 90.95%); opacity: 0.82" title="-0.043">r</span><span style="background-color: hsl(120, 100.00%, 84.10%); opacity: 0.85" title="0.095">g</span><span style="background-color: hsl(120, 100.00%, 81.25%); opacity: 0.87" title="0.121">i</span><span style="background-color: hsl(120, 100.00%, 93.71%); opacity: 0.81" title="0.025">c</span><span style="background-color: hsl(120, 100.00%, 94.13%); opacity: 0.81" title="0.023">a</span><span style="background-color: hsl(120, 100.00%, 90.56%); opacity: 0.83" title="0.045">lly</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.001">.</span><span style="background-color: hsl(120, 100.00%, 84.71%); opacity: 0.85" title="0.090"> </span><span style="background-color: hsl(120, 100.00%, 69.33%); opacity: 0.94" title="0.244">w</span><span style="background-color: hsl(120, 100.00%, 71.30%); opacity: 0.92" title="0.222">h</span><span style="background-color: hsl(120, 100.00%, 69.11%); opacity: 0.94" title="0.246">e</span><span style="background-color: hsl(120, 100.00%, 82.47%); opacity: 0.86" title="0.110">n</span><span style="background-color: hsl(0, 100.00%, 84.26%); opacity: 0.85" title="-0.094"> </span><span style="background-color: hsl(0, 100.00%, 68.65%); opacity: 0.94" title="-0.252">i</span><span style="background-color: hsl(0, 100.00%, 82.00%); opacity: 0.86" title="-0.114"> </span><span style="background-color: hsl(120, 100.00%, 74.52%); opacity: 0.90" title="0.187">w</span><span style="background-color: hsl(120, 100.00%, 80.43%); opacity: 0.87" title="0.128">a</span><span style="background-color: hsl(120, 100.00%, 74.79%); opacity: 0.90" title="0.184">s</span><span style="background-color: hsl(120, 100.00%, 84.21%); opacity: 0.85" title="0.095"> </span><span style="background-color: hsl(0, 100.00%, 85.04%); opacity: 0.85" title="-0.087">i</span><span style="background-color: hsl(0, 100.00%, 86.89%); opacity: 0.84" title="-0.072">n,</span><span style="background-color: hsl(0, 100.00%, 81.26%); opacity: 0.87" title="-0.121"> t</span><span style="background-color: hsl(120, 100.00%, 92.02%); opacity: 0.82" title="0.036">he</span><span style="background-color: hsl(120, 100.00%, 75.95%); opacity: 0.90" title="0.172"> </span><span style="background-color: hsl(120, 100.00%, 73.00%); opacity: 0.91" title="0.203">x</span><span style="background-color: hsl(120, 100.00%, 82.18%); opacity: 0.86" title="0.112">-</span><span style="background-color: hsl(0, 100.00%, 87.51%); opacity: 0.84" title="-0.068">ra</span><span style="background-color: hsl(0, 100.00%, 98.73%); opacity: 0.80" title="-0.003">y</span><span style="background-color: hsl(120, 100.00%, 94.81%); opacity: 0.81" title="0.019"> </span><span style="background-color: hsl(0, 100.00%, 90.50%); opacity: 0.83" title="-0.046">t</span><span style="opacity: 0.80">e</span><span style="background-color: hsl(0, 100.00%, 88.78%); opacity: 0.83" title="-0.058">ch</span><span style="opacity: 0.80"> happen</span><span style="background-color: hsl(0, 100.00%, 87.62%); opacity: 0.84" title="-0.067">ed</span><span style="background-color: hsl(0, 100.00%, 91.32%); opacity: 0.82" title="-0.040"> </span><span style="background-color: hsl(120, 100.00%, 94.44%); opacity: 0.81" title="0.021">t</span><span style="background-color: hsl(120, 100.00%, 91.73%); opacity: 0.82" title="0.038">o</span><span style="opacity: 0.80"> men</span><span style="background-color: hsl(120, 100.00%, 87.50%); opacity: 0.84" title="0.068">ti</span><span style="opacity: 0.80">on</span><span style="background-color: hsl(0, 100.00%, 95.40%); opacity: 0.81" title="-0.016"> t</span><span style="opacity: 0.80">h</span><span style="background-color: hsl(120, 100.00%, 97.38%); opacity: 0.80" title="0.007">a</span><span style="background-color: hsl(120, 100.00%, 82.39%); opacity: 0.86" title="0.110">t </span><span style="background-color: hsl(120, 100.00%, 85.61%); opacity: 0.85" title="0.083">s</span><span style="background-color: hsl(120, 100.00%, 78.70%); opacity: 0.88" title="0.145">h</span><span style="background-color: hsl(120, 100.00%, 83.30%); opacity: 0.86" title="0.102">e</span><span style="background-color: hsl(120, 100.00%, 83.50%); opacity: 0.86" title="0.101">'</span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.026">d</span><span style="background-color: hsl(120, 100.00%, 90.79%); opacity: 0.82" title="0.044"> </span><span style="background-color: hsl(120, 100.00%, 81.36%); opacity: 0.87" title="0.120">h</span><span style="background-color: hsl(120, 100.00%, 84.33%); opacity: 0.85" title="0.094">a</span><span style="background-color: hsl(120, 100.00%, 79.84%); opacity: 0.88" title="0.134">d </span><span style="background-color: hsl(120, 100.00%, 93.07%); opacity: 0.82" title="0.029">k</span><span style="background-color: hsl(0, 100.00%, 85.28%); opacity: 0.85" title="-0.086">id</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.003">n</span><span style="background-color: hsl(120, 100.00%, 90.01%); opacity: 0.83" title="0.049">ey</span><span style="background-color: hsl(120, 100.00%, 88.57%); opacity: 0.83" title="0.060">
    s</span><span style="background-color: hsl(120, 100.00%, 92.83%); opacity: 0.82" title="0.031">t</span><span style="opacity: 0.80">o</span><span style="background-color: hsl(120, 100.00%, 93.55%); opacity: 0.81" title="0.026">n</span><span style="background-color: hsl(120, 100.00%, 83.34%); opacity: 0.86" title="0.102">e</span><span style="background-color: hsl(120, 100.00%, 73.45%); opacity: 0.91" title="0.199">s</span><span style="background-color: hsl(120, 100.00%, 75.28%); opacity: 0.90" title="0.179"> </span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.114">a</span><span style="background-color: hsl(120, 100.00%, 91.52%); opacity: 0.82" title="0.039">n</span><span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.021">d</span><span style="background-color: hsl(0, 100.00%, 88.08%); opacity: 0.84" title="-0.063"> </span><span style="background-color: hsl(0, 100.00%, 83.33%); opacity: 0.86" title="-0.102">ch</span><span style="background-color: hsl(120, 100.00%, 77.04%); opacity: 0.89" title="0.161">i</span><span style="background-color: hsl(120, 100.00%, 77.52%); opacity: 0.89" title="0.157">l</span><span style="background-color: hsl(120, 100.00%, 85.22%); opacity: 0.85" title="0.086">d</span><span style="background-color: hsl(0, 100.00%, 97.80%); opacity: 0.80" title="-0.006">r</span><span style="background-color: hsl(120, 100.00%, 98.18%); opacity: 0.80" title="0.004">e</span><span style="background-color: hsl(0, 100.00%, 98.32%); opacity: 0.80" title="-0.004">n,</span><span style="background-color: hsl(0, 100.00%, 94.82%); opacity: 0.81" title="-0.019"> </span><span style="background-color: hsl(0, 100.00%, 89.79%); opacity: 0.83" title="-0.051">an</span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.063">d</span><span style="background-color: hsl(0, 100.00%, 82.32%); opacity: 0.86" title="-0.111"> </span><span style="background-color: hsl(0, 100.00%, 84.52%); opacity: 0.85" title="-0.092">t</span><span style="background-color: hsl(0, 100.00%, 96.05%); opacity: 0.81" title="-0.013">he</span><span style="background-color: hsl(0, 100.00%, 86.97%); opacity: 0.84" title="-0.072"> </span><span style="background-color: hsl(0, 100.00%, 83.70%); opacity: 0.86" title="-0.099">c</span><span style="background-color: hsl(0, 100.00%, 83.33%); opacity: 0.86" title="-0.102">h</span><span style="background-color: hsl(120, 100.00%, 77.86%); opacity: 0.89" title="0.153">il</span><span style="background-color: hsl(120, 100.00%, 84.54%); opacity: 0.85" title="0.092">d</span><span style="opacity: 0.80">birth</span><span style="background-color: hsl(120, 100.00%, 88.28%); opacity: 0.83" title="0.062"> </span><span style="background-color: hsl(120, 100.00%, 79.90%); opacity: 0.87" title="0.133">h</span><span style="background-color: hsl(120, 100.00%, 94.88%); opacity: 0.81" title="0.019">u</span><span style="background-color: hsl(0, 100.00%, 81.94%); opacity: 0.86" title="-0.114">r</span><span style="background-color: hsl(120, 100.00%, 91.75%); opacity: 0.82" title="0.037">t</span><span style="background-color: hsl(120, 100.00%, 84.24%); opacity: 0.85" title="0.094"> </span><span style="background-color: hsl(120, 100.00%, 94.40%); opacity: 0.81" title="0.021">le</span><span style="background-color: hsl(120, 100.00%, 97.98%); opacity: 0.80" title="0.005">ss</span><span style="opacity: 0.80">.</span>
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

    {'score': 0.98259788563297445, 'mean_KL_divergence': 0.039099930047723261}




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
                            0.5456
                            
                        </td>
                        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                            kidney
                        </td>
                    </tr>
                
                    <tr style="background-color: hsl(120, 100.00%, 82.40%); border: none;">
                        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                            0.4544
                            
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
    <text text-anchor="middle" x="397" y="-239.4" font-family="Times,serif" font-size="14.00">gini = 0.1587</text>
    <text text-anchor="middle" x="397" y="-223.4" font-family="Times,serif" font-size="14.00">samples = 100.0%</text>
    <text text-anchor="middle" x="397" y="-207.4" font-family="Times,serif" font-size="14.00">value = [0.01, 0.03, 0.92, 0.04]</text>
    </g>
    <!-- 1 -->
    <g id="node2" class="node"><title>1</title>
    <polygon fill="none" stroke="black" points="390,-164 200,-164 200,-92 390,-92 390,-164"/>
    <text text-anchor="middle" x="295" y="-147.4" font-family="Times,serif" font-size="14.00">pain &lt;= 0.5</text>
    <text text-anchor="middle" x="295" y="-131.4" font-family="Times,serif" font-size="14.00">gini = 0.3891</text>
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
    <text text-anchor="middle" x="500" y="-131.4" font-family="Times,serif" font-size="14.00">gini = 0.0462</text>
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
    <text text-anchor="middle" x="95" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.5252</text>
    <text text-anchor="middle" x="95" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 28.4%</text>
    <text text-anchor="middle" x="95" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.04, 0.14, 0.65, 0.16]</text>
    </g>
    <!-- 1&#45;&gt;2 -->
    <g id="edge4" class="edge"><title>1&#45;&gt;2</title>
    <path fill="none" stroke="black" d="M223.36,-91.8966C202.462,-81.6566 179.763,-70.534 159.346,-60.5294"/>
    <polygon fill="black" stroke="black" points="160.66,-57.2758 150.14,-56.0186 157.58,-63.5617 160.66,-57.2758"/>
    </g>
    <!-- 3 -->
    <g id="node6" class="node"><title>3</title>
    <polygon fill="none" stroke="black" points="384,-56 208,-56 208,-0 384,-0 384,-56"/>
    <text text-anchor="middle" x="296" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.0438</text>
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
    <text text-anchor="middle" x="499" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.1182</text>
    <text text-anchor="middle" x="499" y="-23.4" font-family="Times,serif" font-size="14.00">samples = 22.8%</text>
    <text text-anchor="middle" x="499" y="-7.4" font-family="Times,serif" font-size="14.00">value = [0.01, 0.02, 0.94, 0.03]</text>
    </g>
    <!-- 4&#45;&gt;5 -->
    <g id="edge10" class="edge"><title>4&#45;&gt;5</title>
    <path fill="none" stroke="black" d="M499.642,-91.8966C499.557,-83.6325 499.467,-74.7936 499.382,-66.4314"/>
    <polygon fill="black" stroke="black" points="502.879,-66.1371 499.277,-56.1734 495.88,-66.2086 502.879,-66.1371"/>
    </g>
    <!-- 6 -->
    <g id="node12" class="node"><title>6</title>
    <polygon fill="none" stroke="black" points="782,-56 612,-56 612,-0 782,-0 782,-56"/>
    <text text-anchor="middle" x="697" y="-39.4" font-family="Times,serif" font-size="14.00">gini = 0.0109</text>
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
    0.013 alt.atheism
    0.023 comp.graphics
    0.892 sci.med
    0.071 soc.religion.christian
    
    only 'pain' removed:
    0.002 alt.atheism
    0.004 comp.graphics
    0.980 sci.med
    0.014 soc.religion.christian


As expected, after removing both words probability of ``sci.med``
decreased, though not as much as our simple decision tree predicted (to
0.9 instead of 0.64). Removing ``pain`` provided exactly the same effect
as predicted - probability of ``sci.med`` became ``0.98``.
