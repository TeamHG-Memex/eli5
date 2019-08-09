
.. code:: ipython3

    %load_ext autoreload
    %autoreload 2

Explaining Keras text classifier predictions with Grad-CAM
==========================================================

We will explain text classification predicictions using Grad-CAM. We
will use the IMDB dataset available at keras and the financial dataset,
loading pretrained models.

Grad-CAM shows what's important in input, using a hidden layer and a
target class.

First some imports

.. code:: ipython3

    import os
    
    import numpy as np
    import pandas as pd
    from IPython.display import display, HTML
    
    # you may want to keep logging enabled when doing your own work
    import logging
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR) # disable Tensorflow warnings for this tutorial
    import warnings
    warnings.simplefilter("ignore") # disable Keras warnings for this tutorial
    import keras
    
    import eli5


.. parsed-literal::

    Using TensorFlow backend.


.. code:: ipython3

    # we need this to load some of the local modules
    
    old = os.getcwd()
    os.chdir('..')

Explaining sentiment classification
-----------------------------------

This is common in tutorials. A binary classification task with only one
output. In this case high (1) is positive, low (0) is negative. We will
use the IMDB dataset and a recurrent model, word level tokenization.

Load our model (available in ELI5).

.. code:: ipython3

    model = keras.models.load_model('tests/estimators/keras_sentiment_classifier/keras_sentiment_classifier.h5')
    model.summary()


.. parsed-literal::

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, None, 8)           80000     
    _________________________________________________________________
    masking_1 (Masking)          (None, None, 8)           0         
    _________________________________________________________________
    masking_2 (Masking)          (None, None, 8)           0         
    _________________________________________________________________
    masking_3 (Masking)          (None, None, 8)           0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, None, 128)         37376     
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, None, 64)          41216     
    _________________________________________________________________
    bidirectional_3 (Bidirection (None, 32)                10368     
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 264       
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 9         
    =================================================================
    Total params: 169,233
    Trainable params: 169,233
    Non-trainable params: 0
    _________________________________________________________________


Load some sample data. We have a module that will do preprocessing, etc
for us. Check the relevant package to learn more. For your own models
you will have to do your own preprocessing

.. code:: ipython3

    import tests.estimators.keras_sentiment_classifier.keras_sentiment_classifier \
    as keras_sentiment_classifier

.. code:: ipython3

    (x_train, y_train), (x_test, y_test) = keras_sentiment_classifier.prepare_train_test_dataset()

Confirming the accuracy of the model

.. code:: ipython3

    print(model.metrics_names)
    model.evaluate(x_test, y_test)


.. parsed-literal::

    ['loss', 'acc']
    25000/25000 [==============================] - 95s 4ms/step




.. parsed-literal::

    [0.4319177031707764, 0.81504]



Looks good? Let's go on and check one of the test samples.

.. code:: ipython3

    doc = x_test[0:1]
    print(doc)
    
    tokens = keras_sentiment_classifier.vectorized_to_tokens(doc)
    print(tokens)


.. parsed-literal::

    [[   1  591  202   14   31    6  717   10   10    2    2    5    4  360
         7    4  177 5760  394  354    4  123    9 1035 1035 1035   10   10
        13   92  124   89  488 7944  100   28 1668   14   31   23   27 7479
        29  220  468    8  124   14  286  170    8  157   46    5   27  239
        16  179    2   38   32   25 7944  451  202   14    6  717    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0]]
    [['<START>', 'please', 'give', 'this', 'one', 'a', 'miss', 'br', 'br', '<OOV>', '<OOV>', 'and', 'the', 'rest', 'of', 'the', 'cast', 'rendered', 'terrible', 'performances', 'the', 'show', 'is', 'flat', 'flat', 'flat', 'br', 'br', 'i', "don't", 'know', 'how', 'michael', 'madison', 'could', 'have', 'allowed', 'this', 'one', 'on', 'his', 'plate', 'he', 'almost', 'seemed', 'to', 'know', 'this', "wasn't", 'going', 'to', 'work', 'out', 'and', 'his', 'performance', 'was', 'quite', '<OOV>', 'so', 'all', 'you', 'madison', 'fans', 'give', 'this', 'a', 'miss', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']]


Check the prediction

.. code:: ipython3

    model.predict(doc)




.. parsed-literal::

    array([[0.1622659]], dtype=float32)



As expected, looks pretty low accuracy.

Now let's explain what got us this result with ELI5. We need to pass the
model, the input, and the associated tokens that will be highlighted.

.. code:: ipython3

    eli5.show_prediction(model, doc, tokens=tokens)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> &lt;OOV&gt; &lt;OOV&gt; </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> the rest </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the cast </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> terrible </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> flat flat flat </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> could have allowed this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> almost seemed to </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> this wasn&#x27;t going to </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> out </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> &lt;OOV&gt; so </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Let's try a custom input

.. code:: ipython3

    s = "hello this is great but not so great"
    doc_s, tokens_s = keras_sentiment_classifier.string_to_vectorized(s)
    print(doc_s, tokens_s)


.. parsed-literal::

    [[   1 4825   14    9   87   21   24   38   87]] [['<START>' 'hello' 'this' 'is' 'great' 'but' 'not' 'so' 'great']]


Notice that this model does not require fixed length input. We do not
need to pad this sample.

.. code:: ipython3

    model.predict(doc_s)




.. parsed-literal::

    array([[0.5912496]], dtype=float32)



.. code:: ipython3

    eli5.show_prediction(model, doc_s, tokens=tokens_s)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 94.46%); opacity: 0.81" title="0.001">hello</span><span style="opacity: 0.80"> this </span><span style="background-color: hsl(120, 100.00%, 70.23%); opacity: 0.93" title="0.015">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span><span style="opacity: 0.80"> but not so </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




The ``counterfactual`` and ``relu`` arguments
---------------------------------------------

What did we see in the last section? Grad-CAM shows what makes a class
score "go up". So we are only seeing the "positive" parts.

To "fix" this, we can pass two boolean arguments.

``counterfactual`` shows the "opposite", what makes the score "go down"
(set to ``True`` to enable).

.. code:: ipython3

    eli5.show_prediction(model, doc_s, tokens=tokens_s, relu=False)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 77.80%); opacity: 0.89" title="-0.010">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.46%); opacity: 0.81" title="0.001">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.81%); opacity: 0.84" title="-0.005">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.23%); opacity: 0.93" title="0.015">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.58%); opacity: 0.85" title="-0.005">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.79%); opacity: 0.96" title="-0.018">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.60%); opacity: 0.83" title="-0.004">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




For the test sample

.. code:: ipython3

    eli5.show_prediction(model, doc, tokens=tokens, counterfactual=True)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 90.36%); opacity: 0.83" title="0.003">&lt;START&gt;</span><span style="opacity: 0.80"> please </span><span style="background-color: hsl(120, 100.00%, 94.90%); opacity: 0.81" title="0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> one a miss br br </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.87%); opacity: 0.86" title="0.006">rest</span><span style="opacity: 0.80"> of </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.46%); opacity: 0.82" title="0.002">cast</span><span style="opacity: 0.80"> rendered </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.020">terrible</span><span style="opacity: 0.80"> performances </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> show is </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> br br i </span><span style="background-color: hsl(120, 100.00%, 91.05%); opacity: 0.82" title="0.002">don&#x27;t</span><span style="opacity: 0.80"> know how </span><span style="background-color: hsl(120, 100.00%, 89.51%); opacity: 0.83" title="0.003">michael</span><span style="opacity: 0.80"> madison </span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.51%); opacity: 0.83" title="0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> one on his plate he </span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.39%); opacity: 0.89" title="0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.001">to</span><span style="opacity: 0.80"> know </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.38%); opacity: 0.85" title="0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.001">to</span><span style="opacity: 0.80"> work </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.000">out</span><span style="opacity: 0.80"> and his performance </span><span style="background-color: hsl(120, 100.00%, 93.49%); opacity: 0.81" title="0.001">was</span><span style="opacity: 0.80"> quite </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.97%); opacity: 0.82" title="0.002">so</span><span style="opacity: 0.80"> all you madison fans </span><span style="background-color: hsl(120, 100.00%, 94.90%); opacity: 0.81" title="0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> a miss </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




``relu`` filters out the negative scores and only shows what makes the
predicted score go up (set to ``False`` to disable).

.. code:: ipython3

    eli5.show_prediction(model, doc, tokens=tokens, relu=False)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 90.36%); opacity: 0.83" title="-0.003">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.77%); opacity: 0.83" title="0.003">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.87%); opacity: 0.86" title="-0.006">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.002">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.46%); opacity: 0.82" title="-0.002">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.66%); opacity: 0.91" title="0.011">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.020">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.002">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.12%); opacity: 0.84" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.52%); opacity: 0.83" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.80%); opacity: 0.81" title="0.001">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.05%); opacity: 0.82" title="-0.002">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.92%); opacity: 0.81" title="0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.80%); opacity: 0.80" title="-0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.09%); opacity: 0.82" title="-0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.11%); opacity: 0.82" title="0.002">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.57%); opacity: 0.80" title="-0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.39%); opacity: 0.89" title="-0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.38%); opacity: 0.85" title="-0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.90%); opacity: 0.81" title="-0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.002">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.55%); opacity: 0.80" title="-0.000">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.33%); opacity: 0.89" title="0.009">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.49%); opacity: 0.81" title="-0.001">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.31%); opacity: 0.83" title="0.003">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.97%); opacity: 0.82" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.02%); opacity: 0.86" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.09%); opacity: 0.86" title="0.006">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Green is positive, red is negative, white is neutral. We can see what
made the network decide that is is a negative example.

What happens if we pass both ``counterfactual`` and ``relu``?

.. code:: ipython3

    eli5.show_prediction(model, doc, tokens=tokens, relu=False, counterfactual=True)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 90.36%); opacity: 0.83" title="0.003">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.77%); opacity: 0.83" title="-0.003">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.90%); opacity: 0.81" title="0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.87%); opacity: 0.80" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.74%); opacity: 0.80" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.24%); opacity: 0.81" title="-0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.75%); opacity: 0.80" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.75%); opacity: 0.80" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.82%); opacity: 0.82" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.87%); opacity: 0.86" title="0.006">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.49%); opacity: 0.82" title="-0.002">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.46%); opacity: 0.82" title="0.002">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.66%); opacity: 0.91" title="-0.011">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.020">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.00%); opacity: 0.82" title="-0.002">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.12%); opacity: 0.84" title="-0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.52%); opacity: 0.83" title="-0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.75%); opacity: 0.80" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.75%); opacity: 0.80" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.001">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.05%); opacity: 0.82" title="0.002">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.07%); opacity: 0.83" title="-0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.92%); opacity: 0.81" title="-0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.51%); opacity: 0.83" title="0.003">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.29%); opacity: 0.83" title="-0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.51%); opacity: 0.83" title="0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.87%); opacity: 0.80" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.88%); opacity: 0.82" title="-0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.43%); opacity: 0.80" title="-0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.20%); opacity: 0.81" title="-0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.11%); opacity: 0.82" title="-0.002">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.39%); opacity: 0.89" title="0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.07%); opacity: 0.83" title="-0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.38%); opacity: 0.85" title="0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.73%); opacity: 0.82" title="-0.002">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.000">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.82%); opacity: 0.82" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.43%); opacity: 0.80" title="-0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.33%); opacity: 0.89" title="-0.009">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.49%); opacity: 0.81" title="0.001">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.31%); opacity: 0.83" title="-0.003">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.97%); opacity: 0.82" title="0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.14%); opacity: 0.81" title="-0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.02%); opacity: 0.86" title="-0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.29%); opacity: 0.83" title="-0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.09%); opacity: 0.86" title="-0.006">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.90%); opacity: 0.81" title="0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.74%); opacity: 0.80" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.24%); opacity: 0.81" title="-0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Notice how the colors (green and red) are inverted.

Removing padding with ``pad_value`` and ``padding`` arguments
-------------------------------------------------------------

Often when working with text, each example is padded, whether because
the model expects input with a certain length, or to have all samples be
the same length to put them in a batch.

We can remove padding by specifying two arguments. The first is
``pad_value``, the padding token such as ``<PAD>`` or a numeric value
such as ``0`` for ``doc``. The second argument is ``padding``, which
should be set to either ``pre`` (padding is done before actual text) or
``post`` (padding is done after actual text).

.. code:: ipython3

    eli5.show_prediction(model, doc, tokens=tokens, relu=False, pad_value='<PAD>', padding='post')




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 90.36%); opacity: 0.83" title="-0.003">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.77%); opacity: 0.83" title="0.003">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.87%); opacity: 0.86" title="-0.006">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.002">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.46%); opacity: 0.82" title="-0.002">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.66%); opacity: 0.91" title="0.011">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.020">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.002">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.12%); opacity: 0.84" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.52%); opacity: 0.83" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.80%); opacity: 0.81" title="0.001">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.05%); opacity: 0.82" title="-0.002">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.92%); opacity: 0.81" title="0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.80%); opacity: 0.80" title="-0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.09%); opacity: 0.82" title="-0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.11%); opacity: 0.82" title="0.002">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.57%); opacity: 0.80" title="-0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.39%); opacity: 0.89" title="-0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.38%); opacity: 0.85" title="-0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.90%); opacity: 0.81" title="-0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.002">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.55%); opacity: 0.80" title="-0.000">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.33%); opacity: 0.89" title="0.009">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.49%); opacity: 0.81" title="-0.001">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.31%); opacity: 0.83" title="0.003">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.97%); opacity: 0.82" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.02%); opacity: 0.86" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.09%); opacity: 0.86" title="0.006">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Now the explanation is shorter. This is useful if the input has a lot of
padding.

Choosing a hidden layer to do Grad-CAM on
-----------------------------------------

Grad-CAM requires a hidden layer to do its calculations on. This is
controlled by the ``layer`` argument. We can pass the layer (as an int
index, string name, or a keras Layer instance) explicitly, or let ELI5
attempt to find a good layer to do Grad-CAM on automatically.

.. code:: ipython3

    for layer in model.layers:
        name = layer.name
        print(name)
        if 'masking' not in layer.name:
            e = eli5.show_prediction(model,
                                     doc,
                                     tokens=tokens,
                                     layer=layer,
                                     relu=False, 
                                     pad_value='<PAD>', 
                                     padding='post')
            display(e) # if using in a loop, we need these two explicit IPython calls


.. parsed-literal::

    embedding_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 90.36%); opacity: 0.83" title="-0.003">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.77%); opacity: 0.83" title="0.003">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.87%); opacity: 0.86" title="-0.006">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.002">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.46%); opacity: 0.82" title="-0.002">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.66%); opacity: 0.91" title="0.011">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.020">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.002">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.12%); opacity: 0.84" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.52%); opacity: 0.83" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.80%); opacity: 0.81" title="0.001">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.05%); opacity: 0.82" title="-0.002">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.92%); opacity: 0.81" title="0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.80%); opacity: 0.80" title="-0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.09%); opacity: 0.82" title="-0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.11%); opacity: 0.82" title="0.002">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.57%); opacity: 0.80" title="-0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.39%); opacity: 0.89" title="-0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.38%); opacity: 0.85" title="-0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.90%); opacity: 0.81" title="-0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.002">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.55%); opacity: 0.80" title="-0.000">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.33%); opacity: 0.89" title="0.009">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.49%); opacity: 0.81" title="-0.001">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.31%); opacity: 0.83" title="0.003">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.97%); opacity: 0.82" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.02%); opacity: 0.86" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.09%); opacity: 0.86" title="0.006">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    masking_1
    masking_2
    masking_3
    bidirectional_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 96.21%); opacity: 0.81" title="-0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.23%); opacity: 0.82" title="0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.56%); opacity: 0.81" title="-0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.90%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.19%); opacity: 0.82" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.75%); opacity: 0.82" title="0.000">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.24%); opacity: 0.81" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.82%); opacity: 0.81" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.91%); opacity: 0.82" title="0.000">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.72%); opacity: 0.84" title="-0.000">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.34%); opacity: 0.83" title="0.000">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.87%); opacity: 0.81" title="-0.000">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.36%); opacity: 0.89" title="0.001">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.22%); opacity: 0.96" title="-0.002">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.48%); opacity: 0.81" title="0.000">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.28%); opacity: 0.81" title="0.000">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.08%); opacity: 0.81" title="0.000">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.40%); opacity: 0.83" title="-0.000">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.25%); opacity: 0.82" title="-0.000">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.94%); opacity: 0.82" title="-0.000">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.24%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.29%); opacity: 0.81" title="-0.000">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.94%); opacity: 0.84" title="-0.001">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.50%); opacity: 0.81" title="-0.000">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.22%); opacity: 0.86" title="-0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.98%); opacity: 0.88" title="-0.001">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.33%); opacity: 0.81" title="-0.000">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.62%); opacity: 0.90" title="-0.001">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.59%); opacity: 0.88" title="-0.001">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.53%); opacity: 0.89" title="-0.001">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.43%); opacity: 0.82" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.81%); opacity: 0.81" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.35%); opacity: 0.82" title="0.000">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.63%); opacity: 0.81" title="-0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.83%); opacity: 0.81" title="-0.000">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.23%); opacity: 0.84" title="-0.001">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.80%); opacity: 0.85" title="-0.001">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.58%); opacity: 0.90" title="-0.001">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.000">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.99%); opacity: 0.81" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.32%); opacity: 0.88" title="-0.001">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.28%); opacity: 0.86" title="-0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.36%); opacity: 0.83" title="-0.000">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.38%); opacity: 0.83" title="-0.000">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.05%); opacity: 0.86" title="-0.001">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.68%); opacity: 0.82" title="-0.000">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.19%); opacity: 0.86" title="-0.001">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.14%); opacity: 0.85" title="-0.001">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.74%); opacity: 0.93" title="-0.002">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.77%); opacity: 0.91" title="-0.001">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.68%); opacity: 0.85" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 67.28%); opacity: 0.95" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.63%); opacity: 0.93" title="-0.002">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.10%); opacity: 0.93" title="-0.002">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.07%); opacity: 0.88" title="-0.001">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.003">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.37%); opacity: 0.94" title="-0.002">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.62%); opacity: 0.87" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 71.47%); opacity: 0.92" title="-0.002">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.54%); opacity: 0.93" title="-0.002">miss</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    bidirectional_2



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.78%); opacity: 0.81" title="-0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.92%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.79%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.06%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.64%); opacity: 0.83" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.19%); opacity: 0.83" title="-0.001">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.70%); opacity: 0.89" title="-0.004">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.28%); opacity: 0.89" title="-0.004">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.26%); opacity: 0.83" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.11%); opacity: 0.91" title="-0.005">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 72.53%); opacity: 0.92" title="-0.006">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.52%); opacity: 0.94" title="-0.007">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.62%); opacity: 0.93" title="-0.006">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.26%); opacity: 0.84" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.20%); opacity: 0.93" title="-0.006">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.33%); opacity: 0.94" title="-0.006">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.08%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.30%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.91%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.35%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.07%); opacity: 0.91" title="-0.005">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.81%); opacity: 0.90" title="-0.005">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.35%); opacity: 0.85" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.37%); opacity: 0.89" title="-0.004">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.91%); opacity: 0.89" title="-0.004">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.55%); opacity: 0.84" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.63%); opacity: 0.89" title="-0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.30%); opacity: 0.88" title="-0.004">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.01%); opacity: 0.88" title="-0.004">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.57%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.76%); opacity: 0.83" title="-0.002">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.36%); opacity: 0.86" title="-0.003">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.65%); opacity: 0.83" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.34%); opacity: 0.87" title="-0.003">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.38%); opacity: 0.87" title="-0.003">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.59%); opacity: 0.88" title="-0.004">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.86%); opacity: 0.88" title="-0.004">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.83%); opacity: 0.84" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.72%); opacity: 0.88" title="-0.004">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.55%); opacity: 0.87" title="-0.003">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.32%); opacity: 0.83" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.43%); opacity: 0.87" title="-0.003">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.89%); opacity: 0.87" title="-0.003">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.99%); opacity: 0.84" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.03%); opacity: 0.84" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.29%); opacity: 0.88" title="-0.004">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.41%); opacity: 0.90" title="-0.005">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.69%); opacity: 0.91" title="-0.005">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.69%); opacity: 0.84" title="-0.002">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.50%); opacity: 0.94" title="-0.006">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 67.84%); opacity: 0.95" title="-0.007">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.70%); opacity: 0.96" title="-0.008">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.26%); opacity: 0.89" title="-0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.009">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.31%); opacity: 0.91" title="-0.005">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.62%); opacity: 0.86" title="-0.003">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.33%); opacity: 0.93" title="-0.006">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.84%); opacity: 0.94" title="-0.007">miss</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    bidirectional_3



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.162">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.27%); opacity: 1.00" title="-0.161">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.71%); opacity: 0.90" title="-0.080">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.15%); opacity: 0.85" title="-0.039">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.04%); opacity: 0.90" title="-0.078">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.28%); opacity: 0.89" title="-0.072">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.66%); opacity: 0.88" title="-0.066">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.74%); opacity: 0.84" title="-0.030">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.65%); opacity: 0.83" title="-0.027">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.27%); opacity: 0.83" title="-0.028">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.89%); opacity: 0.82" title="-0.020">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.87%); opacity: 0.82" title="-0.017">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.78%); opacity: 0.80" title="-0.003">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.60%); opacity: 0.80" title="-0.003">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.81%); opacity: 0.81" title="-0.006">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.36%); opacity: 0.80" title="-0.003">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.97%); opacity: 0.82" title="-0.014">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.39%); opacity: 0.84" title="-0.035">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.41%); opacity: 0.88" title="-0.067">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 71.57%); opacity: 0.92" title="-0.100">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.95%); opacity: 0.85" title="-0.044">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 64.37%); opacity: 0.97" title="-0.137">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.04%); opacity: 0.95" title="-0.118">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.98%); opacity: 0.84" title="-0.033">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.88%); opacity: 0.83" title="-0.026">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.44%); opacity: 0.83" title="-0.024">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.21%); opacity: 0.83" title="-0.022">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.11%); opacity: 0.83" title="-0.025">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.49%); opacity: 0.94" title="-0.115">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.36%); opacity: 0.94" title="-0.116">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.93%); opacity: 0.85" title="-0.044">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.01%); opacity: 0.87" title="-0.060">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.07%); opacity: 0.84" title="-0.032">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.99%); opacity: 0.80" title="-0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.19%); opacity: 0.80" title="0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.31%); opacity: 0.82" title="0.015">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.57%); opacity: 0.83" title="0.027">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.12%); opacity: 0.81" title="0.008">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.004">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.96%); opacity: 0.86" title="-0.048">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.95%); opacity: 0.85" title="-0.044">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 66.28%); opacity: 0.96" title="-0.127">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 64.29%); opacity: 0.97" title="-0.138">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 62.35%); opacity: 0.98" title="-0.149">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.45%); opacity: 1.00" title="-0.160">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.51%); opacity: 0.91" title="-0.085">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.04%); opacity: 0.90" title="-0.083">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.04%); opacity: 0.85" title="-0.040">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 61.59%); opacity: 0.99" title="-0.153">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 62.68%); opacity: 0.98" title="-0.147">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.23%); opacity: 0.88" title="-0.064">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.50%); opacity: 0.93" title="-0.105">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.03%); opacity: 0.90" title="-0.083">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.65%); opacity: 0.84" title="-0.030">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.44%); opacity: 0.83" title="-0.024">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.02%); opacity: 0.85" title="-0.040">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.35%); opacity: 0.84" title="-0.031">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.88%); opacity: 0.83" title="-0.023">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.11%); opacity: 0.81" title="-0.006">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.87%); opacity: 0.82" title="-0.014">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.23%); opacity: 0.81" title="-0.010">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.75%); opacity: 0.81" title="-0.007">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.04%); opacity: 0.80" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.38%); opacity: 0.80" title="-0.003">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.78%); opacity: 0.80" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.53%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.63%); opacity: 0.80" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.88%); opacity: 0.80" title="-0.000">miss</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    dense_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.963">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 61.51%); opacity: 0.99" title="-0.911">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.25%); opacity: 0.89" title="-0.430">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.59%); opacity: 0.84" title="-0.202">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.20%); opacity: 0.88" title="-0.378">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.20%); opacity: 0.87" title="-0.352">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.23%); opacity: 0.87" title="-0.327">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.09%); opacity: 0.83" title="-0.150">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.75%); opacity: 0.83" title="-0.138">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.30%); opacity: 0.83" title="-0.166">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.16%); opacity: 0.83" title="-0.149">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.79%); opacity: 0.84" title="-0.198">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.98%); opacity: 0.82" title="-0.115">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.63%); opacity: 0.86" title="-0.292">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.84%); opacity: 0.85" title="-0.241">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.06%); opacity: 0.81" title="-0.063">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.75%); opacity: 0.83" title="-0.138">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.62%); opacity: 0.82" title="-0.086">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.10%); opacity: 0.81" title="-0.035">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.82%); opacity: 0.81" title="-0.026">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.58%); opacity: 0.80" title="-0.008">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.08%); opacity: 0.80" title="-0.023">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.21%); opacity: 0.80" title="-0.021">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.77%); opacity: 0.80" title="-0.007">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.84%); opacity: 0.80" title="-0.006">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.90%); opacity: 0.80" title="-0.006">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.16%); opacity: 0.80" title="-0.004">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.22%); opacity: 0.80" title="-0.003">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.10%); opacity: 0.80" title="-0.012">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.26%); opacity: 0.80" title="-0.011">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.03%); opacity: 0.80" title="-0.005">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.61%); opacity: 0.80" title="-0.008">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.80%); opacity: 0.80" title="-0.006">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.39%); opacity: 0.80" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.003">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.48%); opacity: 0.80" title="-0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.82%); opacity: 0.80" title="-0.000">allowed</span><span style="opacity: 0.80"> this one on his plate he almost seemed to know this wasn&#x27;t going to work out and his performance was quite &lt;OOV&gt; so all you madison fans give this a miss</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    dense_2



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">miss</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



If you don't get good explanations from ELI5 out of the box, it may be
worth looking into this parameter. We advice to pick layers that contain
"spatial or temporal" information, i.e. NOT dense/fully-connected or
merge layers.

Notice that when explaining the final dense layer node (there is only 1
output), we get an "all green" explanation. You need to hover over the
explanation to see the actual value. It seems off because there are no
"negative" values here and the colouring is not gradual.

Explaining multiple classes
---------------------------

A multi-class model trained on the finanial dataset. Character-level
tokenization. Convolutional network.

.. code:: ipython3

    # multiclass model (*target, layer - conv/others, diff. types of expls, padding and its effect)

.. code:: ipython3

    model2 = keras.models.load_model('tests/estimators/keras_multiclass_text_classifier/keras_multiclass_text_classifier.h5')
    model2.summary()


.. parsed-literal::

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 3193, 8)           816       
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 3179, 128)         15488     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 3179, 128)         0         
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 1589, 128)         0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 1580, 128)         163968    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 1580, 128)         0         
    _________________________________________________________________
    average_pooling1d_1 (Average (None, 790, 128)          0         
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 786, 128)          82048     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 786, 128)          0         
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 393, 128)          0         
    _________________________________________________________________
    global_average_pooling1d_1 ( (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 32)                4128      
    _________________________________________________________________
    dense_2 (Dense)              (None, 11)                363       
    =================================================================
    Total params: 266,811
    Trainable params: 266,811
    Non-trainable params: 0
    _________________________________________________________________


.. code:: ipython3

    import tests.estimators.keras_multiclass_text_classifier.keras_multiclass_text_classifier \
    as keras_multiclass_text_classifier

.. code:: ipython3

    (x_train, x_test), (y_train, y_test) = keras_multiclass_text_classifier.prepare_train_test_dataset()

Possible classes

.. code:: ipython3

    keras_multiclass_text_classifier.labels_index




.. parsed-literal::

    {'Debt collection': 0,
     'Consumer Loan': 1,
     'Mortgage': 2,
     'Credit card': 3,
     'Credit reporting': 4,
     'Student loan': 5,
     'Bank account or service': 6,
     'Payday loan': 7,
     'Money transfers': 8,
     'Other financial service': 9,
     'Prepaid card': 10}



Again check the metrics.

.. code:: ipython3

    print(model2.metrics_names)
    model2.evaluate(x_test, y_test)


.. parsed-literal::

    ['loss', 'acc']
    500/500 [==============================] - 7s 13ms/step




.. parsed-literal::

    [0.6319513120651246, 0.7999999990463257]



Let's explain one of the test samples

.. code:: ipython3

    doc = x_test[0:1]
    tokens = keras_multiclass_text_classifier.vectorized_to_tokens(doc)
    s = keras_multiclass_text_classifier.tokens_to_string(tokens)
    
    print(len(doc[0]))
    limit = 150
    print(doc[0, :limit])
    print(tokens[0, :limit])
    print(s[0][:limit+800])


.. parsed-literal::

    3193
    [38 15 21  3  7  2 20  8  7  5  7 15  8  5 14  2 11  3  9 25  8 15  3 11
      2 15 14  5  8 16 11  2 11  8 16 17 14  4  5  7  3  6 17 11 14 18  2  4
      6  2 12  5 25  3  2  5  2 14  6  5  7  2 21  8  4 12  2 16  3  2 58  2
     13  3 11 19  8  4  3  2 16 18  2  7  3 25  3  9  2 12  5 25  8  7 22  2
     13  6  7  3  2 24 17 11  8  7  3 11 11  2 21  8  4 12  2  4 12  3 16  2
      6  9  2 12  5 25  8  7 22  2 24  3  3  7  2  7  6  4  8 20  8  3 13  2
      6 20  2 11  5  8]
    ['O' 'c' 'w' 'e' 'n' ' ' 'f' 'i' 'n' 'a' 'n' 'c' 'i' 'a' 'l' ' ' 's' 'e'
     'r' 'v' 'i' 'c' 'e' 's' ' ' 'c' 'l' 'a' 'i' 'm' 's' ' ' 's' 'i' 'm' 'u'
     'l' 't' 'a' 'n' 'e' 'o' 'u' 's' 'l' 'y' ' ' 't' 'o' ' ' 'h' 'a' 'v' 'e'
     ' ' 'a' ' ' 'l' 'o' 'a' 'n' ' ' 'w' 'i' 't' 'h' ' ' 'm' 'e' ' ' '(' ' '
     'd' 'e' 's' 'p' 'i' 't' 'e' ' ' 'm' 'y' ' ' 'n' 'e' 'v' 'e' 'r' ' ' 'h'
     'a' 'v' 'i' 'n' 'g' ' ' 'd' 'o' 'n' 'e' ' ' 'b' 'u' 's' 'i' 'n' 'e' 's'
     's' ' ' 'w' 'i' 't' 'h' ' ' 't' 'h' 'e' 'm' ' ' 'o' 'r' ' ' 'h' 'a' 'v'
     'i' 'n' 'g' ' ' 'b' 'e' 'e' 'n' ' ' 'n' 'o' 't' 'i' 'f' 'i' 'e' 'd' ' '
     'o' 'f' ' ' 's' 'a' 'i']
    Ocwen financial services claims simultaneously to have a loan with me ( despite my never having done business with them or having been notified of said loan ) and to have written off said loan with the IRS in XX/XX/XXXX. Further, they continue to insert themselves in a legal case I have against XXXX XXXX XXXX company regarding my foreclosure. Ocwen has claimed in a legal deposition that they are the current holder of an unsigned original, legally executed Note. XXXX has claimed that the holder of the note is the owner of the property. However, Ocwen appears to have discharged the Note according to the IRS. This Note was discharged previously by IndyMac Mortgage Services in XX/XX/XXXX ( which Ocwen is aware of as the evidence of this was shown to them at a legal deposition ). Ocwen appears to have applied a court ordered use and occupancy payment made out to XXXX to a loan ( number XXXX XXXX which Ocwen pretends to have with me. I have r


Notice that the padding length is quite long. We are also dealing with
character-level tokenization - our tokens are single characters, not
words.

Let's check what the model predicts (to which category the financial
complaint belongs).

.. code:: ipython3

    preds = model2.predict(doc)
    print(preds)
    y = np.argmax(preds)
    print(y)
    keras_multiclass_text_classifier.decode_output(y)


.. parsed-literal::

    [[7.4966592e-03 9.7562626e-08 9.9250317e-01 9.1982411e-12 5.3569739e-08
      4.8417964e-10 9.6964792e-10 4.0114050e-09 5.9291594e-10 3.4063903e-13
      3.9474773e-19]]
    2




.. parsed-literal::

    'Mortgage'



And the ground truth:

.. code:: ipython3

    y_truth = y_test[0]
    print(y_truth)
    keras_multiclass_text_classifier.decode_output(y_truth)


.. parsed-literal::

    [0 0 1 0 0 0 0 0 0 0 0]




.. parsed-literal::

    'Mortgage'



Now let's explain this prediction with ELI5. Enable relu to not see
other classes.

.. code:: ipython3

    eli5.show_prediction(model2, doc, tokens=tokens, pad_value='<PAD>', padding='post')




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n financial </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">rvic</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> claim</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">imultan</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">u</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ly t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> a l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ith m</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.000">(</span><span style="opacity: 0.80"> d</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">it</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> my n</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">v</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r havin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> d</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> bu</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">in</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">ss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ith th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">m </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r havin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> b</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">ee</span><span style="opacity: 0.80">n n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">tifi</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">aid l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an ) and t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ritt</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ff </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">aid l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ith th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> IR</span><span style="background-color: hsl(120, 100.00%, 60.09%); opacity: 1.00" title="0.000">S</span><span style="opacity: 0.80"> in XX/XX/XXXX. Furth</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r, th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ntinu</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> in</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">rt th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">m</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">lv</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> in a l</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">al ca</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> I hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> a</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">ain</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">t XXXX XXXX XXXX c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">m</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">any r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">ardin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> my f</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">cl</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ur</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">. </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n ha</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> claim</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d in a l</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">al d</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">iti</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n that th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> curr</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">nt h</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ld</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f an un</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ri</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">inal, l</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">ally </span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">x</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">cut</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d </span><span style="background-color: hsl(120, 100.00%, 78.58%); opacity: 0.88" title="0.000">N</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">. XXXX ha</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> claim</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d that th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> h</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ld</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">rty. </span><span style="background-color: hsl(120, 100.00%, 81.62%); opacity: 0.87" title="0.000">H</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">v</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r, </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n a</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">pp</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ar</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> di</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">char</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.58%); opacity: 0.88" title="0.000">N</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> acc</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">rdin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> IR</span><span style="background-color: hsl(120, 100.00%, 60.09%); opacity: 1.00" title="0.000">S</span><span style="opacity: 0.80">. Thi</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.58%); opacity: 0.88" title="0.000">N</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> di</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">char</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">vi</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">u</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ly by IndyMac M</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">rt</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.09%); opacity: 1.00" title="0.000">S</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">rvic</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> in XX/XX/XXXX </span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.000">(</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">hich </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> a</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">vid</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">nc</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f thi</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">h</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">n t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">m at a l</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">al d</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">iti</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n ). </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n a</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">pp</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ar</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> a</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">pp</span><span style="opacity: 0.80">li</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d a c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">urt </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">rd</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d u</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ccu</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">ancy </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">aym</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">nt mad</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ut t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> XXXX t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> a l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an </span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.000">(</span><span style="opacity: 0.80"> numb</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r XXXX XXXX </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">hich </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">nd</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ith m</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">. I hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">iv</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">tificati</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n that any l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> tran</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">f</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">rr</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n at any tim</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">, n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">uld thi</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> ha</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">pp</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">inc</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t </span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.000">(</span><span style="opacity: 0.80"> and d</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t claim ) any inv</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">lv</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">m</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">nt until l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> aft</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r a f</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">cl</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ur</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> had alr</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ady tak</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">lac</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">rty. Furth</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r, </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n did n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t </span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.000">(</span><span style="opacity: 0.80"> acc</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">rdin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ir </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">n t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">tim</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ny at d</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">iti</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n ) b</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">m</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> h</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ld</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f any </span><span style="background-color: hsl(120, 100.00%, 78.58%); opacity: 0.88" title="0.000">N</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r L</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an thr</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">u</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">h th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">urcha</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">ss</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f IndyMac M</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">rt</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.09%); opacity: 1.00" title="0.000">S</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">rvic</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> fr</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">m XXXX </span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.000">(</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r any </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r </span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ntity ), rath</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r, th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y claim thi</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> ha</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">pp</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> a r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ult </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f XXXX </span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">ivin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">m an </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ri</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">inal c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">y </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f a l</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">ally </span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">x</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">cut</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d </span><span style="background-color: hsl(120, 100.00%, 78.58%); opacity: 0.88" title="0.000">N</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">h</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> it </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> h</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ld in a vault fr</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">m th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> tim</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f f</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">cl</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ur</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">. Th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> v</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ry fact</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> in di</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">ut</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> in a c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">urt </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f la</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80"> at th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> m</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">m</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">nt and th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> ca</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ndin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> a</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">pp</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">al. 
    In filin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> a tax f</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r claimin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> di</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">char</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> a n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> that th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y did n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t had and had b</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">ee</span><span style="opacity: 0.80">n </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">vi</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">u</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ly di</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">char</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d, </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">uilty </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f tax fraud a</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">ain</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">t th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> IR</span><span style="background-color: hsl(120, 100.00%, 60.09%); opacity: 1.00" title="0.000">S</span><span style="opacity: 0.80">. In claimin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> a l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ith m</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> and/</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r attach</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> my </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">rty, </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">mmittin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> fraud. If </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> claimin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> hav</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> a l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ith m</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">, I </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> n</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">v</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">tifi</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f thi</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> fact a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">quir</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d by f</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ral la</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80"> and I di</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">ut</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> any </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">uch claim. By a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">ss</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">nin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> a </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">aym</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">nt t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> a l</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">an </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">hich d</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> n&#x27;t </span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">xi</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">t, </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">mmittin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> fraud, </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">inc</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">imultan</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">u</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ly claim that th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y di</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">char</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> d</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">bt la</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">t y</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ar and th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> ch</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ck </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t mad</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ut t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n Financial it &#x27;</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">nt </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ub</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">idiary </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">lat</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">m</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">ani</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">hich c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">titut</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> ch</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ck fraud. Furth</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r, </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> vi</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">latin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">urt </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">rd</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r f</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r u</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">ccu</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">ancy, </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">hich th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t </span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ntitl</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">inc</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">art </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> l</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">al acti</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n. Furth</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r, </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> cl</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">arly c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">irin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ith XXXX and i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> c</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">tin</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80"> m</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> m</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">n</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y. Finally, </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">O</span><span style="opacity: 0.80">c</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">n </span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">vi</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">u</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">ly n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">tifi</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ir tax fraud by r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">t</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d l</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">tt</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">r, </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">y ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> alr</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ady a</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.32%); opacity: 0.82" title="0.000">(</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r </span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80">h</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">uld b</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> a</span><span style="background-color: hsl(120, 100.00%, 89.74%); opacity: 0.83" title="0.000">w</span><span style="opacity: 0.80">ar</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> ) that th</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ir b</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">havi</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">r i</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000">s</span><span style="opacity: 0.80"> ina</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">pp</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 94.27%); opacity: 0.81" title="0.000">p</span><span style="opacity: 0.80">riat</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80"> and lik</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">ly ill</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.92%); opacity: 0.82" title="0.000">g</span><span style="opacity: 0.80">al. 
    </span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Our own example

.. code:: ipython3

    s = "the IRS is afterr my car loan"
    doc_s, tokens_s = keras_multiclass_text_classifier.string_to_vectorized(s)
    print(doc_s)
    print(tokens_s[0, :50]) # note that this model requires fixed length input


.. parsed-literal::

    [[ 4 12  3 ...  0  0  0]]
    ['t' 'h' 'e' ' ' 'I' 'R' 'S' ' ' 'i' 's' ' ' 'a' 'f' 't' 'e' 'r' 'r' ' '
     'm' 'y' ' ' 'c' 'a' 'r' ' ' 'l' 'o' 'a' 'n' '<PAD>' '<PAD>' '<PAD>'
     '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
     '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>']


.. code:: ipython3

    preds = model2.predict(doc_s)
    print(preds)
    keras_multiclass_text_classifier.decode_output(preds)


.. parsed-literal::

    [[0.09576575 0.27872923 0.10852851 0.03327851 0.11653358 0.1867436
      0.02678595 0.13854526 0.00900717 0.00178243 0.00429991]]




.. parsed-literal::

    'Consumer Loan'



.. code:: ipython3

    eli5.show_prediction(model2, doc_s, tokens=tokens_s, pad_value='<PAD>', padding='post')
    
    # TODO: would be good to show predicted label




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Choosing a classification target to focus on
--------------------------------------------

.. code:: ipython3

    debt_idx = 0
    loan_idx = 1

.. code:: ipython3

    eli5.show_prediction(model2, doc_s, tokens=tokens_s, pad_value='<PAD>', padding='post', targets=[debt_idx])




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 97.31%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 91.02%); opacity: 0.82" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 93.80%); opacity: 0.81" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.001">I</span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.000">R</span><span style="background-color: hsl(120, 100.00%, 77.35%); opacity: 0.89" title="0.000">S</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 81.77%); opacity: 0.87" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 88.49%); opacity: 0.83" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 93.84%); opacity: 0.81" title="0.000">a</span><span style="opacity: 0.80">f</span><span style="background-color: hsl(120, 100.00%, 97.31%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 93.80%); opacity: 0.81" title="0.000">e</span><span style="opacity: 0.80">rr</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 92.32%); opacity: 0.82" title="0.000">m</span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 78.56%); opacity: 0.88" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 93.84%); opacity: 0.81" title="0.000">a</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 92.12%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 81.00%); opacity: 0.87" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 90.31%); opacity: 0.83" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 93.84%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 81.37%); opacity: 0.87" title="0.000">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Sensible?

How it works - ``explain_prediction`` and ``format_as_html``.
-------------------------------------------------------------

.. code:: ipython3

    # heatmap, tokens, weighted_spans, interpolation_kind, etc.

.. code:: ipython3

    E = eli5.explain_prediction(model2, doc_s, tokens=tokens_s, pad_value='<PAD>', padding='post')

Looking at the ``Explanation`` object

.. code:: ipython3

    repr(E)




.. parsed-literal::

    "Explanation(estimator='sequential_1', description='\\nGrad-CAM visualization for classification tasks; \\noutput is explanation object that contains a heatmap.\\n', error='', method='Grad-CAM', is_regression=False, targets=[TargetExplanation(target=1, feature_weights=None, proba=None, score=0.27872923, weighted_spans=WeightedSpans(docs_weighted_spans=[DocWeightedSpans(document='the IRS is afterr my car loan', spans=[('t', [(0, 1)], 0.0), ('h', [(1, 2)], 0.0), ('e', [(2, 3)], 0.0), (' ', [(3, 4)], 0.0), ('I', [(4, 5)], 0.0), ('R', [(5, 6)], 2.540649802540429e-05), ('S', [(6, 7)], 0.0), (' ', [(7, 8)], 0.0), ('i', [(8, 9)], 0.0), ('s', [(9, 10)], 0.0), (' ', [(10, 11)], 0.0), ('a', [(11, 12)], 0.0), ('f', [(12, 13)], 6.950748502276838e-05), ('t', [(13, 14)], 0.0), ('e', [(14, 15)], 0.0), ('r', [(15, 16)], 0.000238344761442022), ('r', [(16, 17)], 0.000238344761442022), (' ', [(17, 18)], 0.0), ('m', [(18, 19)], 0.0), ('y', [(19, 20)], 0.0), (' ', [(20, 21)], 0.0), ('c', [(21, 22)], 0.0), ('a', [(22, 23)], 0.0), ('r', [(23, 24)], 0.000238344761442022), (' ', [(24, 25)], 0.0), ('l', [(25, 26)], 0.0), ('o', [(26, 27)], 0.0), ('a', [(27, 28)], 0.0), ('n', [(28, 29)], 0.0)], preserve_density=None, vec_name=None)], other=None), heatmap=array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,\n       0.00000000e+00, 2.54064980e-05, 0.00000000e+00, 0.00000000e+00,\n       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,\n       6.95074850e-05, 0.00000000e+00, 0.00000000e+00, 2.38344761e-04,\n       2.38344761e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,\n       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.38344761e-04,\n       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,\n       0.00000000e+00]))], feature_importances=None, decision_tree=None, highlight_spaces=None, transition_features=None, image=None)"



We can get the predicted class and the value for the prediction

.. code:: ipython3

    target = E.targets[0]
    print(target.target, target.score)


.. parsed-literal::

    1 0.27872923


The highlighting for each token is stored in a ``WeightedSpans`` object
(specifically the ``DocWeightedSpans`` object)

.. code:: ipython3

    weighted_spans = target.weighted_spans
    print(weighted_spans)
    
    doc_ws = weighted_spans.docs_weighted_spans[0]
    print(doc_ws)


.. parsed-literal::

    WeightedSpans(docs_weighted_spans=[DocWeightedSpans(document='the IRS is afterr my car loan', spans=[('t', [(0, 1)], 0.0), ('h', [(1, 2)], 0.0), ('e', [(2, 3)], 0.0), (' ', [(3, 4)], 0.0), ('I', [(4, 5)], 0.0), ('R', [(5, 6)], 2.540649802540429e-05), ('S', [(6, 7)], 0.0), (' ', [(7, 8)], 0.0), ('i', [(8, 9)], 0.0), ('s', [(9, 10)], 0.0), (' ', [(10, 11)], 0.0), ('a', [(11, 12)], 0.0), ('f', [(12, 13)], 6.950748502276838e-05), ('t', [(13, 14)], 0.0), ('e', [(14, 15)], 0.0), ('r', [(15, 16)], 0.000238344761442022), ('r', [(16, 17)], 0.000238344761442022), (' ', [(17, 18)], 0.0), ('m', [(18, 19)], 0.0), ('y', [(19, 20)], 0.0), (' ', [(20, 21)], 0.0), ('c', [(21, 22)], 0.0), ('a', [(22, 23)], 0.0), ('r', [(23, 24)], 0.000238344761442022), (' ', [(24, 25)], 0.0), ('l', [(25, 26)], 0.0), ('o', [(26, 27)], 0.0), ('a', [(27, 28)], 0.0), ('n', [(28, 29)], 0.0)], preserve_density=None, vec_name=None)], other=None)
    DocWeightedSpans(document='the IRS is afterr my car loan', spans=[('t', [(0, 1)], 0.0), ('h', [(1, 2)], 0.0), ('e', [(2, 3)], 0.0), (' ', [(3, 4)], 0.0), ('I', [(4, 5)], 0.0), ('R', [(5, 6)], 2.540649802540429e-05), ('S', [(6, 7)], 0.0), (' ', [(7, 8)], 0.0), ('i', [(8, 9)], 0.0), ('s', [(9, 10)], 0.0), (' ', [(10, 11)], 0.0), ('a', [(11, 12)], 0.0), ('f', [(12, 13)], 6.950748502276838e-05), ('t', [(13, 14)], 0.0), ('e', [(14, 15)], 0.0), ('r', [(15, 16)], 0.000238344761442022), ('r', [(16, 17)], 0.000238344761442022), (' ', [(17, 18)], 0.0), ('m', [(18, 19)], 0.0), ('y', [(19, 20)], 0.0), (' ', [(20, 21)], 0.0), ('c', [(21, 22)], 0.0), ('a', [(22, 23)], 0.0), ('r', [(23, 24)], 0.000238344761442022), (' ', [(24, 25)], 0.0), ('l', [(25, 26)], 0.0), ('o', [(26, 27)], 0.0), ('a', [(27, 28)], 0.0), ('n', [(28, 29)], 0.0)], preserve_density=None, vec_name=None)


Observe the ``document`` attribute and ``spans``

.. code:: ipython3

    print(doc_ws.document)
    print(doc_ws.spans)


.. parsed-literal::

    the IRS is afterr my car loan
    [('t', [(0, 1)], 0.0), ('h', [(1, 2)], 0.0), ('e', [(2, 3)], 0.0), (' ', [(3, 4)], 0.0), ('I', [(4, 5)], 0.0), ('R', [(5, 6)], 2.540649802540429e-05), ('S', [(6, 7)], 0.0), (' ', [(7, 8)], 0.0), ('i', [(8, 9)], 0.0), ('s', [(9, 10)], 0.0), (' ', [(10, 11)], 0.0), ('a', [(11, 12)], 0.0), ('f', [(12, 13)], 6.950748502276838e-05), ('t', [(13, 14)], 0.0), ('e', [(14, 15)], 0.0), ('r', [(15, 16)], 0.000238344761442022), ('r', [(16, 17)], 0.000238344761442022), (' ', [(17, 18)], 0.0), ('m', [(18, 19)], 0.0), ('y', [(19, 20)], 0.0), (' ', [(20, 21)], 0.0), ('c', [(21, 22)], 0.0), ('a', [(22, 23)], 0.0), ('r', [(23, 24)], 0.000238344761442022), (' ', [(24, 25)], 0.0), ('l', [(25, 26)], 0.0), ('o', [(26, 27)], 0.0), ('a', [(27, 28)], 0.0), ('n', [(28, 29)], 0.0)]


The ``document`` is the "stringified" version of ``tokens``. If you have
a custom "tokens -> string" algorithm you may want to set this attribute
yourself.

The ``spans`` object is a list of weights for each character in
``document``. We use the indices in ``document`` string to indicate
which characters should be weighted with a specific value.

The weights come from the ``heatmap`` object found on each item in
``targets``.

.. code:: ipython3

    heatmap = target.heatmap
    print(heatmap)
    print(len(heatmap))
    
    print(len(doc_ws.spans))


.. parsed-literal::

    [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00 2.54064980e-05 0.00000000e+00 0.00000000e+00
     0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     6.95074850e-05 0.00000000e+00 0.00000000e+00 2.38344761e-04
     2.38344761e-04 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00 0.00000000e+00 0.00000000e+00 2.38344761e-04
     0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00]
    29
    29


You can think of this as an array of "importances" in the tokens array
(after padding is removed).

Let's format this. HTML formatter is what should be used here.

.. code:: ipython3

    import eli5.formatters.fields as fields
    F = eli5.format_as_html(E, show=fields.WEIGHTS)

We pass a ``show`` argument to not display the method name or its
description (Grad-CAM). See ``eli5.format_as_html()`` for a list of all
supported arguments.

The output is an HTML-encoded string.

.. code:: ipython3

    repr(F)




.. parsed-literal::

    '\'\\n    <style>\\n    table.eli5-weights tr:hover {\\n        filter: brightness(85%);\\n    }\\n</style>\\n\\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n        \\n\\n    \\n\\n        \\n            \\n                \\n                \\n            \\n        \\n\\n        \\n\\n\\n    <p style="margin-bottom: 2.5em; margin-top:-0.5em;">\\n        <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>\\n    </p>\\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n\\n\\n\''



Display it in an IPython notebook

.. code:: ipython3

    display(HTML(F))



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
                
                    
                    
                
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



The ``interpolation_kind`` argument
-----------------------------------

Heatmap does not match shape of tokens. We want to control how the
resizing is done.

Getting back to sentiment classification

.. code:: ipython3

    print(tokens.shape, len(heatmap))


.. parsed-literal::

    (1, 3193) 29


.. code:: ipython3

    model2.get_layer(index=3).output_shape




.. parsed-literal::

    (None, 1589, 128)



.. code:: ipython3

    kinds = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']

.. code:: ipython3

    for kind in kinds:
        print(kind)
        H = eli5.show_prediction(model2, doc_s, tokens=tokens_s, pad_value='<PAD>', padding='post', 
                                 interpolation_kind=kind,
                                 )
        display(H)


.. parsed-literal::

    linear



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    nearest



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    zero



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    slinear



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    quadratic



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">i</span><span style="opacity: 0.80">s</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">t</span><span style="opacity: 0.80">e</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">m</span><span style="opacity: 0.80">y</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">c</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">l</span><span style="opacity: 0.80">o</span><span style="opacity: 0.80">a</span><span style="opacity: 0.80">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    cubic



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">t</span><span style="opacity: 0.80">h</span><span style="opacity: 0.80">e</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">i</span><span style="opacity: 0.80">s </span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">m</span><span style="opacity: 0.80">y</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">c</span><span style="opacity: 0.80">a</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">l</span><span style="opacity: 0.80">o</span><span style="opacity: 0.80">a</span><span style="opacity: 0.80">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    previous



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    next



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">the I</span><span style="background-color: hsl(120, 100.00%, 81.99%); opacity: 0.86" title="0.000">R</span><span style="opacity: 0.80">S is a</span><span style="background-color: hsl(120, 100.00%, 63.57%); opacity: 0.97" title="0.000">f</span><span style="opacity: 0.80">te</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">rr</span><span style="opacity: 0.80"> my ca</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



The results are roughly the same. If highlighting seems off this
argument may be a thing to try.

Notes on results
----------------

Multi-label classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Does not work
