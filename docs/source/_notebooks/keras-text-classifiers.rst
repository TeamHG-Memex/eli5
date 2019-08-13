
Explaining Keras text classifier predictions with Grad-CAM
==========================================================

We can use ELI5 to explain text-based classifiers, i.e. models that take
in a text and assign it to some class. Common examples include sentiment
classification, labelling into categories, etc.

The underlying method used is 'Grad-CAM'
(https://arxiv.org/abs/1610.02391). This technique shows what parts of
the input are the most important to the predicted result, by overlaying
a "heatmap" over the original input.

See also the tutorial for images
(https://eli5.readthedocs.io/en/latest/tutorials/keras-image-classifiers.html).
Certain sections such as 'removing softmax' and 'comparing different
models' are relevant for text as well.

Set up
------

First some imports

.. code:: ipython3

    import os
    
    import numpy as np
    import pandas as pd
    from IPython.display import display, HTML  # our explanations will be formatted in HTML
    
    # you may want to keep logging enabled when doing your own work
    import logging
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR) # disable Tensorflow warnings for this tutorial
    import warnings
    warnings.simplefilter("ignore") # disable Keras warnings for this tutorial
    import keras
    from keras.preprocessing.sequence import pad_sequences
    
    import eli5


.. parsed-literal::

    Using TensorFlow backend.


The rest of what we need in this tutorial is stored in the
``tests/estimators`` package, whose source you can check for your own
reference. You may need extra steps here to load your custom model and
data.

.. code:: ipython3

    # we need to go back to top level in order to import some local ELI5 modules
    
    old = os.getcwd()
    os.chdir('..')

Explaining binary (sentiment) classifications
---------------------------------------------

In binary classification there is only one possible class to which a
piece of text can either belong to or not. In sentiment classification,
that class is whether the text is "positive" (belongs to the class) or
"negative" (doesn't belong to the class).

In this example we will have a recurrent model with word level
tokenization, trained on the IMDB dataset
(https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification).
The model has one output node that gives probabilities. Output close to
1 is positive, and close to 0 is negative.

See
https://www.tensorflow.org/beta/tutorials/text/text\_classification\_rnn
for a simple example of how to build such a model and prepare its input.

Let's load our pre-trained model

.. code:: ipython3

    binary_model = keras.models.load_model('tests/estimators/keras_sentiment_classifier/keras_sentiment_classifier.h5')
    binary_model.summary()


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


Load our test and train data. We have a module that will do
preprocessing for us. For your own usage you may have to do your own
preprocessing.

.. code:: ipython3

    import tests.estimators.keras_sentiment_classifier.keras_sentiment_classifier \
    as keras_sentiment_classifier

.. code:: ipython3

    (x_train, y_train), (x_test, y_test) = keras_sentiment_classifier.prepare_train_test_dataset()

Confirm the accuracy of the model

.. code:: ipython3

    print(binary_model.metrics_names)
    binary_model.evaluate(x_test, y_test)


.. parsed-literal::

    ['loss', 'acc']
    25000/25000 [==============================] - 88s 4ms/step




.. parsed-literal::

    [0.4319177031707764, 0.81504]



Looks good? Let's go on and check one of the test samples.

.. code:: ipython3

    test_review = x_test[0:1]
    print(test_review)
    
    test_review_t = keras_sentiment_classifier.vectorized_to_tokens(test_review)
    print(test_review_t)


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

    binary_model.predict(test_review)




.. parsed-literal::

    array([[0.1622659]], dtype=float32)



As expected, looks pretty low accuracy.

Now let's explain what got us this result with ELI5. We need to pass the
model, the input, and the associated tokens that will be highlighted.

.. code:: ipython3

    eli5.show_prediction(binary_model, test_review, tokens=test_review_t)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> &lt;OOV&gt; &lt;OOV&gt; </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> the rest </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the cast </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> terrible </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> flat flat flat </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> could have allowed this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> almost seemed to </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> this wasn&#x27;t going to </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> out </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> &lt;OOV&gt; so </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




What we are seeing is what makes the prediction "go up", i.e. the
"positive" words (check the next section to see how to show positive AND
negative words with the ``relu`` argument).

Hover over the highlighted words to see their "weight".

Let's try a custom input

.. code:: ipython3

    s = "hello this is great but not so great"
    review, review_t = keras_sentiment_classifier.string_to_vectorized(s)
    print(review, review_t, sep='\n')


.. parsed-literal::

    [[   1 4825   14    9   87   21   24   38   87]]
    [['<START>' 'hello' 'this' 'is' 'great' 'but' 'not' 'so' 'great']]


Notice that this model does not require fixed length input. We do not
need to pad this sample.

.. code:: ipython3

    binary_model.predict(review)




.. parsed-literal::

    array([[0.5912496]], dtype=float32)



Neutral as expected.

What makes the score go up?

.. code:: ipython3

    eli5.show_prediction(binary_model, review, tokens=review_t)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 94.46%); opacity: 0.81" title="0.001">hello</span><span style="opacity: 0.80"> this </span><span style="background-color: hsl(120, 100.00%, 70.23%); opacity: 0.93" title="0.015">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span><span style="opacity: 0.80"> but not so </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Let's try add padding to the sample

.. code:: ipython3

    review_t_padded = pad_sequences(review_t, maxlen=128, value='<PAD>', padding='pre', dtype=object)
    review_padded = keras_sentiment_classifier.tokens_to_vectorized(review_t_padded)
    print(review_t_padded, review_padded)
    
    eli5.show_prediction(binary_model, review_padded, tokens=review_t_padded)


.. parsed-literal::

    [['<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<START>' 'hello' 'this' 'is' 'great' 'but' 'not' 'so'
      'great']] [[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    1 4825   14    9   87   21   24
        38   87]]




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> &lt;START&gt; hello this </span><span style="background-color: hsl(120, 100.00%, 75.96%); opacity: 0.90" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.006">great</span><span style="opacity: 0.80"> but not so </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.006">great</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




As expected special words like ``<PAD>`` shouldn't have an effect on the
explanation.

Modify explanations with the ``relu`` and ``counterfactual`` arguments
----------------------------------------------------------------------

In the last section we only saw the "positive" words in our input, what
made the class score "go up". To "fix" this and see the "negative" words
too, we can pass two boolean arguments.

``relu`` (default ``True``) only shows what makes the predicted score go
up and discards the effect of counter-evidence or other classes in case
of multiclass classification (set to ``False`` to disable).

.. code:: ipython3

    eli5.show_prediction(binary_model, review, tokens=review_t, relu=False)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 77.80%); opacity: 0.89" title="-0.010">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.46%); opacity: 0.81" title="0.001">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.81%); opacity: 0.84" title="-0.005">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.23%); opacity: 0.93" title="0.015">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.58%); opacity: 0.85" title="-0.005">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.79%); opacity: 0.96" title="-0.018">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.60%); opacity: 0.83" title="-0.004">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Green is positive, red is negative, white is neutral. We see how the
input has conflicting sentiment and thus the model sensibly predicted a
score close to 0.5.

And for the test sample

.. code:: ipython3

    eli5.show_prediction(binary_model, test_review, tokens=test_review_t, relu=False)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 90.36%); opacity: 0.83" title="-0.003">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.77%); opacity: 0.83" title="0.003">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.87%); opacity: 0.86" title="-0.006">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.002">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.46%); opacity: 0.82" title="-0.002">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.66%); opacity: 0.91" title="0.011">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.020">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.002">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.12%); opacity: 0.84" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.52%); opacity: 0.83" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.80%); opacity: 0.81" title="0.001">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.05%); opacity: 0.82" title="-0.002">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.92%); opacity: 0.81" title="0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.80%); opacity: 0.80" title="-0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.09%); opacity: 0.82" title="-0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.11%); opacity: 0.82" title="0.002">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.57%); opacity: 0.80" title="-0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.39%); opacity: 0.89" title="-0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.38%); opacity: 0.85" title="-0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.90%); opacity: 0.81" title="-0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.002">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.55%); opacity: 0.80" title="-0.000">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.33%); opacity: 0.89" title="0.009">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.49%); opacity: 0.81" title="-0.001">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.31%); opacity: 0.83" title="0.003">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.97%); opacity: 0.82" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.02%); opacity: 0.86" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.09%); opacity: 0.86" title="0.006">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




We can see what made the network decide that is is a negative example.

Another argument ``counterfactual`` (default ``False``) highlights the
counter-evidence for a class, or what makes the score "go down" (set to
``True`` to enable). This is mentioned in the Grad-CAM paper
(https://arxiv.org/abs/1610.02391).

.. code:: ipython3

    eli5.show_prediction(binary_model, test_review, tokens=test_review_t, counterfactual=True)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 90.36%); opacity: 0.83" title="0.003">&lt;START&gt;</span><span style="opacity: 0.80"> please </span><span style="background-color: hsl(120, 100.00%, 94.90%); opacity: 0.81" title="0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> one a miss br br </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> and </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.87%); opacity: 0.86" title="0.006">rest</span><span style="opacity: 0.80"> of </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.46%); opacity: 0.82" title="0.002">cast</span><span style="opacity: 0.80"> rendered </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.020">terrible</span><span style="opacity: 0.80"> performances </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> show is </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.15%); opacity: 0.82" title="0.002">flat</span><span style="opacity: 0.80"> br br i </span><span style="background-color: hsl(120, 100.00%, 91.05%); opacity: 0.82" title="0.002">don&#x27;t</span><span style="opacity: 0.80"> know how </span><span style="background-color: hsl(120, 100.00%, 89.51%); opacity: 0.83" title="0.003">michael</span><span style="opacity: 0.80"> madison </span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.51%); opacity: 0.83" title="0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> one on his plate he </span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.39%); opacity: 0.89" title="0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.001">to</span><span style="opacity: 0.80"> know </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.38%); opacity: 0.85" title="0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.001">to</span><span style="opacity: 0.80"> work </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.000">out</span><span style="opacity: 0.80"> and his performance </span><span style="background-color: hsl(120, 100.00%, 93.49%); opacity: 0.81" title="0.001">was</span><span style="opacity: 0.80"> quite </span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.97%); opacity: 0.82" title="0.002">so</span><span style="opacity: 0.80"> all you madison fans </span><span style="background-color: hsl(120, 100.00%, 94.90%); opacity: 0.81" title="0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> a miss </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




This shows the "negative" words in green.

What happens if we pass both ``counterfactual`` and ``relu``?

.. code:: ipython3

    eli5.show_prediction(binary_model, test_review, tokens=test_review_t, relu=False, counterfactual=True)




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

When working with text, often sample input is padded or truncated to a
certain length, whether because the model only takes fixed-length input,
or because we want to put all the samples in a batch.

We can remove padding by specifying two arguments. The first is
``pad_value``, the padding token such as ``<PAD>`` in ``tokens`` or a
numeric value such as ``0`` for ``doc`` input. The second argument is
``padding``, which should be set to either ``pre`` (to remove padding
that comes before the actual text) or ``post`` (to remove padding that
comes after the actual text).

.. code:: ipython3

    print(test_review_t)


.. parsed-literal::

    [['<START>', 'please', 'give', 'this', 'one', 'a', 'miss', 'br', 'br', '<OOV>', '<OOV>', 'and', 'the', 'rest', 'of', 'the', 'cast', 'rendered', 'terrible', 'performances', 'the', 'show', 'is', 'flat', 'flat', 'flat', 'br', 'br', 'i', "don't", 'know', 'how', 'michael', 'madison', 'could', 'have', 'allowed', 'this', 'one', 'on', 'his', 'plate', 'he', 'almost', 'seemed', 'to', 'know', 'this', "wasn't", 'going', 'to', 'work', 'out', 'and', 'his', 'performance', 'was', 'quite', '<OOV>', 'so', 'all', 'you', 'madison', 'fans', 'give', 'this', 'a', 'miss', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']]


Notice that the padding word used here is ``<PAD>`` and that it comes
after the text.

.. code:: ipython3

    eli5.show_prediction(binary_model, test_review, tokens=test_review_t, 
                        pad_value='<PAD>', padding='post', relu=False)




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

We can also pass padding as a number into our input ``doc``.

.. code:: ipython3

    print(test_review)


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


Notice the number used for padding is ``0``.

.. code:: ipython3

    eli5.show_prediction(binary_model, test_review, tokens=test_review_t, 
                         pad_value=0, padding='post', relu=False)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 90.36%); opacity: 0.83" title="-0.003">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.77%); opacity: 0.83" title="0.003">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.87%); opacity: 0.86" title="-0.006">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.002">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.46%); opacity: 0.82" title="-0.002">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.66%); opacity: 0.91" title="0.011">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.020">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.002">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.12%); opacity: 0.84" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.52%); opacity: 0.83" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.80%); opacity: 0.81" title="0.001">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.05%); opacity: 0.82" title="-0.002">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.92%); opacity: 0.81" title="0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.80%); opacity: 0.80" title="-0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.09%); opacity: 0.82" title="-0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.11%); opacity: 0.82" title="0.002">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.57%); opacity: 0.80" title="-0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.39%); opacity: 0.89" title="-0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.38%); opacity: 0.85" title="-0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.90%); opacity: 0.81" title="-0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.002">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.55%); opacity: 0.80" title="-0.000">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.33%); opacity: 0.89" title="0.009">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.49%); opacity: 0.81" title="-0.001">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.31%); opacity: 0.83" title="0.003">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.97%); opacity: 0.82" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.02%); opacity: 0.86" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.09%); opacity: 0.86" title="0.006">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Let's try our pre-padded sample

.. code:: ipython3

    print(review_t_padded)
    eli5.show_prediction(binary_model, review_padded, tokens=review_t_padded, 
                         relu=False, pad_value='<PAD>', padding='pre')


.. parsed-literal::

    [['<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>' '<PAD>'
      '<PAD>' '<PAD>' '<START>' 'hello' 'this' 'is' 'great' 'but' 'not' 'so'
      'great']]




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 77.44%); opacity: 0.89" title="-0.003">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.79%); opacity: 0.81" title="-0.000">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.35%); opacity: 0.87" title="-0.002">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.96%); opacity: 0.90" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.006">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.66%); opacity: 0.88" title="-0.002">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.63%); opacity: 0.96" title="-0.005">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.45%); opacity: 0.87" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.006">great</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Useful!

Explaining multiclass model predictions
---------------------------------------

In multiple classification tasks a piece of text is classified into a
single class (we still have only one predicted label) from a number of
classes (not just one as in binary classification).

In this tutorial we have a multiclass model trained on the US consumer
finanial complaints dataset
(https://www.kaggle.com/cfpb/us-consumer-finance-complaints). We have
used character-level tokenization and a convolutional network that takes
fixed-length input. For this model the output will be a vector (since we
have many classes). The entry with the highest value will be the
"predicted" class.

.. code:: ipython3

    multicls_model = keras.models.load_model('tests/estimators/keras_multiclass_text_classifier/keras_multiclass_text_classifier.h5')
    multicls_model.summary()


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

Again check the metrics.

.. code:: ipython3

    print(multicls_model.metrics_names)
    multicls_model.evaluate(x_test, y_test)


.. parsed-literal::

    ['loss', 'acc']
    500/500 [==============================] - 9s 19ms/step




.. parsed-literal::

    [0.6319513120651246, 0.7999999990463257]



Let's see the possible classes that consumer complaint narratives can
fall into

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



Let's explain one of the test samples

.. code:: ipython3

    test_complaint = x_test[0:1]  # we need to keep the batch dimension
    test_complaint_t = keras_multiclass_text_classifier.vectorized_to_tokens(test_complaint)
    s = keras_multiclass_text_classifier.tokens_to_string(test_complaint_t)
    
    print(len(test_complaint[0]))
    limit = 150  # the input is quite long so just print the beginning
    print(test_complaint[0, :limit])
    print(test_complaint_t[0, :limit])
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


Let's check what the model predicts (to which category the financial
complaint belongs)

.. code:: ipython3

    preds = multicls_model.predict(test_complaint)
    print(preds)  # score for each class
    y = np.argmax(preds)  # take the maximum class
    print(y)
    keras_multiclass_text_classifier.decode_output(y)


.. parsed-literal::

    [[7.4966592e-03 9.7562626e-08 9.9250317e-01 9.1982411e-12 5.3569739e-08
      4.8417964e-10 9.6964792e-10 4.0114050e-09 5.9291594e-10 3.4063903e-13
      3.9474773e-19]]
    2




.. parsed-literal::

    'Mortgage'



And the ground truth

.. code:: ipython3

    y_truth = y_test[0]
    print(y_truth)
    keras_multiclass_text_classifier.decode_output(y_truth)


.. parsed-literal::

    [0 0 1 0 0 0 0 0 0 0 0]




.. parsed-literal::

    'Mortgage'



Seems reasonable!

Now let's explain this prediction with ELI5.

.. code:: ipython3

    eli5.show_prediction(multicls_model, test_complaint, tokens=test_complaint_t, 
                         pad_value='<PAD>', padding='post')




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
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




Note that we do not set ``relu`` to ``False`` because then we would see
other classes.

Our own example

.. code:: ipython3

    s = """first of all I should not be charged and debted for the private car loan"""
    
    complaint, complaint_t = keras_multiclass_text_classifier.string_to_vectorized(s)
    print(complaint)
    print(complaint_t[0, :50])  # note that this model requires fixed length input


.. parsed-literal::

    [[20  8  9 ...  0  0  0]]
    ['f' 'i' 'r' 's' 't' ' ' 'o' 'f' ' ' 'a' 'l' 'l' ' ' 'I' ' ' 's' 'h' 'o'
     'u' 'l' 'd' ' ' 'n' 'o' 't' ' ' 'b' 'e' ' ' 'c' 'h' 'a' 'r' 'g' 'e' 'd'
     ' ' 'a' 'n' 'd' ' ' 'd' 'e' 'b' 't' 'e' 'd' ' ' 'f' 'o']


.. code:: ipython3

    preds = multicls_model.predict(complaint)
    print(preds)
    print(keras_multiclass_text_classifier.decode_output(preds))
    
    eli5.show_prediction(multicls_model, complaint, tokens=complaint_t, 
                         pad_value='<PAD>', padding='post')


.. parsed-literal::

    [[0.12006541 0.29358572 0.07228176 0.02435291 0.0842614  0.22937107
      0.01472037 0.15292141 0.00499524 0.00131895 0.00212576]]
    Consumer Loan




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80">st o</span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80"> all I should not be cha</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">g</span><span style="opacity: 0.80">ed and debted </span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80">o</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80"> the p</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80">ivate ca</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    




.. code:: ipython3

    # TODO: would be good to show predicted label

Choosing a classification target to focus on via ``targets``
------------------------------------------------------------

In the last text we saw that it could be classified into more than just
one category.

We can use ELI5 to "force" the network to classify the input into a
certain category, and then highlight evidence for that category.

We use the ``targets`` argument for this. We pass a list that contains
integer indices. Those indices represent a class in the final output
layer.

Let's check two sensible categories

.. code:: ipython3

    debt_idx = 0  # we get this from the labels index
    loan_idx = 1

.. code:: ipython3

    print('debt collection')
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t, targets=[debt_idx],
                         pad_value='<PAD>', padding='post'))
    
    print('consumer loan')
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t, targets=[loan_idx],
                         pad_value='<PAD>', padding='post'))


.. parsed-literal::

    debt collection



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 88.87%); opacity: 0.83" title="0.000">i</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 92.81%); opacity: 0.82" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 98.17%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.97%); opacity: 0.81" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 92.15%); opacity: 0.82" title="0.000">ll</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.001">I</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 92.81%); opacity: 0.82" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 95.00%); opacity: 0.81" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 95.97%); opacity: 0.81" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 74.92%); opacity: 0.90" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 92.15%); opacity: 0.82" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 95.97%); opacity: 0.81" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.17%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 79.83%); opacity: 0.88" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 87.16%); opacity: 0.84" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 95.00%); opacity: 0.81" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.000">a</span><span style="opacity: 0.80">rg</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 79.83%); opacity: 0.88" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 98.17%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 95.97%); opacity: 0.81" title="0.000">o</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.17%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 95.00%); opacity: 0.81" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="opacity: 0.80">pr</span><span style="background-color: hsl(120, 100.00%, 88.87%); opacity: 0.83" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 67.58%); opacity: 0.95" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.17%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 87.16%); opacity: 0.84" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.000">a</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 95.75%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 92.15%); opacity: 0.82" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 95.97%); opacity: 0.81" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 96.05%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.000">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    consumer loan



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80">st o</span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80"> all I should not be cha</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">g</span><span style="opacity: 0.80">ed and debted </span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80">o</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80"> the p</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80">ivate ca</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



Sensible at least a little bit?

Note that we can use the IPython ``display()`` call to render HTML if it
is not the last value in a call.

Choosing a hidden layer to do Grad-CAM on with ``layer``
--------------------------------------------------------

Grad-CAM requires a hidden layer to do its calculations on and produce a
heatmap. This is controlled by the ``layer`` argument. We can pass the
layer (as an int index, string name, or a keras Layer instance)
explicitly, or let ELI5 attempt to find a good layer for us
automatically.

.. code:: ipython3

    from keras.layers import (  # some of the layers we may want to check
        Embedding,
        Conv1D,
        MaxPool1D,
        AveragePooling1D,
        GlobalAveragePooling1D,
        Dense,
    )

.. code:: ipython3

    desired = (Embedding, Conv1D, MaxPool1D, AveragePooling1D, GlobalAveragePooling1D, Dense)
    
    for layer in multicls_model.layers:
        print(layer.name, layer.output_shape)
        if isinstance(layer, desired):
            html = eli5.show_prediction(multicls_model, complaint, tokens=complaint_t, layer=layer,
                                        pad_value='<PAD>', padding='post')
            display(html)  # if using a loop we also need a display call


.. parsed-literal::

    embedding_1 (None, 3193, 8)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80">st o</span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80"> all I should not be cha</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">g</span><span style="opacity: 0.80">ed and debted </span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80">o</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80"> the p</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80">ivate ca</span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    conv1d_1 (None, 3179, 128)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 92.35%); opacity: 0.82" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 95.21%); opacity: 0.81" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 95.87%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 95.70%); opacity: 0.81" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 94.73%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 76.63%); opacity: 0.89" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 97.96%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 94.20%); opacity: 0.81" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 78.17%); opacity: 0.88" title="0.000">I</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 83.65%); opacity: 0.86" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 89.89%); opacity: 0.83" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 89.39%); opacity: 0.83" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 78.52%); opacity: 0.88" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 97.90%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.03%); opacity: 0.81" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t </span><span style="background-color: hsl(120, 100.00%, 92.34%); opacity: 0.82" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 92.37%); opacity: 0.82" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 75.04%); opacity: 0.90" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.000">g</span><span style="opacity: 0.80">ed and</span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.000">d</span><span style="opacity: 0.80">e</span><span style="background-color: hsl(120, 100.00%, 85.30%); opacity: 0.85" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 84.95%); opacity: 0.85" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 91.70%); opacity: 0.82" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 93.96%); opacity: 0.81" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 95.44%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 89.04%); opacity: 0.83" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 84.28%); opacity: 0.85" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 89.84%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 94.06%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 86.69%); opacity: 0.84" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 85.63%); opacity: 0.85" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 97.12%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.02%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 87.55%); opacity: 0.84" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 87.60%); opacity: 0.84" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 89.01%); opacity: 0.83" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 91.64%); opacity: 0.82" title="0.000">v</span><span style="background-color: hsl(120, 100.00%, 89.93%); opacity: 0.83" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 70.92%); opacity: 0.93" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 87.10%); opacity: 0.84" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 87.31%); opacity: 0.84" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 61.29%); opacity: 0.99" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 80.05%); opacity: 0.87" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 80.09%); opacity: 0.87" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 92.09%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 80.18%); opacity: 0.87" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 90.96%); opacity: 0.82" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 92.88%); opacity: 0.82" title="0.000">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    dropout_1 (None, 3179, 128)
    max_pooling1d_1 (None, 1589, 128)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 96.24%); opacity: 0.81" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 88.98%); opacity: 0.83" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 92.03%); opacity: 0.82" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 95.68%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 90.27%); opacity: 0.83" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 90.42%); opacity: 0.83" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 96.44%); opacity: 0.81" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 85.11%); opacity: 0.85" title="0.000">I</span><span style="background-color: hsl(120, 100.00%, 97.24%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 88.96%); opacity: 0.83" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 91.59%); opacity: 0.82" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 96.16%); opacity: 0.81" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 97.92%); opacity: 0.80" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 91.31%); opacity: 0.82" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 87.50%); opacity: 0.84" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 95.84%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.56%); opacity: 0.80" title="0.000">n</span><span style="opacity: 0.80">ot</span><span style="background-color: hsl(120, 100.00%, 97.63%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 84.27%); opacity: 0.85" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 92.89%); opacity: 0.82" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.42%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 84.49%); opacity: 0.85" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 86.88%); opacity: 0.84" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 94.41%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.41%); opacity: 0.80" title="0.000">r</span><span style="opacity: 0.80">ged and de</span><span style="background-color: hsl(120, 100.00%, 87.76%); opacity: 0.84" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 88.57%); opacity: 0.83" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 90.58%); opacity: 0.83" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 91.68%); opacity: 0.82" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 95.19%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 82.23%); opacity: 0.86" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 87.80%); opacity: 0.84" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 88.91%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 88.22%); opacity: 0.83" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 87.63%); opacity: 0.84" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 97.41%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 97.85%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 75.03%); opacity: 0.90" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 93.40%); opacity: 0.82" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 95.57%); opacity: 0.81" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 62.73%); opacity: 0.98" title="0.000">v</span><span style="background-color: hsl(120, 100.00%, 80.30%); opacity: 0.87" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 75.41%); opacity: 0.90" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 79.00%); opacity: 0.88" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 89.01%); opacity: 0.83" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 82.98%); opacity: 0.86" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 82.53%); opacity: 0.86" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 91.92%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 80.95%); opacity: 0.87" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 87.00%); opacity: 0.84" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 93.62%); opacity: 0.81" title="0.000">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    conv1d_2 (None, 1580, 128)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 97.78%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.34%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 97.92%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.16%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.86%); opacity: 0.81" title="0.003">I</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.15%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">b</span><span style="opacity: 0.80">e charged </span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 94.66%); opacity: 0.81" title="0.004">n</span><span style="background-color: hsl(120, 100.00%, 95.34%); opacity: 0.81" title="0.003">d</span><span style="background-color: hsl(120, 100.00%, 97.50%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 93.88%); opacity: 0.81" title="0.005">d</span><span style="background-color: hsl(120, 100.00%, 93.47%); opacity: 0.82" title="0.005">e</span><span style="background-color: hsl(120, 100.00%, 83.96%); opacity: 0.85" title="0.018">b</span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.008">t</span><span style="background-color: hsl(120, 100.00%, 91.16%); opacity: 0.82" title="0.008">e</span><span style="background-color: hsl(120, 100.00%, 89.25%); opacity: 0.83" title="0.010">d</span><span style="background-color: hsl(120, 100.00%, 94.47%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 83.02%); opacity: 0.86" title="0.020">f</span><span style="background-color: hsl(120, 100.00%, 87.69%); opacity: 0.84" title="0.012">o</span><span style="background-color: hsl(120, 100.00%, 87.35%); opacity: 0.84" title="0.013">r</span><span style="background-color: hsl(120, 100.00%, 93.68%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 86.69%); opacity: 0.84" title="0.014">t</span><span style="background-color: hsl(120, 100.00%, 80.93%); opacity: 0.87" title="0.023">h</span><span style="background-color: hsl(120, 100.00%, 88.58%); opacity: 0.83" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.067">p</span><span style="background-color: hsl(120, 100.00%, 87.15%); opacity: 0.84" title="0.013">r</span><span style="background-color: hsl(120, 100.00%, 76.60%); opacity: 0.89" title="0.031">i</span><span style="background-color: hsl(120, 100.00%, 62.63%); opacity: 0.98" title="0.061">v</span><span style="background-color: hsl(120, 100.00%, 89.05%); opacity: 0.83" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 87.79%); opacity: 0.84" title="0.012">t</span><span style="background-color: hsl(120, 100.00%, 90.40%); opacity: 0.83" title="0.009">e</span><span style="background-color: hsl(120, 100.00%, 95.41%); opacity: 0.81" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 85.47%); opacity: 0.85" title="0.016">c</span><span style="background-color: hsl(120, 100.00%, 94.90%); opacity: 0.81" title="0.004">a</span><span style="background-color: hsl(120, 100.00%, 96.04%); opacity: 0.81" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.88%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.61%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.000">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    dropout_2 (None, 1580, 128)
    average_pooling1d_1 (None, 790, 128)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 97.52%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 97.17%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.68%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.33%); opacity: 0.81" title="0.003">I</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">d</span><span style="opacity: 0.80"> not be char</span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 97.96%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 96.94%); opacity: 0.81" title="0.002">a</span><span style="background-color: hsl(120, 100.00%, 93.46%); opacity: 0.82" title="0.007">n</span><span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.005">d</span><span style="background-color: hsl(120, 100.00%, 96.84%); opacity: 0.81" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 92.55%); opacity: 0.82" title="0.008">d</span><span style="background-color: hsl(120, 100.00%, 92.59%); opacity: 0.82" title="0.008">e</span><span style="background-color: hsl(120, 100.00%, 82.25%); opacity: 0.86" title="0.029">b</span><span style="background-color: hsl(120, 100.00%, 89.76%); opacity: 0.83" title="0.013">t</span><span style="background-color: hsl(120, 100.00%, 90.37%); opacity: 0.83" title="0.012">e</span><span style="background-color: hsl(120, 100.00%, 88.51%); opacity: 0.83" title="0.015">d</span><span style="background-color: hsl(120, 100.00%, 94.15%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 82.06%); opacity: 0.86" title="0.029">f</span><span style="background-color: hsl(120, 100.00%, 87.09%); opacity: 0.84" title="0.018">o</span><span style="background-color: hsl(120, 100.00%, 86.93%); opacity: 0.84" title="0.018">r</span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 86.61%); opacity: 0.84" title="0.019">t</span><span style="background-color: hsl(120, 100.00%, 80.80%); opacity: 0.87" title="0.032">h</span><span style="background-color: hsl(120, 100.00%, 88.32%); opacity: 0.83" title="0.016">e</span><span style="background-color: hsl(120, 100.00%, 93.62%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.091">p</span><span style="background-color: hsl(120, 100.00%, 87.24%); opacity: 0.84" title="0.018">r</span><span style="background-color: hsl(120, 100.00%, 76.32%); opacity: 0.89" title="0.043">i</span><span style="background-color: hsl(120, 100.00%, 62.44%); opacity: 0.98" title="0.083">v</span><span style="background-color: hsl(120, 100.00%, 89.55%); opacity: 0.83" title="0.013">a</span><span style="background-color: hsl(120, 100.00%, 88.80%); opacity: 0.83" title="0.015">t</span><span style="background-color: hsl(120, 100.00%, 91.50%); opacity: 0.82" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 96.11%); opacity: 0.81" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 88.30%); opacity: 0.83" title="0.016">c</span><span style="background-color: hsl(120, 100.00%, 96.06%); opacity: 0.81" title="0.003">a</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 98.56%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 97.60%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.001">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    conv1d_3 (None, 786, 128)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">first</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.04%); opacity: 0.80" title="0.003">I</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">h</span><span style="opacity: 0.80">ould</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.004">n</span><span style="background-color: hsl(120, 100.00%, 97.50%); opacity: 0.80" title="0.004">o</span><span style="background-color: hsl(120, 100.00%, 96.87%); opacity: 0.81" title="0.005">t</span><span style="background-color: hsl(120, 100.00%, 97.91%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 89.21%); opacity: 0.83" title="0.031">b</span><span style="background-color: hsl(120, 100.00%, 93.90%); opacity: 0.81" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 96.06%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 82.45%); opacity: 0.86" title="0.062">c</span><span style="background-color: hsl(120, 100.00%, 85.16%); opacity: 0.85" title="0.049">h</span><span style="background-color: hsl(120, 100.00%, 89.91%); opacity: 0.83" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 87.49%); opacity: 0.84" title="0.038">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.200">g</span><span style="background-color: hsl(120, 100.00%, 88.73%); opacity: 0.83" title="0.033">e</span><span style="background-color: hsl(120, 100.00%, 87.37%); opacity: 0.84" title="0.039">d</span><span style="background-color: hsl(120, 100.00%, 93.94%); opacity: 0.81" title="0.014"> </span><span style="background-color: hsl(120, 100.00%, 89.26%); opacity: 0.83" title="0.031">a</span><span style="background-color: hsl(120, 100.00%, 83.15%); opacity: 0.86" title="0.058">n</span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.033">d</span><span style="background-color: hsl(120, 100.00%, 94.68%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 89.69%); opacity: 0.83" title="0.029">d</span><span style="background-color: hsl(120, 100.00%, 91.80%); opacity: 0.82" title="0.021">e</span><span style="background-color: hsl(120, 100.00%, 84.29%); opacity: 0.85" title="0.053">b</span><span style="background-color: hsl(120, 100.00%, 92.83%); opacity: 0.82" title="0.017">t</span><span style="background-color: hsl(120, 100.00%, 94.63%); opacity: 0.81" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.011">d</span><span style="background-color: hsl(120, 100.00%, 97.86%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 95.10%); opacity: 0.81" title="0.010">f</span><span style="background-color: hsl(120, 100.00%, 97.52%); opacity: 0.80" title="0.004">o</span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">t</span><span style="opacity: 0.80">he priva</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 96.70%); opacity: 0.81" title="0.006">c</span><span style="background-color: hsl(120, 100.00%, 98.13%); opacity: 0.80" title="0.003">a</span><span style="background-color: hsl(120, 100.00%, 98.01%); opacity: 0.80" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.10%); opacity: 0.80" title="0.003">l</span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.07%); opacity: 0.80" title="0.003">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    dropout_3 (None, 786, 128)
    max_pooling1d_2 (None, 393, 128)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">first of all I sh</span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 92.69%); opacity: 0.82" title="0.029">u</span><span style="background-color: hsl(120, 100.00%, 96.18%); opacity: 0.81" title="0.012">l</span><span style="background-color: hsl(120, 100.00%, 95.93%); opacity: 0.81" title="0.013">d</span><span style="background-color: hsl(120, 100.00%, 97.66%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 92.13%); opacity: 0.82" title="0.032">n</span><span style="background-color: hsl(120, 100.00%, 93.84%); opacity: 0.81" title="0.023">o</span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.026">t</span><span style="background-color: hsl(120, 100.00%, 96.33%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 84.06%); opacity: 0.85" title="0.089">b</span><span style="background-color: hsl(120, 100.00%, 91.90%); opacity: 0.82" title="0.034">e</span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.016"> </span><span style="background-color: hsl(120, 100.00%, 79.62%); opacity: 0.88" title="0.126">c</span><span style="background-color: hsl(120, 100.00%, 83.61%); opacity: 0.86" title="0.092">h</span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.050">a</span><span style="background-color: hsl(120, 100.00%, 87.13%); opacity: 0.84" title="0.065">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.330">g</span><span style="background-color: hsl(120, 100.00%, 89.19%); opacity: 0.83" title="0.051">e</span><span style="background-color: hsl(120, 100.00%, 88.43%); opacity: 0.83" title="0.056">d</span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.018"> </span><span style="background-color: hsl(120, 100.00%, 91.11%); opacity: 0.82" title="0.038">a</span><span style="background-color: hsl(120, 100.00%, 86.66%); opacity: 0.84" title="0.069">n</span><span style="background-color: hsl(120, 100.00%, 91.47%); opacity: 0.82" title="0.036">d</span><span style="background-color: hsl(120, 100.00%, 96.25%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 93.07%); opacity: 0.82" title="0.027">d</span><span style="background-color: hsl(120, 100.00%, 94.37%); opacity: 0.81" title="0.020">e</span><span style="background-color: hsl(120, 100.00%, 88.92%); opacity: 0.83" title="0.053">b</span><span style="background-color: hsl(120, 100.00%, 94.75%); opacity: 0.81" title="0.018">t</span><span style="background-color: hsl(120, 100.00%, 95.92%); opacity: 0.81" title="0.013">e</span><span style="background-color: hsl(120, 100.00%, 96.02%); opacity: 0.81" title="0.012">d</span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 96.41%); opacity: 0.81" title="0.011">f</span><span style="background-color: hsl(120, 100.00%, 98.26%); opacity: 0.80" title="0.004">o</span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.002">t</span><span style="background-color: hsl(120, 100.00%, 98.47%); opacity: 0.80" title="0.003">h</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.002">p</span><span style="opacity: 0.80">r</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(120, 100.00%, 97.48%); opacity: 0.80" title="0.006">v</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.002">a</span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.003">t</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 96.27%); opacity: 0.81" title="0.011">c</span><span style="background-color: hsl(120, 100.00%, 98.10%); opacity: 0.80" title="0.004">a</span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.005">r</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.005">l</span><span style="background-color: hsl(120, 100.00%, 98.14%); opacity: 0.80" title="0.004">o</span><span style="background-color: hsl(120, 100.00%, 98.44%); opacity: 0.80" title="0.003">a</span><span style="background-color: hsl(120, 100.00%, 97.59%); opacity: 0.80" title="0.006">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    global_average_pooling1d_1 (None, 128)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">first of all I should not be charged and debted for</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.82%); opacity: 0.81" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 91.78%); opacity: 0.82" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 93.71%); opacity: 0.81" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 95.89%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 70.21%); opacity: 0.93" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 89.19%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 77.39%); opacity: 0.89" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 87.69%); opacity: 0.84" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 85.02%); opacity: 0.85" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 85.97%); opacity: 0.84" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 91.79%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 66.20%); opacity: 0.96" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 83.54%); opacity: 0.86" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 80.42%); opacity: 0.87" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 90.06%); opacity: 0.83" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 75.11%); opacity: 0.90" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 77.88%); opacity: 0.89" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 79.81%); opacity: 0.88" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 66.04%); opacity: 0.96" title="0.001">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    dense_1 (None, 32)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 79.62%); opacity: 0.88" title="0.060">f</span><span style="background-color: hsl(120, 100.00%, 73.12%); opacity: 0.91" title="0.089">i</span><span style="background-color: hsl(120, 100.00%, 85.94%); opacity: 0.84" title="0.035">r</span><span style="background-color: hsl(120, 100.00%, 73.49%); opacity: 0.91" title="0.087">s</span><span style="background-color: hsl(120, 100.00%, 86.14%); opacity: 0.84" title="0.034">t</span><span style="background-color: hsl(120, 100.00%, 93.31%); opacity: 0.82" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 86.34%); opacity: 0.84" title="0.034">o</span><span style="background-color: hsl(120, 100.00%, 80.60%); opacity: 0.87" title="0.056">f</span><span style="background-color: hsl(120, 100.00%, 93.45%); opacity: 0.82" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 88.24%); opacity: 0.83" title="0.027">a</span><span style="background-color: hsl(120, 100.00%, 84.49%); opacity: 0.85" title="0.040">l</span><span style="background-color: hsl(120, 100.00%, 84.61%); opacity: 0.85" title="0.040">l</span><span style="background-color: hsl(120, 100.00%, 93.64%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.156">I</span><span style="background-color: hsl(120, 100.00%, 93.74%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 75.76%); opacity: 0.90" title="0.076">s</span><span style="background-color: hsl(120, 100.00%, 81.90%); opacity: 0.86" title="0.050">h</span><span style="background-color: hsl(120, 100.00%, 87.44%); opacity: 0.84" title="0.030">o</span><span style="background-color: hsl(120, 100.00%, 61.57%); opacity: 0.99" title="0.148">u</span><span style="background-color: hsl(120, 100.00%, 85.56%); opacity: 0.85" title="0.036">l</span><span style="background-color: hsl(120, 100.00%, 87.75%); opacity: 0.84" title="0.029">d</span><span style="background-color: hsl(120, 100.00%, 94.09%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 82.78%); opacity: 0.86" title="0.047">n</span><span style="background-color: hsl(120, 100.00%, 88.06%); opacity: 0.84" title="0.028">o</span><span style="background-color: hsl(120, 100.00%, 88.17%); opacity: 0.84" title="0.027">t</span><span style="background-color: hsl(120, 100.00%, 94.30%); opacity: 0.81" title="0.010"> </span><span style="background-color: hsl(120, 100.00%, 77.93%); opacity: 0.89" title="0.067">b</span><span style="background-color: hsl(120, 100.00%, 89.86%); opacity: 0.83" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 94.45%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 78.53%); opacity: 0.88" title="0.064">c</span><span style="background-color: hsl(120, 100.00%, 83.99%); opacity: 0.85" title="0.042">h</span><span style="background-color: hsl(120, 100.00%, 90.24%); opacity: 0.83" title="0.021">a</span><span style="background-color: hsl(120, 100.00%, 89.02%); opacity: 0.83" title="0.025">r</span><span style="background-color: hsl(120, 100.00%, 66.46%); opacity: 0.96" title="0.122">g</span><span style="background-color: hsl(120, 100.00%, 90.53%); opacity: 0.83" title="0.020">e</span><span style="background-color: hsl(120, 100.00%, 89.35%); opacity: 0.83" title="0.024">d</span><span style="background-color: hsl(120, 100.00%, 94.87%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.019">a</span><span style="background-color: hsl(120, 100.00%, 85.24%); opacity: 0.85" title="0.038">n</span><span style="background-color: hsl(120, 100.00%, 89.79%); opacity: 0.83" title="0.022">d</span><span style="background-color: hsl(120, 100.00%, 95.09%); opacity: 0.81" title="0.008"> </span><span style="background-color: hsl(120, 100.00%, 90.01%); opacity: 0.83" title="0.022">d</span><span style="background-color: hsl(120, 100.00%, 91.31%); opacity: 0.82" title="0.018">e</span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.052">b</span><span style="background-color: hsl(120, 100.00%, 90.35%); opacity: 0.83" title="0.020">t</span><span style="background-color: hsl(120, 100.00%, 91.61%); opacity: 0.82" title="0.017">e</span><span style="background-color: hsl(120, 100.00%, 90.58%); opacity: 0.83" title="0.020">d</span><span style="background-color: hsl(120, 100.00%, 95.48%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 86.87%); opacity: 0.84" title="0.032">f</span><span style="background-color: hsl(120, 100.00%, 90.93%); opacity: 0.82" title="0.019">o</span><span style="background-color: hsl(120, 100.00%, 91.05%); opacity: 0.82" title="0.018">r</span><span style="background-color: hsl(120, 100.00%, 95.71%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 91.29%); opacity: 0.82" title="0.018">t</span><span style="background-color: hsl(120, 100.00%, 87.72%); opacity: 0.84" title="0.029">h</span><span style="background-color: hsl(120, 100.00%, 92.55%); opacity: 0.82" title="0.014">e</span><span style="background-color: hsl(120, 100.00%, 95.94%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 74.62%); opacity: 0.90" title="0.082">p</span><span style="background-color: hsl(120, 100.00%, 91.90%); opacity: 0.82" title="0.016">r</span><span style="background-color: hsl(120, 100.00%, 84.85%); opacity: 0.85" title="0.039">i</span><span style="background-color: hsl(120, 100.00%, 75.77%); opacity: 0.90" title="0.076">v</span><span style="background-color: hsl(120, 100.00%, 93.20%); opacity: 0.82" title="0.012">a</span><span style="background-color: hsl(120, 100.00%, 92.40%); opacity: 0.82" title="0.015">t</span><span style="background-color: hsl(120, 100.00%, 93.42%); opacity: 0.82" title="0.012">e</span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 86.29%); opacity: 0.84" title="0.034">c</span><span style="background-color: hsl(120, 100.00%, 93.76%); opacity: 0.81" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 93.04%); opacity: 0.82" title="0.013">r</span><span style="background-color: hsl(120, 100.00%, 96.68%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 92.18%); opacity: 0.82" title="0.015">l</span><span style="background-color: hsl(120, 100.00%, 93.44%); opacity: 0.82" title="0.012">o</span><span style="background-color: hsl(120, 100.00%, 94.35%); opacity: 0.81" title="0.010">a</span><span style="background-color: hsl(120, 100.00%, 91.02%); opacity: 0.82" title="0.019">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    dense_2 (None, 11)



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">f</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 96.94%); opacity: 0.81" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 98.03%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.38%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.002">f</span><span style="background-color: hsl(120, 100.00%, 98.44%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 96.94%); opacity: 0.81" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 95.62%); opacity: 0.81" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 95.32%); opacity: 0.81" title="0.003">l</span><span style="background-color: hsl(120, 100.00%, 97.93%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 86.13%); opacity: 0.84" title="0.012">I</span><span style="background-color: hsl(120, 100.00%, 97.70%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 90.56%); opacity: 0.83" title="0.007">s</span><span style="background-color: hsl(120, 100.00%, 92.56%); opacity: 0.82" title="0.005">h</span><span style="background-color: hsl(120, 100.00%, 94.57%); opacity: 0.81" title="0.003">o</span><span style="background-color: hsl(120, 100.00%, 82.58%); opacity: 0.86" title="0.017">u</span><span style="background-color: hsl(120, 100.00%, 93.14%); opacity: 0.82" title="0.004">l</span><span style="background-color: hsl(120, 100.00%, 93.92%); opacity: 0.81" title="0.004">d</span><span style="background-color: hsl(120, 100.00%, 96.94%); opacity: 0.81" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 90.71%); opacity: 0.82" title="0.007">n</span><span style="background-color: hsl(120, 100.00%, 93.30%); opacity: 0.82" title="0.004">o</span><span style="background-color: hsl(120, 100.00%, 93.09%); opacity: 0.82" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 86.13%); opacity: 0.84" title="0.012">b</span><span style="background-color: hsl(120, 100.00%, 93.40%); opacity: 0.82" title="0.004">e</span><span style="background-color: hsl(120, 100.00%, 96.26%); opacity: 0.81" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 85.02%); opacity: 0.85" title="0.013">c</span><span style="background-color: hsl(120, 100.00%, 88.45%); opacity: 0.83" title="0.009">h</span><span style="background-color: hsl(120, 100.00%, 92.73%); opacity: 0.82" title="0.005">a</span><span style="background-color: hsl(120, 100.00%, 91.55%); opacity: 0.82" title="0.006">r</span><span style="background-color: hsl(120, 100.00%, 73.37%); opacity: 0.91" title="0.030">g</span><span style="background-color: hsl(120, 100.00%, 92.24%); opacity: 0.82" title="0.005">e</span><span style="background-color: hsl(120, 100.00%, 91.00%); opacity: 0.82" title="0.006">d</span><span style="background-color: hsl(120, 100.00%, 95.54%); opacity: 0.81" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 91.77%); opacity: 0.82" title="0.006">a</span><span style="background-color: hsl(120, 100.00%, 86.38%); opacity: 0.84" title="0.012">n</span><span style="background-color: hsl(120, 100.00%, 90.30%); opacity: 0.83" title="0.007">d</span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 89.95%); opacity: 0.83" title="0.008">d</span><span style="background-color: hsl(120, 100.00%, 91.00%); opacity: 0.82" title="0.006">e</span><span style="background-color: hsl(120, 100.00%, 80.27%); opacity: 0.87" title="0.020">b</span><span style="background-color: hsl(120, 100.00%, 89.44%); opacity: 0.83" title="0.008">t</span><span style="background-color: hsl(120, 100.00%, 90.56%); opacity: 0.83" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 89.11%); opacity: 0.83" title="0.008">d</span><span style="background-color: hsl(120, 100.00%, 94.62%); opacity: 0.81" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 83.95%); opacity: 0.85" title="0.015">f</span><span style="background-color: hsl(120, 100.00%, 88.62%); opacity: 0.83" title="0.009">o</span><span style="background-color: hsl(120, 100.00%, 88.45%); opacity: 0.83" title="0.009">r</span><span style="background-color: hsl(120, 100.00%, 94.31%); opacity: 0.81" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 88.13%); opacity: 0.84" title="0.010">t</span><span style="background-color: hsl(120, 100.00%, 82.80%); opacity: 0.86" title="0.016">h</span><span style="background-color: hsl(120, 100.00%, 89.27%); opacity: 0.83" title="0.008">e</span><span style="background-color: hsl(120, 100.00%, 94.00%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 61.43%); opacity: 0.99" title="0.052">p</span><span style="background-color: hsl(120, 100.00%, 87.34%); opacity: 0.84" title="0.010">r</span><span style="background-color: hsl(120, 100.00%, 75.67%); opacity: 0.90" title="0.027">i</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.054">v</span><span style="background-color: hsl(120, 100.00%, 88.45%); opacity: 0.83" title="0.009">a</span><span style="background-color: hsl(120, 100.00%, 86.73%); opacity: 0.84" title="0.011">t</span><span style="background-color: hsl(120, 100.00%, 88.19%); opacity: 0.84" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 93.40%); opacity: 0.82" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 73.93%); opacity: 0.91" title="0.029">c</span><span style="background-color: hsl(120, 100.00%, 87.79%); opacity: 0.84" title="0.010">a</span><span style="background-color: hsl(120, 100.00%, 85.98%); opacity: 0.84" title="0.012">r</span><span style="background-color: hsl(120, 100.00%, 93.11%); opacity: 0.82" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 83.26%); opacity: 0.86" title="0.016">l</span><span style="background-color: hsl(120, 100.00%, 85.53%); opacity: 0.85" title="0.013">o</span><span style="background-color: hsl(120, 100.00%, 87.14%); opacity: 0.84" title="0.011">a</span><span style="background-color: hsl(120, 100.00%, 78.90%); opacity: 0.88" title="0.022">n</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



Now this looks better. It should make sense for a Convolutional network
that later layers pick up "higher level" information than earlier "lower
level" layers. If you don't get good explanations from ELI5 out of the
box, it may be worth looking into this parameter. We advice to pick
layers that contain "spatial or temporal" information, i.e. NOT
dense/fully-connected or merge layers, but recurrent, convolutional, or
embedding layers.

What's up with the final dense layers? They do not have spatial
information so it's mostly a visualization of the activations of each
node, ignoring the underlying tokens. Hover over to see the actual
values (though some parts seem bright green, they may not have a high
weight - the color scale is "relative").

Resizing the heatmap with the ``interpolation_kind`` argument
-------------------------------------------------------------

In the last section we learned that we use a hidden layer to generate a
heatmap of activations. However, notice that some of the layers have
dimensions different from the tokens dimension.

.. code:: ipython3

    print(complaint_t.shape)
    print(multicls_model.get_layer(index=-5).output_shape)  # <--- an arbitrary layer


.. parsed-literal::

    (1, 3193)
    (None, 786, 128)


``tokens`` is length ``3193``, while the layer's temporal dimension is
``786``. This difference makes Grad-CAM explanations "coarse" or
approximate.

We have to resize the heatmap in order to lay it over the input tokens.

The resizing method can be controlled with the ``interpolation_kind``
argument. This is one of the strings listed under
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
for the ``kind`` argument. The default is ``linear``.

Let's check the possible interpolations (going back to sentiment
classification).

.. code:: ipython3

    kinds = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']

.. code:: ipython3

    for kind in kinds:
        print(kind)
        html = eli5.show_prediction(binary_model, test_review, tokens=test_review_t, interpolation_kind=kind)
        display(html)


.. parsed-literal::

    linear



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
            
    
        
    
            
    
            
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> &lt;OOV&gt; &lt;OOV&gt; </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> the rest </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the cast </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> terrible </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> flat flat flat </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> could have allowed this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> almost seemed to </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> this wasn&#x27;t going to </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> out </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> &lt;OOV&gt; so </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt;</span>
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
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> &lt;OOV&gt; &lt;OOV&gt; </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> the rest </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the cast </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> terrible </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> flat flat flat </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> could have allowed this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> almost seemed to </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> this wasn&#x27;t going to </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> out </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> &lt;OOV&gt; so </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt;</span>
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
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> &lt;OOV&gt; &lt;OOV&gt; </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> the rest </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the cast </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> terrible </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> flat flat flat </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> could have allowed this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> almost seemed to </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> this wasn&#x27;t going to </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> out </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> &lt;OOV&gt; so </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt;</span>
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
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> &lt;OOV&gt; &lt;OOV&gt; </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> the rest </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the cast </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> terrible </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> flat flat flat </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> could have allowed this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> almost seemed to </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> this wasn&#x27;t going to </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> out </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> &lt;OOV&gt; so </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt;</span>
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
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">give</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> &lt;OOV&gt; </span><span style="opacity: 0.80">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">the</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the </span><span style="opacity: 0.80">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">flat</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">flat</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">could</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">have</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">allowed</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">almost</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">seemed</span><span style="opacity: 0.80"> to </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">this</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">wasn&#x27;t</span><span style="opacity: 0.80"> going </span><span style="opacity: 0.80">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">give</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> &lt;PAD&gt;</span>
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
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">give</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">the</span><span style="opacity: 0.80"> rest </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the </span><span style="opacity: 0.80">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">flat</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">flat</span><span style="opacity: 0.80"> flat </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">could</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">have</span><span style="opacity: 0.80"> allowed </span><span style="opacity: 0.80">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">almost</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">seemed</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">this</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">wasn&#x27;t</span><span style="opacity: 0.80"> going to </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> out </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">give</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> &lt;PAD&gt; </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> &lt;PAD&gt; </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> &lt;PAD&gt; &lt;PAD&gt; </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="opacity: 0.80">&lt;PAD&gt;</span><span style="opacity: 0.80"> &lt;PAD&gt;</span>
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
            <span style="opacity: 0.80">&lt;START&gt; </span><span style="background-color: hsl(120, 100.00%, 84.46%); opacity: 0.85" title="0.003">please</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> &lt;OOV&gt; &lt;OOV&gt; </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> the rest </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.002">of</span><span style="opacity: 0.80"> the cast </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.011">rendered</span><span style="opacity: 0.80"> terrible </span><span style="background-color: hsl(120, 100.00%, 87.85%); opacity: 0.84" title="0.002">performances</span><span style="opacity: 0.80"> the </span><span style="background-color: hsl(120, 100.00%, 78.91%); opacity: 0.88" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.08%); opacity: 0.85" title="0.003">is</span><span style="opacity: 0.80"> flat flat flat </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.10%); opacity: 0.82" title="0.001">i</span><span style="opacity: 0.80"> don&#x27;t </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.001">how</span><span style="opacity: 0.80"> michael </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> could have allowed this </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.15%); opacity: 0.84" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.71%); opacity: 0.82" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.49%); opacity: 0.84" title="0.002">he</span><span style="opacity: 0.80"> almost seemed to </span><span style="background-color: hsl(120, 100.00%, 84.91%); opacity: 0.85" title="0.003">know</span><span style="opacity: 0.80"> this wasn&#x27;t going to </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">work</span><span style="opacity: 0.80"> out </span><span style="background-color: hsl(120, 100.00%, 86.06%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.09%); opacity: 0.81" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 64.05%); opacity: 0.97" title="0.009">performance</span><span style="opacity: 0.80"> was </span><span style="background-color: hsl(120, 100.00%, 83.77%); opacity: 0.86" title="0.003">quite</span><span style="opacity: 0.80"> &lt;OOV&gt; so </span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.22%); opacity: 0.91" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.73%); opacity: 0.86" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.31%); opacity: 0.91" title="0.006">fans</span><span style="opacity: 0.80"> give this </span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.25%); opacity: 0.82" title="0.001">miss</span><span style="opacity: 0.80"> &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt;</span>
        </p>
    
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    
    



.. parsed-literal::

    next


The results are roughly the same and usually you do not need to modify
this argument. Nevertheless, you can try if the overlay seems off.

How it works - ``explain_prediction()`` and ``format_as_html()``.
-----------------------------------------------------------------

What we have seen so far is calls to ``show_prediction()``. What this
function actually does is call ``explain_prediction()`` to produce an
``Explanation`` object, and then passes that object to
``format_as_html()`` to produce highlighted HTML.

Let's check each of these steps

.. code:: ipython3

    E = eli5.explain_prediction(binary_model, review, tokens=review_t)

This is an ``Explanation`` object

.. code:: ipython3

    repr(E)

We can get the predicted class and the value for the prediction

.. code:: ipython3

    target = E.targets[0]
    print(target.target, target.score)

We can also check the produced Grad-CAM ``heatmap`` found on each item
in ``targets``. You can think of this as an array of "importances" for
tokens (after padding is removed and the heatmap is resized).

.. code:: ipython3

    heatmap = target.heatmap
    print(heatmap)
    print(len(heatmap))

The highlighting for each token is stored in a ``WeightedSpans`` object
(specifically the ``DocWeightedSpans`` object)

.. code:: ipython3

    weighted_spans = target.weighted_spans
    print(weighted_spans)
    
    doc_ws = weighted_spans.docs_weighted_spans[0]
    print(doc_ws)

Observe the ``document`` attribute and ``spans``

.. code:: ipython3

    print(doc_ws.document)
    print(doc_ws.spans)

The ``document`` is the "stringified" version of ``tokens``. If you have
a custom "tokens -> string" algorithm you may want to set this attribute
yourself.

The ``spans`` object is a list of weights for each character in
``document``. We use the indices in ``document`` string to indicate
which characters should be weighted with a specific value.

Let's format this. HTML formatter is what should be used here.

.. code:: ipython3

    import eli5.formatters.fields as fields
    F = eli5.format_as_html(E, show=fields.WEIGHTS)

We pass a ``show`` argument to not display the method name or its
description ("Grad-CAM"). See ``eli5.format_as_html()`` for a list of
all supported arguments.

The output is an HTML-encoded string.

.. code:: ipython3

    repr(F)

Convert the string to an HTML object and display it in an IPython
notebook

.. code:: ipython3

    display(HTML(F))

Notes on results
----------------

In general, this is experimental work. Unlike for images, there is not
much talk about Grad-CAM applied to text.

``layer`` is probably a very important argument as we currently use
basic heuristics to pick a suitable layer. Thus explanations may not
look as good for your own model.

Multi-label classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Did not really work for us. Got non-sensical explanations. Send comment
if can do it.
