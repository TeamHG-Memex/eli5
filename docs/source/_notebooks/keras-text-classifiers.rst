Explaining Keras text classifier predictions with Grad-CAM
==========================================================

We can use ELI5 to explain text-based classifiers, i.e. models that take
in a text and assign it to some class. Common examples include sentiment
classification, labelling into categories, etc.

The underlying method used is ‘Grad-CAM’
(https://arxiv.org/abs/1610.02391). This technique shows what parts of
the input are the most important to the predicted result, by overlaying
a “heatmap” over the original input.

See also the tutorial for images
(https://eli5.readthedocs.io/en/latest/tutorials/keras-image-classifiers.html).
Certain sections such as ‘removing softmax’ and ‘comparing different
models’ are relevant for text as well.

**This is experimental work. Unlike for images, this is not based on any
paper.**

Set up
------

First some imports

.. code:: ipython3

    import os
    import os.path
    import sys
    # access packages in top level eli5 directory
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    
    import logging
    import warnings
    
    import numpy as np
    import pandas as pd
    from IPython.display import display, HTML  # our explanations will be formatted in HTML
    
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR) # disable Tensorflow warnings for this tutorial
    
    import keras
    warnings.simplefilter("ignore") # disable Keras warnings for this tutorial
    from keras.preprocessing.sequence import pad_sequences
    
    import eli5


.. code:: ipython3

    # for reproducibility, the tutorial was ran with the following versions
    print('python', sys.version_info)
    print('keras', keras.__version__)
    print('tensorflow', tf.__version__)
    print('numpy', np.__version__)
    print('pandas', pd.__version__)


.. parsed-literal::

    python sys.version_info(major=3, minor=7, micro=7, releaselevel='final', serial=0)
    keras 2.2.5
    tensorflow 1.14.0
    numpy 1.19.0
    pandas 1.0.5


The rest of what we need in this tutorial is stored in the
``tests/estimators`` package, whose source you can check for your own
reference. You may need extra steps here to load your custom model and
data.

Explaining binary (sentiment) classifications
---------------------------------------------

In binary classification there is only one possible class to which a
piece of text can either belong to or not. In sentiment classification,
that class is whether the text is “positive” (belongs to the class) or
“negative” (doesn’t belong to the class).

In this example we will have a recurrent model with word level
tokenization, trained on the IMDB dataset
(https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification).
The model has one output node that gives probabilities. Output close to
1 is positive, and close to 0 is negative.

See
https://www.tensorflow.org/beta/tutorials/text/text_classification_rnn
for a simple example of how to build such a model and prepare its input.

For exact details of how we trained our model and what data we used see
https://www.kaggle.com/tobalt/keras-text-model-sentiment or the
``tests/estimators/keras_sentiment_classifier/keras_sentiment_classifier.ipynb``
file in the ELI5 repo.

.. code:: ipython3

    import tests.estimators.keras_sentiment_classifier.keras_sentiment_classifier \
        as keras_sentiment_classifier

Let’s load our pre-trained model

.. code:: ipython3

    binary_model = keras.models.load_model(keras_sentiment_classifier.MODEL)
    binary_model.summary()


.. parsed-literal::

    Model: "sequential_1"
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

    (x_train, y_train), (x_test, y_test) = keras_sentiment_classifier.prepare_train_test_dataset()

Confirm the accuracy of the model

.. code:: ipython3

    print(binary_model.metrics_names)
    loss, acc = binary_model.evaluate(x_test, y_test)
    print(loss, acc)
    
    print('Accuracy: ', acc)


.. parsed-literal::

    ['loss', 'acc']
    25000/25000 [==============================] - 43s 2ms/step
    0.4319177031707764 0.81504
    Accuracy:  0.81504


Looks good? Let’s go on and check one of the test samples.

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



As expected, looks pretty low score.

Now let’s explain what got us this result with ELI5. We need to pass the
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
            <span style="opacity: 0.80">&lt;START&gt; please give this one a miss br br &lt;OOV&gt; &lt;OOV&gt; and the rest of the cast rendered terrible performances the show is flat flat flat br br i don&#x27;t know how michael madison could have allowed this one on his plate he almost seemed to know this wasn&#x27;t going to work out and his performance was quite &lt;OOV&gt; so all you madison fans give this a miss &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt;</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




That’s unexpected. The input seems to have nothing that makes the
predicted score *go up*. (See the next section for an explanation.)

Let’s try a custom input

.. code:: ipython3

    s = 'hello this is great but not so great'
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
            <span style="background-color: hsl(120, 100.00%, 62.45%); opacity: 0.98" title="0.047">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.69%); opacity: 1.00" title="0.050">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.25%); opacity: 1.00" title="0.051">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.052">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.26%); opacity: 0.89" title="0.024">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 67.04%); opacity: 0.95" title="0.039">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.83%); opacity: 0.93" title="0.033">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.37%); opacity: 0.91" title="0.029">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.66%); opacity: 0.85" title="0.012">great</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




Now this is something. The words highlighted in green show what makes
the score “go up”, i.e. the “positive” words (check the next section to
see how to show positive AND negative words with the ``relu`` argument).

Even though the explanation is bright green, the actual prediction is
not very positive. Hover over the highlighted words to see their actual
“weight”.

Modify explanations with the ``relu`` and ``counterfactual`` arguments
----------------------------------------------------------------------

In the last section we only saw the “positive” words in our input, what
made the class score “go up”. To “fix” this and see the “negative” words
too, we can pass two boolean arguments.

``relu`` (default ``True``) only shows what makes the predicted score go
up and discards the effect of counter-evidence or other classes in case
of multiclass classification (set to ``False`` to disable). Under the
hood, this discards negative gradients / negative pixels (which are
likely to belong to other classes according to the Grad-CAM paper
(https://arxiv.org/abs/1610.02391)).

.. code:: ipython3

    eli5.show_prediction(binary_model, review, tokens=review_t, relu=False)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 62.45%); opacity: 0.98" title="0.047">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.69%); opacity: 1.00" title="0.050">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.25%); opacity: 1.00" title="0.051">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.052">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.26%); opacity: 0.89" title="0.024">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 67.04%); opacity: 0.95" title="0.039">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.83%); opacity: 0.93" title="0.033">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.37%); opacity: 0.91" title="0.029">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.66%); opacity: 0.85" title="0.012">great</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




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
            <span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.78%); opacity: 0.81" title="-0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.92%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.79%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.06%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.64%); opacity: 0.83" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.19%); opacity: 0.83" title="-0.001">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.70%); opacity: 0.89" title="-0.004">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.28%); opacity: 0.89" title="-0.004">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.26%); opacity: 0.83" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.11%); opacity: 0.91" title="-0.005">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 72.53%); opacity: 0.92" title="-0.006">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.52%); opacity: 0.94" title="-0.007">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.62%); opacity: 0.93" title="-0.006">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.26%); opacity: 0.84" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.20%); opacity: 0.93" title="-0.006">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.33%); opacity: 0.94" title="-0.006">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.08%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.30%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.91%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.35%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.07%); opacity: 0.91" title="-0.005">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.81%); opacity: 0.90" title="-0.005">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.35%); opacity: 0.85" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.37%); opacity: 0.89" title="-0.004">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.91%); opacity: 0.89" title="-0.004">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.55%); opacity: 0.84" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.63%); opacity: 0.89" title="-0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.30%); opacity: 0.88" title="-0.004">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.01%); opacity: 0.88" title="-0.004">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.57%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.76%); opacity: 0.83" title="-0.002">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.36%); opacity: 0.86" title="-0.003">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.65%); opacity: 0.83" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.34%); opacity: 0.87" title="-0.003">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.38%); opacity: 0.87" title="-0.003">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.59%); opacity: 0.88" title="-0.004">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.86%); opacity: 0.88" title="-0.004">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.83%); opacity: 0.84" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.72%); opacity: 0.88" title="-0.004">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.55%); opacity: 0.87" title="-0.003">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.32%); opacity: 0.83" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.43%); opacity: 0.87" title="-0.003">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.89%); opacity: 0.87" title="-0.003">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.99%); opacity: 0.84" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.03%); opacity: 0.84" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.29%); opacity: 0.88" title="-0.004">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.41%); opacity: 0.90" title="-0.005">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.69%); opacity: 0.91" title="-0.005">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.69%); opacity: 0.84" title="-0.002">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.50%); opacity: 0.94" title="-0.006">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 67.84%); opacity: 0.95" title="-0.007">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.70%); opacity: 0.96" title="-0.008">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.26%); opacity: 0.89" title="-0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.009">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.31%); opacity: 0.91" title="-0.005">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.62%); opacity: 0.86" title="-0.003">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.33%); opacity: 0.93" title="-0.006">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.84%); opacity: 0.94" title="-0.007">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.83%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.70%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.46%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.34%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.23%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.13%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.03%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.83%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.74%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.65%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.56%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.47%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.38%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.30%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.21%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.13%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.05%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.96%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.72%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.64%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.57%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.49%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.41%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.34%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.26%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.19%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.12%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.05%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.91%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.85%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.78%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.72%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.67%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.61%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.56%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.51%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.46%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.41%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.37%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.33%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.30%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.27%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.24%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.22%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.21%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.21%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.23%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.27%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.33%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.45%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.64%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.94%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.43%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.18%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




Green is positive, red is negative, white is neutral.

We can see what made the network decide that is is a negative example,
and why in the previous section there were no highlighted words
(according to the explanation there are no positive words).

Another argument ``counterfactual`` (default ``False``) highlights the
counter-evidence for a class, or what makes the score “go down” (set to
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
            <span style="background-color: hsl(120, 100.00%, 97.48%); opacity: 0.80" title="0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.03%); opacity: 0.80" title="0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.92%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.51%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.53%); opacity: 0.81" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.79%); opacity: 0.81" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.06%); opacity: 0.81" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.96%); opacity: 0.82" title="0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.13%); opacity: 0.82" title="0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.001">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.70%); opacity: 0.89" title="0.004">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.28%); opacity: 0.89" title="0.004">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.26%); opacity: 0.83" title="0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.11%); opacity: 0.91" title="0.005">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 72.53%); opacity: 0.92" title="0.006">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 68.52%); opacity: 0.94" title="0.007">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.62%); opacity: 0.93" title="0.006">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.26%); opacity: 0.84" title="0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.20%); opacity: 0.93" title="0.006">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 69.33%); opacity: 0.94" title="0.006">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.08%); opacity: 0.85" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.30%); opacity: 0.85" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.35%); opacity: 0.83" title="0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.82%); opacity: 0.83" title="0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.07%); opacity: 0.91" title="0.005">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.81%); opacity: 0.90" title="0.005">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.35%); opacity: 0.85" title="0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.37%); opacity: 0.89" title="0.004">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.91%); opacity: 0.89" title="0.004">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.55%); opacity: 0.84" title="0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.63%); opacity: 0.89" title="0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.30%); opacity: 0.88" title="0.004">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.01%); opacity: 0.88" title="0.004">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.57%); opacity: 0.82" title="0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.76%); opacity: 0.83" title="0.002">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.36%); opacity: 0.86" title="0.003">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.65%); opacity: 0.83" title="0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.34%); opacity: 0.87" title="0.003">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.38%); opacity: 0.87" title="0.003">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.59%); opacity: 0.88" title="0.004">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.86%); opacity: 0.88" title="0.004">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.83%); opacity: 0.84" title="0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.13%); opacity: 0.84" title="0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.72%); opacity: 0.88" title="0.004">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.55%); opacity: 0.87" title="0.003">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.32%); opacity: 0.83" title="0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.43%); opacity: 0.87" title="0.003">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.89%); opacity: 0.87" title="0.003">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.99%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.03%); opacity: 0.84" title="0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.29%); opacity: 0.88" title="0.004">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.41%); opacity: 0.90" title="0.005">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.69%); opacity: 0.91" title="0.005">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.69%); opacity: 0.84" title="0.002">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 69.50%); opacity: 0.94" title="0.006">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 67.84%); opacity: 0.95" title="0.007">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 65.70%); opacity: 0.96" title="0.008">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.26%); opacity: 0.89" title="0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.009">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.31%); opacity: 0.91" title="0.005">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.62%); opacity: 0.86" title="0.003">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.33%); opacity: 0.93" title="0.006">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 68.84%); opacity: 0.94" title="0.007">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.97%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.83%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.70%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.58%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.46%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.34%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.23%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.03%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.93%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.74%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.65%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.56%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.47%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.38%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.30%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.21%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.13%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.05%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.88%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.80%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.57%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.49%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.41%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.34%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.26%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.19%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.12%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.98%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.91%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.85%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.78%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.72%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.61%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.51%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.46%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.41%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.37%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.33%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.30%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.27%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.24%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.22%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.23%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.27%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.33%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.45%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.64%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.94%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.18%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




This shows the “negative” words in green, i.e. inverts the classes.

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
            <span style="background-color: hsl(120, 100.00%, 97.48%); opacity: 0.80" title="0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.03%); opacity: 0.80" title="0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.92%); opacity: 0.80" title="0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.51%); opacity: 0.81" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.53%); opacity: 0.81" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.79%); opacity: 0.81" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.06%); opacity: 0.81" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.96%); opacity: 0.82" title="0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.13%); opacity: 0.82" title="0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.64%); opacity: 0.83" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.19%); opacity: 0.83" title="0.001">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.70%); opacity: 0.89" title="0.004">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.28%); opacity: 0.89" title="0.004">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.26%); opacity: 0.83" title="0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.11%); opacity: 0.91" title="0.005">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 72.53%); opacity: 0.92" title="0.006">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 68.52%); opacity: 0.94" title="0.007">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.62%); opacity: 0.93" title="0.006">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.26%); opacity: 0.84" title="0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.20%); opacity: 0.93" title="0.006">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 69.33%); opacity: 0.94" title="0.006">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.08%); opacity: 0.85" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.30%); opacity: 0.85" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.91%); opacity: 0.85" title="0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.35%); opacity: 0.83" title="0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.82%); opacity: 0.83" title="0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.07%); opacity: 0.91" title="0.005">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 74.81%); opacity: 0.90" title="0.005">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.35%); opacity: 0.85" title="0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.37%); opacity: 0.89" title="0.004">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.91%); opacity: 0.89" title="0.004">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.55%); opacity: 0.84" title="0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.63%); opacity: 0.89" title="0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.30%); opacity: 0.88" title="0.004">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.01%); opacity: 0.88" title="0.004">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.57%); opacity: 0.82" title="0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.76%); opacity: 0.83" title="0.002">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.36%); opacity: 0.86" title="0.003">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.65%); opacity: 0.83" title="0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.34%); opacity: 0.87" title="0.003">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.38%); opacity: 0.87" title="0.003">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.59%); opacity: 0.88" title="0.004">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.86%); opacity: 0.88" title="0.004">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.83%); opacity: 0.84" title="0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.13%); opacity: 0.84" title="0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.41%); opacity: 0.82" title="0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.72%); opacity: 0.88" title="0.004">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.55%); opacity: 0.87" title="0.003">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.32%); opacity: 0.83" title="0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.43%); opacity: 0.87" title="0.003">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 80.89%); opacity: 0.87" title="0.003">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.99%); opacity: 0.84" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 87.03%); opacity: 0.84" title="0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.29%); opacity: 0.88" title="0.004">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.41%); opacity: 0.90" title="0.005">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.69%); opacity: 0.91" title="0.005">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.69%); opacity: 0.84" title="0.002">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 69.50%); opacity: 0.94" title="0.006">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 67.84%); opacity: 0.95" title="0.007">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 65.70%); opacity: 0.96" title="0.008">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 77.26%); opacity: 0.89" title="0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.009">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.31%); opacity: 0.91" title="0.005">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 82.62%); opacity: 0.86" title="0.003">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.33%); opacity: 0.93" title="0.006">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 68.84%); opacity: 0.94" title="0.007">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.97%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.83%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.70%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.58%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.46%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.34%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.23%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.03%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.93%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.83%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.74%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.65%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.56%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.47%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.38%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.30%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.21%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.13%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.05%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.88%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.80%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.64%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.57%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.49%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.41%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.34%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.26%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.19%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.12%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.05%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.98%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.91%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.85%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.78%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.72%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.61%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.51%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.46%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.41%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.37%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.33%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.30%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.27%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.24%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.22%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.21%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.23%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.27%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.33%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.45%); opacity: 0.82" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.64%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.94%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.43%); opacity: 0.81" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.18%); opacity: 0.81" title="0.000">&lt;PAD&gt;</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




We do not see it in this example, but the colors (green and red) should
be inverted.

Choosing a hidden layer to do Grad-CAM on with ``layer``
--------------------------------------------------------

Grad-CAM requires a hidden layer to do its calculations on and produce a
heatmap. This is controlled by the ``layer`` argument. We can pass the
layer as an int index, string name, or a keras Layer instance.

If you do not pass ``layer``, ELI5 tries to find a good layer for us
automatically. By default we search backwards through the flattened list
of layers, starting from the output layer, for a layer such as
Convolutional or Recurrent. Searching backwards means that we follow the
Grad-CAM method closer, as opposed to some other Gradient-based method.

However, note that we found that using lower layers (closer to the
input) for word-level tokenization text models gave better results.

If you don’t get good explanations from ELI5 out of the box, it may be
worth looking into this parameter. We advice to pick layers that contain
“spatial or temporal” information, i.e. NOT dense/fully-connected or
merge layers, but recurrent, convolutional, or embedding layers.

.. code:: ipython3

    layer = 'embedding_1'
    print(layer)
    display(eli5.show_prediction(binary_model, review, tokens=review_t, relu=False, layer=layer))
    
    layer = 'bidirectional_1'
    print(layer)
    display(eli5.show_prediction(binary_model, review, tokens=review_t, relu=False, layer=layer))
    
    layer = 'bidirectional_2'
    print(layer)
    display(eli5.show_prediction(binary_model, review, tokens=review_t, relu=False, layer=layer))
    
    layer = 'bidirectional_3'
    print(layer)
    display(eli5.show_prediction(binary_model, review, tokens=review_t, relu=False, layer=layer))


.. parsed-literal::

    embedding_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 77.80%); opacity: 0.89" title="-0.010">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.46%); opacity: 0.81" title="0.001">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.81%); opacity: 0.84" title="-0.005">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.23%); opacity: 0.93" title="0.015">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.58%); opacity: 0.85" title="-0.005">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.79%); opacity: 0.96" title="-0.018">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.60%); opacity: 0.83" title="-0.004">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.023">great</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    bidirectional_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 75.25%); opacity: 0.90" title="0.010">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 69.00%); opacity: 0.94" title="0.013">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 68.92%); opacity: 0.94" title="0.013">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.019">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 69.77%); opacity: 0.93" title="0.013">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 68.07%); opacity: 0.94" title="0.014">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 69.91%); opacity: 0.93" title="0.013">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 61.29%); opacity: 0.99" title="0.018">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 66.59%); opacity: 0.95" title="0.015">great</span>
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
            <span style="background-color: hsl(120, 100.00%, 62.45%); opacity: 0.98" title="0.047">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.69%); opacity: 1.00" title="0.050">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.25%); opacity: 1.00" title="0.051">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.052">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.26%); opacity: 0.89" title="0.024">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 67.04%); opacity: 0.95" title="0.039">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.83%); opacity: 0.93" title="0.033">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.37%); opacity: 0.91" title="0.029">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.66%); opacity: 0.85" title="0.012">great</span>
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
            <span style="background-color: hsl(120, 100.00%, 78.21%); opacity: 0.88" title="0.015">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.42%); opacity: 0.83" title="-0.005">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.45%); opacity: 0.85" title="-0.008">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.85%); opacity: 0.80" title="-0.001">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.001">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.32%); opacity: 1.00" title="0.034">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.53%); opacity: 0.89" title="0.016">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.035">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.04%); opacity: 0.81" title="-0.002">great</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



If the ``eli5.show_prediction()`` call is not the last statement in a
cell (or if it is in a loop), you can use the IPython ``display()`` to
force-show the explanation.

The test sample

.. code:: ipython3

    layer = 'embedding_1'
    print(layer)
    display(eli5.show_prediction(binary_model, test_review, tokens=test_review_t, relu=False, layer=layer))
    
    layer = 'bidirectional_1'
    print(layer)
    display(eli5.show_prediction(binary_model, test_review, tokens=test_review_t, relu=False, layer=layer))
    
    layer = 'bidirectional_2'
    print(layer)
    display(eli5.show_prediction(binary_model, test_review, tokens=test_review_t, relu=False, layer=layer))
    
    layer = 'bidirectional_3'
    print(layer)
    display(eli5.show_prediction(binary_model, test_review, tokens=test_review_t, relu=False, layer=layer))


.. parsed-literal::

    embedding_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 90.36%); opacity: 0.83" title="-0.003">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.77%); opacity: 0.83" title="0.003">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.87%); opacity: 0.86" title="-0.006">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.002">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.46%); opacity: 0.82" title="-0.002">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.66%); opacity: 0.91" title="0.011">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.020">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.002">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 86.12%); opacity: 0.84" title="0.004">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.52%); opacity: 0.83" title="0.003">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.15%); opacity: 0.82" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.80%); opacity: 0.81" title="0.001">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.05%); opacity: 0.82" title="-0.002">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.92%); opacity: 0.81" title="0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.80%); opacity: 0.80" title="-0.000">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.09%); opacity: 0.82" title="-0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.51%); opacity: 0.83" title="-0.003">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.88%); opacity: 0.82" title="0.002">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.001">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.11%); opacity: 0.82" title="0.002">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.57%); opacity: 0.80" title="-0.000">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.39%); opacity: 0.89" title="-0.009">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.07%); opacity: 0.83" title="0.003">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.38%); opacity: 0.85" title="-0.005">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.90%); opacity: 0.81" title="-0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.15%); opacity: 0.81" title="-0.001">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.73%); opacity: 0.82" title="0.002">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.55%); opacity: 0.80" title="-0.000">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.82%); opacity: 0.82" title="0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.33%); opacity: 0.89" title="0.009">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.49%); opacity: 0.81" title="-0.001">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.31%); opacity: 0.83" title="0.003">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.36%); opacity: 0.80" title="-0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.97%); opacity: 0.82" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.001">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.02%); opacity: 0.86" title="0.006">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.003">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.09%); opacity: 0.86" title="0.006">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.90%); opacity: 0.81" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.06%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    bidirectional_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 96.21%); opacity: 0.81" title="-0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.23%); opacity: 0.82" title="0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.56%); opacity: 0.81" title="-0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.90%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.85%); opacity: 0.82" title="0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.19%); opacity: 0.82" title="0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.75%); opacity: 0.82" title="0.000">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.24%); opacity: 0.81" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.82%); opacity: 0.81" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.000">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.91%); opacity: 0.82" title="0.000">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.72%); opacity: 0.84" title="-0.000">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.34%); opacity: 0.83" title="0.000">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.87%); opacity: 0.81" title="-0.000">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.36%); opacity: 0.89" title="0.001">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.22%); opacity: 0.96" title="-0.002">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 93.48%); opacity: 0.81" title="0.000">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.28%); opacity: 0.81" title="0.000">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 94.08%); opacity: 0.81" title="0.000">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.40%); opacity: 0.83" title="-0.000">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.25%); opacity: 0.82" title="-0.000">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.94%); opacity: 0.82" title="-0.000">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 98.24%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.29%); opacity: 0.81" title="-0.000">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.94%); opacity: 0.84" title="-0.001">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.50%); opacity: 0.81" title="-0.000">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.22%); opacity: 0.86" title="-0.001">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.98%); opacity: 0.88" title="-0.001">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.33%); opacity: 0.81" title="-0.000">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.62%); opacity: 0.90" title="-0.001">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.59%); opacity: 0.88" title="-0.001">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.53%); opacity: 0.89" title="-0.001">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.43%); opacity: 0.82" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.81%); opacity: 0.81" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.35%); opacity: 0.82" title="0.000">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.63%); opacity: 0.81" title="-0.000">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.83%); opacity: 0.81" title="-0.000">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.23%); opacity: 0.84" title="-0.001">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.80%); opacity: 0.85" title="-0.001">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.58%); opacity: 0.90" title="-0.001">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.31%); opacity: 0.81" title="-0.000">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.99%); opacity: 0.81" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.32%); opacity: 0.88" title="-0.001">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.28%); opacity: 0.86" title="-0.001">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.36%); opacity: 0.83" title="-0.000">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.38%); opacity: 0.83" title="-0.000">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.05%); opacity: 0.86" title="-0.001">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.68%); opacity: 0.82" title="-0.000">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.19%); opacity: 0.86" title="-0.001">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.14%); opacity: 0.85" title="-0.001">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.74%); opacity: 0.93" title="-0.002">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.77%); opacity: 0.91" title="-0.001">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.68%); opacity: 0.85" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 67.28%); opacity: 0.95" title="-0.002">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.63%); opacity: 0.93" title="-0.002">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.10%); opacity: 0.93" title="-0.002">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.07%); opacity: 0.88" title="-0.001">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.003">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.37%); opacity: 0.94" title="-0.002">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.62%); opacity: 0.87" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 71.47%); opacity: 0.92" title="-0.002">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.54%); opacity: 0.93" title="-0.002">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.88%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.69%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.49%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.28%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.07%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.86%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.66%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.45%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.24%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.03%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.83%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.62%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.42%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.21%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.01%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.81%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.61%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.41%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.21%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.01%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.81%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.62%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.42%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.23%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.04%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.85%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.66%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.47%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.29%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.10%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.92%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.75%); opacity: 0.82" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.57%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.40%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.23%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.06%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.90%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.74%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.58%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.43%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.28%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.14%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.00%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.86%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.73%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.61%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.48%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.37%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.25%); opacity: 0.83" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.15%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.04%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.95%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.86%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.77%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.69%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.62%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.55%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.50%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.45%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.40%); opacity: 0.84" title="-0.000">&lt;PAD&gt;</span>
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
            <span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.78%); opacity: 0.81" title="-0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.92%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.79%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.06%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.64%); opacity: 0.83" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.19%); opacity: 0.83" title="-0.001">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.70%); opacity: 0.89" title="-0.004">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.28%); opacity: 0.89" title="-0.004">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.26%); opacity: 0.83" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.11%); opacity: 0.91" title="-0.005">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 72.53%); opacity: 0.92" title="-0.006">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.52%); opacity: 0.94" title="-0.007">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.62%); opacity: 0.93" title="-0.006">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.26%); opacity: 0.84" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.20%); opacity: 0.93" title="-0.006">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.33%); opacity: 0.94" title="-0.006">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.08%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.30%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.91%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.35%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.07%); opacity: 0.91" title="-0.005">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.81%); opacity: 0.90" title="-0.005">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.35%); opacity: 0.85" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.37%); opacity: 0.89" title="-0.004">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.91%); opacity: 0.89" title="-0.004">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.55%); opacity: 0.84" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.63%); opacity: 0.89" title="-0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.30%); opacity: 0.88" title="-0.004">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.01%); opacity: 0.88" title="-0.004">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.57%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.76%); opacity: 0.83" title="-0.002">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.36%); opacity: 0.86" title="-0.003">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.65%); opacity: 0.83" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.34%); opacity: 0.87" title="-0.003">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.38%); opacity: 0.87" title="-0.003">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.59%); opacity: 0.88" title="-0.004">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.86%); opacity: 0.88" title="-0.004">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.83%); opacity: 0.84" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.72%); opacity: 0.88" title="-0.004">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.55%); opacity: 0.87" title="-0.003">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.32%); opacity: 0.83" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.43%); opacity: 0.87" title="-0.003">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.89%); opacity: 0.87" title="-0.003">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.99%); opacity: 0.84" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.03%); opacity: 0.84" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.29%); opacity: 0.88" title="-0.004">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.41%); opacity: 0.90" title="-0.005">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.69%); opacity: 0.91" title="-0.005">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.69%); opacity: 0.84" title="-0.002">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.50%); opacity: 0.94" title="-0.006">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 67.84%); opacity: 0.95" title="-0.007">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.70%); opacity: 0.96" title="-0.008">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.26%); opacity: 0.89" title="-0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.009">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.31%); opacity: 0.91" title="-0.005">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.62%); opacity: 0.86" title="-0.003">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.33%); opacity: 0.93" title="-0.006">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.84%); opacity: 0.94" title="-0.007">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.97%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.83%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.70%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.58%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.46%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.34%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.23%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.13%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.03%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.93%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.83%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.74%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.65%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.56%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.47%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.38%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.30%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.21%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.13%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.05%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.96%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.88%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.80%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.72%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.64%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.57%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.49%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.41%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.34%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.26%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.19%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.12%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.05%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.98%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.91%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.85%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.78%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.72%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.67%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.61%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.56%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.51%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.46%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.41%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.37%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.33%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.30%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.27%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.24%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.22%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.21%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.21%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.23%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.27%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.33%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.45%); opacity: 0.82" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.64%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.94%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.43%); opacity: 0.81" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.18%); opacity: 0.81" title="-0.000">&lt;PAD&gt;</span>
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
            <span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.162">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.27%); opacity: 1.00" title="-0.161">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.71%); opacity: 0.90" title="-0.080">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.15%); opacity: 0.85" title="-0.039">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.04%); opacity: 0.90" title="-0.078">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.28%); opacity: 0.89" title="-0.072">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.66%); opacity: 0.88" title="-0.066">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.74%); opacity: 0.84" title="-0.030">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.65%); opacity: 0.83" title="-0.027">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.27%); opacity: 0.83" title="-0.028">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.89%); opacity: 0.82" title="-0.020">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 91.87%); opacity: 0.82" title="-0.017">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.78%); opacity: 0.80" title="-0.003">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.60%); opacity: 0.80" title="-0.003">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.81%); opacity: 0.81" title="-0.006">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.36%); opacity: 0.80" title="-0.003">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.97%); opacity: 0.82" title="-0.014">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.39%); opacity: 0.84" title="-0.035">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.41%); opacity: 0.88" title="-0.067">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 71.57%); opacity: 0.92" title="-0.100">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.95%); opacity: 0.85" title="-0.044">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 64.37%); opacity: 0.97" title="-0.137">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.04%); opacity: 0.95" title="-0.118">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.98%); opacity: 0.84" title="-0.033">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.88%); opacity: 0.83" title="-0.026">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.44%); opacity: 0.83" title="-0.024">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.21%); opacity: 0.83" title="-0.022">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.11%); opacity: 0.83" title="-0.025">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.49%); opacity: 0.94" title="-0.115">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.36%); opacity: 0.94" title="-0.116">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.93%); opacity: 0.85" title="-0.044">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.01%); opacity: 0.87" title="-0.060">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.07%); opacity: 0.84" title="-0.032">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.99%); opacity: 0.80" title="-0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.19%); opacity: 0.80" title="0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 92.31%); opacity: 0.82" title="0.015">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.57%); opacity: 0.83" title="0.027">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 95.12%); opacity: 0.81" title="0.008">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.004">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.96%); opacity: 0.86" title="-0.048">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.95%); opacity: 0.85" title="-0.044">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 66.28%); opacity: 0.96" title="-0.127">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 64.29%); opacity: 0.97" title="-0.138">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 62.35%); opacity: 0.98" title="-0.149">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.45%); opacity: 1.00" title="-0.160">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.51%); opacity: 0.91" title="-0.085">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.04%); opacity: 0.90" title="-0.083">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.04%); opacity: 0.85" title="-0.040">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 61.59%); opacity: 0.99" title="-0.153">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 62.68%); opacity: 0.98" title="-0.147">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.23%); opacity: 0.88" title="-0.064">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.50%); opacity: 0.93" title="-0.105">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.03%); opacity: 0.90" title="-0.083">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.65%); opacity: 0.84" title="-0.030">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.44%); opacity: 0.83" title="-0.024">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.02%); opacity: 0.85" title="-0.040">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.35%); opacity: 0.84" title="-0.031">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.88%); opacity: 0.83" title="-0.023">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.11%); opacity: 0.81" title="-0.006">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.87%); opacity: 0.82" title="-0.014">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.23%); opacity: 0.81" title="-0.010">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.75%); opacity: 0.81" title="-0.007">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.04%); opacity: 0.80" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.38%); opacity: 0.80" title="-0.003">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.78%); opacity: 0.80" title="-0.001">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.53%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.63%); opacity: 0.80" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.88%); opacity: 0.80" title="-0.000">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.88%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.66%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.48%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.33%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.25%); opacity: 0.80" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.34%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.44%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.54%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.65%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.73%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.82%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.94%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.95%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.97%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.89%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.90%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.92%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.96%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.93%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.87%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.85%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.87%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.90%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.93%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.92%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.85%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.80%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.75%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.68%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.55%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.44%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.33%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.27%); opacity: 0.80" title="-0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.39%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.51%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.66%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.81%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.81%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.81%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.80%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.80%); opacity: 0.80" title="-0.000">&lt;PAD&gt;</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



The last two dense layers

.. code:: ipython3

    layer = 'dense_1'
    print(layer)
    display(eli5.show_prediction(binary_model, test_review, tokens=test_review_t, relu=False, layer=layer))
    
    layer = 'dense_2'
    print(layer)
    display(eli5.show_prediction(binary_model, test_review, tokens=test_review_t, relu=False, layer=layer))


.. parsed-literal::

    dense_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.963">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 61.51%); opacity: 0.99" title="-0.911">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.25%); opacity: 0.89" title="-0.430">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.59%); opacity: 0.84" title="-0.202">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.20%); opacity: 0.88" title="-0.378">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.20%); opacity: 0.87" title="-0.352">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.23%); opacity: 0.87" title="-0.327">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.09%); opacity: 0.83" title="-0.150">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.75%); opacity: 0.83" title="-0.138">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.30%); opacity: 0.83" title="-0.166">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.16%); opacity: 0.83" title="-0.149">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.79%); opacity: 0.84" title="-0.198">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.98%); opacity: 0.82" title="-0.115">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.63%); opacity: 0.86" title="-0.292">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 84.84%); opacity: 0.85" title="-0.241">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.06%); opacity: 0.81" title="-0.063">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.75%); opacity: 0.83" title="-0.138">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.62%); opacity: 0.82" title="-0.086">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.10%); opacity: 0.81" title="-0.035">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.82%); opacity: 0.81" title="-0.026">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.58%); opacity: 0.80" title="-0.008">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.08%); opacity: 0.80" title="-0.023">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.21%); opacity: 0.80" title="-0.021">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.77%); opacity: 0.80" title="-0.007">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.84%); opacity: 0.80" title="-0.006">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.90%); opacity: 0.80" title="-0.006">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.16%); opacity: 0.80" title="-0.004">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.22%); opacity: 0.80" title="-0.003">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.10%); opacity: 0.80" title="-0.012">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.26%); opacity: 0.80" title="-0.011">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.03%); opacity: 0.80" title="-0.005">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.61%); opacity: 0.80" title="-0.008">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 98.80%); opacity: 0.80" title="-0.006">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.39%); opacity: 0.80" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.23%); opacity: 0.80" title="-0.003">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.48%); opacity: 0.80" title="-0.002">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 99.82%); opacity: 0.80" title="-0.000">allowed</span><span style="opacity: 0.80"> this one on his plate he almost seemed to know this wasn&#x27;t going to work out and his performance was quite &lt;OOV&gt; so all you madison fans give this a miss &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; </span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.001">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">&lt;PAD&gt;</span>
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
            <span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87" title="0.054">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.162">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 84.84%); opacity: 0.85" title="0.041">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.38%); opacity: 0.90" title="0.081">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">&lt;PAD&gt;</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



What’s up with the final dense layers? They do not have spatial
information so it’s mostly a visualization of the activations of each
node, ignoring the underlying tokens.

Removing padding with ``pad_value`` or ``pad_token`` arguments
--------------------------------------------------------------

When working with text, often sample input is padded or truncated to a
certain length, whether because the model only takes fixed-length input,
or because we want to put all the samples in a batch.

We can remove padding by specifying the value used for the padding
symbol. We can either specify ``pad_value``, a numeric value such as
``0`` for ``doc`` input, or ``pad_token``, the padding token such as
``<PAD>`` in ``tokens``.

.. code:: ipython3

    print(test_review_t)


.. parsed-literal::

    [['<START>', 'please', 'give', 'this', 'one', 'a', 'miss', 'br', 'br', '<OOV>', '<OOV>', 'and', 'the', 'rest', 'of', 'the', 'cast', 'rendered', 'terrible', 'performances', 'the', 'show', 'is', 'flat', 'flat', 'flat', 'br', 'br', 'i', "don't", 'know', 'how', 'michael', 'madison', 'could', 'have', 'allowed', 'this', 'one', 'on', 'his', 'plate', 'he', 'almost', 'seemed', 'to', 'know', 'this', "wasn't", 'going', 'to', 'work', 'out', 'and', 'his', 'performance', 'was', 'quite', '<OOV>', 'so', 'all', 'you', 'madison', 'fans', 'give', 'this', 'a', 'miss', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']]


Notice that the padding word used here is ``<PAD>`` and that it comes
after the text.

.. code:: ipython3

    eli5.show_prediction(binary_model, test_review, tokens=test_review_t, 
                        pad_token='<PAD>', relu=False)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.78%); opacity: 0.81" title="-0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.92%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.79%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.06%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.64%); opacity: 0.83" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.19%); opacity: 0.83" title="-0.001">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.70%); opacity: 0.89" title="-0.004">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.28%); opacity: 0.89" title="-0.004">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.26%); opacity: 0.83" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.11%); opacity: 0.91" title="-0.005">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 72.53%); opacity: 0.92" title="-0.006">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.52%); opacity: 0.94" title="-0.007">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.62%); opacity: 0.93" title="-0.006">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.26%); opacity: 0.84" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.20%); opacity: 0.93" title="-0.006">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.33%); opacity: 0.94" title="-0.006">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.08%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.30%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.91%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.35%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.07%); opacity: 0.91" title="-0.005">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.81%); opacity: 0.90" title="-0.005">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.35%); opacity: 0.85" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.37%); opacity: 0.89" title="-0.004">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.91%); opacity: 0.89" title="-0.004">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.55%); opacity: 0.84" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.63%); opacity: 0.89" title="-0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.30%); opacity: 0.88" title="-0.004">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.01%); opacity: 0.88" title="-0.004">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.57%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.76%); opacity: 0.83" title="-0.002">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.36%); opacity: 0.86" title="-0.003">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.65%); opacity: 0.83" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.34%); opacity: 0.87" title="-0.003">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.38%); opacity: 0.87" title="-0.003">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.59%); opacity: 0.88" title="-0.004">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.86%); opacity: 0.88" title="-0.004">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.83%); opacity: 0.84" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.72%); opacity: 0.88" title="-0.004">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.55%); opacity: 0.87" title="-0.003">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.32%); opacity: 0.83" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.43%); opacity: 0.87" title="-0.003">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.89%); opacity: 0.87" title="-0.003">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.99%); opacity: 0.84" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.03%); opacity: 0.84" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.29%); opacity: 0.88" title="-0.004">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.41%); opacity: 0.90" title="-0.005">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.69%); opacity: 0.91" title="-0.005">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.69%); opacity: 0.84" title="-0.002">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.50%); opacity: 0.94" title="-0.006">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 67.84%); opacity: 0.95" title="-0.007">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.70%); opacity: 0.96" title="-0.008">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.26%); opacity: 0.89" title="-0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.009">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.31%); opacity: 0.91" title="-0.005">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.62%); opacity: 0.86" title="-0.003">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.33%); opacity: 0.93" title="-0.006">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.84%); opacity: 0.94" title="-0.007">miss</span>
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
                         pad_value=0, relu=False)




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 97.48%); opacity: 0.80" title="-0.000">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.03%); opacity: 0.80" title="-0.000">please</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.78%); opacity: 0.81" title="-0.000">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 97.92%); opacity: 0.80" title="-0.000">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 96.51%); opacity: 0.81" title="-0.000">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.53%); opacity: 0.81" title="-0.000">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 94.54%); opacity: 0.81" title="-0.001">miss</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.79%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 95.06%); opacity: 0.81" title="-0.000">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.96%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.13%); opacity: 0.82" title="-0.001">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.64%); opacity: 0.83" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 90.19%); opacity: 0.83" title="-0.001">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.70%); opacity: 0.89" title="-0.004">rest</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.28%); opacity: 0.89" title="-0.004">of</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.26%); opacity: 0.83" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.11%); opacity: 0.91" title="-0.005">cast</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 72.53%); opacity: 0.92" title="-0.006">rendered</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.52%); opacity: 0.94" title="-0.007">terrible</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.62%); opacity: 0.93" title="-0.006">performances</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.26%); opacity: 0.84" title="-0.002">the</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.20%); opacity: 0.93" title="-0.006">show</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.33%); opacity: 0.94" title="-0.006">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.08%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.30%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.91%); opacity: 0.85" title="-0.002">flat</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.35%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.82%); opacity: 0.83" title="-0.001">br</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.07%); opacity: 0.91" title="-0.005">i</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 74.81%); opacity: 0.90" title="-0.005">don&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 85.35%); opacity: 0.85" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.37%); opacity: 0.89" title="-0.004">how</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.91%); opacity: 0.89" title="-0.004">michael</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.55%); opacity: 0.84" title="-0.002">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.63%); opacity: 0.89" title="-0.004">could</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.30%); opacity: 0.88" title="-0.004">have</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.01%); opacity: 0.88" title="-0.004">allowed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.57%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.76%); opacity: 0.83" title="-0.002">one</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.36%); opacity: 0.86" title="-0.003">on</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.65%); opacity: 0.83" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.34%); opacity: 0.87" title="-0.003">plate</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.38%); opacity: 0.87" title="-0.003">he</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.59%); opacity: 0.88" title="-0.004">almost</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.86%); opacity: 0.88" title="-0.004">seemed</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.83%); opacity: 0.84" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.13%); opacity: 0.84" title="-0.002">know</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 92.41%); opacity: 0.82" title="-0.001">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 79.72%); opacity: 0.88" title="-0.004">wasn&#x27;t</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.55%); opacity: 0.87" title="-0.003">going</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 88.32%); opacity: 0.83" title="-0.002">to</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 81.43%); opacity: 0.87" title="-0.003">work</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 80.89%); opacity: 0.87" title="-0.003">out</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.99%); opacity: 0.84" title="-0.002">and</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 87.03%); opacity: 0.84" title="-0.002">his</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 78.29%); opacity: 0.88" title="-0.004">performance</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 75.41%); opacity: 0.90" title="-0.005">was</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.69%); opacity: 0.91" title="-0.005">quite</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.69%); opacity: 0.84" title="-0.002">&lt;OOV&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 69.50%); opacity: 0.94" title="-0.006">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 67.84%); opacity: 0.95" title="-0.007">all</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.70%); opacity: 0.96" title="-0.008">you</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 77.26%); opacity: 0.89" title="-0.004">madison</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.009">fans</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 73.31%); opacity: 0.91" title="-0.005">give</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 82.62%); opacity: 0.86" title="-0.003">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 70.33%); opacity: 0.93" title="-0.006">a</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.84%); opacity: 0.94" title="-0.007">miss</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




Let’s try to add padding to the sample and explain that

.. code:: ipython3

    review_t_padded = pad_sequences(review_t, maxlen=128, value='<PAD>', dtype=object)
    review_padded = keras_sentiment_classifier.tokens_to_vectorized(review_t_padded)
    print(review_t_padded)
    print(review_padded)
    
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
      'great']]
    [[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
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
            <span style="opacity: 0.80">&lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;PAD&gt; &lt;START&gt; hello this is great but not so great</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




It looks like ``<PAD>`` had some effect on the explanation. Removing it
from the explanation

.. code:: ipython3

    eli5.show_prediction(binary_model, review_padded, tokens=review_t_padded, 
                         relu=False, pad_token='<PAD>')




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(0, 100.00%, 60.00%); opacity: 1.00" title="-0.046">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.32%); opacity: 1.00" title="-0.045">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 60.76%); opacity: 0.99" title="-0.044">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 61.40%); opacity: 0.99" title="-0.043">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 76.77%); opacity: 0.89" title="-0.021">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 63.33%); opacity: 0.98" title="-0.040">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 65.14%); opacity: 0.96" title="-0.038">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 68.05%); opacity: 0.95" title="-0.033">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 83.01%); opacity: 0.86" title="-0.013">great</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




That’s something. Though the model still gives different results
compared to the explanation given for the not-padded ``review`` array.
That is because we feed the input as it is, but only remove padding from
the results.

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
“predicted” class.

For full details of how we trained the model and the data check
https://www.kaggle.com/tobalt/keras-text-model-multiclass or the
``tests/estimators/keras_multiclass_text_classifier/keras_multiclass_text_classifier.ipynb``
file in the ELI5 repo.

.. code:: ipython3

    import tests.estimators.keras_multiclass_text_classifier.keras_multiclass_text_classifier \
        as keras_multiclass_text_classifier

Load the model

.. code:: ipython3

    multicls_model = keras.models.load_model(keras_multiclass_text_classifier.MODEL)
    multicls_model.summary()


.. parsed-literal::

    Model: "sequential_1"
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

    (x_train, x_test), (y_train, y_test) = keras_multiclass_text_classifier.prepare_train_test_dataset()

Again check the metrics.

.. code:: ipython3

    print(multicls_model.metrics_names)
    loss, acc = multicls_model.evaluate(x_test, y_test)
    print(loss, acc)
    
    print('Accuracy:', acc)


.. parsed-literal::

    ['loss', 'acc']
    500/500 [==============================] - 2s 5ms/step
    0.6319513120651246 0.7999999990463257
    Accuracy: 0.7999999990463257


Let’s see the possible classes that consumer complaint narratives can
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



Let’s explain one of the test samples

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


Let’s check what the model predicts (to which category the financial
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

Now let’s explain this prediction with ELI5.

.. code:: ipython3

    eli5.show_prediction(multicls_model, test_complaint, tokens=test_complaint_t, pad_token='<PAD>')




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">Ocwen fin</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 98.11%); opacity: 0.80" title="0.002">v</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.01%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">m</span><span style="background-color: hsl(120, 100.00%, 97.74%); opacity: 0.80" title="0.003">u</span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.47%); opacity: 0.80" title="0.003">u</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 98.21%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 97.53%); opacity: 0.80" title="0.003">y</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.64%); opacity: 0.80" title="0.003">v</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.78%); opacity: 0.81" title="0.004">(</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.85%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">v</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">n</span><span style="opacity: 0.80">g done b</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.52%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.002">v</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.23%); opacity: 0.81" title="0.010">b</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.82%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 96.52%); opacity: 0.81" title="0.005">f</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 97.80%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 96.92%); opacity: 0.81" title="0.004">f</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.008">)</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 96.95%); opacity: 0.81" title="0.004">v</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.003">w</span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.11%); opacity: 0.80" title="0.004">f</span><span style="background-color: hsl(120, 100.00%, 97.31%); opacity: 0.80" title="0.003">f</span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.77%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 100.00%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 100.00%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 100.00%); opacity: 0.80" title="0.000">t</span><span style="opacity: 0.80">h the IRS in XX/XX/XXXX. Further, they continue to insert themselves in a legal case I have against XXXX XXXX</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">X</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">X</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">X</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">X</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">m</span><span style="opacity: 0.80">pany</span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.41%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.39%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 96.87%); opacity: 0.81" title="0.004">g</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.58%); opacity: 0.81" title="0.005">m</span><span style="background-color: hsl(120, 100.00%, 96.83%); opacity: 0.81" title="0.004">y</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.26%); opacity: 0.81" title="0.005">f</span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.17%); opacity: 0.80" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.002">c</span><span style="background-color: hsl(120, 100.00%, 97.88%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.28%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.004">u</span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 94.49%); opacity: 0.81" title="0.009">.</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.96%); opacity: 0.81" title="0.008">O</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 97.80%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.43%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.02%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 95.53%); opacity: 0.81" title="0.007">,</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 97.57%); opacity: 0.80" title="0.003">g</span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.41%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 97.91%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 91.79%); opacity: 0.82" title="0.016">x</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.39%); opacity: 0.80" title="0.002">u</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 92.23%); opacity: 0.82" title="0.015">N</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.47%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.46%); opacity: 0.81" title="0.005">.</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.39%); opacity: 0.80" title="0.002">X</span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.002">X</span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.002">X</span><span style="background-color: hsl(120, 100.00%, 98.26%); opacity: 0.80" title="0.002">X</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 97.77%); opacity: 0.80" title="0.003">m</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000">s</span><span style="opacity: 0.80"> the owner of the pr</span><span style="background-color: hsl(120, 100.00%, 100.00%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 98.34%); opacity: 0.80" title="0.002">.</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 81.33%); opacity: 0.87" title="0.053">H</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 97.31%); opacity: 0.80" title="0.003">v</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 95.50%); opacity: 0.81" title="0.007">,</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.45%); opacity: 0.81" title="0.007">O</span><span style="background-color: hsl(120, 100.00%, 98.65%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.15%); opacity: 0.80" title="0.004">v</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.82%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 97.79%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.61%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 90.59%); opacity: 0.83" title="0.020">N</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.59%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.13%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.08%); opacity: 0.81" title="0.006">I</span><span style="background-color: hsl(120, 100.00%, 89.54%); opacity: 0.83" title="0.023">R</span><span style="background-color: hsl(120, 100.00%, 92.37%); opacity: 0.82" title="0.015">S</span><span style="background-color: hsl(120, 100.00%, 96.63%); opacity: 0.81" title="0.005">.</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 83.80%); opacity: 0.86" title="0.043">T</span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 90.47%); opacity: 0.83" title="0.020">N</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.30%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.84%); opacity: 0.81" title="0.006">b</span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.18%); opacity: 0.81" title="0.010">I</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.21%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 97.17%); opacity: 0.80" title="0.004">y</span><span style="background-color: hsl(120, 100.00%, 81.18%); opacity: 0.87" title="0.053">M</span><span style="background-color: hsl(120, 100.00%, 98.66%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.60%); opacity: 0.80" title="0.003">c</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 77.00%); opacity: 0.89" title="0.071">M</span><span style="background-color: hsl(120, 100.00%, 98.12%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 97.80%); opacity: 0.80" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.002">t</span><span style="background-color: hsl(120, 100.00%, 96.00%); opacity: 0.81" title="0.006">g</span><span style="background-color: hsl(120, 100.00%, 98.65%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.005">g</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 87.91%); opacity: 0.84" title="0.028">S</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.01%); opacity: 0.80" title="0.002">v</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">i</span><span style="opacity: 0.80">n XX</span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.001">/</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000">X</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">X</span><span style="background-color: hsl(120, 100.00%, 97.67%); opacity: 0.80" title="0.003">/</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">X</span><span style="background-color: hsl(120, 100.00%, 99.22%); opacity: 0.80" title="0.001">X</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">X</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">X</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 93.73%); opacity: 0.81" title="0.011">(</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.97%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.56%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.24%); opacity: 0.81" title="0.007">O</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 97.82%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 100.00%); opacity: 0.80" title="0.000"> </span><span style="opacity: 0.80">of as the eviden</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80">s sh</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.68%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.47%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.13%); opacity: 0.81" title="0.008">)</span><span style="background-color: hsl(120, 100.00%, 97.38%); opacity: 0.80" title="0.003">.</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.92%); opacity: 0.81" title="0.004">O</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.33%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.30%); opacity: 0.80" title="0.002">p</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80"> have applied a court ordered use and occupancy </span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.38%); opacity: 0.80" title="0.002">u</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.94%); opacity: 0.80" title="0.002">X</span><span style="background-color: hsl(120, 100.00%, 97.85%); opacity: 0.80" title="0.002">X</span><span style="background-color: hsl(120, 100.00%, 97.76%); opacity: 0.80" title="0.003">X</span><span style="background-color: hsl(120, 100.00%, 97.67%); opacity: 0.80" title="0.003">X</span><span style="background-color: hsl(120, 100.00%, 99.47%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.10%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 91.78%); opacity: 0.82" title="0.016">(</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.001">b</span><span style="opacity: 0.80">er XXXX XXXX</span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.58%); opacity: 0.81" title="0.009">O</span><span style="background-color: hsl(120, 100.00%, 98.24%); opacity: 0.80" title="0.002">c</span><span style="background-color: hsl(120, 100.00%, 97.14%); opacity: 0.80" title="0.004">w</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.02%); opacity: 0.80" title="0.004">p</span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">m</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">. I have rec</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 97.42%); opacity: 0.80" title="0.003">f</span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.002">c</span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.85%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.54%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.04%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 98.68%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.25%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(120, 100.00%, 96.23%); opacity: 0.81" title="0.005">y</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 97.20%); opacity: 0.80" title="0.003">l</span><span style="background-color: hsl(120, 100.00%, 98.13%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 96.33%); opacity: 0.81" title="0.005">w</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.08%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 97.90%); opacity: 0.80" title="0.002">f</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.65%); opacity: 0.81" title="0.009">O</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.002">c</span><span style="background-color: hsl(120, 100.00%, 97.13%); opacity: 0.80" title="0.004">w</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000">t</span><span style="opacity: 0.80">ime, nor cou</span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.32%); opacity: 0.81" title="0.010">O</span><span style="background-color: hsl(120, 100.00%, 98.20%); opacity: 0.80" title="0.002">c</span><span style="background-color: hsl(120, 100.00%, 96.99%); opacity: 0.80" title="0.004">w</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.004">w</span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.16%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 93.27%); opacity: 0.82" title="0.012">(</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.00%); opacity: 0.81" title="0.010">)</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.47%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 97.26%); opacity: 0.80" title="0.003">u</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.28%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 98.85%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.00%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.22%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">d</span><span style="opacity: 0.80">y ta</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">k</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000"> </span><span style="opacity: 0.80">the prop</span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">.</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.74%); opacity: 0.81" title="0.006">F</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.004">,</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.005">O</span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.01%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 97.80%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.50%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.47%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 88.59%); opacity: 0.83" title="0.026">(</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.003">g</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.85%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.80%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.16%); opacity: 0.81" title="0.010">)</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.38%); opacity: 0.81" title="0.005">b</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.77%); opacity: 0.80" title="0.003">f</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.06%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 89.96%); opacity: 0.83" title="0.022">N</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.156">L</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.79%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 98.48%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.13%); opacity: 0.80" title="0.004">u</span><span style="background-color: hsl(120, 100.00%, 96.98%); opacity: 0.80" title="0.004">g</span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.81%); opacity: 0.80" title="0.002">p</span><span style="background-color: hsl(120, 100.00%, 98.08%); opacity: 0.80" title="0.002">u</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.53%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.001">I</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 94.57%); opacity: 0.81" title="0.009">M</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 91.19%); opacity: 0.82" title="0.018">M</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 97.27%); opacity: 0.80" title="0.003">g</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 96.54%); opacity: 0.81" title="0.005">g</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 79.98%); opacity: 0.87" title="0.058">S</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 97.72%); opacity: 0.80" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 94.14%); opacity: 0.81" title="0.010">v</span><span style="background-color: hsl(120, 100.00%, 98.08%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.003">c</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 97.83%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.60%); opacity: 0.81" title="0.005">f</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.04%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">X</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">X</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">X</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">X</span><span style="opacity: 0.80"> ( or any other entity ), rather, th</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.47%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">r</span><span style="opacity: 0.80">esult of XXXX giving them an original copy of a l</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.59%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 92.68%); opacity: 0.82" title="0.014">x</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.85%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.12%); opacity: 0.80" title="0.002">u</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 88.83%); opacity: 0.83" title="0.025">N</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.61%); opacity: 0.80" title="0.003">w</span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">v</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.47%); opacity: 0.80" title="0.000">m</span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">m</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">f forecl</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.76%); opacity: 0.81" title="0.004">.</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 79.31%); opacity: 0.88" title="0.061">T</span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.99%); opacity: 0.81" title="0.006">v</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.53%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 97.61%); opacity: 0.80" title="0.003">y</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.43%); opacity: 0.80" title="0.002">f</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 100.00%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">i</span><span style="opacity: 0.80">n a court of law </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.77%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.64%); opacity: 0.80" title="0.003">p</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.44%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 97.50%); opacity: 0.80" title="0.003">g</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.54%); opacity: 0.80" title="0.003">p</span><span style="background-color: hsl(120, 100.00%, 97.58%); opacity: 0.80" title="0.003">p</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.005">.</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 89.57%); opacity: 0.83" title="0.023">
    </span><span style="background-color: hsl(120, 100.00%, 97.96%); opacity: 0.80" title="0.002">I</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">l</span><span style="opacity: 0.80">ing a tax for cl</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80">rge a note that they did not had and had been</span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.83%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 97.85%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 95.12%); opacity: 0.81" title="0.008">,</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.71%); opacity: 0.81" title="0.009">O</span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">c</span><span style="background-color: hsl(120, 100.00%, 97.28%); opacity: 0.80" title="0.003">w</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.73%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.30%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.92%); opacity: 0.81" title="0.004">g</span><span style="background-color: hsl(120, 100.00%, 97.10%); opacity: 0.80" title="0.004">u</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 97.46%); opacity: 0.80" title="0.003">y</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.64%); opacity: 0.80" title="0.003">f</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 94.42%); opacity: 0.81" title="0.009">x</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.39%); opacity: 0.80" title="0.002">u</span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.78%); opacity: 0.80" title="0.003">g</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.54%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 92.69%); opacity: 0.82" title="0.014">I</span><span style="background-color: hsl(120, 100.00%, 81.81%); opacity: 0.86" title="0.051">R</span><span style="background-color: hsl(120, 100.00%, 88.08%); opacity: 0.84" title="0.028">S</span><span style="background-color: hsl(120, 100.00%, 95.29%); opacity: 0.81" title="0.007">.</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.009">I</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 98.65%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 92.35%); opacity: 0.82" title="0.015">/</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">h</span><span style="opacity: 0.80">ed to my</span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">,</span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">O</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.78%); opacity: 0.80" title="0.003">m</span><span style="background-color: hsl(120, 100.00%, 97.56%); opacity: 0.80" title="0.003">m</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 97.00%); opacity: 0.80" title="0.004">g</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.00%); opacity: 0.80" title="0.004">f</span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.77%); opacity: 0.80" title="0.003">u</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 96.35%); opacity: 0.81" title="0.005">.</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.08%); opacity: 0.81" title="0.006">I</span><span style="background-color: hsl(120, 100.00%, 98.56%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.003">O</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 97.88%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.61%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 96.47%); opacity: 0.81" title="0.005">v</span><span style="background-color: hsl(120, 100.00%, 99.22%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.61%); opacity: 0.80" title="0.001">w</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.81%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 95.02%); opacity: 0.81" title="0.008">,</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 93.33%); opacity: 0.82" title="0.012">I</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.36%); opacity: 0.80" title="0.003">w</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.82%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 96.79%); opacity: 0.81" title="0.004">v</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.99%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">d of thi</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.98%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.94%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 82.46%); opacity: 0.86" title="0.048">q</span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.54%); opacity: 0.81" title="0.007">b</span><span style="background-color: hsl(120, 100.00%, 98.44%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.75%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80">l law and I dispute any such claim. B</span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 97.97%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 97.19%); opacity: 0.80" title="0.004">g</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.90%); opacity: 0.81" title="0.004">p</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.30%); opacity: 0.80" title="0.003">y</span><span style="background-color: hsl(120, 100.00%, 97.17%); opacity: 0.80" title="0.004">m</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.12%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.62%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.55%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.52%); opacity: 0.80" title="0.003">w</span><span style="background-color: hsl(120, 100.00%, 98.53%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.17%); opacity: 0.80" title="0.002">c</span><span style="background-color: hsl(120, 100.00%, 98.29%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.79%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 98.44%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 97.92%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.38%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(120, 100.00%, 65.82%); opacity: 0.96" title="0.125">&#x27;</span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 84.17%); opacity: 0.85" title="0.042">x</span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(120, 100.00%, 98.03%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 98.75%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 93.65%); opacity: 0.81" title="0.011">,</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.95%); opacity: 0.81" title="0.008">O</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.29%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.66%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 98.47%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.25%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.93%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.82%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 96.98%); opacity: 0.80" title="0.004">g</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.57%); opacity: 0.81" title="0.005">f</span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 96.61%); opacity: 0.81" title="0.005">u</span><span style="background-color: hsl(120, 100.00%, 97.73%); opacity: 0.80" title="0.003">d</span><span style="background-color: hsl(120, 100.00%, 93.11%); opacity: 0.82" title="0.013">,</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.24%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 98.91%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.30%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.55%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.42%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.89%); opacity: 0.80" title="0.000">r</span><span style="opacity: 0.80">ged the debt last year and the check</span><span style="background-color: hsl(120, 100.00%, 100.00%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.93%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.18%); opacity: 0.80" title="0.002">m</span><span style="background-color: hsl(120, 100.00%, 99.31%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.04%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.91%); opacity: 0.80" title="0.002">u</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.97%); opacity: 0.81" title="0.006">O</span><span style="background-color: hsl(120, 100.00%, 98.81%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.14%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.014">F</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.93%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(120, 100.00%, 98.53%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 60.30%); opacity: 1.00" title="0.154">&#x27;</span><span style="background-color: hsl(120, 100.00%, 97.39%); opacity: 0.80" title="0.003">s</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 94.99%); opacity: 0.81" title="0.008">p</span><span style="background-color: hsl(120, 100.00%, 98.21%); opacity: 0.80" title="0.002">a</span><span style="background-color: hsl(120, 100.00%, 97.43%); opacity: 0.80" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.002">n</span><span style="background-color: hsl(120, 100.00%, 98.19%); opacity: 0.80" title="0.002">t</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.11%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.02%); opacity: 0.80" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.31%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 97.67%); opacity: 0.80" title="0.003">u</span><span style="background-color: hsl(120, 100.00%, 95.37%); opacity: 0.81" title="0.007">b</span><span style="background-color: hsl(120, 100.00%, 99.40%); opacity: 0.80" title="0.000">s</span><span style="opacity: 0.80">idiary or related companies which constitutes</span><span style="background-color: hsl(120, 100.00%, 99.97%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 95.91%); opacity: 0.81" title="0.006">k</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 97.81%); opacity: 0.80" title="0.002">.</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.36%); opacity: 0.81" title="0.010">F</span><span style="background-color: hsl(120, 100.00%, 98.58%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.18%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 95.15%); opacity: 0.81" title="0.008">,</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.04%); opacity: 0.81" title="0.008">O</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 97.55%); opacity: 0.80" title="0.003">w</span><span style="background-color: hsl(120, 100.00%, 99.22%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.92%); opacity: 0.81" title="0.004">v</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.01%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.54%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.22%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.10%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.85%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 97.60%); opacity: 0.80" title="0.003">u</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.07%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">r</span><span style="opacity: 0.80"> for use and occupancy, which they are not entitled to since </span><span style="background-color: hsl(120, 100.00%, 99.95%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.74%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.79%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.90%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">g</span><span style="background-color: hsl(120, 100.00%, 100.00%); opacity: 0.80" title="0.000">a</span><span style="opacity: 0.80">l action. Further, O</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000">w</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.22%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.98%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.56%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.02%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 97.78%); opacity: 0.80" title="0.003">p</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 98.72%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 96.60%); opacity: 0.81" title="0.005">g</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.005">w</span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">i</span><span style="background-color: hsl(120, 100.00%, 98.56%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 97.92%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 96.68%); opacity: 0.81" title="0.004">X</span><span style="background-color: hsl(120, 100.00%, 97.00%); opacity: 0.80" title="0.004">X</span><span style="background-color: hsl(120, 100.00%, 97.33%); opacity: 0.80" title="0.003">X</span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.003">X</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.15%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.22%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.41%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">m</span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">m</span><span style="background-color: hsl(120, 100.00%, 99.81%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.80%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.83%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 98.59%); opacity: 0.80" title="0.001">.</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.73%); opacity: 0.81" title="0.004">F</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.63%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.68%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 96.88%); opacity: 0.81" title="0.004">,</span><span style="background-color: hsl(120, 100.00%, 99.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 96.50%); opacity: 0.81" title="0.005">O</span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">c</span><span style="background-color: hsl(120, 100.00%, 98.18%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.002">w</span><span style="background-color: hsl(120, 100.00%, 99.39%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 97.98%); opacity: 0.80" title="0.002">v</span><span style="background-color: hsl(120, 100.00%, 99.33%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.62%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.70%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.03%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 97.44%); opacity: 0.80" title="0.003">f</span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.13%); opacity: 0.80" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.63%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 96.79%); opacity: 0.81" title="0.004">f</span><span style="background-color: hsl(120, 100.00%, 99.43%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.89%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.002">h</span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.90%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 94.21%); opacity: 0.81" title="0.010">x</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">u</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002">b</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">y</span><span style="background-color: hsl(120, 100.00%, 99.88%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.77%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.001">g</span><span style="background-color: hsl(120, 100.00%, 99.59%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.61%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.56%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.13%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.10%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 95.29%); opacity: 0.81" title="0.007">,</span><span style="background-color: hsl(120, 100.00%, 99.56%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">s</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.64%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.001">y</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000"> </span><span style="opacity: 0.80">are already aware ( or should be aware ) that their behavior </span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.84%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.92%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.76%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.09%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.45%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.001">p</span><span style="background-color: hsl(120, 100.00%, 99.37%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.49%); opacity: 0.80" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 99.58%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.47%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.68%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.88%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.27%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 90.59%); opacity: 0.83" title="0.020">k</span><span style="background-color: hsl(120, 100.00%, 99.48%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.85%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.40%); opacity: 0.80" title="0.002">y</span><span style="background-color: hsl(120, 100.00%, 99.67%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.84%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.87%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 99.44%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.99%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.004">.</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 88.21%); opacity: 0.83" title="0.027">
    </span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




Note that we do not set ``relu`` to ``False`` because we would see other
classes.

Our own example

.. code:: ipython3

    s = """first of all I should not be charged and debited for the private car loan"""
    complaint, complaint_t = keras_multiclass_text_classifier.string_to_vectorized(s)
    print(complaint)
    print(complaint_t[0, :50])  # note that this model requires fixed length input


.. parsed-literal::

    [[20  8  9 ...  0  0  0]]
    ['f' 'i' 'r' 's' 't' ' ' 'o' 'f' ' ' 'a' 'l' 'l' ' ' 'I' ' ' 's' 'h' 'o'
     'u' 'l' 'd' ' ' 'n' 'o' 't' ' ' 'b' 'e' ' ' 'c' 'h' 'a' 'r' 'g' 'e' 'd'
     ' ' 'a' 'n' 'd' ' ' 'd' 'e' 'b' 'i' 't' 'e' 'd' ' ' 'f']


.. code:: ipython3

    preds = multicls_model.predict(complaint)
    print(preds)
    print(keras_multiclass_text_classifier.decode_output(preds))
    
    eli5.show_prediction(multicls_model, complaint, tokens=complaint_t, pad_token='<PAD>')


.. parsed-literal::

    [[0.10190239 0.29990542 0.07907407 0.02289054 0.08175348 0.23877561
      0.01516608 0.15204951 0.00509677 0.00121074 0.00217532]]
    Consumer Loan




.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">f</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 97.22%); opacity: 0.80" title="0.004">f</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.003">a</span><span style="background-color: hsl(120, 100.00%, 97.01%); opacity: 0.80" title="0.005">l</span><span style="background-color: hsl(120, 100.00%, 96.82%); opacity: 0.81" title="0.005">l</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 90.42%); opacity: 0.83" title="0.025">I</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.015">s</span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.011">h</span><span style="background-color: hsl(120, 100.00%, 95.59%); opacity: 0.81" title="0.008">o</span><span style="background-color: hsl(120, 100.00%, 84.32%); opacity: 0.85" title="0.050">u</span><span style="background-color: hsl(120, 100.00%, 93.31%); opacity: 0.82" title="0.015">l</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.014">d</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 90.03%); opacity: 0.83" title="0.026">n</span><span style="background-color: hsl(120, 100.00%, 92.77%); opacity: 0.82" title="0.017">o</span><span style="background-color: hsl(120, 100.00%, 92.51%); opacity: 0.82" title="0.018">t</span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 83.74%); opacity: 0.86" title="0.053">b</span><span style="background-color: hsl(120, 100.00%, 91.95%); opacity: 0.82" title="0.019">e</span><span style="background-color: hsl(120, 100.00%, 95.28%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 80.25%); opacity: 0.87" title="0.070">c</span><span style="background-color: hsl(120, 100.00%, 84.00%); opacity: 0.85" title="0.052">h</span><span style="background-color: hsl(120, 100.00%, 89.48%); opacity: 0.83" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 87.30%); opacity: 0.84" title="0.037">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.192">g</span><span style="background-color: hsl(120, 100.00%, 88.81%); opacity: 0.83" title="0.031">e</span><span style="background-color: hsl(120, 100.00%, 87.54%); opacity: 0.84" title="0.036">d</span><span style="background-color: hsl(120, 100.00%, 94.07%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 89.50%); opacity: 0.83" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 83.33%); opacity: 0.86" title="0.055">n</span><span style="background-color: hsl(120, 100.00%, 88.63%); opacity: 0.83" title="0.032">d</span><span style="background-color: hsl(120, 100.00%, 94.61%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.029">d</span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 81.93%); opacity: 0.86" title="0.062">b</span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.037">i</span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.020">t</span><span style="background-color: hsl(120, 100.00%, 93.96%); opacity: 0.81" title="0.013">e</span><span style="background-color: hsl(120, 100.00%, 94.48%); opacity: 0.81" title="0.011">d</span><span style="background-color: hsl(120, 100.00%, 98.05%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 96.39%); opacity: 0.81" title="0.006">f</span><span style="background-color: hsl(120, 100.00%, 97.96%); opacity: 0.80" title="0.003">o</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="opacity: 0.80">the private c</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.18%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.97%); opacity: 0.80" title="0.003">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




Let’s check all the layers. Maybe some will give better-looking
explanations.

.. code:: ipython3

    layer = 'embedding_1'
    print(layer)
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t,
                                 pad_token='<PAD>', layer=layer))
    
    layer = 'conv1d_1'
    print(layer)
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t,
                                 pad_token='<PAD>', layer=layer))
    
    layer = 'max_pooling1d_1'
    print(layer)
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t,
                                 pad_token='<PAD>', layer=layer))
    
    layer = 'conv1d_2'
    print(layer)
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t,
                                 pad_token='<PAD>', layer=layer))
    
    layer = 'average_pooling1d_1'
    print(layer)
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t,
                                 pad_token='<PAD>', layer=layer))
    
    layer = 'conv1d_3'
    print(layer)
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t,
                                 pad_token='<PAD>', layer=layer))
    
    layer = 'max_pooling1d_2'
    print(layer)
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t,
                                 pad_token='<PAD>', layer=layer))
    
    layer = 'global_average_pooling1d_1'
    print(layer)
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t,
                                 pad_token='<PAD>', layer=layer))


.. parsed-literal::

    embedding_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 91.67%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80">i</span><span style="background-color: hsl(120, 100.00%, 90.33%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80">st o</span><span style="background-color: hsl(120, 100.00%, 91.67%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80"> all I should not be cha</span><span style="background-color: hsl(120, 100.00%, 90.33%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">g</span><span style="opacity: 0.80">ed and debited </span><span style="background-color: hsl(120, 100.00%, 91.67%); opacity: 0.82" title="0.000">f</span><span style="opacity: 0.80">o</span><span style="background-color: hsl(120, 100.00%, 90.33%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80"> the p</span><span style="background-color: hsl(120, 100.00%, 90.33%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80">ivate ca</span><span style="background-color: hsl(120, 100.00%, 90.33%); opacity: 0.83" title="0.000">r</span><span style="opacity: 0.80"> loan</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    conv1d_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 98.56%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 92.31%); opacity: 0.82" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 94.76%); opacity: 0.81" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 95.61%); opacity: 0.81" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 95.62%); opacity: 0.81" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 94.72%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 77.03%); opacity: 0.89" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 97.96%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.68%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.82%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 94.70%); opacity: 0.81" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 78.28%); opacity: 0.88" title="0.000">I</span><span style="background-color: hsl(120, 100.00%, 99.23%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 84.07%); opacity: 0.85" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 98.05%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 89.96%); opacity: 0.83" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 89.84%); opacity: 0.83" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 99.51%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 78.86%); opacity: 0.88" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 97.93%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 95.90%); opacity: 0.81" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">o</span><span style="opacity: 0.80">t </span><span style="background-color: hsl(120, 100.00%, 90.66%); opacity: 0.83" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 91.50%); opacity: 0.82" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.86%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 60.85%); opacity: 0.99" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 82.49%); opacity: 0.86" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 97.75%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.38%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 98.57%); opacity: 0.80" title="0.000">g</span><span style="opacity: 0.80">ed a</span><span style="background-color: hsl(120, 100.00%, 87.44%); opacity: 0.84" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 88.74%); opacity: 0.83" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 97.74%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.14%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">e</span><span style="opacity: 0.80">b</span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 86.00%); opacity: 0.84" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 91.29%); opacity: 0.82" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 93.99%); opacity: 0.81" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 95.38%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 88.31%); opacity: 0.83" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 84.36%); opacity: 0.85" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 89.65%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 94.18%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 86.67%); opacity: 0.84" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 85.24%); opacity: 0.85" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 96.98%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 95.89%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 87.97%); opacity: 0.84" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 87.53%); opacity: 0.84" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 91.50%); opacity: 0.82" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 88.44%); opacity: 0.83" title="0.000">v</span><span style="background-color: hsl(120, 100.00%, 89.70%); opacity: 0.83" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 71.19%); opacity: 0.93" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 87.02%); opacity: 0.84" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 87.66%); opacity: 0.84" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 80.78%); opacity: 0.87" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 80.51%); opacity: 0.87" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 80.04%); opacity: 0.87" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 83.95%); opacity: 0.85" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 90.68%); opacity: 0.82" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 92.48%); opacity: 0.82" title="0.000">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    max_pooling1d_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 95.41%); opacity: 0.81" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 95.47%); opacity: 0.81" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 96.87%); opacity: 0.81" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 89.89%); opacity: 0.83" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 92.80%); opacity: 0.82" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 96.15%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 91.41%); opacity: 0.82" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 91.47%); opacity: 0.82" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 98.78%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.87%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 97.08%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 96.46%); opacity: 0.81" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 98.28%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 86.75%); opacity: 0.84" title="0.000">I</span><span style="background-color: hsl(120, 100.00%, 97.56%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 90.22%); opacity: 0.83" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 92.49%); opacity: 0.82" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 96.57%); opacity: 0.81" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 98.14%); opacity: 0.80" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 92.27%); opacity: 0.82" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 88.88%); opacity: 0.83" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 96.30%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.83%); opacity: 0.80" title="0.000">n</span><span style="opacity: 0.80">ot</span><span style="background-color: hsl(120, 100.00%, 97.69%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 84.66%); opacity: 0.85" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 93.25%); opacity: 0.82" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 88.84%); opacity: 0.83" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 92.80%); opacity: 0.82" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 97.06%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.16%); opacity: 0.80" title="0.000">r</span><span style="opacity: 0.80">ged </span><span style="background-color: hsl(120, 100.00%, 94.21%); opacity: 0.81" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 83.55%); opacity: 0.86" title="0.000">n</span><span style="background-color: hsl(120, 100.00%, 90.72%); opacity: 0.82" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 97.71%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.50%); opacity: 0.80" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 90.85%); opacity: 0.82" title="0.000">b</span><span style="background-color: hsl(120, 100.00%, 87.78%); opacity: 0.84" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 91.93%); opacity: 0.82" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 94.37%); opacity: 0.81" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 93.13%); opacity: 0.82" title="0.000">d</span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 86.40%); opacity: 0.84" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 88.88%); opacity: 0.83" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 88.29%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 94.18%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 91.26%); opacity: 0.82" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 95.76%); opacity: 0.81" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 98.14%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 98.11%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 87.17%); opacity: 0.84" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 96.06%); opacity: 0.81" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 91.66%); opacity: 0.82" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 74.76%); opacity: 0.90" title="0.000">v</span><span style="background-color: hsl(120, 100.00%, 87.44%); opacity: 0.84" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 79.22%); opacity: 0.88" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 80.23%); opacity: 0.87" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 89.21%); opacity: 0.83" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 82.85%); opacity: 0.86" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 82.45%); opacity: 0.86" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 92.52%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 84.58%); opacity: 0.85" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 89.09%); opacity: 0.83" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 92.79%); opacity: 0.82" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 93.05%); opacity: 0.82" title="0.000">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    conv1d_2



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 99.20%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.38%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 96.53%); opacity: 0.81" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 97.98%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.14%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.24%); opacity: 0.80" title="0.000">f</span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.19%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.20%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.24%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.32%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.05%); opacity: 0.80" title="0.002">I</span><span style="background-color: hsl(120, 100.00%, 99.85%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.78%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.96%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.91%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 99.54%); opacity: 0.80" title="0.000">u</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 98.68%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.86%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.21%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.21%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.47%); opacity: 0.80" title="0.001">b</span><span style="background-color: hsl(120, 100.00%, 99.06%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.60%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.97%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 99.71%); opacity: 0.80" title="0.000">h</span><span style="opacity: 0.80">ar</span><span style="background-color: hsl(120, 100.00%, 96.40%); opacity: 0.81" title="0.002">g</span><span style="background-color: hsl(120, 100.00%, 98.04%); opacity: 0.80" title="0.001">e</span><span style="background-color: hsl(120, 100.00%, 96.79%); opacity: 0.81" title="0.002">d</span><span style="background-color: hsl(120, 100.00%, 97.99%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 96.03%); opacity: 0.81" title="0.002">a</span><span style="background-color: hsl(120, 100.00%, 93.45%); opacity: 0.82" title="0.005">n</span><span style="background-color: hsl(120, 100.00%, 95.47%); opacity: 0.81" title="0.003">d</span><span style="background-color: hsl(120, 100.00%, 97.86%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 95.31%); opacity: 0.81" title="0.003">d</span><span style="background-color: hsl(120, 100.00%, 95.35%); opacity: 0.81" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 88.21%); opacity: 0.83" title="0.012">b</span><span style="background-color: hsl(120, 100.00%, 89.48%); opacity: 0.83" title="0.010">i</span><span style="background-color: hsl(120, 100.00%, 91.62%); opacity: 0.82" title="0.007">t</span><span style="background-color: hsl(120, 100.00%, 91.79%); opacity: 0.82" title="0.007">e</span><span style="background-color: hsl(120, 100.00%, 89.76%); opacity: 0.83" title="0.009">d</span><span style="background-color: hsl(120, 100.00%, 94.58%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 83.20%); opacity: 0.86" title="0.019">f</span><span style="background-color: hsl(120, 100.00%, 87.92%); opacity: 0.84" title="0.012">o</span><span style="background-color: hsl(120, 100.00%, 87.50%); opacity: 0.84" title="0.013">r</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 86.70%); opacity: 0.84" title="0.014">t</span><span style="background-color: hsl(120, 100.00%, 81.04%); opacity: 0.87" title="0.023">h</span><span style="background-color: hsl(120, 100.00%, 88.41%); opacity: 0.83" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.066">p</span><span style="background-color: hsl(120, 100.00%, 86.91%); opacity: 0.84" title="0.013">r</span><span style="background-color: hsl(120, 100.00%, 81.35%); opacity: 0.87" title="0.022">i</span><span style="background-color: hsl(120, 100.00%, 60.83%); opacity: 0.99" title="0.064">v</span><span style="background-color: hsl(120, 100.00%, 89.04%); opacity: 0.83" title="0.010">a</span><span style="background-color: hsl(120, 100.00%, 87.58%); opacity: 0.84" title="0.012">t</span><span style="background-color: hsl(120, 100.00%, 89.46%); opacity: 0.83" title="0.010">e</span><span style="background-color: hsl(120, 100.00%, 94.81%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 82.68%); opacity: 0.86" title="0.020">c</span><span style="background-color: hsl(120, 100.00%, 93.84%); opacity: 0.81" title="0.005">a</span><span style="background-color: hsl(120, 100.00%, 95.10%); opacity: 0.81" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 98.38%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 97.86%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.64%); opacity: 0.80" title="0.001">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    average_pooling1d_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 98.36%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 98.24%); opacity: 0.80" title="0.001">i</span><span style="background-color: hsl(120, 100.00%, 98.69%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 97.35%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 98.53%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.76%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.41%); opacity: 0.80" title="0.001">f</span><span style="background-color: hsl(120, 100.00%, 99.53%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.17%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.94%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.71%); opacity: 0.80" title="0.002">I</span><span style="background-color: hsl(120, 100.00%, 99.72%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.29%); opacity: 0.80" title="0.000">s</span><span style="background-color: hsl(120, 100.00%, 99.86%); opacity: 0.80" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 99.52%); opacity: 0.80" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 97.42%); opacity: 0.80" title="0.002">u</span><span style="background-color: hsl(120, 100.00%, 98.67%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.59%); opacity: 0.80" title="0.001">d</span><span style="background-color: hsl(120, 100.00%, 99.28%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 97.86%); opacity: 0.80" title="0.001">n</span><span style="background-color: hsl(120, 100.00%, 98.48%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.01%); opacity: 0.80" title="0.001">b</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 99.87%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 99.26%); opacity: 0.80" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.001">h</span><span style="background-color: hsl(120, 100.00%, 99.11%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 98.74%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 93.90%); opacity: 0.81" title="0.006">g</span><span style="background-color: hsl(120, 100.00%, 97.51%); opacity: 0.80" title="0.002">e</span><span style="background-color: hsl(120, 100.00%, 96.43%); opacity: 0.81" title="0.003">d</span><span style="background-color: hsl(120, 100.00%, 97.93%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 95.96%); opacity: 0.81" title="0.004">a</span><span style="background-color: hsl(120, 100.00%, 93.29%); opacity: 0.82" title="0.007">n</span><span style="background-color: hsl(120, 100.00%, 95.20%); opacity: 0.81" title="0.005">d</span><span style="background-color: hsl(120, 100.00%, 97.62%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 94.47%); opacity: 0.81" title="0.006">d</span><span style="background-color: hsl(120, 100.00%, 94.24%); opacity: 0.81" title="0.006">e</span><span style="background-color: hsl(120, 100.00%, 85.77%); opacity: 0.85" title="0.021">b</span><span style="background-color: hsl(120, 100.00%, 88.00%); opacity: 0.84" title="0.017">i</span><span style="background-color: hsl(120, 100.00%, 90.77%); opacity: 0.82" title="0.012">t</span><span style="background-color: hsl(120, 100.00%, 91.19%); opacity: 0.82" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 89.23%); opacity: 0.83" title="0.014">d</span><span style="background-color: hsl(120, 100.00%, 94.40%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 82.76%); opacity: 0.86" title="0.028">f</span><span style="background-color: hsl(120, 100.00%, 87.61%); opacity: 0.84" title="0.018">o</span><span style="background-color: hsl(120, 100.00%, 87.27%); opacity: 0.84" title="0.018">r</span><span style="background-color: hsl(120, 100.00%, 93.65%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 86.77%); opacity: 0.84" title="0.019">t</span><span style="background-color: hsl(120, 100.00%, 81.17%); opacity: 0.87" title="0.032">h</span><span style="background-color: hsl(120, 100.00%, 88.46%); opacity: 0.83" title="0.016">e</span><span style="background-color: hsl(120, 100.00%, 93.65%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.094">p</span><span style="background-color: hsl(120, 100.00%, 87.18%); opacity: 0.84" title="0.019">r</span><span style="background-color: hsl(120, 100.00%, 81.89%); opacity: 0.86" title="0.030">i</span><span style="background-color: hsl(120, 100.00%, 61.39%); opacity: 0.99" title="0.089">v</span><span style="background-color: hsl(120, 100.00%, 89.38%); opacity: 0.83" title="0.014">a</span><span style="background-color: hsl(120, 100.00%, 89.04%); opacity: 0.83" title="0.015">t</span><span style="background-color: hsl(120, 100.00%, 91.37%); opacity: 0.82" title="0.011">e</span><span style="background-color: hsl(120, 100.00%, 95.82%); opacity: 0.81" title="0.004"> </span><span style="background-color: hsl(120, 100.00%, 86.09%); opacity: 0.84" title="0.021">c</span><span style="background-color: hsl(120, 100.00%, 94.67%); opacity: 0.81" title="0.005">a</span><span style="background-color: hsl(120, 100.00%, 95.33%); opacity: 0.81" title="0.004">r</span><span style="background-color: hsl(120, 100.00%, 98.51%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.27%); opacity: 0.80" title="0.001">l</span><span style="background-color: hsl(120, 100.00%, 98.66%); opacity: 0.80" title="0.001">o</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 98.52%); opacity: 0.80" title="0.001">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    conv1d_3



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">f</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 97.22%); opacity: 0.80" title="0.004">f</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.003">a</span><span style="background-color: hsl(120, 100.00%, 97.01%); opacity: 0.80" title="0.005">l</span><span style="background-color: hsl(120, 100.00%, 96.82%); opacity: 0.81" title="0.005">l</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 90.42%); opacity: 0.83" title="0.025">I</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.015">s</span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.011">h</span><span style="background-color: hsl(120, 100.00%, 95.59%); opacity: 0.81" title="0.008">o</span><span style="background-color: hsl(120, 100.00%, 84.32%); opacity: 0.85" title="0.050">u</span><span style="background-color: hsl(120, 100.00%, 93.31%); opacity: 0.82" title="0.015">l</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.014">d</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 90.03%); opacity: 0.83" title="0.026">n</span><span style="background-color: hsl(120, 100.00%, 92.77%); opacity: 0.82" title="0.017">o</span><span style="background-color: hsl(120, 100.00%, 92.51%); opacity: 0.82" title="0.018">t</span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 83.74%); opacity: 0.86" title="0.053">b</span><span style="background-color: hsl(120, 100.00%, 91.95%); opacity: 0.82" title="0.019">e</span><span style="background-color: hsl(120, 100.00%, 95.28%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 80.25%); opacity: 0.87" title="0.070">c</span><span style="background-color: hsl(120, 100.00%, 84.00%); opacity: 0.85" title="0.052">h</span><span style="background-color: hsl(120, 100.00%, 89.48%); opacity: 0.83" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 87.30%); opacity: 0.84" title="0.037">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.192">g</span><span style="background-color: hsl(120, 100.00%, 88.81%); opacity: 0.83" title="0.031">e</span><span style="background-color: hsl(120, 100.00%, 87.54%); opacity: 0.84" title="0.036">d</span><span style="background-color: hsl(120, 100.00%, 94.07%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 89.50%); opacity: 0.83" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 83.33%); opacity: 0.86" title="0.055">n</span><span style="background-color: hsl(120, 100.00%, 88.63%); opacity: 0.83" title="0.032">d</span><span style="background-color: hsl(120, 100.00%, 94.61%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.029">d</span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 81.93%); opacity: 0.86" title="0.062">b</span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.037">i</span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.020">t</span><span style="background-color: hsl(120, 100.00%, 93.96%); opacity: 0.81" title="0.013">e</span><span style="background-color: hsl(120, 100.00%, 94.48%); opacity: 0.81" title="0.011">d</span><span style="background-color: hsl(120, 100.00%, 98.05%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 96.39%); opacity: 0.81" title="0.006">f</span><span style="background-color: hsl(120, 100.00%, 97.96%); opacity: 0.80" title="0.003">o</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="opacity: 0.80">the private c</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.18%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.97%); opacity: 0.80" title="0.003">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    max_pooling1d_2



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 98.42%); opacity: 0.80" title="0.003">f</span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.004">i</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.003">r</span><span style="background-color: hsl(120, 100.00%, 96.67%); opacity: 0.81" title="0.009">s</span><span style="background-color: hsl(120, 100.00%, 98.05%); opacity: 0.80" title="0.004">t</span><span style="background-color: hsl(120, 100.00%, 98.96%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 97.68%); opacity: 0.80" title="0.006">o</span><span style="background-color: hsl(120, 100.00%, 96.44%); opacity: 0.81" title="0.010">f</span><span style="background-color: hsl(120, 100.00%, 98.71%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 97.31%); opacity: 0.80" title="0.007">a</span><span style="background-color: hsl(120, 100.00%, 95.95%); opacity: 0.81" title="0.012">l</span><span style="background-color: hsl(120, 100.00%, 95.50%); opacity: 0.81" title="0.014">l</span><span style="background-color: hsl(120, 100.00%, 97.95%); opacity: 0.80" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 85.86%); opacity: 0.85" title="0.074">I</span><span style="background-color: hsl(120, 100.00%, 97.60%); opacity: 0.80" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 90.00%); opacity: 0.83" title="0.045">s</span><span style="background-color: hsl(120, 100.00%, 92.00%); opacity: 0.82" title="0.033">h</span><span style="background-color: hsl(120, 100.00%, 93.90%); opacity: 0.81" title="0.022">o</span><span style="background-color: hsl(120, 100.00%, 79.44%); opacity: 0.88" title="0.126">u</span><span style="background-color: hsl(120, 100.00%, 91.57%); opacity: 0.82" title="0.035">l</span><span style="background-color: hsl(120, 100.00%, 92.27%); opacity: 0.82" title="0.031">d</span><span style="background-color: hsl(120, 100.00%, 95.99%); opacity: 0.81" title="0.012"> </span><span style="background-color: hsl(120, 100.00%, 87.50%); opacity: 0.84" title="0.062">n</span><span style="background-color: hsl(120, 100.00%, 90.77%); opacity: 0.82" title="0.040">o</span><span style="background-color: hsl(120, 100.00%, 90.30%); opacity: 0.83" title="0.043">t</span><span style="background-color: hsl(120, 100.00%, 95.07%); opacity: 0.81" title="0.016"> </span><span style="background-color: hsl(120, 100.00%, 79.94%); opacity: 0.87" title="0.122">b</span><span style="background-color: hsl(120, 100.00%, 90.34%); opacity: 0.83" title="0.043">e</span><span style="background-color: hsl(120, 100.00%, 94.46%); opacity: 0.81" title="0.019"> </span><span style="background-color: hsl(120, 100.00%, 77.63%); opacity: 0.89" title="0.142">c</span><span style="background-color: hsl(120, 100.00%, 82.59%); opacity: 0.86" title="0.099">h</span><span style="background-color: hsl(120, 100.00%, 88.94%); opacity: 0.83" title="0.052">a</span><span style="background-color: hsl(120, 100.00%, 87.05%); opacity: 0.84" title="0.065">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.326">g</span><span style="background-color: hsl(120, 100.00%, 89.00%); opacity: 0.83" title="0.052">e</span><span style="background-color: hsl(120, 100.00%, 87.97%); opacity: 0.84" title="0.059">d</span><span style="background-color: hsl(120, 100.00%, 94.38%); opacity: 0.81" title="0.020"> </span><span style="background-color: hsl(120, 100.00%, 90.27%); opacity: 0.83" title="0.043">a</span><span style="background-color: hsl(120, 100.00%, 84.90%); opacity: 0.85" title="0.081">n</span><span style="background-color: hsl(120, 100.00%, 89.95%); opacity: 0.83" title="0.045">d</span><span style="background-color: hsl(120, 100.00%, 95.36%); opacity: 0.81" title="0.015"> </span><span style="background-color: hsl(120, 100.00%, 91.07%); opacity: 0.82" title="0.038">d</span><span style="background-color: hsl(120, 100.00%, 92.86%); opacity: 0.82" title="0.028">e</span><span style="background-color: hsl(120, 100.00%, 86.20%); opacity: 0.84" title="0.071">b</span><span style="background-color: hsl(120, 100.00%, 90.89%); opacity: 0.82" title="0.039">i</span><span style="background-color: hsl(120, 100.00%, 94.58%); opacity: 0.81" title="0.019">t</span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.012">e</span><span style="background-color: hsl(120, 100.00%, 96.75%); opacity: 0.81" title="0.009">d</span><span style="background-color: hsl(120, 100.00%, 99.08%); opacity: 0.80" title="0.001"> </span><span style="opacity: 0.80">for the private c</span><span style="background-color: hsl(120, 100.00%, 99.73%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.65%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.95%); opacity: 0.80" title="0.002">a</span><span style="background-color: hsl(120, 100.00%, 98.09%); opacity: 0.80" title="0.004">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



.. parsed-literal::

    global_average_pooling1d_1



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="opacity: 0.80">first of all I should not be charged and debited fo</span><span style="background-color: hsl(120, 100.00%, 97.88%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 98.12%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 94.67%); opacity: 0.81" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 90.52%); opacity: 0.83" title="0.000">h</span><span style="background-color: hsl(120, 100.00%, 93.11%); opacity: 0.82" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 95.65%); opacity: 0.81" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 69.10%); opacity: 0.94" title="0.000">p</span><span style="background-color: hsl(120, 100.00%, 88.96%); opacity: 0.83" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 82.82%); opacity: 0.86" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.001">v</span><span style="background-color: hsl(120, 100.00%, 87.78%); opacity: 0.84" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 85.22%); opacity: 0.85" title="0.000">t</span><span style="background-color: hsl(120, 100.00%, 86.23%); opacity: 0.84" title="0.000">e</span><span style="background-color: hsl(120, 100.00%, 91.98%); opacity: 0.82" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 67.09%); opacity: 0.95" title="0.000">c</span><span style="background-color: hsl(120, 100.00%, 84.03%); opacity: 0.85" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 81.05%); opacity: 0.87" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 90.40%); opacity: 0.83" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 76.03%); opacity: 0.90" title="0.000">l</span><span style="background-color: hsl(120, 100.00%, 78.74%); opacity: 0.88" title="0.000">o</span><span style="background-color: hsl(120, 100.00%, 80.62%); opacity: 0.87" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 67.47%); opacity: 0.95" title="0.000">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



It should make sense for a Convolutional network that later layers pick
up “higher level” information than earlier “lower level” layers (such as
the Embedding layer that only highlights characters).

Choosing a classification target to focus on via ``targets``
------------------------------------------------------------

In the last text we saw that it could be classified into more than just
one category.

We can use ELI5 to “force” the network to classify the input into a
certain category, and then highlight evidence for that category.

We use the ``targets`` argument for this. We pass a list that contains
integer indices. Those indices represent a class in the final output
layer.

Let’s check two sensible categories

.. code:: ipython3

    debt_idx = 0  # we get this from the labels' index
    loan_idx = 1

.. code:: ipython3

    print('debt collection')
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t, 
                                 targets=[debt_idx], pad_token='<PAD>'))
    
    print('consumer loan')
    display(eli5.show_prediction(multicls_model, complaint, tokens=complaint_t, 
                                 targets=[loan_idx], pad_token='<PAD>'))


.. parsed-literal::

    debt collection



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 90.90%); opacity: 0.82" title="0.019">f</span><span style="background-color: hsl(120, 100.00%, 89.91%); opacity: 0.83" title="0.021">i</span><span style="background-color: hsl(120, 100.00%, 92.28%); opacity: 0.82" title="0.015">r</span><span style="background-color: hsl(120, 100.00%, 84.13%); opacity: 0.85" title="0.041">s</span><span style="background-color: hsl(120, 100.00%, 91.02%); opacity: 0.82" title="0.018">t</span><span style="background-color: hsl(120, 100.00%, 95.31%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 89.70%); opacity: 0.83" title="0.022">o</span><span style="background-color: hsl(120, 100.00%, 84.37%); opacity: 0.85" title="0.040">f</span><span style="background-color: hsl(120, 100.00%, 94.38%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 89.49%); opacity: 0.83" title="0.023">a</span><span style="background-color: hsl(120, 100.00%, 85.63%); opacity: 0.85" title="0.036">l</span><span style="background-color: hsl(120, 100.00%, 85.22%); opacity: 0.85" title="0.037">l</span><span style="background-color: hsl(120, 100.00%, 93.68%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.153">I</span><span style="background-color: hsl(120, 100.00%, 93.75%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 75.80%); opacity: 0.90" title="0.075">s</span><span style="background-color: hsl(120, 100.00%, 81.93%); opacity: 0.86" title="0.049">h</span><span style="background-color: hsl(120, 100.00%, 88.07%); opacity: 0.84" title="0.027">o</span><span style="background-color: hsl(120, 100.00%, 66.12%); opacity: 0.96" title="0.121">u</span><span style="background-color: hsl(120, 100.00%, 88.31%); opacity: 0.83" title="0.026">l</span><span style="background-color: hsl(120, 100.00%, 91.03%); opacity: 0.82" title="0.018">d</span><span style="background-color: hsl(120, 100.00%, 96.00%); opacity: 0.81" title="0.006"> </span><span style="background-color: hsl(120, 100.00%, 89.11%); opacity: 0.83" title="0.024">n</span><span style="background-color: hsl(120, 100.00%, 93.02%); opacity: 0.82" title="0.013">o</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.011">t</span><span style="background-color: hsl(120, 100.00%, 97.36%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.016">b</span><span style="background-color: hsl(120, 100.00%, 97.39%); opacity: 0.80" title="0.003">e</span><span style="background-color: hsl(120, 100.00%, 99.36%); opacity: 0.80" title="0.000"> </span><span style="opacity: 0.80">charged and debited for the private car loan</span>
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
            <span style="opacity: 0.80">f</span><span style="background-color: hsl(120, 100.00%, 99.46%); opacity: 0.80" title="0.000">i</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000">r</span><span style="background-color: hsl(120, 100.00%, 98.45%); opacity: 0.80" title="0.002">s</span><span style="background-color: hsl(120, 100.00%, 99.00%); opacity: 0.80" title="0.001">t</span><span style="background-color: hsl(120, 100.00%, 99.35%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 98.35%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 97.22%); opacity: 0.80" title="0.004">f</span><span style="background-color: hsl(120, 100.00%, 98.92%); opacity: 0.80" title="0.001"> </span><span style="background-color: hsl(120, 100.00%, 97.89%); opacity: 0.80" title="0.003">a</span><span style="background-color: hsl(120, 100.00%, 97.01%); opacity: 0.80" title="0.005">l</span><span style="background-color: hsl(120, 100.00%, 96.82%); opacity: 0.81" title="0.005">l</span><span style="background-color: hsl(120, 100.00%, 98.60%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 90.42%); opacity: 0.83" title="0.025">I</span><span style="background-color: hsl(120, 100.00%, 98.37%); opacity: 0.80" title="0.002"> </span><span style="background-color: hsl(120, 100.00%, 93.19%); opacity: 0.82" title="0.015">s</span><span style="background-color: hsl(120, 100.00%, 94.54%); opacity: 0.81" title="0.011">h</span><span style="background-color: hsl(120, 100.00%, 95.59%); opacity: 0.81" title="0.008">o</span><span style="background-color: hsl(120, 100.00%, 84.32%); opacity: 0.85" title="0.050">u</span><span style="background-color: hsl(120, 100.00%, 93.31%); opacity: 0.82" title="0.015">l</span><span style="background-color: hsl(120, 100.00%, 93.67%); opacity: 0.81" title="0.014">d</span><span style="background-color: hsl(120, 100.00%, 96.74%); opacity: 0.81" title="0.005"> </span><span style="background-color: hsl(120, 100.00%, 90.03%); opacity: 0.83" title="0.026">n</span><span style="background-color: hsl(120, 100.00%, 92.77%); opacity: 0.82" title="0.017">o</span><span style="background-color: hsl(120, 100.00%, 92.51%); opacity: 0.82" title="0.018">t</span><span style="background-color: hsl(120, 100.00%, 96.13%); opacity: 0.81" title="0.007"> </span><span style="background-color: hsl(120, 100.00%, 83.74%); opacity: 0.86" title="0.053">b</span><span style="background-color: hsl(120, 100.00%, 91.95%); opacity: 0.82" title="0.019">e</span><span style="background-color: hsl(120, 100.00%, 95.28%); opacity: 0.81" title="0.009"> </span><span style="background-color: hsl(120, 100.00%, 80.25%); opacity: 0.87" title="0.070">c</span><span style="background-color: hsl(120, 100.00%, 84.00%); opacity: 0.85" title="0.052">h</span><span style="background-color: hsl(120, 100.00%, 89.48%); opacity: 0.83" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 87.30%); opacity: 0.84" title="0.037">r</span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.192">g</span><span style="background-color: hsl(120, 100.00%, 88.81%); opacity: 0.83" title="0.031">e</span><span style="background-color: hsl(120, 100.00%, 87.54%); opacity: 0.84" title="0.036">d</span><span style="background-color: hsl(120, 100.00%, 94.07%); opacity: 0.81" title="0.013"> </span><span style="background-color: hsl(120, 100.00%, 89.50%); opacity: 0.83" title="0.028">a</span><span style="background-color: hsl(120, 100.00%, 83.33%); opacity: 0.86" title="0.055">n</span><span style="background-color: hsl(120, 100.00%, 88.63%); opacity: 0.83" title="0.032">d</span><span style="background-color: hsl(120, 100.00%, 94.61%); opacity: 0.81" title="0.011"> </span><span style="background-color: hsl(120, 100.00%, 89.29%); opacity: 0.83" title="0.029">d</span><span style="background-color: hsl(120, 100.00%, 91.10%); opacity: 0.82" title="0.022">e</span><span style="background-color: hsl(120, 100.00%, 81.93%); opacity: 0.86" title="0.062">b</span><span style="background-color: hsl(120, 100.00%, 87.28%); opacity: 0.84" title="0.037">i</span><span style="background-color: hsl(120, 100.00%, 91.89%); opacity: 0.82" title="0.020">t</span><span style="background-color: hsl(120, 100.00%, 93.96%); opacity: 0.81" title="0.013">e</span><span style="background-color: hsl(120, 100.00%, 94.48%); opacity: 0.81" title="0.011">d</span><span style="background-color: hsl(120, 100.00%, 98.05%); opacity: 0.80" title="0.003"> </span><span style="background-color: hsl(120, 100.00%, 96.39%); opacity: 0.81" title="0.006">f</span><span style="background-color: hsl(120, 100.00%, 97.96%); opacity: 0.80" title="0.003">o</span><span style="background-color: hsl(120, 100.00%, 98.49%); opacity: 0.80" title="0.002">r</span><span style="background-color: hsl(120, 100.00%, 99.57%); opacity: 0.80" title="0.000"> </span><span style="opacity: 0.80">the private c</span><span style="background-color: hsl(120, 100.00%, 99.50%); opacity: 0.80" title="0.000">a</span><span style="background-color: hsl(120, 100.00%, 99.05%); opacity: 0.80" title="0.001">r</span><span style="background-color: hsl(120, 100.00%, 99.38%); opacity: 0.80" title="0.000"> </span><span style="background-color: hsl(120, 100.00%, 98.18%); opacity: 0.80" title="0.002">l</span><span style="background-color: hsl(120, 100.00%, 98.46%); opacity: 0.80" title="0.002">o</span><span style="background-color: hsl(120, 100.00%, 98.70%); opacity: 0.80" title="0.001">a</span><span style="background-color: hsl(120, 100.00%, 97.97%); opacity: 0.80" title="0.003">n</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



Sensible at least a little bit?

Note that we can use the IPython ``display()`` call to render HTML if it
is not the last value in a call.

How it works - ``explain_prediction()`` and ``format_as_html()``.
-----------------------------------------------------------------

What we have seen so far is calls to ``show_prediction()``. What this
function actually does is call ``explain_prediction()`` to produce an
``Explanation`` object, and then passes that object to
``format_as_html()`` to produce highlighted HTML.

Let’s check each of these steps

.. code:: ipython3

    E = eli5.explain_prediction(binary_model, review, tokens=review_t)

This is an ``Explanation`` object

.. code:: ipython3

    repr(E)




.. parsed-literal::

    "Explanation(estimator='sequential_1', description='\\nGrad-CAM visualization for classification tasks; \\noutput is explanation object that contains a heatmap.\\n', error='', method='Grad-CAM', is_regression=False, targets=[TargetExplanation(target=0, feature_weights=None, proba=None, score=0.5912496, weighted_spans=WeightedSpans(docs_weighted_spans=[DocWeightedSpans(document='<START> hello this is great but not so great', spans=[('<START>', [(0, 7)], 0.04707520170978796), ('hello', [(8, 13)], 0.050255952907264145), ('this', [(14, 18)], 0.051052169244030665), ('is', [(19, 21)], 0.051517811452868045), ('great', [(22, 27)], 0.048895430725679034), ('but', [(28, 31)], 0.03907256391948977), ('not', [(32, 35)], 0.03281831360072829), ('so', [(36, 38)], 0.028806375945350737), ('great', [(39, 44)], 0.023793572530848905)], preserve_density=None, vec_name=None)], other=None), heatmap=array([0.0470752 , 0.05025595, 0.05105217, 0.05151781, 0.04889543,\n       0.03907256, 0.03281831, 0.02880638, 0.02379357]))], feature_importances=None, decision_tree=None, highlight_spaces=None, transition_features=None, image=None, layer=<keras.layers.wrappers.Bidirectional object at 0x7f7354366dd0>)"



We can check the name of the hidden layer that was used for producing
the heatmap

.. code:: ipython3

    E.layer




.. parsed-literal::

    <keras.layers.wrappers.Bidirectional at 0x7f7354366dd0>



We can get the predicted class and the value for the prediction

.. code:: ipython3

    target = E.targets[0]
    print(target.target, target.score)


.. parsed-literal::

    0 0.5912496


We can also check the produced Grad-CAM ``heatmap`` found on each item
in ``targets``. You can think of this as an array of “importances” for
tokens (after padding is removed and the heatmap is resized).

.. code:: ipython3

    heatmap = target.heatmap
    print(heatmap)
    print(len(heatmap))


.. parsed-literal::

    [0.0470752  0.05025595 0.05105217 0.05151781 0.04889543 0.03907256
     0.03281831 0.02880638 0.02379357]
    9


The highlighting for each token is stored in a ``WeightedSpans`` object
(specifically the ``DocWeightedSpans`` object)

.. code:: ipython3

    weighted_spans = target.weighted_spans
    print(weighted_spans)
    
    doc_ws = weighted_spans.docs_weighted_spans[0]
    print(doc_ws)


.. parsed-literal::

    WeightedSpans(docs_weighted_spans=[DocWeightedSpans(document='<START> hello this is great but not so great', spans=[('<START>', [(0, 7)], 0.04707520170978796), ('hello', [(8, 13)], 0.050255952907264145), ('this', [(14, 18)], 0.051052169244030665), ('is', [(19, 21)], 0.051517811452868045), ('great', [(22, 27)], 0.048895430725679034), ('but', [(28, 31)], 0.03907256391948977), ('not', [(32, 35)], 0.03281831360072829), ('so', [(36, 38)], 0.028806375945350737), ('great', [(39, 44)], 0.023793572530848905)], preserve_density=None, vec_name=None)], other=None)
    DocWeightedSpans(document='<START> hello this is great but not so great', spans=[('<START>', [(0, 7)], 0.04707520170978796), ('hello', [(8, 13)], 0.050255952907264145), ('this', [(14, 18)], 0.051052169244030665), ('is', [(19, 21)], 0.051517811452868045), ('great', [(22, 27)], 0.048895430725679034), ('but', [(28, 31)], 0.03907256391948977), ('not', [(32, 35)], 0.03281831360072829), ('so', [(36, 38)], 0.028806375945350737), ('great', [(39, 44)], 0.023793572530848905)], preserve_density=None, vec_name=None)


Observe the ``document`` attribute and ``spans``

.. code:: ipython3

    print(doc_ws.document)
    print(doc_ws.spans)


.. parsed-literal::

    <START> hello this is great but not so great
    [('<START>', [(0, 7)], 0.04707520170978796), ('hello', [(8, 13)], 0.050255952907264145), ('this', [(14, 18)], 0.051052169244030665), ('is', [(19, 21)], 0.051517811452868045), ('great', [(22, 27)], 0.048895430725679034), ('but', [(28, 31)], 0.03907256391948977), ('not', [(32, 35)], 0.03281831360072829), ('so', [(36, 38)], 0.028806375945350737), ('great', [(39, 44)], 0.023793572530848905)]


The ``document`` is the “stringified” version of ``tokens``. If you have
a custom “tokens -> string” algorithm you may want to set this attribute
yourself.

The ``spans`` object is a list of weights for each character in
``document``. We use the indices in ``document`` string to indicate
which characters should be weighted with a specific value.

Let’s format this. HTML formatter is what should be used here.

.. code:: ipython3

    import eli5.formatters.fields as fields
    F = eli5.format_as_html(E, show=fields.WEIGHTS)

We pass a ``show`` argument to not display the method name or its
description (“Grad-CAM”). See ``eli5.format_as_html()`` for a list of
all supported arguments.

The output is an HTML-encoded string.

.. code:: ipython3

    repr(F)




.. parsed-literal::

    '\'\\n    <style>\\n    table.eli5-weights tr:hover {\\n        filter: brightness(85%);\\n    }\\n</style>\\n\\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n        \\n\\n    \\n\\n        \\n            \\n                \\n                \\n            \\n        \\n\\n        \\n\\n\\n    <p style="margin-bottom: 2.5em; margin-top:-0.5em;">\\n        <span style="background-color: hsl(120, 100.00%, 62.45%); opacity: 0.98" title="0.047">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.69%); opacity: 1.00" title="0.050">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.25%); opacity: 1.00" title="0.051">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.052">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.26%); opacity: 0.89" title="0.024">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 67.04%); opacity: 0.95" title="0.039">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.83%); opacity: 0.93" title="0.033">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.37%); opacity: 0.91" title="0.029">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.66%); opacity: 0.85" title="0.012">great</span>\\n    </p>\\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n    \\n\\n\\n\\n\''



Convert the string to an HTML object and display it in an IPython
notebook

.. code:: ipython3

    display(HTML(F))



.. raw:: html

    
        <style>
        table.eli5-weights tr:hover {
            filter: brightness(85%);
        }
    </style>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        <p style="margin-bottom: 2.5em; margin-top:-0.5em;">
            <span style="background-color: hsl(120, 100.00%, 62.45%); opacity: 0.98" title="0.047">&lt;START&gt;</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.69%); opacity: 1.00" title="0.050">hello</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.25%); opacity: 1.00" title="0.051">this</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.052">is</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 76.26%); opacity: 0.89" title="0.024">great</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 67.04%); opacity: 0.95" title="0.039">but</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.83%); opacity: 0.93" title="0.033">not</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 73.37%); opacity: 0.91" title="0.029">so</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 85.66%); opacity: 0.85" title="0.012">great</span>
        </p>
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



Notes on results
----------------

Multi-label classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Did not really work for us. Got non-sensical explanations. Feel free to
send feedback if you could explain multi-label models.
