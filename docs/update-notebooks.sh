#!/usr/bin/env bash

# scikit-learn text processing tutorial
jupyter nbconvert \
        --to rst \
        --stdout \
        '../notebooks/Debugging scikit-learn text classification pipeline.ipynb' \
        > source/_notebooks/debug-sklearn-text.rst

sed -i '' 's/InvertableHashingVectorizer\./:class:`~.InvertableHashingVectorizer`/g' \
    source/_notebooks/debug-sklearn-text.rst

sed -i '' 's/eli5.explain\\_weights/:func:`eli5.explain_weights`/g' \
    source/_notebooks/debug-sklearn-text.rst

# sklearn-crfsuite tutorial
jupyter nbconvert \
        --to rst \
        --stdout \
        '../notebooks/sklearn-crfsuite.ipynb' \
        > source/_notebooks/debug-sklearn-crfsuite.rst

sed -i '' 's/class="eli5-transition-features"/class="docutils"/g' \
    source/_notebooks/debug-sklearn-crfsuite.rst
