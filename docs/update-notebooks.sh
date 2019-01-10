#!/usr/bin/env bash

# scikit-learn text processing tutorial
jupyter nbconvert \
        --to rst \
        --stdout \
        '../notebooks/Debugging scikit-learn text classification pipeline.ipynb' \
        > source/_notebooks/debug-sklearn-text.rst

sed -i '' 's/``InvertableHashingVectorizer``/:class:`~.InvertableHashingVectorizer`/g' \
    source/_notebooks/debug-sklearn-text.rst

sed -i '' 's/``eli5.show_weights``/:func:`eli5.show_weights`/g' \
    source/_notebooks/debug-sklearn-text.rst

# sklearn-crfsuite tutorial
jupyter nbconvert \
        --to rst \
        --stdout \
        '../notebooks/sklearn-crfsuite.ipynb' \
        > source/_notebooks/debug-sklearn-crfsuite.rst

sed -i '' 's/class="eli5-transition-features"/class="docutils"/g' \
    source/_notebooks/debug-sklearn-crfsuite.rst


# TextExplainer (debugging black-box text classifiers)
jupyter nbconvert \
        --to rst \
        --stdout \
        '../notebooks/TextExplainer.ipynb' \
        > source/_notebooks/text-explainer.rst

sed -i '' 's/``TextExplainer``/:class:`~.TextExplainer`/g' \
    source/_notebooks/text-explainer.rst

sed -i '' 's/``TextExplainer.fit``/:meth:`~.TextExplainer.fit`/g' \
    source/_notebooks/text-explainer.rst

sed -i '' 's/``MaskingTextSampler``/:class:`~.MaskingTextSampler`/g' \
    source/_notebooks/text-explainer.rst

sed -i '' 's/``MaskingTextSamplers``/:class:`~.MaskingTextSamplers`/g' \
    source/_notebooks/text-explainer.rst

sed -i '' 's/<svg width="7..pt" height="280pt"/<svg width="520pt" height="180pt"/g' \
    source/_notebooks/text-explainer.rst

# xgboost-titanic tutorial
jupyter nbconvert \
        --to rst \
        --stdout \
        '../notebooks/xgboost-titanic.ipynb' \
        > source/_notebooks/xgboost-titanic.rst
sed -i '' 's/``eli5.show_weights``/:func:`eli5.show_weights`/g' \
    source/_notebooks/xgboost-titanic.rst
sed -i '' 's/``eli5.show_prediction``/:func:`eli5.show_prediction`/g' \
    source/_notebooks/xgboost-titanic.rst

# LIME
#jupyter nbconvert \
#        --to rst \
#        '../notebooks/LIME and synthetic data.ipynb'
#rm -r source/_notebooks/LIME
#mv '../notebooks/LIME and synthetic data_files' 'source/_notebooks/LIME and synthetic data_files'
#mv '../notebooks/LIME and synthetic data.rst' source/_notebooks/lime-synthetic.rst
