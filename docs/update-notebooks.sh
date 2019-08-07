#!/usr/bin/env bash

# pandoc is required to convert to rst

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


# Keras Grad-CAM (keras-image-classifiers)
# we execute the notebook as a form of testing
PYTHONPATH=$PWD/.. jupyter nbconvert \
        --to rst \
        --ExecutePreprocessor.timeout=180 \
        --execute \
        '../notebooks/keras-image-classifiers.ipynb'
mv ../notebooks/keras-image-classifiers.rst \
    source/_notebooks/
rm -r source/_notebooks/keras-image-classifiers_files
mv ../notebooks/keras-image-classifiers_files/ \
    source/_notebooks/
sed -i 's&.. image:: keras-image-classifiers_files/&.. image:: ../_notebooks/keras-image-classifiers_files/&g' \
    source/_notebooks/keras-image-classifiers.rst


# Keras text Grad-CAM (keras-text-classifiers)
# PYTHONPATH=$PWD/.. jupyter nbconvert \
#         --to rst \
#         --ExecutePreprocessor.timeout=180 \
#         --execute \
#         '../notebooks/'
# mv ../notebooks/keras-image-classifiers.rst \
#     source/_notebooks/
jupyter nbconvert \
        --to rst \
        --stdout \
        '../notebooks/keras-text-classifiers.ipynb' \
        > source/_notebooks/keras-text-classifiers.rst


# to only execute one section: (replace # Keras with section comment)
# sed -n '/# Keras/,/^$/p' update-notebooks.sh | bash