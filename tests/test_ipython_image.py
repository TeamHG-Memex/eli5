# -*- coding: utf-8 -*-

import pytest
pytest.importorskip('IPython')

import numpy as np

import eli5
from eli5.base import Explanation
from eli5 import ipython_image


def test_display_prediction_image_nodeps():
    # FIXME: problems with subsequent runs
    # # mock missing import
    # eli5.ipython_image.format_as_image = None 

    # mock explanation with images
    mock_expl = Explanation(
        'mock_estimator',
        # image=PIL.Image.open('tests/images/cat_dog.jpg'),
        image='somenonsense', # mock image (should be a PIL Image)
        heatmap=np.zeros((7, 7)),
    )

    formatted_expl = ipython_image.display_prediction_image(mock_expl)
    assert mock_expl is formatted_expl