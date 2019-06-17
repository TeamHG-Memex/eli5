# -*- coding: utf-8 -*-
from typing import Any

from PIL import Image # type: ignore

from .base import Explanation
from .formatters import format_as_image


def display_prediction_image(expl, **format_kwargs):
    # type: (Explanation, **Any) -> Image
    """ 
    Show the heatmap and image overlay as a PIL image
    displayable in an IPython cell.

    Requires ``matplotlib`` and ``Pillow`` as extra dependencies.
    
    Parameters
    ----------
    expl : Explanation
        Explanation object with ``image`` and ``heatmap`` attributes set.
    
    format_kwargs : **kwargs
        Keyword arguments passed to :func:`eli5.format_as_image`

    Returns
    -------
    PIL.Image.Image
        Final image with the heatmap over it, as a Pillow Image object
        that can be displayed in an IPython cell.

        Note that to display the image in a loop, function, or other case,
        use IPython.display.display::

            from IPython.display import display
            for cls_idx in [0, 432]:
                display(eli5.show_prediction(clf, doc, targets=[cls_idx]))
    """
    overlay = format_as_image(expl, **format_kwargs)
    return overlay