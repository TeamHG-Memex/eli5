# -*- coding: utf-8 -*-

try:
    from PIL import Image # type: ignore
except ImportError:
    PIL = None

from .base import Explanation
try:
    from .formatters import format_as_image
except ImportError:
    # matplotlib or Pillow is not available
    format_as_image = None


def display_prediction_image(expl, **format_kwargs):
    # type: (Explanation, **Any) -> Image
    """ 
    Show the heatmap and image overlay as a PIL image
    displayable in an IPython cell.

    Requires ``matplotlib`` and ``Pillow`` as extra dependencies.
    
    Parameters
    ----------
    expl : Explanation


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
    if format_as_image is None:
        # image display implementation not available
        # no formatting is done
        print('Dependencies are missing. No formatting will be done.')
        return expl
    else:
        overlay = format_as_image(expl, **format_kwargs)
        return overlay