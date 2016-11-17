# -*- coding: utf-8 -*-
"""
Functions to convert explanations to human-digestable formats.

TODO: IPython integration, customizability.
"""

from .text import format_as_text
from .html import format_as_html, format_html_styles
from . import fields
from .features import FormattedFeatureName
from .as_dict import format_as_dict
