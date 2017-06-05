# -*- coding: utf-8 -*-
"""
Functions to convert explanations to human-digestible formats.

TODO: IPython integration, customizability.
"""

from .text import format_as_text
from .html import format_as_html, format_html_styles
from .as_dict import format_as_dict
from . import fields
from .features import FormattedFeatureName
