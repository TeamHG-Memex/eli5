# -*- coding: utf-8 -*-

from jinja2 import Environment, PackageLoader


template_env = Environment(
    loader=PackageLoader('eli5', 'templates'),
    extensions=['jinja2.ext.with_'])


def format_as_html(explanation, include_styles=True, force_weights=False):
    template = template_env.get_template('explain.html')
    return template.render(
        include_styles=include_styles,
        force_weights=force_weights,
        **explanation)


def format_html_styles():
    return template_env.get_template('styles.html').render()
