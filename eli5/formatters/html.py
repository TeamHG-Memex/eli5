# -*- coding: utf-8 -*-

from jinja2 import Environment, PackageLoader


template_env = Environment(
    loader=PackageLoader('eli5', 'templates'),
    extensions=['jinja2.ext.with_'])


def format_as_html(explanation):
    template = template_env.get_template('explain.html')
    return template.render(**explanation)
