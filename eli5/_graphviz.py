# -*- coding: utf-8 -*-
import graphviz


def is_supported():
    # type: () -> bool
    try:
        graphviz.Graph().pipe('svg')
        return True
    except RuntimeError:
        return False


def dot2svg(dot):
    # type: (str) -> str
    """ Render Graphviz data to SVG """
    svg = graphviz.Source(dot).pipe(format='svg').decode('utf8')  # type: str
    # strip doctype and xml declaration
    svg = svg[svg.index('<svg'):]
    return svg
