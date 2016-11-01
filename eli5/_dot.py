# -*- coding: utf-8 -*-
import graphviz


def dot2svg(dot):
    """ Render Graphviz data to SVG """
    svg = graphviz.Source(dot).pipe(format='svg').decode('utf8')
    # strip doctype and xml declaration
    svg = svg[svg.index('<svg'):]
    return svg
