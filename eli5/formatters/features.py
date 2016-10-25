# -*- coding: utf-8 -*-


class EscapedFeatureName(object):
    """ Feature name that does not need any additional escaping.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return str(self) == other

    def __repr__(self):
        return '<{} {!r}>'.format(self.__class__.__name__, self.value)
