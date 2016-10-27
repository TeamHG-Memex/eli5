# -*- coding: utf-8 -*-


class FormattedFeatureName(object):
    """ Feature name that does not need any additional escaping.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return (isinstance(other, FormattedFeatureName) and
                self.value == other.value)

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return '<{} {!r}>'.format(self.__class__.__name__, self.value)
