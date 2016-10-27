# -*- coding: utf-8 -*-

import six


class FormattedFeatureName(object):
    """ Feature name that does not need any additional escaping.
    """
    def __init__(self, value):
        if not isinstance(value, six.string_types):
            raise TypeError('"value" must be a string, got {} instead'
                            .format(type(value)))
        self.value = value

    def format(self):
        return self.value

    def __eq__(self, other):
        return (isinstance(other, FormattedFeatureName) and
                self.value == other.value)

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return '<{} {!r}>'.format(self.__class__.__name__, self.value)
