class State(object):
    __slots__ = ("c", "h")

    def __init__(self, c, h):
        self.c = c
        self.h = h

    def __repr__(self):
        return "{0}({1}, {2})".format(self.__class__.__name__,
                                      self.c,
                                      self.h)
