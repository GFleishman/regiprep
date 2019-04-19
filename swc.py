#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------- Imports ----------------------------------------------
import numpy as np

class swc_point:

    def __init__(self, identity):
        self.identity = identity
        self.type = 0
        self.coords = np.zeros(3)
        self.radius = 0
        self.parent = 0

    def translate(self, translation):
        self.coords += translation

    def print(self):
        ret = '{}'.format(self.identity)
        ret += ' {}'.format(self.type)
        ret += ' {} {} {}'.format(self.coords[0],
                                  self.coords[1],
                                  self.coords[2])
        ret += ' {}'.format(self.radius)
        ret += ' {}\n'.format(self.parent)
        return ret


class swc_object:

    def __init__(self):
        self.points = []
        self.offset = np.zeros(3)

    def add_point(self, point):
        if type(point) is swc_point:
            self.points.append(point)
        else:
            raise ValueError()

    def translate(self, translation):
        self.offset -= translation
        for p in self.points:
            p.translate(translation)
