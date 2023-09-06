#import numpy as np

def Index2Coord(index, x_range=30):
    x = index // x_range
    y = index % x_range
    return x, y

def Coord2Index(x, y, x_range=30):
    index = x * x_range + y
    return index