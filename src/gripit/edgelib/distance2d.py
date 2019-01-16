from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import math


# Calculates the distance between the midpoint of a smaller line 
# from a larger line.


def distance2d(line1, line2):
    # LineFeature = [y1 x1 y2 x2 length slope];
    if line1[4] < line2[4]:
        big_line = line1
        other_line = line2
    else:
        big_line = line2
        other_line = line1

    # midpoints
    x0 = (big_line[1] + big_line[3]) / 2
    y0 = (big_line[0] + big_line[2]) / 2

    x1 = other_line[1]
    y1 = other_line[0]

    x2 = other_line[3]
    y2 = other_line[2]

    return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
