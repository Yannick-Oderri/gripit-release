from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# -------------------------------------------------------------------------------
# Name:        lineseg
# Purpose:     Forms straight line segments
#              from an edge list.
# -------------------------------------------------------------------------------
#
# Usage:    seglist = lineseg(edgelist, tol)
#
# Arguments:
#           edgelist - List of arrays of edgelists where each edgelist
#                      is an Nx2 array of (row, col) coordinates.
#           tol      - Maximum deviation from straight line before a
#                      segment is broken in two (measured in pixels).
#
# Returns:
#           seglist  - A list of arrays where each seglist is a subsampling of
#                      its corresponding edgelist such that straight line
#                      segments points do not deviate from the original points by
#                      more than tol.
#
# See also: MAXLINEDEV
#
# Author: Jordan Zhu
#
# May 2016 - Original version.
# July 2016 - Reshapes the input array for better processing.

from builtins import range
from future import standard_library
standard_library.install_aliases()
import numpy as np
from gripit.edgelib.maxlinedev import maxlinedev
import gripit.edgelib.util as util

def line_seg(edgelist, tol):
    num_contours = edgelist.shape[0]

    # reshape array -
    temp = []
    for i in range(num_contours):
        arr = np.squeeze(edgelist[i])
        temp.append(arr)


    seglist = []

    for i in range(num_contours):
        num_edges = edgelist[i].shape[0]

        # Fill in the x and y coordinate matrices.
        x = np.empty(num_edges)
        y = np.empty(num_edges)
        np.copyto(x, temp[i][:, 0])
        np.copyto(y, temp[i][:, 1])

        # Beginning and endpoints in edge segment being considered.
        first = 0
        last = num_edges - 1

        # We can add the first point right away since
        # it's going to be the beginning of any created edge segment.
        list = np.array([[x[first], y[first]]], dtype=np.int32)
        num_pts = 1

        while first < last:
            # Find the size and index of maximum deviation.
            (maxdev, index) = maxlinedev(x[first:last], y[first:last])

            while maxdev > tol:
                # Shorten the line to point of max deviation.
                last = first + index
                # Double check
                (maxdev, index) = maxlinedev(x[first:last], y[first:last])
            # end - while

            num_pts += 1
            list = np.concatenate((list, np.array([[x[last], y[last]]], dtype=np.int32)))

            first = last
            last = num_edges - 1
        # end-while

        # Add the edge segment lists to a seglist container.
        # list = np.asarray(list, dtype=np.int32)
        seglist.append(list)
    # end-for
    seglist = np.array(seglist)

    # ADVANCED SLICING
    for i in range(seglist.shape[0]):
        util.swap_cols(seglist[i], 0, 1)

    return seglist
# end-lineseg


#
# Algorithm:
#   Create line segments, given an list of pixel arrays
#   and a tolerance for how close the line segment should be to the edgels.
#
#   Run it for every set of edges in the list.
#   Create an output array to store the created line segments.
#
#   For every edge, assign x = first index, y = second index
#   First = 0, start of the edge array; last = length
#
#   Count the number of points
#   Add the first point to the seg list, since this is guaranteed
#   to be in the edge segment.
#
#   Loop while there are still segments to check.
#   Check every point in this iterations edge array and find point of max deviation.
#   If this point is higher than the tolerance,
#       break the line at this point.
#       Check the remaining points.
#
#   Increment the points counter.
#   Add the current last point to the seg list.
#
#   Make the current last point the new first point
#   And make the last point the length of the edge array.
#
