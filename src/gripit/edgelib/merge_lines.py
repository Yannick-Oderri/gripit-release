from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import filter
from builtins import int
from future import standard_library
standard_library.install_aliases()
from math import sqrt, atan, degrees
from numpy import sort, unique, where, unravel_index, append, delete, array, concatenate, r_
from itertools import combinations, chain
from collections import Counter
import copy


def compare(s, t):
    return Counter(s) == Counter(t)


def math_stuff(x1, y1, x2, y2):
    slope = float((y2 - y1) / (x2 - x1) if ((x2 - x1) != 0) else float('inf'))
    line_len = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = degrees(atan(-slope))
    alpha = 180+alpha if alpha < 0 else alpha
    return slope, line_len, alpha


def squeeze_arr(arr):
    res = []
    for i in range(arr.shape[0]):
        res.append(arr[i][0])
    return res


def merge_lp(listpt, pt1, pt2, lind1, lind2):
    line_start = listpt[pt1]
    line_end = listpt[pt2]

    # Delete the listpoints that are going to be merged
    # listpt = delete(listpt, max(pt1, pt2))
    # listpt = delete(listpt, min(pt1, pt2))
    del listpt[max(pt1, pt2)]
    del listpt[min(pt1, pt2)]

    merged = r_[line_start, line_end]
    merged_new = unique(r_[line_start, line_end])
    # print(merged, "%d %d merged" %(pt1, pt2))
    # print(merged_new, "NEW merged")
    # listpt = append(listpt, [array(merged)], axis=0)
    listpt.append(merged_new)
    # print(listpt, "end of listpoint")

    return listpt

"""
def merge_listpoints(listpt, pt1, pt2, px1, px2):
    lp1 = listpt[pt1]
    lp2 = listpt[pt2]
    startpt1 = where(lp1 == px1)[0]
    startpt2 = where(lp1 == px2)[0]
    startpt3 = where(lp2 == px1)[0]
    startpt4 = where(lp2 == px2)[0]

    # print('pt1:', pt1, 'pt2:', pt2)
    print('px1:', px1, 'px2:', px2)
    print('startpt1:', startpt1, 'startpt2:', startpt2, 'startpt3:', startpt3, 'startpt4:', startpt4, '\n')
    # print('lp1', lp1, '\nlp2', lp2)

    if not startpt1 and startpt1.shape[0] < 1:
        # print('if')
        line_start = lp2
        line_end = lp1

        # print(startpt3)
        if startpt3 > 0:
            line_start = line_start[::-1]
        if startpt2 == 0:
            line_end = line_end[::-1]
    else:
        # print('else')
        line_start = lp1
        line_end = lp2

        if startpt1 > 0:
            line_start = line_start[::-1]
        if startpt4 == 0:
            line_end = line_end[::-1]

    # print(listpt)
    # print(listpt[max(pt1, pt2)])
    listpt = delete(listpt, max(pt1, pt2))
    listpt = delete(listpt, min(pt1, pt2))
    # del listpt[max(pt1, pt2)] # delete(listpt, max(pt1, pt2))
    # del listpt[min(pt1, pt2)] # listpt = delete(listpt, min(pt1, pt2))

    # print('listpt length', len(listpt))
    # print('line_start:', line_start)
    # # print(line_start[0][0:-1])
    # print('line_end:', line_end)
    # if len(line_end[0]) == 1:
    #     line_end[0] = array(list(line_end[0]))
    #     print('new line end', line_end[0])
    # print('line_start shape:', line_start[0].shape, 'line_end shape:', line_end[0].shape)
    # print('line start:', line_start[0:-1], '\nline end:', line_end)
    # merged = concatenate((line_start[0:-1], line_end))
    merged = r_[line_start, line_end]
    # print('concatenate:', merged)
    # listpt.append(merged)
    # print('index appended to', len(listpt) - 1)
    listpt = append(listpt, array(merged))
    # print('end of listpt', listpt[-1])
    # print('after length', len(listpt))
    # print(listpt)

    return listpt
"""

def relevant_lines(pairs, lines):
    pt1 = pairs[0]
    pt2 = pairs[1]
    line1 = lines[pt1]
    line2 = lines[pt2]
    alph1 = line1[6]
    alph2 = line2[6]
    temp1 = [line1[8], line1[9]]
    temp2 = [line2[8], line2[9]]
    return pt1, pt2, alph1, alph2, temp1, temp2


def merge_lines(lines, listpt, P):
    thresh = copy.deepcopy(P["sgmnt_threshold"]['value'])
    img_size = copy.deepcopy(P["img_size"])
    # lines format: y1, x1, y2, x2, length, slope, alpha, index, start_pt, end_pt
    # listpt = squeeze_arr(listpt)

    # All lines that can be merged. Merging lines
    # will group them together and then add the grouping to the list
    out = [[n] for n in range(0, lines.shape[0])]
    new_lp = listpt.tolist()

    # Get unique start and end points. These are what we check
    unique_pts = sort(unique(lines[:, 8:10]))

    for index, ptx in enumerate(unique_pts):
        # Test each combination of lines with this
        # point to see which ones we can merge.
        # Formula is combinations w/o repetitions (choose 2)
        pairs = list(combinations(list(where(lines == ptx)[0]), 2))
        # print(pairs)

        # Go to next iteration if there's no combinations
        if not pairs:
            continue
        for i, curr_pair in enumerate(pairs):
            pt1, pt2, alph1, alph2, temp1, temp2 = relevant_lines(curr_pair, lines)
            # Check that the lines are within the threshold and not coincident
            if abs(alph1 - alph2) > thresh or compare(temp1, temp2):
                continue

            # Get the unique start and end points of the two lines
            lind1, lind2 = sort([int(i) for i in list(filter(lambda e: e not in [ptx], chain(temp1 + temp2)))])
            y1, x1 = unravel_index([lind1], img_size, order='F')
            y2, x2 = unravel_index([lind2], img_size, order='F')
            # print('y1', y1, 'x1', x1, 'y2', y2, 'x2', x2)
            # print(unravel_index([lind1], img_size, order='F'))
            slope, line_len, alpha = math_stuff(x1, y1, x2, y2)
            # print('slope', slope)

            # Intersection point is in the middle of the new line
            if min(alph1, alph2) <= alpha <= max(alph1, alph2):
                lines = delete(lines, max(pt1, pt2), axis=0)
                lines = delete(lines, min(pt1, pt2), axis=0)
                val1 = out[pt1]
                val2 = out[pt2]
                del out[max(pt1, pt2)]
                del out[min(pt1, pt2)]

                # Update both lists to reflect the addition of the merged line.
                lines = append(lines, [[int(y1), int(x1) + 1, int(y2), int(x2) + 1, line_len, slope, alpha, 0, lind1, lind2]], axis=0)
                out.append([val1, val2])

                # listpt = merge_listpoints(listpt, pt1, pt2, lind1, lind2)
                listpt = merge_lp(new_lp, pt1, pt2, lind1, lind2)
                # Merged lines, so don't check the other pairs
                break
            else:
                continue

    return lines, array(listpt), array(out)
