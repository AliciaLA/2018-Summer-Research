#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:45:42 2018

@author: Marcos Ortiz

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
"""

import dionysus as ds
import numpy as np

'''
    The corners of five squares in the xy-plane
'''
# Lower left corners
c1 = [0, 0]
c2 = [5, 0]
c3 = [-2, 4]
c4 = [7, 4]
c5 = [2.5, 7]
# Chose the side lengths
s1 = 1
s2 = 1
s3 = 1
s4 = 1
s5 = 1
# The squares with the given side lengths at the given corners.
sq1 = [c1, [c1[0]+s1, c1[1]], [c1[0], c1[1]+s1], [c1[0]+s1, c1[1]+s1]]
sq2 = [c2, [c2[0]+s2, c2[1]], [c2[0], c2[1]+s2], [c2[0]+s2, c2[1]+s2]]
sq3 = [c3, [c3[0]+s3, c3[1]], [c3[0], c3[1]+s3], [c3[0]+s3, c3[1]+s3]]
sq4 = [c4, [c4[0]+s4, c4[1]], [c4[0], c4[1]+s4], [c4[0]+s4, c4[1]+s4]]
sq5 = [c5, [c5[0]+s5, c5[1]], [c5[0], c5[1]+s5], [c5[0]+s5, c5[1]+s5]]

five_squares = np.array(sq1 + sq2 + sq3 + sq4 + sq5)

'''
    Using dionysus, we create a filtration via construction of a Vietoris-Rips
     complex on each collection of points. Then we can compute the homology at
     each  stage of the filtration to establish the persistent homology.

    Info on dionysis:
     http://www.mrzv.org/software/dionysus/
    Quick explanation of filtrations:
     http://mrzv.org/software/dionysus2/tutorial/basics.html#filtration
    Quick explanation of the Vietoris-Rips complex:
     http://mrzv.org/software/dionysus2/tutorial/rips.html

'''

'''
    Step 1
    ------
    Build a filtration based on the Vietoris-Rips complex.
    We give dionysis.fill_rips three inputs
        1. Our data points
        2. The maximum dimension simplex to add to the complex. E.g. if we set
            this to 2 then the algorithm will not fill in a three simplex,
            even if the comlete graph of four vertices arises.
        3. The maximum we want the radius of the balls around each point to
            become.
'''
squares_filtration = ds.fill_rips(five_squares, 2, 12)

# How many simplices are there in this filtration?
print(squares_filtration)

# Uncomment bellow to print a full list of the simplices (and the times they
#    appear) in the filtration.
# for each_simplex in squares_filtration:
#     print(each_simplex)

# Compute the persistence homology of the filtration
squares_persistence = ds.homology_persistence(squares_filtration)

# Organize this information to plot the diagrams
diagram_info = ds.init_diagrams(squares_persistence, squares_filtration)

# Plot the diagrams:
ds.plot.plot_diagram(diagram_info[1], show=True)
ds.plot.plot_bars(diagram_info[1], show=True)
