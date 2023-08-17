"__author__ = Hui Wang"

""" Functions:
 read_csv(path0, name)
"""

import numpy as np
import csv 
#-----------------------------------------------------------------------

def read_csv(name):
    arrays = []
    with open(name, 'r') as f:
        reader = csv.reader(f, delimiter=',') 
        for row in reader: 
            # print(row, len(row))
            row = str(row).replace(';', '')
            row = str(row).replace('[', '')
            row = str(row).replace(']', '')
            row = str(row).replace('\'', '')
            row = str(row).replace('\'', '')
            # print(row, type(row))
            if row =='Pi':
                row = str(np.pi)
            if row =='1.5 * Pi':
                row = str(1.5*np.pi)
            if row =='0.5 * Pi':
                row = str(0.5*np.pi)
            arrays.append(row)

    # print(arrays)
    arr = np.array(arrays[5:])
    phi = np.array([],dtype=float)
    for p in arr:
        phi = np.r_[phi, float(p)]
    # print(name, phi.shape)
    return phi
