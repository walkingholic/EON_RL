# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:39:13 2019

@author: User
"""
import numpy as np

for k in range(60):
    total=0
    data = input()
    for d in data:
        total = total + int(d)
    
    print(total)
