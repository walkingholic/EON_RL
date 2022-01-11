# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:41:36 2019

@author: User
"""

class Request :
    def __init__(self, i, s, d, b, a, h,  warmup):
        self.id = i
        self.source = s
        self.destination = d
        self.bandwidth = b
        self.nos = -1
        self.arrival_time = a
        self.holding_time = h
        self.end_time = a+h
#        self.req_print()
        self.slot_start = -1
        self.slot_end = -1
        self.state = -1
        self.path = {}
        self.warmup = warmup
        self.hop=0

    def req_print(self):
        print('ID:', self.id,  'source: ', self.source,' destination: ', self.destination,' bandwidth: ', self.bandwidth, ' arrival: ', self.arrival_time, ' holding: ',self.holding_time,' end: ', self.end_time,'\n','slot_start: ', self.slot_start, ' slot_end: ',self.slot_end,  ' hop: ',self.hop )
        
        


