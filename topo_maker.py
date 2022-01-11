# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:27:49 2019

@author: User
"""
#{'0':{'1': 1100, '2':1600}, '7':2800 ,'1':{'0':1100 , '2':  }, '2':{'': , '':  }, '3':{'': , '':  }, '4':{'': , '':  },'5':{'': , '':  },'6':{'': , '':  },'7':{'': , '':  },'8':{'': , '':  },'9':{'': , '':  },'10':{'': , '':  },'11':{'': , '':  },'12':{'': , '':  },'13':{'': , '':  }}
#
#in_filename = 'data/json/test_nsf.txt'
#out_filename = 'data/json/test_nsf_temp.json'

in_filename = 'data/json/test_usnet.txt'
out_filename = 'data/json/test_usnet_temp.json'

fi = open(in_filename, 'r')
fo = open(out_filename, 'w')
#line = f.read()
lines  = fi.readlines()
data={}
k="1"
n=1
temp = {}

for line in lines:
    items = line.split('\t') 
#    print(items[0])
#    print(items[1])
#    print(items[2])
    fn = str(int(items[0])+1)
    tn = str(int(items[1])+1)
    di = items[2]
    
    if fn == k :
        temp[tn] = int(di)
        data[fn]=temp
    else:
        k = fn
        temp = {}
        temp[tn] = int(di)
        
fo.write(str(data))

#print(data)
fi.close()
fo.close()
################################################################################
#in_filename = 'data/json/test_nsf_temp.json'
#out_filename = 'data/json/test_nsf.json'


in_filename = 'data/json/test_usnet_temp.json'
out_filename = 'data/json/test_usnet.json'

fi = open(in_filename, 'r')
fo = open(out_filename, 'w')
#line = f.read()
lines  = fi.readlines()
#print('lines ', lines)
data=''
for line in lines:
    items = line.split('\'') 
    print(len(items))
    k=0
    for i in items:        
        if k < len(items)-1:
            data += i+'"'
        else:
            data += i
        k+=1

print(data)
fo.write(data)

fi.close()
fo.close()
