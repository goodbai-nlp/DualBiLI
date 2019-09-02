import sys
import random

input1=str(sys.argv[1])
output1=str(sys.argv[2])
output2=str(sys.argv[3])
f=open(input1,'r',encoding='utf-8')
ff1=open(output1,'w',encoding='utf-8')
ff2=open(output2,'w',encoding='utf-8')
dict1=[]
for line in f:
    vals=line.split()
    if len(vals)==2:
        dict1.append([vals[0].strip(),vals[1].strip()])

random.Random(0).shuffle(dict1)

dict_size=len(dict1)
dict11=dict1[:int(0.8*dict_size)]
dict12=dict1[int(0.8*dict_size):]

for vals in dict11:
    ff1.write(vals[0].strip()+' '+vals[1].strip()+'\n')

for vals in dict12:
    ff2.write(vals[0].strip()+' '+vals[1].strip()+'\n')
