#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: valid_dict.py
@time: 2019/1/28 14:00
"""
import io
import argparse
lang='en'
data='wacky'
path = '/data/dictionaries/{}-{}.{}.dict'.format(lang,lang,data)

with io.open(path, 'r', encoding='utf-8',errors='ignore') as f:
    for line_num, line in enumerate(f):
        if line != line.lower():
            line = line.lower()
        # print(line_num,end=',')
        tmp = line.rstrip().split()
        if not len(tmp) == 2:
            print("Warning: Found {} words in line #{}".format(len(tmp), line_num))
            continue
        word1, word2 = tmp[0], tmp[1]