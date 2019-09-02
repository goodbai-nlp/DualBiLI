#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: evaluator.py
@time: 2018/10/11 10:10
"""

import util
import torch
import numpy as np
import json
import platform
import time
import codecs
import scipy
from torch import Tensor as torch_tensor

from dico_builder import get_candidates, build_dictionary

op_sys = platform.system()
if op_sys == 'Darwin':
    from faiss_master import faiss
elif op_sys == 'Linux':
    import faiss
else:
    raise 'Operating system not supported: %s' % op_sys


class Evaluator:
    def __init__(self, params, src_emb, tgt_emb, use_cuda=False):
        self.params = params
        self.data_dir = params.data_dir
        self.ks = params.ks
        self.methods = params.methods
        self.models = params.models
        self.refine = params.refine
        self.csls_k = params.csls_k
        self.num_refine = params.num_refine

        self.tgt_emb = tgt_emb
        self.src_emb = src_emb

    def dist_mean_cosine(self, src_emb, tgt_emb):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        # build dictionary
        for dico_method in ['csls_knn_10']:
            dico_max_size = 10000
            _params = self.params
            s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
            t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
            dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
            # mean cosine
            if dico is None:
                mean_cosine = -1e9
            else:
                mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
            mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
            print("Mean cosine (%s method, %s build, %i max size): %.5f" % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
            # to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = mean_cosine

            return mean_cosine
