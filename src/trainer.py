#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: trainer.py.py
@time: 2018/10/11 9:53
"""

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from model import AE,Discriminator
from util import *
from timeit import default_timer as timer
#import matplotlib
from torch.autograd import Variable
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import os
import io
import random
import loader
from datetime import timedelta
import json
import copy
from evaluator import Evaluator
from word_translation import get_word_translation_accuracy
from dico_builder import build_dictionary

class CycleBWE(object):
    def __init__(self, params):
        self.params = params
        self.tune_dir = "{}/{}-{}/{}".format(params.exp_id, params.src_lang,params.tgt_lang,params.norm_embeddings)
        self.tune_best_dir = "{}/best".format(self.tune_dir)
        self.tune_export_dir = "{}/export".format(self.tune_dir)
        if self.params.eval_file == 'wiki':
            self.eval_file = '../data/bilingual_dicts/{}-{}.5000-6500.txt'.format(self.params.src_lang,self.params.tgt_lang)
            self.eval_file2 = '../data/bilingual_dicts/{}-{}.5000-6500.txt'.format(self.params.tgt_lang,self.params.src_lang)
        elif self.params.eval_file == 'wacky':
            self.eval_file = '../data/bilingual_dicts/{}-{}.test.txt'.format(self.params.src_lang,self.params.tgt_lang)
            self.eval_file2 = '../data/bilingual_dicts/{}-{}.test.txt'.format(self.params.tgt_lang,self.params.src_lang)
        else:
            print('Invalid eval file!')
        # self.seed = random.randint(0, 1000)
        # self.seed = 41
        # self.initialize_exp(self.seed)

        self.X_AE = AE(params)
        self.Y_AE = AE(params)
        self.D_X = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size,
                            output_size=params.d_output_size)
        self.D_Y = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size,
                            output_size=params.d_output_size)

        self.nets = [self.X_AE, self.Y_AE, self.D_X, self.D_Y]
        self.loss_fn = torch.nn.BCELoss()
        self.loss_fn2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def weights_init(self, m):  # 正交初始化
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0.01)

    def weights_init2(self, m):  # xavier_normal 初始化
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.01)

    def weights_init3(self, m):  # 单位阵初始化
        if isinstance(m, torch.nn.Linear):
            m.weight.data.copy_(torch.diag(torch.ones(self.params.g_input_size)))

    def init_state(self,state=1):
        if torch.cuda.is_available():
            # Move the network and the optimizer to the GPU
            for net in self.nets:
                net.cuda()
            self.loss_fn = self.loss_fn.cuda()
            self.loss_fn2 = self.loss_fn2.cuda()

        if self.params.init=='eye':
            self.X_AE.apply(self.weights_init3)  # 可更改G初始化方式
            self.Y_AE.apply(self.weights_init3)  # 可更改G初始化方式

        elif self.params.init=='orth':
            self.X_AE.apply(self.weights_init)  # 可更改G初始化方式
            self.Y_AE.apply(self.weights_init)
        else:
            print('Invalid init func!')

        #self.D_X.apply(self.weights_init2)
        #self.D_Y.apply(self.weights_init2)

    def orthogonalize(self, W):
        params = self.params
        W.copy_((1 + params.beta) * W - params.beta * W.mm(W.transpose(0, 1).mm(W)))

    def train(self, src_dico, tgt_dico, src_emb, tgt_emb, seed):
        params = self.params
        # Load data
        if not os.path.exists(params.data_dir):
            print("Data path doesn't exists: %s" % params.data_dir)
        if not os.path.exists(self.tune_dir):
            os.makedirs(self.tune_dir)
        if not os.path.exists(self.tune_best_dir):
            os.makedirs(self.tune_best_dir)
        if not os.path.exists(self.tune_export_dir):
            os.makedirs(self.tune_export_dir)

        src_word2id = src_dico[1]
        tgt_word2id = tgt_dico[1]

        en = src_emb
        it = tgt_emb

        params = _get_eval_params(params)
        self.params = params
        eval = Evaluator(params, en,it, torch.cuda.is_available())

        # for seed_index in range(params.num_random_seeds):

        AE_optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(self.X_AE.parameters()) + list(self.Y_AE.parameters())), lr=params.g_learning_rate)
        # AE_optimizer = optim.SGD(G_params, lr=0.1, momentum=0.9)
        # AE_optimizer = optim.Adam(G_params, lr=params.g_learning_rate, betas=(0.9, 0.9))
        # AE_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, list(self.X_AE.parameters()) + list(self.Y_AE.parameters())),lr=params.g_learning_rate,alpha=0.9)
        D_optimizer = optim.SGD(list(self.D_X.parameters()) + list(self.D_Y.parameters()), lr=params.d_learning_rate)
        # D_optimizer = optim.Adam(D_params, lr=params.d_learning_rate, betas=(0.5, 0.9))
        # D_optimizer = optim.RMSprop(list(self.D_X.parameters()) + list(self.D_Y.parameters()), lr=params.d_learning_rate , alpha=0.9)

        # D_X=nn.DataParallel(D_X)
        # D_Y=nn.DataParallel(D_Y)
        # true_dict = get_true_dict(params.data_dir)
        D_A_acc_epochs = []
        D_B_acc_epochs = []
        D_A_loss_epochs = []
        D_B_loss_epochs = []
        G_AB_loss_epochs = []
        G_BA_loss_epochs = []
        G_AB_recon_epochs = []
        G_BA_recon_epochs = []
        L_Z_loss_epoches = []

        acc1_epochs = []
        acc2_epochs = []

        csls_epochs = []
        f_csls_epochs = []
        b_csls_epochs = []
        best_valid_metric = -100

        # logs for plotting later
        log_file = open("log_src_tgt.txt", "w")  # Being overwritten in every loop, not really required
        log_file.write("epoch, dis_loss, dis_acc, g_loss\n")

        try:
            for epoch in range(self.params.num_epochs):
                D_A_losses = []
                D_B_losses = []
                G_AB_losses = []
                G_AB_recon = []
                G_BA_losses = []
                G_adv_losses = []
                G_BA_recon = []
                L_Z_losses = []
                d_losses = []
                g_losses = []
                hit_A = 0
                hit_B = 0
                total = 0
                start_time = timer()
                # lowest_loss = 1e5
                # label_D = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())
                label_D = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())
                label_D[:params.mini_batch_size] = 1 - params.smoothing
                label_D[params.mini_batch_size:] = params.smoothing

                label_G = to_variable(torch.FloatTensor(params.mini_batch_size).zero_())
                label_G = label_G + 1 - params.smoothing

                for mini_batch in range(0, params.iters_in_epoch // params.mini_batch_size):
                    for d_index in range(params.d_steps):
                        D_optimizer.zero_grad()  # Reset the gradients
                        self.D_X.train()
                        self.D_Y.train()

                        #print('D_X:', self.D_X.map1.weight.data)
                        #print('D_Y:', self.D_Y.map1.weight.data)

                        view_X, view_Y = self.get_batch_data_fast_new(en, it)
                        # Discriminator X
                        #print('View_Y',view_Y)
                        fake_X= self.Y_AE.encode(view_Y).detach()
                        #print('fakeX',fake_X)
                        input = torch.cat([view_X, fake_X], 0)

                        pred_A = self.D_X(input)
                        #print('Pred_A',pred_A)
                        D_A_loss = self.loss_fn(pred_A, label_D)
                        # print(view_Y)
                        # Discriminator Y
                        # print('View_X',view_X)
                        fake_Y = self.X_AE.encode(view_X).detach()
                        # print('fakeY:',fake_Y)

                        input = torch.cat([view_Y, fake_Y], 0)
                        pred_B = self.D_Y(input)
                        # print('Pred_B', pred_B)
                        D_B_loss = self.loss_fn(pred_B, label_D)

                        D_loss = (1.0) * D_A_loss + params.gate * D_B_loss

                        D_loss.backward()  # compute/store gradients, but don't change params
                        d_losses.append(to_numpy(D_loss.data))
                        D_A_losses.append(to_numpy(D_A_loss.data))
                        D_B_losses.append(to_numpy(D_B_loss.data))

                        discriminator_decision_A = to_numpy(pred_A.data)
                        hit_A += np.sum(discriminator_decision_A[:params.mini_batch_size] >= 0.5)
                        hit_A += np.sum(discriminator_decision_A[params.mini_batch_size:] < 0.5)

                        discriminator_decision_B = to_numpy(pred_B.data)
                        hit_B += np.sum(discriminator_decision_B[:params.mini_batch_size] >= 0.5)
                        hit_B += np.sum(discriminator_decision_B[params.mini_batch_size:] < 0.5)

                        D_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

                        # Clip weights
                        _clip(self.D_X, params.clip_value)
                        _clip(self.D_Y, params.clip_value)
                        # print('D_loss',d_losses)

                        sys.stdout.write("[%d/%d] :: Discriminator Loss: %.3f \r" % (
                            mini_batch, params.iters_in_epoch // params.mini_batch_size,
                            np.asscalar(np.mean(d_losses))))
                        sys.stdout.flush()

                    total += 2 * params.mini_batch_size * params.d_steps

                    for g_index in range(params.g_steps):
                        # 2. Train G on D's response (but DO NOT train D on these labels)
                        AE_optimizer.zero_grad()
                        self.D_X.eval()
                        self.D_Y.eval()
                        view_X, view_Y = self.get_batch_data_fast_new(en, it)

                        # Generator X_AE
                        ## adversarial loss
                        Y_fake = self.X_AE.encode(view_X)
                        # X_recon = self.X_AE.decode(X_Z)
                        # Y_fake = self.Y_AE.encode(X_Z)
                        pred_Y = self.D_Y(Y_fake)
                        L_adv_X = self.loss_fn(pred_Y, label_G)

                        X_Cycle = self.Y_AE.encode(Y_fake)
                        L_Cycle_X = 1.0 - torch.mean(self.loss_fn2(view_X, X_Cycle))

                        # L_recon_X = 1.0 - torch.mean(self.loss_fn2(view_X, X_recon))
                        # L_G_AB = L_adv_X + params.recon_weight * L_recon_X

                        # Generator Y_AE
                        # adversarial loss
                        X_fake = self.Y_AE.encode(view_Y)
                        pred_X = self.D_X(X_fake)
                        L_adv_Y = self.loss_fn(pred_X, label_G)

                        ### Cycle Loss
                        Y_Cycle = self.X_AE.encode(X_fake)
                        L_Cycle_Y = 1.0 - torch.mean(self.loss_fn2(view_Y, Y_Cycle))

                        # L_recon_Y = 1.0 - torch.mean(self.loss_fn2(view_Y, Y_recon))
                        # L_G_BA = L_adv_Y + params.recon_weight * L_recon_Y
                        # L_Z = 1.0 - torch.mean(self.loss_fn2(X_Z, Y_Z))

                        # G_loss = L_G_AB + L_G_BA + L_Z
                        G_loss = params.adv_weight * ( params.gate * L_adv_X + (1.0) * L_adv_Y) + \
                                 params.cycle_weight * (L_Cycle_X+L_Cycle_Y)

                        G_loss.backward()

                        g_losses.append(to_numpy(G_loss.data))
                        G_AB_losses.append(to_numpy(L_adv_X.data))
                        G_BA_losses.append(to_numpy(L_adv_Y.data))
                        G_adv_losses.append(to_numpy(L_adv_Y.data))
                        G_AB_recon.append(to_numpy(L_Cycle_X.data))
                        G_BA_recon.append(to_numpy(L_Cycle_Y.data))

                        AE_optimizer.step()  # Only optimizes G's parameters
                        self.orthogonalize(self.X_AE.map1.weight.data)
                        self.orthogonalize(self.Y_AE.map1.weight.data)

                        sys.stdout.write(
                            "[%d/%d] ::                                     Generator Loss: %.3f \r" % (
                                mini_batch, params.iters_in_epoch // params.mini_batch_size,
                                np.asscalar(np.mean(g_losses))))
                        sys.stdout.flush()

                '''for each epoch'''
                D_A_acc_epochs.append(hit_A / total)
                D_B_acc_epochs.append(hit_B / total)
                G_AB_loss_epochs.append(np.asscalar(np.mean(G_AB_losses)))
                G_BA_loss_epochs.append(np.asscalar(np.mean(G_BA_losses)))
                D_A_loss_epochs.append(np.asscalar(np.mean(D_A_losses)))
                D_B_loss_epochs.append(np.asscalar(np.mean(D_B_losses)))
                G_AB_recon_epochs.append(np.asscalar(np.mean(G_AB_recon)))
                G_BA_recon_epochs.append(np.asscalar(np.mean(G_BA_recon)))
                # L_Z_loss_epoches.append(np.asscalar(np.mean(L_Z_losses)))

                print(
                    "Epoch {} : Discriminator Loss: {:.3f}, Discriminator Accuracy: {:.3f}, Generator Loss: {:.3f}, Time elapsed {:.2f} mins".
                        format(epoch, np.asscalar(np.mean(d_losses)), 0.5 * (hit_A + hit_B) / total,
                               np.asscalar(np.mean(g_losses)),
                               (timer() - start_time) / 60))

                # lr decay
                # g_optim_state = AE_optimizer.state_dict()
                # old_lr = g_optim_state['param_groups'][0]['lr']
                # g_optim_state['param_groups'][0]['lr'] = max(old_lr * params.lr_decay, params.lr_min)
                # AE_optimizer.load_state_dict(g_optim_state)
                # print("Changing the learning rate: {} -> {}".format(old_lr, g_optim_state['param_groups'][0]['lr']))
                # d_optim_state = D_optimizer.state_dict()
                # d_optim_state['param_groups'][0]['lr'] = max(
                #     d_optim_state['param_groups'][0]['lr'] * params.lr_decay, params.lr_min)
                # D_optimizer.load_state_dict(d_optim_state)
                #     d_optim_state['param_groups'][0]['lr'] * params.lr_decay, params.lr_min)
                # D_optimizer.load_state_dict(d_optim_state)

                if (epoch + 1) % params.print_every == 0:
                    # No need for discriminator weights
                    # torch.save(d.state_dict(), 'discriminator_weights_en_es_{}.t7'.format(epoch))

                    # all_precisions = eval.get_all_precisions(G_AB(src_emb.weight).data)
                    Vec_xy = self.X_AE.encode(Variable(en))
                    Vec_xyx = self.Y_AE.encode(Vec_xy)
                    Vec_yx = self.Y_AE.encode(Variable(it))
                    Vec_yxy = self.X_AE.encode(Vec_yx)


                    mstart_time = timer()

                    # for method in ['csls_knn_10']:
                    for method in [params.eval_method]:
                        results = get_word_translation_accuracy(
                            params.src_lang, src_word2id, Vec_xy.data,
                            params.tgt_lang, tgt_word2id, it,
                            method=method,
                            dico_eval=self.eval_file,
                            device=params.cuda_device
                        )
                        acc1 = results[0][1]
                        results = get_word_translation_accuracy(
                            params.tgt_lang, tgt_word2id, Vec_yx.data,
                            params.src_lang, src_word2id, en,
                            method=method,
                            dico_eval=self.eval_file2,
                            device=params.cuda_device
                        )
                        acc2 = results[0][1]
                        print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
                        print('Method:{} test_score:{:.4f}-{:.4f}'.format(method, acc1, acc2))
                    '''
                    # for method in ['csls_knn_10']:
                    for method in [params.eval_method]:
                        results = get_word_translation_accuracy(
                            params.src_lang, src_word2id, Vec_xyx.data,
                            params.src_lang, src_word2id, en,
                            method=method,
                            dico_eval='/data/dictionaries/{}-{}.wacky.dict'.format(params.src_lang,params.src_lang),
                            device=params.cuda_device
                        )
                        acc11 = results[0][1]
                    # for method in ['csls_knn_10']:
                    for method in [params.eval_method]:
                        results = get_word_translation_accuracy(
                            params.tgt_lang, tgt_word2id, Vec_yxy.data,
                            params.tgt_lang, tgt_word2id, it,
                            method=method,
                            dico_eval='/data/dictionaries/{}-{}.wacky.dict'.format(params.tgt_lang,params.tgt_lang),
                            device=params.cuda_device
                        )
                        acc22 = results[0][1]
                    print('Valid:{} score:{:.4f}-{:.4f}'.format(method, acc11, acc22))
                    avg_valid = (acc11+acc22)/2.0
                    # valid_x = torch.mean(self.loss_fn2(en, Vec_xyx.data))
                    # valid_y = torch.mean(self.loss_fn2(it, Vec_yxy.data))
                    # avg_valid = (valid_x+valid_y)/2.0
                    '''
                    # csls = 0
                    f_csls = eval.dist_mean_cosine(Vec_xy.data, it)
                    b_csls = eval.dist_mean_cosine(Vec_yx.data, en)
                    csls = (f_csls+b_csls)/2.0
                    # csls = eval.calc_unsupervised_criterion(X_Z)
                    if csls > best_valid_metric:
                        print("New csls value: {}".format(csls))
                        best_valid_metric = csls
                        fp = open(self.tune_dir + "/best/seed_{}_dico_{}_epoch_{}_acc_{:.3f}-{:.3f}.tmp".format(seed,params.dico_build, epoch, acc1,acc2), 'w')
                        fp.close()
                        torch.save(self.X_AE.state_dict(),self.tune_dir+'/best/seed_{}_dico_{}_best_X.t7'.format(seed,params.dico_build))
                        torch.save(self.Y_AE.state_dict(),self.tune_dir+'/best/seed_{}_dico_{}_best_Y.t7'.format(seed,params.dico_build))
                        torch.save(self.D_X.state_dict(),self.tune_dir+'/best/seed_{}_dico_{}_best_Dx.t7'.format(seed,params.dico_build))
                        torch.save(self.D_Y.state_dict(),self.tune_dir+'/best/seed_{}_dico_{}_best_Dy.t7'.format(seed,params.dico_build))
                    # print(json.dumps(all_precisions))
                    # p_1 = all_precisions['validation']['adv']['without-ref']['nn'][1]
                    # p_1 = all_precisions['validation']['adv']['without-ref']['csls'][1]
                    # log_file.write(str(results) + "\n")
                    # print('Method: nn score:{:.4f}'.format(acc))
                    # Saving generator weights
                    # torch.save(X_AE.state_dict(), tune_dir+'/G_AB_seed_{}_mf_{}_lr_{}_p@1_{:.3f}.t7'.format(seed,params.most_frequent_sampling_size,params.g_learning_rate,acc))
                    # torch.save(Y_AE.state_dict(), tune_dir+'/G_BA_seed_{}_mf_{}_lr_{}_p@1_{:.3f}.t7'.format(seed,params.most_frequent_sampling_size,params.g_learning_rate,acc))
                    fp = open(self.tune_dir +
                              "/seed_{}_epoch_{}_acc_{:.3f}-{:.3f}_valid_{:.4f}.tmp".format(seed,epoch,acc1,acc2,csls), 'w')
                    fp.close()
                    acc1_epochs.append(acc1)
                    acc2_epochs.append(acc2)
                    csls_epochs.append(csls)
                    f_csls_epochs.append(f_csls)
                    b_csls_epochs.append(b_csls)

            csls_fb, epoch_fb = max([(score, index) for index, score in enumerate(csls_epochs)])
            fp = open(self.tune_dir + "/best/seed_{}_epoch_{}_{:.3f}_{:.3f}_{:.3f}.cslsfb".format(seed, epoch_fb,acc1_epochs[epoch_fb],acc2_epochs[epoch_fb], csls_fb), 'w')
            fp.close()
            csls_f, epoch_f = max([(score, index) for index, score in enumerate(f_csls_epochs)])
            fp = open(self.tune_dir + "/best/seed_{}_epoch_{}_{:.3f}_{:.3f}_{:.3f}.cslsf".format(seed, epoch_f,acc1_epochs[epoch_f],acc2_epochs[epoch_f],csls_f), 'w')
            fp.close()
            csls_b, epoch_b = max([(score, index) for index, score in enumerate(b_csls_epochs)])
            fp = open(self.tune_dir + "/best/seed_{}_epoch_{}_{:.3f}_{:.3f}_{:.3f}.cslsb".format(seed, epoch_b,acc1_epochs[epoch_b],acc2_epochs[epoch_b],csls_b), 'w')
            fp.close()
            '''

            # Save the plot for discriminator accuracy and generator loss
            fig = plt.figure()
            plt.plot(range(0, len(D_A_acc_epochs)), D_A_acc_epochs, color='b', label='D_A')
            plt.plot(range(0, len(D_B_acc_epochs)), D_B_acc_epochs, color='r', label='D_B')
            plt.ylabel('D_accuracy')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_D_acc.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(D_A_loss_epochs)), D_A_loss_epochs, color='b', label='D_A')
            plt.plot(range(0, len(D_B_loss_epochs)), D_B_loss_epochs, color='r', label='D_B')
            plt.ylabel('D_losses')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_D_loss.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(G_AB_loss_epochs)), G_AB_loss_epochs, color='b', label='G_AB')
            plt.plot(range(0, len(G_BA_loss_epochs)), G_BA_loss_epochs, color='r', label='G_BA')
            plt.ylabel('G_losses')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_G_loss.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(G_AB_recon_epochs)), G_AB_recon_epochs, color='b', label='G_AB')
            plt.plot(range(0, len(G_BA_recon_epochs)), G_BA_recon_epochs, color='r', label='G_BA')
            plt.ylabel('G_Cycle_loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_G_Cycle.png'.format(seed))

            # fig = plt.figure()
            # plt.plot(range(0, len(L_Z_loss_epoches)), L_Z_loss_epoches, color='b', label='L_Z')
            # plt.ylabel('L_Z_loss')
            # plt.xlabel('epochs')
            # plt.legend()
            # fig.savefig(tune_dir + '/seed_{}_stage_{}_L_Z.png'.format(seed,stage))

            fig = plt.figure()
            plt.plot(range(0, len(acc1_epochs)), acc1_epochs, color='b', label='trans_acc1')
            plt.plot(range(0, len(acc2_epochs)), acc2_epochs, color='r', label='trans_acc2')
            plt.ylabel('trans_acc')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_trans_acc.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(csls_epochs)), csls_epochs, color='b', label='csls')
            plt.plot(range(0, len(f_csls_epochs)), f_csls_epochs, color='r', label='csls_f')
            plt.plot(range(0, len(b_csls_epochs)), b_csls_epochs, color='g', label='csls_b')
            plt.ylabel('csls')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_csls.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(g_losses)), g_losses, color='b', label='G_loss')
            plt.ylabel('g_loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_g_loss.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(d_losses)), d_losses, color='b', label='csls')
            plt.ylabel('D_loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_d_loss.png'.format(seed))
            plt.close('all')
            '''

        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            torch.save(self.X_AE.state_dict(), 'g_model_interrupt.t7')
            torch.save(self.D_X.state_dict(), 'd_model_interrupt.t7')
            log_file.close()
            exit()

        log_file.close()
        return self.X_AE

    def get_batch_data_fast_new(self, emb_en, emb_it):

        params = self.params
        random_en_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        random_it_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        #print(random_en_indices)
        #print(random_it_indices)
        en_batch = to_variable(emb_en)[random_en_indices.cuda()]
        it_batch = to_variable(emb_it)[random_it_indices.cuda()]
        return en_batch, it_batch

    def export(self,src_dico,tgt_dico,emb_en,emb_it,seed,export_emb=False):
        params = _get_eval_params(self.params)
        eval = Evaluator(params, emb_en, emb_it, torch.cuda.is_available())
        # Export adversarial dictionaries
        optim_X_AE = AE(params).cuda()
        optim_Y_AE = AE(params).cuda()
        print('Loading pre-trained models...')
        optim_X_AE.load_state_dict(torch.load(self.tune_dir +'/best/seed_{}_dico_{}_best_X.t7'.format(seed,params.dico_build)))
        optim_Y_AE.load_state_dict(torch.load(self.tune_dir +'/best/seed_{}_dico_{}_best_Y.t7'.format(seed,params.dico_build)))
        X_Z = optim_X_AE.encode(Variable(emb_en)).data
        Y_Z = optim_Y_AE.encode(Variable(emb_it)).data

        mstart_time = timer()
        for method in ['nn','csls_knn_10']:
            results = get_word_translation_accuracy(
                params.src_lang, src_dico[1], X_Z,
                params.tgt_lang, tgt_dico[1], emb_it,
                method=method,
                dico_eval=self.eval_file,
                device=params.cuda_device
            )
            acc1 = results[0][1]
            results = get_word_translation_accuracy(
                params.tgt_lang, tgt_dico[1], Y_Z,
                params.src_lang, src_dico[1], emb_en,
                method=method,
                dico_eval=self.eval_file2,
                device=params.cuda_device
            )
            acc2 = results[0][1]

        # csls = 0
            print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
            print('Method:{} score:{:.4f}-{:.4f}'.format(method, acc1, acc2))
        
        f_csls = eval.dist_mean_cosine(X_Z, emb_it)
        b_csls = eval.dist_mean_cosine(Y_Z, emb_en)
        csls = (f_csls+b_csls)/2.0
        print("Seed:{},ACC:{:.4f}-{:.4f},CSLS_FB:{:.6f}".format(seed,acc1,acc2,csls))
        #'''
        print('Building dictionaries...')
        params.dico_build = "S2T&T2S"
        params.dico_method = "csls_knn_10"
        X_Z = X_Z / X_Z.norm(2, 1, keepdim=True).expand_as(X_Z)
        emb_it = emb_it / emb_it.norm(2, 1, keepdim=True).expand_as(emb_it)
        f_dico_induce = build_dictionary(X_Z, emb_it, params)
        f_dico_induce = f_dico_induce.cpu().numpy()
        Y_Z = Y_Z / Y_Z.norm(2, 1, keepdim=True).expand_as(Y_Z)
        emb_en = emb_en / emb_en.norm(2, 1, keepdim=True).expand_as(emb_en)
        b_dico_induce = build_dictionary(Y_Z, emb_en, params)
        b_dico_induce = b_dico_induce.cpu().numpy()

        f_dico_set = set([(a,b) for a,b in f_dico_induce])
        b_dico_set = set([(b,a) for a,b in b_dico_induce])

        intersect = list(f_dico_set & b_dico_set)
        union = list(f_dico_set | b_dico_set)

        with io.open(self.tune_dir + '/export/{}-{}.dict'.format(params.src_lang, params.tgt_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in f_dico_induce:
                f.write('{} {}\n'.format(src_dico[0][item[0]], tgt_dico[0][item[1]]))

        with io.open(self.tune_dir + '/export/{}-{}.dict'.format(params.tgt_lang, params.src_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in b_dico_induce:
                f.write('{} {}\n'.format(tgt_dico[0][item[0]], src_dico[0][item[1]]))

        with io.open(self.tune_dir + '/export/{}-{}.intersect'.format(params.src_lang, params.tgt_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in intersect:
                f.write('{} {}\n'.format(src_dico[0][item[0]], tgt_dico[0][item[1]]))

        with io.open(self.tune_dir + '/export/{}-{}.intersect'.format(params.tgt_lang, params.src_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in intersect:
                f.write('{} {}\n'.format(tgt_dico[0][item[1]],src_dico[0][item[0]]))

        with io.open(self.tune_dir + '/export/{}-{}.union'.format(params.src_lang, params.tgt_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in union:
                f.write('{} {}\n'.format(src_dico[0][item[0]], tgt_dico[0][item[1]]))

        with io.open(self.tune_dir + '/export/{}-{}.union'.format(params.tgt_lang, params.src_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in union:
                f.write('{} {}\n'.format(tgt_dico[0][item[1]], src_dico[0][item[0]]))
        
        if export_emb:
            print('Exporting {}-{}.{}'.format(params.src_lang,params.tgt_lang,params.src_lang))
            loader.export_embeddings(src_dico[0],X_Z,
                                     path=self.tune_dir+'/export/{}-{}.{}'.format(params.src_lang,params.tgt_lang,params.src_lang),eformat='txt')
            print('Exporting {}-{}.{}'.format(params.src_lang,params.tgt_lang,params.tgt_lang))
            loader.export_embeddings(tgt_dico[0],emb_it,
                                     path=self.tune_dir+'/export/{}-{}.{}'.format(params.src_lang,params.tgt_lang,params.tgt_lang),eformat='txt')
            print('Exporting {}-{}.{}'.format(params.tgt_lang,params.src_lang,params.tgt_lang))
            loader.export_embeddings(tgt_dico[0],Y_Z,
                                     path=self.tune_dir+'/export/{}-{}.{}'.format(params.tgt_lang,params.src_lang,params.tgt_lang),eformat='txt')
            print('Exporting {}-{}.{}'.format(params.tgt_lang,params.src_lang,params.src_lang))
            loader.export_embeddings(src_dico[0],emb_en,
                                     path=self.tune_dir+'/export/{}-{}.{}'.format(params.tgt_lang,params.src_lang,params.src_lang),eformat='txt')
        #'''
def _init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)

def to_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)

def to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()

def _clip(d, clip):
    if clip > 0:
        for x in d.parameters():
            x.data.clamp_(-clip, clip)


def _get_eval_params(params):
    params = copy.deepcopy(params)
    params.ks = [1]
    params.methods = ['csls']
    params.models = ['adv']
    params.refine = ['without-ref']

    params.eval_method = "nn"
    params.cuda = True
    params.d_learning_rate = 0.1
    params.d_steps = 5
    params.g_learning_rate = 0.1
    params.iters_in_epoch = 75000
    #params.iters_in_epoch = 32
    # params.num_epochs=1
    params.gate = 1  # change
    params.g_size = 300
    params.adv_weight = 1
    params.cycle_weight = 1
    return params
