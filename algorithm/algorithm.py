#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import torch
import os
import time
from agent_ppo.model.model import NetworkModelLearner
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, device, logger, monitor):
        self.device = device
        self.model = NetworkModelLearner().to(self.device)
        self.lr = Config.START_LR
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.logger = logger
        self.monitor = monitor
        self.last_report_monitor_time = 0
        self.label_size = Config.ACTION_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        # NOTE Added parameters
        self.value_clip = Config.VALUE_CLIP
        self.minibatch_size = Config.MINIBATCH_SIZE
        self.n_epoch = Config.N_EPOCH
        self.dual_clip : float | None = Config.DUAL_CLIP

        assert (
            self.dual_clip is None or self.dual_clip > 1.0
        ), f"Dual-clip PPO parameter should greater than 1.0 but got {self.dual_clip}"

        self.norm_adv : bool = True
        # self.anneal_lr : bool = Config.ANNEAL_LR NOTE 不知道怎么获取 learn 是第几次 update，难以实现学习率衰减


    def learn(self, list_sample_data):
        results = {}
        self.model.train()
        # self.optimizer.zero_grad()

        list_npdata = [torch.as_tensor(sample_data.npdata, device=self.device) for sample_data in list_sample_data]
        _input_datas = torch.stack(list_npdata, dim=0)

        data_list = list(self.model.format_data(_input_datas))
        if self.norm_adv:
            adv_idx = 4
            data_list[adv_idx] = (data_list[adv_idx] - data_list[adv_idx].mean()) / (data_list[adv_idx].std() + 1e-8)
        data_list = tuple(data_list)
        # total_loss, info_list = self.compute_loss(data_list, rst_list)
        # results["total_loss"] = total_loss.item()
        # total_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.parameters, 0.5)
        # self.optimizer.step()

        # _info_list = []
        # for info in info_list:
        #     if isinstance(info, list):
        #         _info = [i.detach().cpu().item() if torch.is_tensor(i) else i for i in info]
        #     else:
        #         _info = info.detach().mean().cpu().item() if torch.is_tensor(info) else info
        #     _info_list.append(_info)

        # if self.monitor:
        #     now = time.time()
        #     if now - self.last_report_monitor_time >= 60:
        #         results["value_loss"] = round(_info_list[1], 2)
        #         results["policy_loss"] = round(_info_list[2], 2)
        #         results["entropy_loss"] = round(_info_list[3], 2)
        #         results["reward"] = _info_list[-1]
        #         self.monitor.put_data({os.getpid(): results})
        #         self.last_report_monitor_time = now

        batch_size = len(list_sample_data)
        b_inds = np.arange(batch_size)
        for epoch in range(self.n_epoch):
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                mb_inds_tensor = torch.from_numpy(mb_inds).to(self.device)

                mb_data_list = [d[mb_inds_tensor] for d in data_list]
                # 每个minibatch都重新计算前向传播，避免计算图重复使用
                mb_rst_list = self.model(mb_data_list)
                total_loss, info_list = self.compute_loss(mb_data_list, mb_rst_list)

                #### Monitor Data ####
                with torch.no_grad():
                    results["total_loss"] = total_loss.item()
                ######################

                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, 0.5) # NOTE 设置 gradient clip 参数为 0.5，可能要调节。
                self.optimizer.step()


                #### Monitor Data ####
                with torch.no_grad():
                    _info_list = []
                    for info in info_list:
                        if isinstance(info, list):
                            _info = [i.detach().cpu().item() if torch.is_tensor(i) else i for i in info]
                        else:
                            _info = info.detach().mean().cpu().item() if torch.is_tensor(info) else info
                        _info_list.append(_info)

                    if self.monitor:
                        now = time.time()
                        if now - self.last_report_monitor_time >= 60:
                            results["value_loss"] = round(_info_list[1], 2)
                            results["policy_loss"] = round(_info_list[2], 2)
                            results["entropy_loss"] = round(_info_list[3], 2)
                            results["reward"] = _info_list[-1]
                            self.monitor.put_data({os.getpid(): results})
                            self.last_report_monitor_time = now
                ######################

                

    def compute_loss(self, data_list, rst_list):
        """
        data_list: [feature, reward, old_value, tdret, adv, old_action, old_log_prob, legal_action]
        rst_list: [log_prob, value]
        """
        (
            feature,        # 0
            reward,         # 1
            old_value,      # 2
            tdret,          # 3 NOTE 就是 `Return` = Advantage + Value
            adv,            # 4
            old_action,
            old_log_prob,
            legal_action,
        ) = data_list


        # value loss
        # 价值损失
        value = rst_list[1].squeeze(1) # squeeze(1) 去掉维度为 1 的维度
        old_value = old_value
        adv = adv
        tdret = tdret
        value_clip = old_value + (value - old_value).clamp(-self.clip_param, self.clip_param) # NOTE 已实现 value clip
        value_loss1 = torch.square(tdret - value_clip)
        value_loss2 = torch.square(tdret - value)
        value_loss = 0.5 * torch.maximum(value_loss1, value_loss2).mean() # NOTE 0.5 是常规设计，为了求平方梯度消掉系数。value loss 前的系数在 Config 控制

        # entropy loss
        # 熵损失
        log_prob = rst_list[0] 
        prob = torch.exp(log_prob) # NOTE This is current probability distribution.
                                   # NOTE 不用掩码非法动作，因为在算 rst_list[0] 时，已经将非法动作概率 * 1e-5
        entropy_loss = (-prob * torch.log(prob.clamp(1e-9, 1))).sum(1).mean()


        # policy loss
        # 策略损失
        clip_fracs = []
        one_hot_action = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size)
        new_log_prob = (one_hot_action * log_prob).sum(1, keepdim=True)
        ratio = torch.exp(new_log_prob - old_log_prob)          
        clip_fracs.append((ratio - 1.0).abs().gt(self.clip_param).float().mean())
        surr1 = ratio * adv 
        surr2 = ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        if self.dual_clip:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self.dual_clip * adv)
            policy_loss = -torch.where(adv < 0, clip2, clip1).mean()
        else:
            policy_loss = -torch.minimum(surr1, surr2).mean()


        total_loss = value_loss * self.vf_coef + policy_loss - self.var_beta * entropy_loss
        info_list = [tdret.mean(), value_loss, policy_loss, entropy_loss] + clip_fracs
        info_list += [adv.mean(), adv.std(), reward.mean()]
        return total_loss, info_list
