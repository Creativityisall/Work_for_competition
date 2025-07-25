#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
import random
from agent_ppo.feature.definition import reward_process


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8
        self.reset()

    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        
        # self.history_pos = []
        self.bad_move_ids = set()

        self.global_map = np.full((128, 128), -1, dtype=int)  # 全局地图探索情况
        self.undetected_area = 128 * 128
        self.treasure_pos_list = []
        self.buff_pos_list = []
        self.dest_pos = None
        self.is_dest_pos_found = False

        self.talent_available = False  # 初始尚未加载穿墙技能
        self.talent_cd = float("inf")  # 剩余冷却时间初始置为无穷大

    def _get_pos_feature(self, found, cur_pos, target_pos):
        """
        获取当前位置和目标位置的特征向量
        Args:
            found: 是否找到目标位置（目标是否是随机取的）
            cur_pos: 当前坐标 (x, z)
            target_pos: 目标坐标 (x, z)
        Returns:
            feature: 特征向量，包括：
            - found;
            - norm_x_relative, norm_z_relative;
            - norm_target_x, norm_target_z;
            - norm_distance.
        """
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 127, 0)
        pos_feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
        )
        return pos_feature
    
    def update_pos(self, obs, hero):
        # Update current position
        # Update position norm (cur & last)
        
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 127, 0)

        # # Update history position feature
        # # 更新历史位置及其特征
        # self.history_pos.append(self.cur_pos)
        # if len(self.history_pos) > 10:
        #     self.history_pos.pop(0)
        # self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0]) # NOTE history here means 10 steps before
            
    def update_view(self, obs):
        """
        Map Protocol:
        -1 : 未探索
        0 : 障碍物
        1 : 空地（见论坛）
        2 : treasure
        3 : buff
        4 : destination
        """   
        # update detected area
        map_info = obs["frame_state"]["map"]
        cnt_new_detected = 0
        x0, z0 = self.cur_pos
        for i in range(-25, 25):
            for j in range(-25, 25):
                x, z = x0 + i, z0 + j
                if 0 <= x < 128 and 0 <= z < 128:                    
                    if self.global_map[x, z] == -1:
                        cnt_new_detected += 1
                    self.global_map[x, z] = map_info[x][z] # 0 or 1
        
        # self.detected_area = np.argwhere(self.global_map != -1)
        self.undetected_area -= cnt_new_detected
        assert self.undetected_area == np.sum(self.global_map == -1), \
            f"undetected_area {self.undetected_area} != np.sum(self.global_map == -1) {np.sum(self.global_map == -1)}"

        # update treasure position list
        organs = obs["frame_state"]["organs"]
        for organ in organs:
            if organ["sub_type"] == 1 and organ["status"] == 1:  # treasure
                # XXX 我怀疑取过的宝箱，再经过时还是会看到，只不过状态为 0（不可取）。
                pos = organ["pos"]
                if pos not in self.treasure_pos_list:
                    self.treasure_pos_list.append(pos)    
            elif organ["sub_type"] == 2 and organ["status"] == 1:   # buff
                pos = organ["pos"]
                if pos not in self.buff_pos_list:
                    self.buff_pos_list.append(pos)
        
        # TODO 

        # TODO History ?
        # self.treasure_pos_list = np.argwhere(self.global_map == 2)
        # if self.treasure_pos_list.size > 0:
            # self.dest_pos = self.treasure_pos_list[0]
        # else:
            # self.dest_pos = None

    def process(self, frame_state, last_action):
        obs, _ = frame_state
        hero = obs["frame_state"]["heroes"][0]
        
        # Record step_no
        self.step_no = obs["frame_state"]["step_no"]

        # Process legal action
        self.last_action = last_action 
        self.talent_available = hero["talent"]["status"]
        self.talent_cd = hero["talent"]["cooldown"]
        legal_action = self.get_legal_action(obs) # TODO how to process talent? how to design action space?

        # Process frame state: update self's attributes
        self.update_pos(obs, hero)
        self.update_view(obs)


        # Feature TODO

        # NOTE I think the following should be encoded into feature:
        # 1. 'detected map' 
        # 2. current (normalized) position
        # 3. organ detection status (MAX_TREASURE_NUM bits for treasures + 1 bit for destination + ? bits for buffs + normalized cd for talent)
        # 4. ?? history position feature (10 steps before) ?? 
        # 5. legal action (16 bits for 16 actions, 1 bit for move action)
        feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos , legal_action])
        # XXX modify conf.py !!

        return (
            feature,
            legal_action,
            reward_process(
                step_no=self.step_no,
                cur_pos=self.cur_pos,
                detected_area=self.detected_area,
                treasure_pos_list=self.treasure_pos_list,
                destination_pos=self.dest_pos
                # TODO MORE ??
            ),  
        )
        

    def get_legal_action(self):
        # TODO better legal action design?? talent, action space, conf.py ...
        
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        # legal_action = [self.move_usable] * self.move_action_num
        legal_action = [True] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        # if self.move_usable not in legal_action:
        if True not in legal_action:
            # 如果没有可用的移动动作，则将所有移动动作设置为可用
            self.bad_move_ids = set()
            # return [self.move_usable] * self.move_action_num
            return [True] * self.move_action_num

        return legal_action
