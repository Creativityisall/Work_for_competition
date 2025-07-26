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

T_TREASURE = 1
T_BUFF = 2
T_START = 3
T_DESTINATION = 4
T_OBSTICAL = 0
T_FREE2GO = 1

TREASURE = 2
BUFF = 3
DESTINATION = 4
PLAYER = 10





class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 16
        self.treasure_and_buff_total_cnt = 14
        self.reset()

    def reset(self):
        # frame number
        self.step_no = 0

        # position
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))

        # explore and map
        self.global_map = np.full((128, 128), -1, dtype=int)  
        self.undetected_area = 128 * 128
        self.cnt_new_detected = 0
        
        # treasure and buff
        self.treasure_buf_list = [None] * self.treasure_and_buff_total_cnt  # treasure and buff positions

        # destination
        self.dest_pos = None
        self.is_dest_pos_found = False

        # target 
        self.target_distance_norm = None
        self.target_pos = None
        
        # talent
        self.talent_available = False  # 初始尚未加载穿墙技能
        self.talent_cd = float("inf")  # 剩余冷却时间初始置为无穷大

        # action
        self.last_action = -1
        self.legal_action = [True] * self.move_action_num

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
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def update_pos(self, obs, hero):
        # Update current position
        # Update position norm (cur & last)
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = (
            norm(self.cur_pos[0], 127, 0),
            norm(self.cur_pos[1], 127, 0),
        )
    
    def get_legal_action(self, obs):
        self.legal_action = [True] * (self.move_action_num // 2) + [self.talent_available] * (self.move_action_num // 2) 

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                
                x, z = self.cur_pos[0] + i, self.cur_pos[1] + j
                assert 0 <= x < 128 and 0 <= z < 128, \
                    f"8 grids around cur_pos {self.cur_pos} + ({i}, {j}) should be within the map"
                
                if self.global_map[x][z] == T_OBSTICAL:
                    self.legal_action[self._ij2direction(i, j)] = False    
    
    def update_view(self, obs):
        """
        Update the map under current view.
        """   

        ######## update detected area ########
        self.global_map[self.cur_pos[0], self.cur_pos[1]] = PLAYER

        map_info = obs["map_info"]
        self.cnt_new_detected = 0
        x0, z0 = self.cur_pos
        for i in range(-25, 25 + 1):
            for j in range(-25, 25 + 1):
                x, z = x0 + i, z0 + j
                if 0 <= x < 128 and 0 <= z < 128:                    
                    if self.global_map[x, z] == -1:
                        self.cnt_new_detected += 1
                        self.global_map[x, z] = map_info[25-j]["values"][25+i] # 0 for T_OBSTICAL, 1 for T_FREE2GO
        
        self.undetected_area -= self.cnt_new_detected
        assert self.undetected_area == np.sum(self.global_map == -1), \
            f"undetected_area {self.undetected_area} != np.sum(self.global_map == -1) {np.sum(self.global_map == -1)}"

        ######## update treasure_bu list & destination ########
        organs = obs["frame_state"]["organs"]
        for organ in organs:
            pos = (organ["pos"]["x"], organ["pos"]["z"])
            config_id=organ["config_id"]
            if organ["sub_type"] == T_TREASURE:  # treasure
                # XXX 我怀疑取过的宝箱，再经过时还是会看到，只不过状态为 0（不可取）。
                assert 1 <= config_id <=13 , "Protocol: treasure -> [1, 13]"
                self.global_map[pos[0], pos[1]] = TREASURE if organ["status"] == 1 else T_FREE2GO  # mark as treasure 
                self.treasure_buf_list[config_id] = organ

            elif organ["sub_type"] == T_BUFF:   # buff
                assert config_id == 0, "Protocol: buff -> 0"
                self.global_map[pos[0], pos[1]] = BUFF if organ["status"] == 1 else T_FREE2GO  # mark as buff
                self.treasure_buf_list[config_id] = organ

            elif organ["sub_type"] == T_DESTINATION:  # destination
                assert organ["status"] == 1, "Destination should always be reachable."
                self.global_map[pos[0], pos[1]] = DESTINATION  # mark as destination
                self.dest_pos = pos
                self.is_dest_pos_found = True
            else:
                pass  # ignore other organs



        ######## Record target in `self.target_distance_norm` ########

        # During exploration, target <- nearest treasure or buff; 
        # If exploration finished, target <- destination
        # if no target currently, target_distance_norm <- None
        if self.undetected_area == 0:
            # if exploration finished, then directly set destination as target
            assert self.dest_pos is not None, "Destination should be found if exploration finished."
            self.target_pos = (self.dest_pos[0], self.dest_pos[1])
            self.target_distance_norm = norm(
                self._manhattan_distance(self.cur_pos, self.dest_pos), 2 * 128
            )
        else:
            min_dist = float("inf")
            for organ in self.treasure_buf_list:
                if organ is not None and organ["status"] == 1:  # only consider reachable organs
                    pos = (organ["pos"]["x"], organ["pos"]["z"])
                    dist = self._manhattan_distance(self.cur_pos, pos)
                    if dist < min_dist:
                        min_dist = dist
                        self.target_pos = pos
                        self.target_distance_norm = norm(dist, 2 * 128)  

        
        
    
    def process(self, frame_state, last_action):
        obs, _ = frame_state
        hero = obs["frame_state"]["heroes"][0]
        
        # Record step_no
        self.step_no = obs["frame_state"]["step_no"]

        # Update current position
        self.update_pos(obs, hero)

        # Process legal action
        self.last_action = last_action 
        self.talent_available = hero["talent"]["status"]
        self.talent_cd = hero["talent"]["cooldown"]
        self.get_legal_action(obs) # TODO how to process talent? how to design action space?

        # Process frame state: update self's attributes
        if hasattr(self, 'target_distance_norm'):
            self.last_dist_goal_norm = self.target_distance_norm
        else:
            self.last_dist_goal_norm = None
        self.update_view(obs)
        


        # Feature TODO
        feature = np.concatenate([
            # [self.step_no],                                      # 1,
            [self.cur_pos_norm[0], self.cur_pos_norm[1]] ,       # 2,
            # self.legal_action,                                   # 16,
            # [self.target_pos[0], self.target_pos[1]] if self.target_pos is not None else [0, 0],  # 2, target pos
            # [self.talent_available],                             # 1, 
            # [self.talent_cd / 30.0],                             # 1, normalize talent cooldown to [0, 1]     
            # self.global_map.flatten(),                           # 128*128,
        ])

        return (
            feature,
            self.legal_action,
            reward_process(
                # step_no=self.step_no,
                # cur_pos=self.cur_pos,
                # cur_pos_norm=self.cur_pos_norm,

                # undetected_area=self.undetected_area,
                cnt_new_detected=self.cnt_new_detected, # XXX
                is_exploration_finished=self.undetected_area == 0, # XXX

                dist_goal_norm=self.target_distance_norm, # XXX
                last_dist_goal_norm=self.last_dist_goal_norm, # XXX
                
                # destination_pos=self.dest_pos,
                # has_found_dest=self.is_dest_pos_found,
                
                # legal_action=self.legal_action,
                is_last_action_talent=self.last_action >= 8, # XXX
                # talent_available=self.talent_available,
                # talent_cd=self.talent_cd,
            ),  
        )
        
    def _ij2direction(self, i, j):
        """
        Convert vector (i, j) to relative direction.
        """
        assert abs(i) <= 1 and abs(j) <= 1, f"i={i}, j={j} should be in [-1, 0, 1]"
        assert i != 0 or j != 0, f"i={i}, j={j} should not be (0, 0)"

        if i == 1 and j == 0:
            return 0
        elif i == 1 and j == 1:
            return 1
        elif i == 0 and j == 1:
            return 2
        elif i == -1 and j == 1:
            return 3
        elif i == -1 and j == 0:
            return 4
        elif i == -1 and j == -1:
            return 5
        elif i == 0 and j == -1:
            return 6    
        elif i == 1 and j == -1:
            return 7



                