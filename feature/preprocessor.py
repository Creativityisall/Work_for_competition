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

OBSTICAL = 0
FREE2GO = 1

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

        # agent position
        self.cur_pos = None 
        self.last_pos = None
        self.cur_pos_norm = None
        self.last_pos_norm = None 

        # talent & action
        self.talent_available = False  # 初始尚未加载穿墙技能
        self.talent_cd = float("inf")  # 剩余冷却时间初始置为无穷大
        self.last_action = -1
        self.legal_action = [True] * self.move_action_num

        # treasure and buff
        self.treasure_buf_list = [None] * self.treasure_and_buff_total_cnt  # treasure and buff positions

        # destination
        self.dest_pos = None 
        self.dest_pos_norm = None
        self.dest_rel_pos_norm = None
        self.is_dest_pos_found = False

        # target 
        self.target_pos = None 
        self.target_rel_pos_norm = None
        self.target_pos_norm = None
        self.target_distance_norm = None
        self.last_target_distance_norm = None  # 上一帧的目标距离 norm
        
        # bookkeeping map
        self.global_map = np.full((128, 128), -1, dtype=int)  
        self.undetected_area = 128 * 128
        self.cnt_new_detected = 0
    
    def set_target(self,
        target_pos=None,
        target_rel_pos_norm=None,
        target_pos_norm=None,
        target_distance_norm=None,
        last_target_distance_norm=None,
    ):
        self.target_pos = target_pos
        self.target_rel_pos_norm = target_rel_pos_norm
        self.target_pos_norm = target_pos_norm
        self.target_distance_norm = target_distance_norm
        self.last_target_distance_norm = last_target_distance_norm
    
    def _get_pos_feature(self, found, cur_pos, target_pos):
        """
        计算：目标相对坐标 norm + 目标绝对坐标 norm + 距离 norm
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

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    

    def update_agent_pos(self, obs, hero):
        ######## Update current position and position norm (cur & last) ########
        if self.cur_pos is None:
            self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
            self.cur_pos_norm = (
                norm(self.cur_pos[0], 127, 0),
                norm(self.cur_pos[1], 127, 0),
            )
        else:
            self.last_pos = self.cur_pos
            self.last_pos_norm = self.cur_pos_norm
        
            self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
            self.cur_pos_norm = (
                norm(self.cur_pos[0], 127, 0),
                norm(self.cur_pos[1], 127, 0),
            )

    def update_view(self, obs):
        ######## update detected area ########
        if self.last_pos is not None:
            self.global_map[self.last_pos[0], self.last_pos[1]] = FREE2GO
        
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
                        self.global_map[x, z] = map_info[25-j]["values"][25+i] # 0 for OBSTICAL, 1 for FREE2GO
        
        self.undetected_area -= self.cnt_new_detected
        assert self.undetected_area == np.sum(self.global_map == -1), \
            f"undetected_area {self.undetected_area} != np.sum(self.global_map == -1) {np.sum(self.global_map == -1)}"

    def get_talent_status_and_legal_action(self, hero, last_action):
        self.last_action = last_action 
        self.talent_available = hero["talent"]["status"]
        self.talent_cd = hero["talent"]["cooldown"]
        
        self.legal_action = [True] * (self.move_action_num // 2) + [self.talent_available] * (self.move_action_num // 2) 

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                
                x, z = self.cur_pos[0] + i, self.cur_pos[1] + j
                assert 0 <= x < 128 and 0 <= z < 128, \
                    f"8 grids around cur_pos {self.cur_pos} + ({i}, {j}) should be within the map"
                
                if self.global_map[x][z] == OBSTICAL:
                    self.legal_action[self._ij2direction(i, j)] = False


    def record_treasure_and_buff(self, obs):
        organs = obs["frame_state"]["organs"]
        for organ in organs:
            pos = (organ["pos"]["x"], organ["pos"]["z"])
            config_id=organ["config_id"]

            if organ["sub_type"] == T_TREASURE:  # treasure                
                self.global_map[pos[0], pos[1]] = TREASURE if organ["status"] == 1 else FREE2GO  # bookkeeping: mark treasure 
                # XXX 我怀疑取过的宝箱，再经过时还是会看到，只不过状态为 0（不可取）。所以该宝箱在 bookkeeping 上的位置由宝箱的状态决定，要么是 TREASURE，要么是 T_FREE2GO。
                
                self.treasure_buf_list[config_id] = organ

            elif organ["sub_type"] == T_BUFF:   # buff                
                self.global_map[pos[0], pos[1]] = BUFF if organ["status"] == 1 else FREE2GO  # bookkeeping: mark buff
                
                self.treasure_buf_list[config_id] = organ
    
    def record_destination(self, obs):
        if not self.is_dest_pos_found:
            self.dest_rel_pos_norm = (
                norm(self.dest_pos[0] - self.cur_pos[0], 1, -1),
                norm(self.dest_pos[1] - self.cur_pos[1], 1, -1),
            )
        else:
            organs = obs["frame_state"]["organs"]
            for organ in organs:
                pos = (organ["pos"]["x"], organ["pos"]["z"])
                
                if organ["sub_type"] == T_DESTINATION:  
                    assert organ["status"] == 1, "Destination should always be reachable."
                    
                    self.global_map[pos[0], pos[1]] = DESTINATION  # bookkeeping: mark destination
                    
                    dest_pos_feature = self._get_pos_feature(
                        found=True,
                        cur_pos=self.cur_pos,
                        target_pos=pos,
                    )
                    self.dest_pos = pos
                    self.dest_pos_norm = (dest_pos_feature[3], dest_pos_feature[4])
                    assert self.dest_pos_norm == (norm(pos[0], 127, 0), norm(pos[1], 127, 0))
                    self.dest_rel_pos_norm = (dest_pos_feature[1], dest_pos_feature[2])
                    self.is_dest_pos_found = True
                    break

    def update_target(self):
        """
        我觉得设置目标，唯一的作用是：鼓励agent就近拾取宝箱或增益。据此设计的奖励函数，本质是贪心拾取。但是当探索阶段结束，或者统计个数发现全局宝箱都捡完了，此时要设置目的地为目标，这并不能教会agent如何到达之。不过就这么贪心选取吧。

        每步都选一遍目标。
        
        理想情况：目标应具有连续性，不会乱变，除非目标已经达到（此时地图状态会自动更新，如treasure_buf_list中的项会标记为status=0）或者进入别的阶段（如读秒阶段必须设置终点为目标）

        选择逻辑：
        - 如果步数大于600且已经看到目的地，则直接设置目的地为目标
        - 如果步数大于600且没有看到目的地，则不设置目标，鼓励探索
        - 如果步数小于600且未探测全部地图，则尝试将“很小范围内”的宝箱或增益作为目标
            - 如果没找到很近的宝箱或增益，则继续探索，不设置目标
            - 如果找到了，则设置其中最近者为目标
        - 如果步数小于600且探测全部地图
            - 如果已经收齐全部宝箱（由于已经全部探测，故可以知道宝箱总数和已获取宝箱数），直接设置目的地为目标
                - 如果增益“真的”离得很近，则设置增益为目标？
            - 如果没有集齐，则设置最近宝箱为目标。
        """
        
        def _find_nearest_treasure_or_buff(explore_phase : bool):
            treasure_or_buff_target_pos = None
            min_dist = float("inf")
            for organ in self.treasure_buf_list:
                if organ is None or organ["status"] == 0:
                    continue
                pos = (organ["pos"]["x"], organ["pos"]["z"])
                dist = self._manhattan_distance(self.cur_pos, pos)
                if explore_phase:
                    if dist <= 10 / (2*128) and dist < min_dist:
                        min_dist = dist
                        treasure_or_buff_target_pos = pos
                else:
                    if dist < min_dist:
                        min_dist = dist
                        treasure_or_buff_target_pos = pos

            return treasure_or_buff_target_pos
        
        if self.step_no > 600:
            # 步数很大了，无脑终点为目标（若还没找到，则继续探索，不设置目标）
            if self.is_dest_pos_found:
                self.set_target(
                    target_pos=self.dest_pos,
                    target_rel_pos_norm=self.dest_rel_pos_norm,
                    target_pos_norm=self.dest_pos_norm,
                    target_distance_norm=norm(
                        self._manhattan_distance(self.cur_pos, self.dest_pos), 
                        2 * 128
                    ),
                    last_target_distance_norm=norm(
                        self._manhattan_distance(self.last_pos, self.dest_pos), 
                        2 * 128
                    ),
                )
            else:
                self.set_target()
        else:
            # 步数小于600，设置目标选择更多
            if self.undetected_area > 0:
                # 探索阶段，试图寻找“非常接近”的宝箱或增益：由于是探索期，额外要求只有真的很近才能贪心。
                treasure_or_buff_target_pos = _find_nearest_treasure_or_buff(explore_phase=True)
                if treasure_or_buff_target_pos is not None:
                    self.set_target(
                        target_pos=treasure_or_buff_target_pos,
                        target_rel_pos_norm=(
                            norm(treasure_or_buff_target_pos[0] - self.cur_pos[0], 1, -1),
                            norm(treasure_or_buff_target_pos[1] - self.cur_pos[1], 1, -1),
                        ),
                        target_pos_norm=(
                            norm(treasure_or_buff_target_pos[0], 127, 0),
                            norm(treasure_or_buff_target_pos[1], 127, 0),
                        ),
                        target_distance_norm=norm(
                            self._manhattan_distance(self.cur_pos, treasure_or_buff_target_pos), 
                            2 * 128
                        ),
                        last_target_distance_norm=norm(
                            self._manhattan_distance(self.last_pos, treasure_or_buff_target_pos), 
                            2 * 128
                        ),
                    )                    
                else:
                    # 没有很近的宝箱或增益，继续探索，不设置目标
                    self.set_target()
            else:
                # 探索结束，判断是否已经收集完所有宝箱
                collected_treasure_cnt = sum(
                    1 for organ in self.treasure_buf_list 
                    if organ is not None and organ["sub_type"] == T_TREASURE and organ["status"] == 0
                )
                total_treasure_cnt = sum(
                    1 for organ in self.treasure_buf_list 
                    if organ is not None and organ["sub_type"] == T_TREASURE
                )
                if total_treasure_cnt == collected_treasure_cnt:
                    # 全部宝箱都收集完了，设置目的地为目标
                    assert self.dest_pos is not None, "Destination should be found if exploration is finished."
                    self.set_target(
                        target_pos=self.dest_pos,
                        target_rel_pos_norm=self.dest_rel_pos_norm, # NOTE This should be updated beforehead, namely record_destination() should be called before update_target()
                        target_pos_norm=self.dest_rel_pos_norm,
                        target_distance_norm=norm(
                            self._manhattan_distance(self.cur_pos, self.dest_pos), 
                            2 * 128
                        ),
                        last_target_distance_norm=norm(
                            self._manhattan_distance(self.last_pos, self.dest_pos), 
                            2 * 128
                        ),
                    )
                else:
                    # 没有收集完所有宝箱，设置最近的宝箱为目标。由于探索已经结束，故没必要额外要求“真近”。
                    treasure_or_buff_target_pos = _find_nearest_treasure_or_buff(explore_phase=False)
                    assert treasure_or_buff_target_pos is not None, "Should find at least one treasure."
                    self.set_target(
                        target_pos=treasure_or_buff_target_pos,
                        target_rel_pos_norm=(
                            norm(treasure_or_buff_target_pos[0] - self.cur_pos[0], 1, -1),
                            norm(treasure_or_buff_target_pos[1] - self.cur_pos[1], 1, -1),
                        ),
                        target_pos_norm=(
                            norm(treasure_or_buff_target_pos[0], 127, 0),
                            norm(treasure_or_buff_target_pos[1], 127, 0),
                        ),
                        target_distance_norm=norm(
                            self._manhattan_distance(self.cur_pos, treasure_or_buff_target_pos), 
                            2 * 128
                        ),
                        last_target_distance_norm=norm(
                            self._manhattan_distance(self.last_pos, treasure_or_buff_target_pos), 
                            2 * 128
                        ),
                    )
                
    
         
    def process(self, frame_state, last_action):
        obs, _ = frame_state
        hero = obs["frame_state"]["heroes"][0]
        
        # Record step_no
        self.step_no = obs["frame_state"]["step_no"]

        # Update -1s in bookkeeping map
        self.update_view(obs)

        # Update agent position
        self.update_agent_pos(obs, hero)

        # Process legal action
        self.get_talent_status_and_legal_action(hero, last_action) 
        
        # Process treasure, buff, destination
        self.record_treasure_and_buff(obs) # NOTE there are bookkeepings in here
        self.record_destination(obs)

        # Update target
        self.update_target()
        
        

        # Feature 
        # NOTE Should be compatible with conf.py configuration
        feature = np.concatenate([
            # agent position
            [self.last_pos_norm[0], self.last_pos_norm[1]],                                                                         # 2
            [self.cur_pos_norm[0], self.cur_pos_norm[1]] ,                                                                          # 2
            
            # action and talent related
            [self.talent_available],                                                                                                # 1 
            [self.talent_cd / 30.0],                                                                                                # 1 (30.0怎么来的?)     
            [self.last_action],                                                                                                     # 1
            self.legal_action,                                                                                                      # 16
            
            # organs (tresure and buff status)
            [-1 if organ is None else organ["status"] for organ in self.treasure_buf_list]                                          # 14

            # destination???
            # ...

            # target
            [self.target_rel_pos_norm[0], self.target_rel_pos_norm[1]] if self.target_rel_pos_norm is not None else [0, 0],         # 2
            [self.target_pos_norm[0], self.target_pos_norm[1]] if self.target_pos is not None else [0, 0],                          # 2
            # XXX 需要 target_distance_norm 和 last_target_distance_norm 吗？
            
            # bookkeeping map
            self.global_map.flatten(),                                                                                              # 128*128
        ])

        return (
            feature,
            self.legal_action,
            reward_process(
                step_no=self.step_no,

                cur_pos=self.cur_pos,
                cur_pos_norm=self.cur_pos_norm,
                last_pos=self.last_pos,
                last_pos_norm=self.last_pos_norm,

                talent_last_action=(self.last_action >= 8), 

                target_pos=self.target_pos,
                target_rel_pos_norm=self.target_rel_pos_norm,
                target_pos_norm=self.target_pos_norm,
                target_distance_norm=self.target_distance_norm,
                last_target_distance_norm=self.last_target_distance_norm,

                undetected_area=self.undetected_area,
                cnt_new_detected=self.cnt_new_detected,
            ),  
        )
    

    



                