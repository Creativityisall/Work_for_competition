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
        self.target_pos_list = [(26, 87), (85, 114), (32, 24), (101, 40), (59, 64)]
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()

        self.has_set_random_target = False
        self.random_target_pos = None

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
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
        )
        return feature
    
    def _generate_target_candidates(self, obs):
        target_candidates = []
        for organ in obs["frame_state"]["organs"]:
            if organ["status"] != 1:
                continue  # 物品可用
            else:
                pos = (organ["pos"]["x"], organ["pos"]["z"])

                if organ["sub_type"] == 4:  # 终点 - 最高优先级
                    target_candidates.append({
                        'pos': pos,
                        'priority': 1,
                        'type': 'end_point',
                        'weight': 1.0
                    })
                elif organ["sub_type"] == 1:  # 宝箱 - 高优先级
                    target_candidates.append({
                        'pos': pos,
                        'priority': 2,
                        'type': 'treasure',
                        'weight': 0.8
                    })
                elif organ["sub_type"] == 2:  # Buff点 - 中等优先级
                    target_candidates.append({
                        'pos': pos,
                        'priority': 3,
                        'type': 'buff',
                        'weight': 0.6
                    })
        if len(target_candidates) > 0:
            # 有非随机目标了，重置随机目标
            self.has_set_random_target = False
            self.random_target_pos = None
            return target_candidates
        
        # 如果没有目标，检查是否已经设置随机目标
        if not self.has_set_random_target:
            def _get_diagonal_target(current_pos, map_size):
                """
                获取对角线方向的远距离目标点
                """                    
                # 计算到四个角的距离，选择最远的角
                corners = [
                    (20, 20),           # 左下角
                    (map_size-20, 20),  # 右下角
                    (20, map_size-20),  # 左上角
                    (map_size-20, map_size-20)  # 右上角
                ]
                
                max_distance = 0
                best_corner = corners[0]
                
                for corner in corners:
                    distance = np.linalg.norm(tuple(y - x for x, y in zip(current_pos, corner)))
                    if distance > max_distance:
                        max_distance = distance
                        best_corner = corner
                
                return best_corner
            def _generate_distant_target():
                # 尝试 10 次生成较远随机目标
                for _ in range(10):
                    target = (np.random.randint(10, 117), np.random.randint(10, 117))
                    if np.linalg.norm(tuple(y - x for x, y in zip(self.cur_pos, target))) > 40:
                        return {
                            'pos': target,
                            'priority': 4,
                            'type': 'random',
                            'weight': 0.5
                        }
                # 若失败，则选择最佳角落位置作为目标
                return {
                    'pos': _get_diagonal_target(self.cur_pos, 128),
                    'priority': 4,
                    'type': 'random',
                    'weight': 0.5
                }
            self.has_set_random_target = True
            self.random_target_pos = _generate_distant_target()
        else:
            # 如果已经设置了随机目标，则使用它
            pass 
        assert self.random_target_pos is not None and self.has_set_random_target, "Random target should be set if no other targets are available."               
        target_candidates.append(self.random_target_pos)
        assert len(target_candidates) == 1, "There should always be at least one target candidate."
        return target_candidates
    
    def _select_target(self, target_candidates):
        best_target = None
        min_distance = float('inf')
        for target in target_candidates:
            distance = np.linalg.norm(tuple(y - x for x, y in zip(self.cur_pos, target['pos'])))
            
            if target['type'] == 'random':
                assert len(target_candidates) == 1, "If the target is random, it should be the only candidate."
                best_target = target
                break
            elif target['type'] == 'end_point':
                best_target = target
                break
            elif target['type'] == 'treasure':
                if min_distance > distance:
                    min_distance = distance
                    best_target = target
            elif target['type'] == 'buff':
                if min_distance > distance:
                    min_distance = distance
                    best_target = target

        assert best_target is not None, "There should always be a best target selected."
        return best_target

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        # Record step_no
        self.step_no = obs["frame_state"]["step_no"]

        # Update current position
        # Update position norm (cur & last)
        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 127, 0)

        # Choose target
        target_candidates = self._generate_target_candidates(obs)
        best_target = self._select_target(target_candidates)
        self.end_pos = best_target['pos']            
        self.is_end_pos_found = True if best_target['type'] == 'end_point' else False  # 真实游戏目标标记为已找到
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # Update history position feature
        # 更新历史位置及其特征
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0]) # NOTE history here means 10 steps before

        # Record last action
        self.last_action = last_action      # NOTE 右边是 agent 的 last_action，作为参数传入 pb2struct；左边是 preprocessor(collector) 记录的 last_action
            
       
    def process(self, frame_state, last_action):
        # Process the frame state and last action to update self's attributes
        self.pb2struct(frame_state, last_action)

        # Legal action
        legal_action = self.get_legal_action()

        # Feature
        feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])

        return (
            feature,
            legal_action,
            reward_process(
                self.feature_end_pos[-1],       # 鼓励靠近目标（目标选取是之前的工作）
                self.feature_history_pos[-1],   # 鼓励远离 10 步之前的历史位置
            ),
        )
        

    def get_legal_action(self):
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
