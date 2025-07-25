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
import random  # <<< ADDED: 导入 random 模块
# <<< MODIFIED: 移除了不再需要的导入
from agent_ppo.feature.definition import reward_process


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8  # TODO:加上超级闪现
        self.reset()

    def reset(self):
        # <<< MODIFIED: 采用新逻辑的 reset 方法
        self.target_pos_list = [(26, 87), (85, 114), (32, 24), (101, 40), (59, 64)]
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()
        # <<< ADDED: 初始化新逻辑所需的属性
        if hasattr(self, 'discovered_treasure_positions'):
            self.discovered_treasure_positions.clear()
        else:
            self.discovered_treasure_positions = set()
        self.total_treasures_discovered = 0
        self.treasures_collected = 0


    def _get_pos_feature(self, found, cur_pos, target_pos):
        # <<< ADDED: 安全检查，防止 target_pos 为 None
        if target_pos is None:
            # 当目标不存在时，返回一个零特征向量
            return np.zeros(6) 
            
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        # <<< MODIFIED: 统一归一化到 [0, 1] 区间，以匹配新逻辑
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

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        
        # <<< MODIFIED BLOCK START: 替换为新的多目标选择逻辑
        current_treasures_in_view = 0
        treasure_positions = set()

        target_candidates = []
        for organ in obs["frame_state"]["organs"]:
            if organ["status"] == 1:
                pos = (organ["pos"]["x"], organ["pos"]["z"])
                if organ["sub_type"] == 4:
                    target_candidates.append({'pos': pos, 'priority': 1, 'type': 'end_point', 'weight': 1.0})
                elif organ["sub_type"] == 1:
                    current_treasures_in_view += 1
                    treasure_positions.add(pos)
                    target_candidates.append({'pos': pos, 'priority': 2, 'type': 'treasure', 'weight': 0.8})
                elif organ["sub_type"] == 2:
                    target_candidates.append({'pos': pos, 'priority': 3, 'type': 'buff', 'weight': 0.6})
        
        if not hasattr(self, 'total_treasures_discovered'):
            self.total_treasures_discovered = current_treasures_in_view
            self.discovered_treasure_positions = treasure_positions.copy()
        else:
            self.discovered_treasure_positions.update(treasure_positions)
            self.total_treasures_discovered = len(self.discovered_treasure_positions)
        
        score_info = obs.get("score_info", {})
        self.treasures_collected = score_info.get("treasure_collected_count", 0)
        
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        self._select_target(target_candidates)
        # <<< MODIFIED BLOCK END

        self.last_pos_norm = self.cur_pos_norm
        # <<< MODIFIED: 统一归一化到 [0, 1] 区间
        self.cur_pos_norm = norm(self.cur_pos, 127, 0)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        self.move_usable = True
        self.last_action = last_action
    
    # <<< ADDED: 完整的新增 _select_target 方法
    def _select_target(self, target_candidates):
        if not target_candidates:
            if len(self.target_pos_list) > 0:
                random_index = random.randrange(len(self.target_pos_list))
                random_pos = self.target_pos_list[random_index]
            else:
                random_x = random.randint(10, 117)
                random_z = random.randint(10, 117)
                random_pos = (random_x, random_z)
            
            random_target = {'pos': random_pos, 'priority': 4, 'type': 'random', 'weight': 0.3}
            target_candidates.append(random_target)
        
        total_treasures = getattr(self, 'total_treasures_discovered', 1)
        treasures_collected = getattr(self, 'treasures_collected', 0)
        collection_progress = treasures_collected / max(total_treasures, 1)
        
        target_candidates.sort(key=lambda x: x['priority'])
        
        best_target = None
        best_score = -float('inf')

        for target in target_candidates:
            distance = np.linalg.norm(tuple(y - x for x, y in zip(self.cur_pos, target['pos'])))
            distance_penalty = distance / (1.41 * 128)
            score = 0
            if target['type'] == 'end_point':
                if collection_progress >= 0.8 and treasures_collected >= 5: score = target['weight'] * (3.0 - distance_penalty)
                elif collection_progress >= 0.5 and treasures_collected >= 2: score = target['weight'] * (1.0 - distance_penalty)
                else: score = target['weight'] * (0.2 - distance_penalty)
            elif target['type'] == 'treasure':
                if collection_progress < 0.8 or treasures_collected <= 2: score = target['weight'] * (2.0 - distance_penalty)
                else: score = target['weight'] * (0.5 - distance_penalty)
            elif target['type'] == 'buff': score = target['weight'] * (0.6 - 0.3 * distance_penalty)
            elif target['type'] == 'random': score = target['weight'] * (0.4 - 0.2 * distance_penalty)
            else: score = target['weight'] * (0.4 - 0.2 * distance_penalty)
            
            if hasattr(self, 'current_target_type') and target['type'] == self.current_target_type:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_target = target
        
        if best_target:
            self.end_pos = best_target['pos']
            self.current_target_type = best_target['type']
            self.target_priority = best_target['priority']
            self.target_weight = best_target['weight']
            self.is_end_pos_found = best_target['type'] == 'end_point'

    # <<< MODIFIED: 函数签名和返回值保持原样
    def process(self, terminated, truncated, obs, extra_info, rewardStateTracker, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        legal_action = self.get_legal_action()
        feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])

        # <<< MODIFIED: 保持您现有的 reward_process 调用方式
        if not frame_state[1]:
            return (feature, legal_action, 0)
            
        game_info = frame_state[1]["game_info"]
        reward = reward_process(rewardStateTracker, game_info['step_no'], terminated, truncated, obs, frame_state[0], extra_info, frame_state[1], self.feature_end_pos[-1], self.feature_history_pos[-1], self.end_pos)

        return (feature, legal_action, reward)

    def get_legal_action(self):
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        legal_action = [self.move_usable] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num

        return legal_action