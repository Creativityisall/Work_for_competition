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

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        # 动态统计当前步骤中发现的宝箱总数
        current_treasures_in_view = 0
        treasure_positions = set()

        # 初始化目标列表和优先级
        target_candidates = []
        for organ in obs["frame_state"]["organs"]:
            if organ["status"] == 1:  # 物品可用
                pos = (organ["pos"]["x"], organ["pos"]["z"])

                if organ["sub_type"] == 4:  # 终点 - 最高优先级
                    target_candidates.append({
                        'pos': pos,
                        'priority': 1,
                        'type': 'end_point',
                        'weight': 1.0
                    })
                elif organ["sub_type"] == 1:  # 宝箱 - 高优先级
                    current_treasures_in_view += 1  # 添加这行
                    treasure_positions.add(pos)     # 添加这行
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
        # 更新历史发现的宝箱总数（累计最大值）
        if not hasattr(self, 'total_treasures_discovered'):
            self.total_treasures_discovered = current_treasures_in_view
            self.discovered_treasure_positions = treasure_positions.copy()
        else:
            # 合并新发现的宝箱位置
            self.discovered_treasure_positions.update(treasure_positions)
            # 更新总数为发现过的宝箱总数
            self.total_treasures_discovered = len(self.discovered_treasure_positions)
        
        # 从score_info获取已收集的宝箱数量
        score_info = obs.get("score_info", {})
        self.treasures_collected = score_info.get("treasure_collected_count", 0)
        
        # 更新历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # 选择目标
        self._select_target(target_candidates) # NOTE this will modify self.end_pos, self.is_end_pos_found etc. 
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # update position norm (cur & last)
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 127, 0)

        # History position feature
        # 历史位置特征
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        # self.move_usable = True           # XXX REDUNDANT! 
        self.last_action = last_action      # NOTE 右边是 agent 的 last_action，作为参数传入 pb2struct；左边是 preprocessor(collector) 记录的 last_action
            
    
    def _select_target(self, target_candidates):
        if not target_candidates:
            random_target = None
        
            if len(self.target_pos_list) > 0:
                # 从预设的目标位置列表中随机选择一个
                random_index = random.randrange(len(self.target_pos_list))
                random_pos = self.target_pos_list[random_index]
            else:
                # 如果目标位置列表也为空，设置一个默认的随机位置
                # 在地图范围内随机生成一个位置 (坐标范围 0-127)
                random_x = random.randint(10, 117)  # 避免边界
                random_z = random.randint(10, 117)  # 避免边界
                random_pos = (random_x, random_z)
            
            # 创建随机目标并加入候选列表
            random_target = {
                'pos': random_pos,
                'priority': 4,  # 随机目标优先级最低
                'type': 'random',
                'weight': 0.3
            }
            target_candidates.append(random_target)
        
        # 计算收集进度
        total_treasures = getattr(self, 'total_treasures_discovered', 1)
        treasures_collected = getattr(self, 'treasures_collected', 0)
        collection_progress = treasures_collected / max(total_treasures, 1)
        # 按优先级排序目标候选列表
        target_candidates.sort(key=lambda x: x['priority'])
        
        # 距离计算和目标选择
        best_target = None
        best_score = -float('inf')

        for target in target_candidates:
            distance = np.linalg.norm(tuple(y - x for x, y in zip(self.cur_pos, target['pos'])))
            
            # 计算综合得分：优先级权重 - 距离惩罚 + 类型奖励
            distance_penalty = distance / (1.41 * 128)  # 归一化距离
            
            if target['type'] == 'end_point':
            # 终点：只有收集足够多宝箱后才高优先级
                if collection_progress >= 0.8:
                    score = target['weight'] * (3.0 - distance_penalty)
                elif collection_progress >= 0.5:
                    score = target['weight'] * (1.0 - distance_penalty)
                else:
                    score = target['weight'] * (0.2 - distance_penalty)
            elif target['type'] == 'treasure':
                # 宝箱：平衡距离和收益
                if collection_progress < 0.8:
                    score = target['weight'] * (2.0 - distance_penalty)
                else:
                    score = target['weight'] * (0.5 - distance_penalty)
            elif target['type'] == 'buff':
                # Buff点：中等吸引力
                score = target['weight'] * (0.6 - 0.3 * distance_penalty)
            elif target['type'] == 'random':
                # 随机目标：最低吸引力，主要用于探索
                score = target['weight'] * (0.4 - 0.2 * distance_penalty)
            else:
                # 其他类型目标
                score = target['weight'] * (0.4 - 0.2 * distance_penalty)
            
            # 如果当前目标类型相同，给予连续性奖励
            if hasattr(self, 'current_target_type') and target['type'] == self.current_target_type:
                score += 0.1  # 连续性奖励
            
            if score > best_score:
                best_score = score
                best_target = target
        
        # 更新目标
        if best_target:
            old_target = getattr(self, 'end_pos', None)
            self.end_pos = best_target['pos']
            self.current_target_type = best_target['type']
            self.target_priority = best_target['priority']
            self.target_weight = best_target['weight']
            
            # 设置found状态
            if best_target['type'] == 'end_point':
                self.is_end_pos_found = True  # 真实游戏目标标记为已找到
            else:
                self.is_end_pos_found = False   # 随机目标标记为未找到真实终点
            
       
    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Feature
        # 特征
        feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])

        target_info = {
            'type': getattr(self, 'current_target_type', 'random'),
            'weight': getattr(self, 'target_weight', 0.3),
            'priority': getattr(self, 'target_priority', 4),
            'treasures_collected': getattr(self, 'treasures_collected', 0),
            'total_treasures': getattr(self, 'total_treasures_discovered', 1),
            'treasure_score': getattr(self, 'treasure_score', 0),
            'step_no': getattr(self, 'step_no', 0),
            'collection_progress': getattr(self, 'treasures_collected', 0) / max(getattr(self, 'total_treasures_discovered', 1), 1)
        }

        return (
            feature,
            legal_action,
            reward_process(
                self.feature_end_pos[-1], 
                self.feature_history_pos[-1], 
                feature_vector=feature,
                target_info=target_info
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
