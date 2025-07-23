
# from enum import Enum
# from typing import List

# class Observation:
#     def __init__(self, frame_state, score_info, map_info, legal_act):
#         self.frame_state = frame_state
#         self.score_info = score_info
#         self.map_info = map_info
#         self.legal_act = legal_act

# class MapInfo:
#     def __init__(self, values: List[int]):
#         self.values = values

# class ScoreInfo:
#     def __init__(self, score, total_score, step_no, treasure_collected_count, treasure_score, buff_count, talent_count):
#         self.score = score
#         self.total_score = total_score
#         self.step_no = step_no
#         self.treasure_collected_count = treasure_collected_count
#         self.treasure_score = treasure_score
#         self.buff_count = buff_count
#         self.talent_count = talent_count

# class ExtraInfo:
#     def __init__(self, result_code, result_class, frame_state, game_info):
#         self.result_code = result_code
#         self.result_class = result_class
#         self.frame_state = frame_state
#         self.game_info = game_info

# class RelativePosition:
#     def __init__(self, direction, l2_distance):
#         self.direction = direction
#         self.l2_distance = l2_distance

# class RelativeDirection(Enum):
#     RELATIVE_DIRECTION_NONE = 0
#     East = 1
#     NorthEast = 2
#     North = 3
#     NorthWest = 4
#     West = 5
#     SouthWest = 6
#     South = 7
#     SouthEast = 8

# class RelativeDistance(Enum):
#     RELATIVE_DISTANCE_NONE = 0
#     VerySmall = 1
#     Small = 2
#     Medium = 3
#     Large = 4
#     VeryLarge = 5

# class FrameState:
#     def __init__(self, step_no, heroes, organs):
#         self.step_no = step_no
#         self.heroes = heroes
#         self.organs = organs

# class GameInfo:
#     def __init__(self, score, total_score, step_no, pos, start_pos, end_pos, treasure_collected_count, treasure_score, treasure_count, buff_count, talent_count, buff_remain_time, buff_duration, map_info, obstacle_id):
#         self.score = score
#         self.total_score = total_score
#         self.step_no = step_no
#         self.pos = pos
#         self.start_pos = start_pos
#         self.end_pos = end_pos
#         self.treasure_collected_count = treasure_collected_count
#         self.treasure_score = treasure_score
#         self.treasure_count = treasure_count
#         self.buff_count = buff_count
#         self.talent_count = talent_count
#         self.buff_remain_time = buff_remain_time
#         self.buff_duration = buff_duration
#         self.map_info = map_info
#         self.obstacle_id = obstacle_id

# class RealmHero:
#     def __init__(self, hero_id, pos, speed_up, talent, buff_remain_time):
#         self.hero_id = hero_id
#         self.pos = pos
#         self.speed_up = speed_up
#         self.talent = talent
#         self.buff_remain_time = buff_remain_time

# class Talent:
#     def __init__(self, talent_type, status, cooldown):
#         self.talent_type = talent_type
#         self.status = status
#         self.cooldown = cooldown

# class RealmOrgan:
#     def __init__(self, sub_type, config_id, status, pos, cooldown, relative_pos):
#         self.sub_type = sub_type
#         self.config_id = config_id
#         self.status = status
#         self.pos = pos
#         self.cooldown = cooldown
#         self.relative_pos = relative_pos

# class Position:
#     def __init__(self, x, z):
#         self.x = x
#         self.z = z