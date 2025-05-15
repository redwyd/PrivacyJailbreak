import random


def get_interval_random_num(intervals, is_begin=False):
    """返回多个区间内的随机数，注意区间左闭右开"""
    old_entity_slice = slice
    total_length = sum(end - start for start, end in intervals)
    random_num = random.randint(0, total_length - 1)
    result = -1
    cumulative_length = 0
    for start, end in intervals:
        if random_num < cumulative_length + (end - start):
            if is_begin:
                result = start  # 只取端点
            else:
                result = random_num - cumulative_length + start  # 区间任意
            # old_entity_slice = slice(start, end)
            break
        cumulative_length += end - start
    return result

def get_interval_random_list(intervals):
    """返回多个区间内某个展开的随机区间"""
    old_entity_slice = slice
    total_length = sum(end - start for start, end in intervals)
    random_num = random.randint(0, total_length - 1)
    entity_list = []
    cumulative_length = 0
    for start, end in intervals:
        if random_num < cumulative_length + (end - start):
            entity_list.extend([i for i in range(start, end)])
            break
        cumulative_length += end - start
    return entity_list