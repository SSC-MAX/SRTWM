def find_best_match(data_dict, target_list):
    """
    在字典中查找与目标列表从左到右匹配程度最高的列表。

    参数:
    data_dict: 包含列表作为值的字典，例如 {'k1': [1, 0, 1]}
    target_list: 用于比较的目标列表，例如 [1, 0, 0]

    返回:
    匹配度最高的列表值。如果字典为空，返回 None。
    """
    best_match_value = None
    max_score = -1

    for key, candidate_list in data_dict.items():
        current_score = 0

        # 使用 zip 同时遍历候选列表和目标列表
        # zip 会自动以较短的那个列表长度为准停止
        for val, target in zip(candidate_list, target_list):
            if val == target:
                current_score += 1
            else:
                # 一旦遇到不匹配，停止当前列表的计数（连续匹配逻辑）
                break

                # 更新最佳匹配
        if current_score > max_score and current_score!=0:
            max_score = current_score
            best_match_value = candidate_list

    return best_match_value


# --- 测试代码 ---

# 输入数据
data = {
    'key1': [0, 0, 1],
    'key2': [0, 1],
    'key3': [0, 0, 0]  # 完全不匹配的情况
}
target = [1, 0, 0]

# 调用函数
result = find_best_match(data, target)

# 打印结果
print(f"目标列表: {target}")
print(f"字典数据: {data}")
print("-" * 20)
print(f"匹配度最高的项: {result}")