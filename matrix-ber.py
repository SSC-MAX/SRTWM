def merge_extracted_messages(items):
    # 取最大长度
    max_len = max(len(item["extracted_message"]) for item in items)

    result = []
    for col in range(max_len):
        ones = zeros = 0

        for item in items:
            msg = item["extracted_message"]
            if col < len(msg):
                if msg[col] == 1:
                    ones += 1
                else:
                    zeros += 1

        # 取多数（若相等，这里默认取 1）
        result.append(1 if ones >= zeros else 0)

    return result
