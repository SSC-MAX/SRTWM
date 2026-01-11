import random

def merge_extracted_messages(items):
    # 取最大长度
    max_len = max(len(item) for item in items)

    result = []
    for col in range(max_len):
        ones = zeros = 0

        for item in items:
            msg = item
            if col < len(msg):
                if msg[col] == 1:
                    ones += 1
                else:
                    zeros += 1

        # 取多数（若相等，这里默认取 1）
        result.append(1 if ones == zeros else random.randint(0,1))

    return result

extracted_message = [
            [
                1,
                0
            ],
            [
                1,
                0
            ],
            [
                1,
                0
            ],
            [
                1
            ],
            [
                1,
                0
            ]
        ]

merge_extracted_messages(extracted_message)