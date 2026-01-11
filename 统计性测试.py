import json

import numpy as np
from scipy.stats import friedmanchisquare, norm
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison


# 假设你已经有多个水印检测器类
# 这里定义一个示例类来模拟你的水印检测器
class YourWatermarkDetector:
    def watermark_detector_fast_bert(self, text, alpha=0.05):
        # 这里是你的水印检测实现
        # 为示例目的，我们返回模拟的z分数
        p = 0.5
        n = len(text.split())
        if n == 0:
            return False, 0.5, 0, 0, 0
        ones = int(n * (0.5 + np.random.normal(0, 0.1)))  # 模拟你的水印检测结果
        ones = max(0, min(n, ones))  # 确保在有效范围内
        z = (ones - p * n) / (n * p * (1 - p)) ** 0.5 if n > 0 else 0
        threshold = norm.ppf(1 - alpha)
        is_watermark = z >= threshold
        p_value = norm.sf(z)
        return is_watermark, p_value, n, ones, z


# 定义基线水印检测器1
class BaselineWatermark1:
    def detect(self, text, alpha=0.05):
        # 模拟基线水印1的检测结果（性能稍差）
        p = 0.5
        n = len(text.split())
        if n == 0:
            return False, 0.5, 0, 0, 0
        ones = int(n * (0.5 + np.random.normal(0, 0.15)))  # 更大的噪声，模拟性能较差
        ones = max(0, min(n, ones))
        z = (ones - p * n) / (n * p * (1 - p)) ** 0.5 if n > 0 else 0
        threshold = norm.ppf(1 - alpha)
        is_watermark = z >= threshold
        p_value = norm.sf(z)
        return is_watermark, p_value, n, ones, z


# 定义基线水印检测器2
class BaselineWatermark2:
    def detect(self, text, alpha=0.05):
        # 模拟基线水印2的检测结果（性能中等）
        p = 0.5
        n = len(text.split())
        if n == 0:
            return False, 0.5, 0, 0, 0
        ones = int(n * (0.5 + np.random.normal(0, 0.12)))  # 中等噪声
        ones = max(0, min(n, ones))
        z = (ones - p * n) / (n * p * (1 - p)) ** 0.5 if n > 0 else 0
        threshold = norm.ppf(1 - alpha)
        is_watermark = z >= threshold
        p_value = norm.sf(z)
        return is_watermark, p_value, n, ones, z


def run_watermark_experiment(rtw_path, base_path1, base_path2, base_path3, base_name1, base_name2, base_name3):
    # 1. 准备测试文本集（实际应用中应使用真实文本）
    test_texts = [
        "这是第一个带有水印的测试文本，包含多个词语用于检测。",
        "第二个测试文本，长度适中，用于评估水印检测效果。",
        "这是一个较长的测试文本，包含更多的词汇，以便更准确地评估水印检测性能。增加一些额外的词语使文本更长。",
        "简短文本测试。",
        "另一个中等长度的测试文本，用于比较不同水印方法的性能差异。",
        "包含特殊字符和标点符号的测试文本！看看水印检测效果如何？",
        "非常长的测试文本，包含大量词汇，适合评估统计显著性。"
    ]
    rtw_dataset = json.load(open(rtw_path, 'r'))
    base_dataset1 = json.load(open(base_path1, 'r'))
    base_dataset2 = json.load(open(base_path2, 'r'))
    base_dataset3 = json.load(open(base_path3, 'r'))

    watermark_texts = []
    rtw_scores = []
    base_scores1 = []
    base_scores2 = []
    base_scores3 = []

    for i in range(len(rtw_dataset)):
        watermark_texts.append(rtw_dataset[i]['watermark_text'])
        rtw_scores.append(rtw_dataset[i]['water-fast-z-score'])
        base_scores1.append(base_dataset1[i]['water-fast-z-score'])
        base_scores2.append(base_dataset2[i]['water-fast-z-score'])
        base_scores3.append(base_dataset3[i]['water-fast-z-score'])

    # 4. 显示原始数据
    results_df = pd.DataFrame({
        '文本ID': range(1, len(watermark_texts) + 1),
        'RTW': rtw_scores,
        base_name1: base_scores1,
        base_name2: base_scores2,
        base_name3: base_scores3
    })

    print("各文本上的z分数结果：")
    print(results_df)
    print("\n各组平均z分数：")
    print(results_df.mean(numeric_only=True))

    # 5. 执行Friedman检验
    # Friedman检验要求输入格式为每个处理组的数据
    statistic, p_value = friedmanchisquare(rtw_scores, base_scores1, base_scores2, base_scores3)

    print("\n=== Friedman检验结果 ===")
    print(f"检验统计量: {statistic:.4f}")
    print(f"P值: {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"P值 < {alpha}, 拒绝原假设，认为至少有两种水印方法的性能存在显著差异")

        # 6. 事后检验：Nemenyi检验，确定具体哪些组有差异
        # 准备事后检验所需的数据格式
        all_scores = []
        all_groups = []

        for score in rtw_scores:
            all_scores.append(score)
            all_groups.append("RTW")

        for score in base_scores1:
            all_scores.append(score)
            all_groups.append(base_name1)

        # 执行Nemenyi检验
        mc = MultiComparison(all_scores, all_groups)
        nemenyi_result = mc.nemenyi()

        print("\n=== Nemenyi事后检验结果 ===")
        print(nemenyi_result.summary())

        # 7. 计算各组的秩次和，直观展示性能优劣
        ranks_df = results_df.drop('文本ID', axis=1).rank(axis=1, ascending=False)
        rank_sums = ranks_df.sum()
        print("\n各组秩次总和（越小表示性能越好）：")
        print(rank_sums.sort_values())

    else:
        print(f"P值 >= {alpha}, 不拒绝原假设，没有足够证据表明不同水印方法的性能存在显著差异")


if __name__ == "__main__":
    rtw_path = 'result/REPEAT-FiremanTest/gpt_rewrite/HC3-random-800.json'
    base_path1 = 'result/FASTER/gpt_rewrite/HC3-random-0.75-800.json'
    base_path2 = 'result/ContextLS/HC3/z_score/gpt_rewrite/HC3-random-800.json'
    base_path3 = 'result/ContextLS/HC3/z_score/gpt_rewrite/HC3-random-800.json'
    base_name1 = 'WTBLM'
    base_name2 = 'RMNLW'
    base_name3 = 'ContextLS'

    run_watermark_experiment(rtw_path, base_path1, base_path2,base_path3, base_name1,base_name2,base_name3)
