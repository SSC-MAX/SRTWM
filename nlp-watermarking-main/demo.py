import os.path
from functools import reduce
from itertools import product
import math
import re
import spacy
import string
import json

import torch

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
from utils.logging import getLogger
from utils.metric import Metric
from torch import cuda

def preprocess_txt(original_text):
    original_text = original_text.replace("\\n", " ")
    original_text = original_text.replace('''"''', " ")
    original_text = original_text.replace("'", " ")
    original_text = original_text.replace("[", " ")
    original_text = original_text.replace("]", " ")
    # original_text = original_text.replace(r"\s+", " ")
    return original_text

def main():
    infill_parser = WatermarkArgs()
    infill_parser.add_argument("--custom_keywords", type=str)
    generic_parser = GenericArgs()
    infill_args, _ = infill_parser.parse_known_args()
    infill_args.mask_select_method = "grammar"
    infill_args.mask_order_by = "dep"
    infill_args.exclude_cc = True
    infill_args.topk = 2
    infill_args.dtype = None
    infill_args.model_name = '/root/autodl-tmp/bert-large-cased'

    generic_args, _ = generic_parser.parse_known_args()

    from utils.dataset_utils import preprocess2sentence
    dataset = json.load(open('data/HC3/dataset-random-800.json'))
    raw_text = preprocess_txt(dataset[329])
    print(f'===original_text:{raw_text}===')
    # 预处理后的句子，格式为[["..."]]
    cover_texts = preprocess2sentence([raw_text], corpus_name="custom",
                                    start_sample_idx=0, cutoff_q=(0.0, 1.0), use_cache=False)

    # you can add your own entity / keyword that should NOT be masked.
    # This list will need to be saved when extracting
    # 自定义关键词列表，它们不会被选作掩码
    infill_args.custom_keywords = ["watermarking", "watermark"]

    DEBUG_MODE = generic_args.debug_mode
    dtype = "custom"
    dirname = f"./results/ours/{dtype}/{generic_args.exp_name}"
    start_sample_idx = 0
    num_sample = generic_args.num_sample

    spacy_tokenizer = spacy.load(generic_args.spacy_model)
    if "trf" in generic_args.spacy_model:
        spacy.require_gpu()
    model = InfillModel(infill_args, dirname=dirname)

    bit_count = 0
    word_count = 0
    upper_bound = 0
    candidate_kwd_cnt = 0
    kwd_match_cnt = 0    # 候选词替换前后关键词相同，则+1
    mask_match_cnt = 0   # 候选词替换前后掩码位置相同，则+1
    sample_cnt = 0
    one_cnt = 0
    zero_cnt = 0

    # select device
    if torch.has_mps:
        device = "mps"
    elif cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metric = Metric(device, **vars(generic_args))

    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    logger = getLogger("DEMO",
                    dir_=dirname,
                    debug_mode=DEBUG_MODE)


    result_dir = os.path.join(dirname, "watermarked.txt")
    if not os.path.exists(result_dir):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)


    for c_idx, sentences in enumerate(cover_texts):
        corpus_level_watermarks = []   # 文本级别的水印列表，其中每个元素为添加了水印的句子
        # 单独处理每个句子
        # s_idx: 句子索引； sen: 待处理的句子
        for s_idx, sen in enumerate(sentences):
            sen = spacy_tokenizer(sen.text.strip())  # 分词
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])  # 提取关键词
            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]

            # 获取所有掩码位置对应的所有候选词id与掩码索引
            agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                                train_flag=False, embed_flag=True)
            # 检查候选词替换前后掩码位置是否改变，若不变，就视为有效水印，加入valid_watermarks
            valid_watermarks = []
            candidate_kwd_cnt = 0   # 有效的候选词组合数目
            tokenized_text = [token.text_with_ws for token in sen]   # 分词列表中所有单词的文本内容

            if len(agg_cwi) > 0:
                for cwi in product(*agg_cwi):   # 获取所有候选词的排列组合方式， cwi为一维列表，表示该组合下的每个掩码位置的候选词
                    wm_text = tokenized_text.copy()    # 保留一份原始文本
                    for m_idx, c_id in zip(mask_idx, cwi):   # 获取掩码位置与一组候选词的排列组合方式
                        wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx]) # 在该组合中用候选词替换原始单词

                    wm_tokenized = spacy_tokenizer("".join(wm_text).strip()) # 将单词重新组合成一个句子

                    # extract keyword of watermark
                    wm_keywords, wm_ent_keywords = model.keyword_module.extract_keyword([wm_tokenized])
                    wm_kwd = wm_keywords[0]
                    wm_ent_kwd = wm_ent_keywords[0]
                    # 获取掩码位置
                    wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)

                    # 检查候选词替换后的关键词是否与替换前相同，若相同，就计数+1
                    kwd_match_flag = set([x.text for x in wm_kwd]) == set([x.text for x in keyword])
                    if kwd_match_flag:
                        kwd_match_cnt += 1

                    # checking whether the watermark can be embedded without the assumption of corruption
                    # 检查候选词替换前后掩码位置是否相同
                    mask_match_flag = len(wm_mask) and set(wm_mask_idx) == set(mask_idx)
                    if mask_match_flag:
                        text2print = [t.text_with_ws for t in wm_tokenized]
                        for m_idx in mask_idx:
                            text2print[m_idx] = f"\033[92m{text2print[m_idx]}\033[00m"
                        valid_watermarks.append(text2print)   # 若替换前后掩码位置相同，就认为是有效水印文本
                        mask_match_cnt += 1

                    sample_cnt += 1
                    candidate_kwd_cnt += 1    # 有效的候选词组合数目+1

            punct_removed = sen.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))  # 去除待处理句子sent中所有标点
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])  # 计算sent中除了空格与停顿词外还有多少单词
            print(f"***Sentence {s_idx}***")
            if len(valid_watermarks) > 1:
                bit_count += math.log2(len(valid_watermarks))
                for vw in valid_watermarks:
                    print("".join(vw))

            if len(valid_watermarks) == 0:  # 若没有合适的水印文本，就将原始文本作为初始的水印文本
                valid_watermarks = [sen.text]

            corpus_level_watermarks.append(valid_watermarks)   # 添加到文本级别的水印列表

            if candidate_kwd_cnt > 0:
                upper_bound += math.log2(candidate_kwd_cnt)  # 候选关键词的比特容量上限

        if word_count:
            logger.info(f"Bpw : {bit_count / word_count:.3f}")

        num_options = reduce(lambda x, y: x * y, [len(vw) for vw in corpus_level_watermarks])
        available_bit = math.floor(math.log2(num_options))
        print(num_options)
        print(f"Input message to embed (max bit:{available_bit}):")
        message = input().replace(" ", "")
        # left pad to available bit if given message is short
        if available_bit > 8:
            print(f"Available bit is large: {available_bit} > 8.. "
                f"We recommend using shorter text segments as it may take a while")
        message = "0" * (available_bit - len(message)) + message
        if len(message) > available_bit:
            print(f"Given message longer than capacity. Truncating...: {len(message)}>{available_bit}")
            message = message[:available_bit]
        # message_decimal = int(message, 2)
        message_decimal = 1
        # breakpoint()
        cnt = 0
        available_candidates = product(*corpus_level_watermarks)
        watermarked_text = next(available_candidates)
        while cnt < message_decimal:
            cnt += 1
            watermarked_text = next(available_candidates)

        print("---- Watermarked text ----")
        for wt in watermarked_text:
            print("".join(wt))


if __name__ == '__main__':
    main()