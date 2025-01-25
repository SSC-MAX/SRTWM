import os.path
from itertools import product
from datasets import load_dataset
import math
import random
import re
import spacy
import string
import sys

import torch
from tqdm.auto import tqdm

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
from utils.dataset_utils import get_result_txt
from utils.logging import getLogger
from utils.metric import Metric
from utils.misc import compute_ber

from utils.dataset_utils import preprocess2sentence, preprocess_txt, get_dataset

import json

random.seed(1230)

infill_parser = WatermarkArgs()
generic_parser = GenericArgs()
infill_args, _ = infill_parser.parse_known_args()
generic_args, _ = generic_parser.parse_known_args()
infill_args, _ = infill_parser.parse_known_args()
infill_args.mask_select_method = "grammar"
infill_args.mask_order_by = "dep"
infill_args.exclude_cc = True
infill_args.topk = 2
infill_args.dtype = None
infill_args.model_name = '/root/autodl-tmp/bert-large-cased'
infill_args.custom_keywords = ["watermarking", "watermark"]
generic_args.embed = True
DEBUG_MODE = generic_args.debug_mode   # => False
dtype = generic_args.dtype   # => imdb

dirname = f"./results/ours/{dtype}/{generic_args.exp_name}"   # => .results/ours/imdb/tmp
start_sample_idx = 0
num_sample = generic_args.num_sample   # => 100

spacy_tokenizer = spacy.load(generic_args.spacy_model)
if "trf" in generic_args.spacy_model:
    spacy.require_gpu()
model = InfillModel(infill_args, dirname=dirname)

dataset = json.load(open('data/HC3/dataset-random-800.json'))

cover_texts = preprocess2sentence(preprocess_txt(dataset), corpus_name="custom", start_sample_idx=0, cutoff_q=(0.0, 1.0), use_cache=False)

print(len(cover_texts))
print(cover_texts[0])

num_sentence = 0
for c_idx, sentences in enumerate(cover_texts):
    num_sentence += len(sentences)
    if num_sentence >= num_sample:
        break

cover_texts = cover_texts[:c_idx]

bit_count = 0
word_count = 0
upper_bound = 0
candidate_kwd_cnt = 0
kwd_match_cnt = 0
mask_match_cnt = 0
sample_cnt = 0
one_cnt = 0
zero_cnt = 0

device = torch.device("cuda")
metric = Metric(device, **vars(generic_args))

if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)

if generic_args.embed:   # 嵌入水印模式
    logger = getLogger("EMBED",
                       dir_=dirname,
                       debug_mode=DEBUG_MODE)

    # 创建结果文件
    result_dir = os.path.join(dirname, "watermarked.txt")
    if not os.path.exists(result_dir):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    print('===嵌入水印===')

    progress_bar = tqdm(range(len(cover_texts)))
    wr = open(result_dir, "w")
    for c_idx, sentences in enumerate(cover_texts):
        print(f'======sentences:{sentences}======')
        for s_idx, sen in enumerate(sentences):
            sen = spacy_tokenizer(sen.text.strip())     # 将句子划分成单词
            # all_keywords: 全体关键词； entity_keyword: 实体关键词
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])   # 提取关键词
            logger.info(f"{c_idx} {s_idx}")
            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]
            agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                                  train_flag=False, embed_flag=True)
            # check if keyword & mask_indices matches
            valid_watermarks = []
            candidate_kwd_cnt = 0
            tokenized_text = [token.text_with_ws for token in sen]

            if len(agg_cwi) > 0:
                for cwi in product(*agg_cwi):
                    wm_text = tokenized_text.copy()
                    for m_idx, c_id in zip(mask_idx, cwi):
                        wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx])

                    wm_tokenized = spacy_tokenizer("".join(wm_text).strip())

                    # extract keyword of watermark
                    wm_keywords, wm_ent_keywords = model.keyword_module.extract_keyword([wm_tokenized])
                    wm_kwd = wm_keywords[0]
                    wm_ent_kwd = wm_ent_keywords[0]
                    wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)

                    kwd_match_flag = set([x.text for x in wm_kwd]) == set([x.text for x in keyword])
                    if kwd_match_flag:
                        kwd_match_cnt += 1

                    # checking whether the watermark can be embedded without the assumption of corruption
                    mask_match_flag = len(wm_mask) and set(wm_mask_idx) == set(mask_idx)
                    if mask_match_flag:
                        valid_watermarks.append(wm_tokenized.text)
                        mask_match_cnt += 1

                    sample_cnt += 1
                    candidate_kwd_cnt += 1
    
            punct_removed = sen.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])
            if len(valid_watermarks) > 1:
                bit_count += math.log2(len(valid_watermarks))
                random_msg_decimal = random.choice(range(len(valid_watermarks)))
                num_digit = math.ceil(math.log2(len(valid_watermarks)))
                random_msg_binary = format(random_msg_decimal, f"0{num_digit}b")

                wm_text = valid_watermarks[random_msg_decimal]
                watermarked_text = "".join(wm_text) if len(mask_idx) > 0 else ""
                message_str = list(random_msg_binary)
                one_cnt += len([i for i in message_str if i =="1"])
                zero_cnt += len([i for i in message_str if i =="0"])

                keys = []
                wm_tokenized = spacy_tokenizer(wm_text)
                for m_idx in mask_idx:
                    keys.append(wm_tokenized[m_idx].text)
                keys_str = ", ".join(keys)
                message_str = ' '.join(message_str) if len(message_str) else ""
                wr.write(f"{c_idx}\t{s_idx}\t \t \t"
                         f"{''.join(wm_text)}\t{keys_str}\t{message_str}\n")
            else:
                original_text = ''.join(tokenized_text)
                wr.write(f"{c_idx}\t{s_idx}\t \t \t"
                         f"{original_text}\t \t \n")

            if candidate_kwd_cnt > 0:
                upper_bound += math.log2(candidate_kwd_cnt)
        progress_bar.update(1)
        if word_count:
            logger.info(f"Bpw : {bit_count / word_count:.3f}")

    wr.close()
    logger.info(infill_args)
    logger.info(f"UB Bpw : {upper_bound / word_count:.3f}")
    logger.info(f"Bpw : {bit_count / word_count:.3f}")
    assert sample_cnt > 0, f"No candidate watermarked sets were created"
    logger.info(f"mask match rate: {mask_match_cnt / sample_cnt:.3f}")
    logger.info(f"kwd match rate: {kwd_match_cnt / sample_cnt :.3f}")
    logger.info(f"zero/one ratio: {zero_cnt / one_cnt :.3f}")
    logger.info(f"calls to LM: {model.call_to_lm}")

    ss_scores = []
    for model_name in ["roberta", "all-MiniLM-L6-v2"]:
        ss_score, ss_dist = metric.compute_ss(result_dir, model_name, cover_texts)
        ss_scores.append(sum(ss_score) / len(ss_score))
        logger.info(f"ss score: {sum(ss_score) / len(ss_score):.3f}")
    nli_score, _, _ = metric.compute_nli(result_dir, cover_texts)
    logger.info(f"nli score: {sum(nli_score) / len(nli_score):.3f}")

    result_dir = os.path.join(dirname, "embed-metrics.txt")
    with open(result_dir, "a") as wr:
        wr.write(str(vars(infill_args))+"\n")
        wr.write(f"num.sample={num_sample}\t bpw={bit_count / word_count}\t "
                 f"ss={ss_scores[0]}\t ss={ss_scores[1]}\t"
                 f"nli={sum(nli_score) / len(nli_score)}\n")
