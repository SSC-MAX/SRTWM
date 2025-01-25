import os.path
from functools import reduce
from itertools import product
import math
import re
import spacy
import string
import json
import os

import torch

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
# from utils.logging import getLogger
from utils.metric import Metric
from torch import cuda
from utils.dataset_utils import preprocess2sentence
from tqdm import tqdm
from utils.misc import compute_ber
from utils.dataset_utils import get_result_txt

# 禁用并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import spacy
from tqdm.auto import tqdm
from itertools import product
from utils.metric import Metric

def extract(input_text, original_message):
    start_sample_idx = 0

    result_dir = os.path.join(dirname, "watermarked.txt")

    clean_watermarked = get_result_txt(result_dir)
    num_corrupted_sen = 0

    prev_c_idx = 0
    agg_msg = []
    bit_error_agg = {"sentence_err_cnt": 0, "sentence_cnt": 0}
    infill_match_cnt = 0
    infill_match_cnt2 = 0

    num_mask_match_cnt = 0
    zero_cnt = 0
    one_cnt = 0
    kwd_match_cnt = 0
    midx_match_cnt = 0
    mword_match_cnt = 0

    # agg_sentences = ""
    # agg_msg = []

    for idx, (c_idx, sen_idx, sub_idset, sub_idx, clean_wm_text, key, msg) in enumerate(clean_watermarked):
        try:
            wm_texts = [clean_wm_text.strip()]
        except IndexError:
            # logger.debug("Create corrupted samples is less than watermarked. Ending extraction...")
            break
        for wm_text in wm_texts:
            num_corrupted_sen += 1
            sen = spacy_tokenizer(wm_text.strip())
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
            # for sanity check, we use the uncorrupted watermarked texts
            sen_ = spacy_tokenizer(clean_wm_text.strip())
            all_keywords_, entity_keywords_ = model.keyword_module.extract_keyword([sen_])

            # logger.info(f"{c_idx} {sen_idx}")
            # extracting states for corrupted
            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]
            agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                                  train_flag=False, embed_flag=True)
            wm_keys = model.tokenizer(" ".join([t.text for t in mask_word]), add_special_tokens=False)['input_ids']

            # extracting states for uncorrupted
            keyword_ = all_keywords_[0]
            ent_keyword_ = entity_keywords_[0]
            agg_cwi_, agg_probs_, tokenized_pt_, (mask_idx_pt_, mask_idx_, mask_word_) = model.run_iter(sen_, keyword_, ent_keyword_,
                                                                                                        train_flag=False, embed_flag=True)
            kwd_match_flag = set([x.text for x in keyword]) == set([x.text for x in keyword_])
            if kwd_match_flag:
                kwd_match_cnt += 1

            midx_match_flag = set(mask_idx) == set(mask_idx_)
            if midx_match_flag:
                midx_match_cnt += 1

            mword_match_flag = set([m.text for m in mask_word]) == set([m.text for m in mask_word_])
            if mword_match_flag:
                mword_match_cnt += 1

            num_mask_match_flag = len(mask_idx) == len(mask_idx_)
            if num_mask_match_flag:
                num_mask_match_cnt += 1

            infill_match_list = []
            if len(agg_cwi) == len(agg_cwi_):
                for a, b in zip(agg_cwi, agg_cwi_):
                    if len(a) == len(b):
                        infill_match_flag = (a == b).all()
                    else:
                        infill_match_flag = False
                    infill_match_list.append(infill_match_flag)
            else:
                infill_match_list.append(False)
            if all(infill_match_list):
                infill_match_cnt += 1

            valid_watermarks = []
            valid_keys = []
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

                    # checking whether the watermark can be embedded
                    mask_match_flag = len(wm_mask) > 0 and set(wm_mask_idx) == set(mask_idx)
                    if mask_match_flag:
                        valid_watermarks.append(wm_tokenized.text)
                        valid_keys.append(torch.stack(cwi).tolist())

            extracted_msg = []
            if len(valid_keys) > 1:
                try:
                    extracted_msg_decimal = valid_keys.index(wm_keys)
                except:
                    similarity = [len(set(wm_keys).intersection(x)) for x in valid_keys]
                    similar_key = max(zip(valid_keys, similarity), key=lambda x: x[1])[0]
                    extracted_msg_decimal = valid_keys.index(similar_key)

                num_digit = math.ceil(math.log2(len(valid_keys)))
                extracted_msg = format(extracted_msg_decimal, f"0{num_digit}b")
                extracted_msg = list(map(int, extracted_msg))

            one_cnt += sum(msg)
            zero_cnt += len(msg) - sum(msg)

            error_cnt, cnt = compute_ber(msg, extracted_msg)
            bit_error_agg['sentence_err_cnt'] = bit_error_agg.get('sentence_err_cnt', 0) + error_cnt
            bit_error_agg['sentence_cnt'] = bit_error_agg.get('sentence_cnt', 0) + cnt

            # logger.info(f"Corrupted sentence: {corrupted_sen}")
            # logger.info(f"original sentence: {sen}")
            # logger.info(f"Extracted msg: {' '.join(extracted_key)}")
            # logger.info(f"Gt msg: {' '.join(key)} \n")
            # agg_sentences = ""
            # agg_msg = []
        # if bit_error_agg['sentence_cnt']:
        #     logger.info(f"BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']}="
        #                 f"{bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")

    # logger.info(f"Sentence BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']}="
    #             f"{bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")
    # logger.info(f"Infill match rate {infill_match_cnt/num_corrupted_sen:.3f}")
    # # logger.info(f"Zero to One Ratio {zero_cnt / one_cnt:3f}")
    # logger.info(f"mask index match rate: {midx_match_cnt / num_corrupted_sen:.3f}")
    # logger.info(f"mask word match rate: {mword_match_cnt / num_corrupted_sen:.3f}")
    # logger.info(f"kwd match rate: {kwd_match_cnt / num_corrupted_sen:.3f}")
    # logger.info(f"num. mask match rate: {num_mask_match_cnt / num_corrupted_sen:.3f}")

def extract_watermark(model, clean_watermarked, corrupted_watermarked=None, corrupted_flag=False, logger=None):
    """
    Extracts the watermark from the watermarked text.

    Args:
    - model: The InfillModel used for watermark extraction.
    - clean_watermarked: List of tuples containing clean watermarked text information.
    - corrupted_watermarked: List of corrupted watermarked texts (optional).
    - corrupted_flag: Boolean indicating if the watermarked text is corrupted.
    - logger: Logger object for logging information.

    Returns:
    - bit_error_agg: Dictionary containing bit error statistics.
    - extracted_msgs: List of extracted messages.
    """
    if corrupted_flag:
        assert corrupted_watermarked is not None, "corrupted_watermarked must be provided if corrupted_flag is True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = Metric(device)

    num_corrupted_sen = 0
    bit_error_agg = {"sentence_err_cnt": 0, "sentence_cnt": 0}
    infill_match_cnt = 0
    infill_match_cnt2 = 0
    num_mask_match_cnt = 0
    zero_cnt = 0
    one_cnt = 0
    kwd_match_cnt = 0
    midx_match_cnt = 0
    mword_match_cnt = 0

    extracted_msgs = []

    for idx, (c_idx, sen_idx, sub_idset, sub_idx, clean_wm_text, key, msg) in enumerate(clean_watermarked):
        try:
            wm_texts = corrupted_watermarked[idx] if corrupted_flag else [clean_wm_text.strip()]
        except IndexError:
            logger.debug("Create corrupted samples is less than watermarked. Ending extraction...")
            break

        for wm_text in wm_texts:
            num_corrupted_sen += 1
            sen = spacy.load("en_core_web_sm")(wm_text.strip())
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])

            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]
            agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                                  train_flag=False, embed_flag=True)

            valid_watermarks = []
            valid_keys = []
            tokenized_text = [token.text_with_ws for token in sen]

            if len(agg_cwi) > 0:
                for cwi in product(*agg_cwi):
                    wm_text = tokenized_text.copy()
                    for m_idx, c_id in zip(mask_idx, cwi):
                        wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx])

                    wm_tokenized = spacy.load("en_core_web_sm")("".join(wm_text).strip())
                    wm_keywords, wm_ent_keywords = model.keyword_module.extract_keyword([wm_tokenized])
                    wm_kwd = wm_keywords[0]
                    wm_ent_kwd = wm_ent_keywords[0]
                    wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)

                    mask_match_flag = len(wm_mask) > 0 and set(wm_mask_idx) == set(mask_idx)
                    if mask_match_flag:
                        valid_watermarks.append(wm_tokenized.text)
                        valid_keys.append(torch.stack(cwi).tolist())

            extracted_msg = []
            if len(valid_keys) > 1:
                try:
                    extracted_msg_decimal = valid_keys.index(wm_keys)
                except:
                    similarity = [len(set(wm_keys).intersection(x)) for x in valid_keys]
                    similar_key = max(zip(valid_keys, similarity), key=lambda x: x[1])[0]
                    extracted_msg_decimal = valid_keys.index(similar_key)

                num_digit = math.ceil(math.log2(len(valid_keys)))
                extracted_msg = format(extracted_msg_decimal, f"0{num_digit}b")
                extracted_msg = list(map(int, extracted_msg))

            one_cnt += sum(msg)
            zero_cnt += len(msg) - sum(msg)

            error_cnt, cnt = compute_ber(msg, extracted_msg)
            bit_error_agg['sentence_err_cnt'] += error_cnt
            bit_error_agg['sentence_cnt'] += cnt

            extracted_msgs.append(extracted_msg)

    if bit_error_agg['sentence_cnt']:
        logger.info(f"BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']}="
                    f"{bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")

    return bit_error_agg, extracted_msgs

def preprocess_txt(original_text):
    original_text = original_text.replace("\\n", " ")
    original_text = original_text.replace('''"''', " ")
    original_text = original_text.replace("'", " ")
    original_text = original_text.replace("[", " ")
    original_text = original_text.replace("]", " ")
    # original_text = original_text.replace(r"\s+", " ")
    return original_text

def main(text, generic_args, dirname, spacy_tokenizer, device, model):
    
    raw_text = text
    cover_texts = preprocess2sentence([raw_text], corpus_name="custom",
                                    start_sample_idx=0, cutoff_q=(0.0, 1.0), use_cache=False)


    DEBUG_MODE = generic_args.debug_mode
    dtype = "custom"
   
    start_sample_idx = 0
    num_sample = generic_args.num_sample

    
    if "trf" in generic_args.spacy_model:
        spacy.require_gpu()
    

    bit_count = 0
    word_count = 0
    upper_bound = 0
    candidate_kwd_cnt = 0
    kwd_match_cnt = 0
    mask_match_cnt = 0
    sample_cnt = 0
    one_cnt = 0
    zero_cnt = 0

    result_dir = os.path.join(dirname, "watermarked.txt")
    if not os.path.exists(result_dir):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)


    for c_idx, sentences in enumerate(cover_texts):
        corpus_level_watermarks = []
        for s_idx, sen in enumerate(sentences):
            sen = spacy_tokenizer(sen.text.strip())
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
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
                        text2print = [t.text_with_ws for t in wm_tokenized]
                        for m_idx in mask_idx:
                            text2print[m_idx] = f"\033[92m{text2print[m_idx]}\033[00m"
                        valid_watermarks.append(text2print)
                        mask_match_cnt += 1

                    sample_cnt += 1
                    candidate_kwd_cnt += 1

            punct_removed = sen.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])
            if len(valid_watermarks) > 1:
                bit_count += math.log2(len(valid_watermarks))

            if len(valid_watermarks) == 0:
                valid_watermarks = [sen.text]

            corpus_level_watermarks.append(valid_watermarks)

            if candidate_kwd_cnt > 0:
                upper_bound += math.log2(candidate_kwd_cnt)

        # if word_count:
        #     logger.info(f"Bpw : {bit_count / word_count:.3f}")

        num_options = reduce(lambda x, y: x * y, [len(vw) for vw in corpus_level_watermarks])
        available_bit = math.floor(math.log2(num_options))
        # print(f"Input message to embed (max bit:{available_bit}):")
        # message = input().replace(" ", "")
       
        message = '10'
        # # left pad to available bit if given message is short
        if available_bit > 8:
            print(f"Available bit is large: {available_bit} > 8.. "
                f"We recommend using shorter text segments as it may take a while")
        if available_bit < 1:
            return raw_text, message
        message = "0" * (available_bit - len(message)) + message
        print(f'======available:{available_bit}======')
        if len(message) > available_bit:
            print(f"Given message longer than capacity. Truncating...: {len(message)}>{available_bit}")
            message = message[:available_bit]
            # print(f'======message:{message}======')
        print(f'===message:{message}===')
        message_decimal = int(message, 2)
        message_decimal = 1
        # print(f'======message_decimal:{message_decimal}======')
        # breakpoint()
        cnt = 0
        available_candidates = product(*corpus_level_watermarks)
        watermarked_text = next(available_candidates)
        while cnt < message_decimal:
            cnt += 1
            watermarked_text = next(available_candidates)

        watermark_text = ''
        filter_text = []
        for wt in watermarked_text:
            for txt in wt:
                if not txt.startswith(chr(27)) and not txt.startswith(chr(10)):
                    watermark_text += txt
        return preprocess_txt(watermark_text), message


if __name__ == '__main__':
    file_name = 'data/HC3/dataset-random-800.json'
    result_file = 'result/nlp/HC3-random-5.json'
    

    infill_parser = WatermarkArgs()
    infill_parser.add_argument("--custom_keywords", type=str)
    generic_parser = GenericArgs()
    infill_args, _ = infill_parser.parse_known_args()
    infill_args.mask_select_method = "grammar"
    infill_args.mask_order_by = "dep"
    infill_args.exclude_cc = True
    infill_args.topk = 4
    infill_args.dtype = None
    infill_args.model_name = '/root/autodl-tmp/bert-large-cased'
    # you can add your own entity / keyword that should NOT be masked.
    # This list will need to be saved when extracting
    infill_args.custom_keywords = ["watermarking", "watermark"]
    dirname = f"./results/ours/{123}/{456}"
    model = InfillModel(infill_args, dirname=dirname)

    generic_args, _ = generic_parser.parse_known_args()
    spacy_tokenizer = spacy.load(generic_args.spacy_model)

    # select device
    if torch.has_mps:
        device = "mps"
    elif cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    dataset = json.load(open(file_name))
    output = []
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    # text = preprocess_txt(dataset[0])
    # watermark_text, message = main(text, generic_args, dirname, spacy_tokenizer, device, model)
    # print(f'==========\n{watermark_text}\nmessage:{message}\n==========')

    for index in tqdm(range(5)):
        
        original_text = preprocess_txt(dataset[index])
        watermark_text, original_message = main(original_text, generic_args, dirname, spacy_tokenizer, device, model)
        output.append({
            'original_text': original_text,
            'watermark_text': watermark_text,
            'original_message': original_message
        })
        # except BaseException as e:
        #     print(f'出现错误==>{e}, index=>{index}')  # 329 485
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
   
    
        
    print('===水印完成===')