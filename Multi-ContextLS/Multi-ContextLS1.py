import random

from utils.contextls_utils import synchronicity_test, substitutability_test, tokenizer, riskset, stop

import json

random.seed(1230)

import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
import torch
from torch import nn
import hashlib
from scipy.stats import norm
from transformers import BertForMaskedLM, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer, BertModel, \
    RobertaModel, AutoTokenizer
import gensim.downloader as api
import Levenshtein
import string
import re
from utils.train_watermark_model import TransformModel
from utils import bias
import numpy as np
import gensim


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub('([。！？\?][”’])([^，。！？\?\n ])', r'\1\n\2', para)
    para = re.sub('(\.{6}|\…{2})([^”’\n])', r'\1\n\2', para)
    para = re.sub('([^。！？\?]*)([:：][^。！？\?\n]*)', r'\1\n\2', para)
    para = re.sub('([。！？\?][”’])$', r'\1\n', para)
    para = para.rstrip()
    return para.split("\n")


def is_subword(token: str):
    return token.startswith('##')


def binary_encoding_function(token):
    hash_value = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)
    random_bit = hash_value % 2
    return random_bit


def is_similar(x, y, threshold=0.5):
    distance = Levenshtein.distance(x, y)
    if distance / max(len(x), len(y)) < threshold:
        return True
    return False


class watermark_model:
    def __init__(self, transform_model_path='/root/RTW913/transform_model.pth', tau_word=0.75, lamda=0.83):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tau_word = tau_word
        self.tau_sent = 0.8
        self.lamda = lamda
        self.cn_tag_black_list = set(
            ['', 'x', 'u', 'j', 'k', 'zg', 'y', 'eng', 'uv', 'uj', 'ud', 'nr', 'nrfg', 'nrt', 'nw', 'nz', 'ns', 'nt',
             'm', 'mq', 'r', 'w', 'PER', 'LOC',
             'ORG'])
        self.en_tag_white_list = set(
            ['MD', 'NN', 'NNS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'RB', 'RBR', 'RBS', 'JJ', 'JJR',
             'JJS'])
        self.embedding_model_path = '/root/autodl-tmp/bert-large-uncased'
        self.roberta_large_mnli_path = '/root/autodl-tmp/roberta-large-mnli'
        self.bert_base_cased_path = '/root/autodl-tmp/bert-base-cased'
        self.tokenizer = BertTokenizer.from_pretrained(f'{self.bert_base_cased_path}', resume_download=True)
        self.model = BertForMaskedLM.from_pretrained(f'{self.bert_base_cased_path}', output_hidden_states=True,
                                                     resume_download=True).to(self.device)
        print(f'======bert-base-cased完毕======')
        self.relatedness_model = RobertaForSequenceClassification.from_pretrained(f'{self.roberta_large_mnli_path}',
                                                                                  resume_download=True).to(self.device)
        self.relatedness_tokenizer = RobertaTokenizer.from_pretrained(f'{self.roberta_large_mnli_path}',
                                                                      resume_download=True)
        self.roberta_model = RobertaModel.from_pretrained(f'{self.roberta_large_mnli_path}',
                                                          resume_download=True).to(self.device)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(f'{self.roberta_large_mnli_path}',
                                                                  resume_download=True)
        print(f'======roberta-large-mnli完毕======')
        self.w2v_model = gensim.models.KeyedVectors.load('/root/autodl-tmp/glove/glove-wiki-gigaword-100.gz')
        print(f'======glove完毕======')
        nltk.data.path.append('/root/autodl-tmp/nltk')
        # nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        print(f'======nltk完毕======')
        self.rng = torch.Generator(device=self.device)
        self.hash_key = 15485863
        self.vocab_size = len(self.tokenizer)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path, resume_download=True)
        self.embedding_model = BertModel.from_pretrained(self.embedding_model_path, resume_download=True).to(
            self.device)
        print(f'======bert-large-uncased完毕======')
        transform_model = TransformModel(input_dim=1024, output_dim=300)
        self.transform_path = transform_model_path
        transform_model.load_state_dict(torch.load(transform_model_path, map_location=self.device))
        self.transform_model = transform_model.to(self.device)
        print(f'======transform_model完毕======')
        self.ori_tokens = []

        self.bits = [1, 0, 1, 0]
        self.num_bits = len(self.bits)

    def get_index_space_list(self, cover_text, tokenizer, f=0):
        """
        cover_text: 待处理的单个句子
        """
        encoded_text = tokenizer(cover_text, add_special_tokens=False)  # 编码
        word_ids = encoded_text._encodings[0].word_ids  # 获取句子中每个单词对应的id
        watermarking_wordset = [None] * len(encoded_text['input_ids'])  # 水印词集
        substituted_idset = []  # 被替换的词的id集合
        substituted_indices = []  # 被替换的索引
        watermarked_text = []  # 水印文本
        message = []  # 嵌入的信息

        latest_embed_index = -1
        index = 1

        # 合法位置
        valid_index = []

        while index < len(encoded_text['input_ids']) - f:  # 从第二个词开始处理，直到划定的f为止
            text = tokenizer.decode(encoded_text['input_ids'][index])  # 还原当前处理的词
            watermarked_text.append(text)  #

            if text in riskset:  # 若该词在风险集中，就不处理该词
                watermarking_wordset[index] = 'riskset'
                index = index + 1
                continue

            valid_indx = [t == index for t in word_ids]
            if sum(valid_indx) >= 2:  # skip this word; subword
                watermarking_wordset[index] = 'subword'
                index = index + 1
                continue

            local_context = encoded_text['input_ids'][:index + 2]  # 获取当前词的局部上下文
            is_synch, words = synchronicity_test(index, local_context)  # 做同步性测试

            if not is_synch:  # 不具备同步性，跳过该词
                watermarking_wordset[index] = 'syn'
                index = index + 1
                continue

            if index - latest_embed_index != f + 1:  # 当前词与上一个词之间的距离不超过f，做可替换性测试
                if not substitutability_test(local_context[:-1], index,
                                             words):  # skip this word if substitutability test fails
                    watermarking_wordset[index] = 'sub'  # 没通过可替换性测试，跳过该词
                    index = index + 1
                    continue

            valid_index.append(index)
            index += 1

        return valid_index, words

    def group_id(self, token_id, sent_embedding):
        h = hashlib.sha256(
            (str(token_id) + str(self.hash_key)).encode()
        ).hexdigest()
        return int(h, 16) % self.num_bits

    def get_greenlist_ids_bert(self, text):
        context_embedding = bias.get_embedding(self.device, self.embedding_model, self.embedding_tokenizer, text)
        with torch.no_grad():
            output = self.transform_model(context_embedding).cpu()[0].detach().numpy()
        similarity_array = bias.scale_vector(output)
        similarity_array = np.tile(similarity_array, (len(self.tokenizer) // 300) + 1)[
                           :len(self.tokenizer)]  # Extend to size of vocabulary
        similarity_array = torch.from_numpy(-similarity_array)
        indices = torch.nonzero(similarity_array > 0)
        greenlist_ids = indices.view(-1).tolist()
        return greenlist_ids

    def cut(self, ori_text, text_len):
        tokens = self.tokenizer.tokenize(ori_text)
        if len(tokens) > text_len + 5:
            ori_text = self.tokenizer.convert_tokens_to_string(tokens[:text_len + 5])
        if len(tokens) < text_len - 5:
            return 'Short'
        return ori_text

    def sent_tokenize(self, ori_text):
        return nltk.sent_tokenize(ori_text)

    def pos_filter(self, tokens, masked_token_index, input_text):
        pos_tags = pos_tag(tokens)
        pos = pos_tags[masked_token_index][1]
        if pos not in self.en_tag_white_list:
            return False
        if is_subword(tokens[masked_token_index]) or is_subword(tokens[masked_token_index + 1]) or (
                tokens[masked_token_index] in self.stop_words or tokens[masked_token_index] in string.punctuation):
            return False
        return True

    def filter_special_candidate(self, top_n_tokens, tokens, masked_token_index, input_text):
        filtered_tokens = [tok for tok in top_n_tokens if
                           tok not in self.stop_words and tok not in string.punctuation and pos_tag([tok])[0][
                               1] in self.en_tag_white_list and not is_subword(tok)]
        base_word = tokens[masked_token_index]

        processed_tokens = [tok for tok in filtered_tokens if not is_similar(tok, base_word)]
        return processed_tokens

    def global_word_sim(self, word, ori_word):
        try:
            global_score = self.w2v_model.similarity(word, ori_word)
        except KeyError:
            global_score = 0
        return global_score

    def sentence_sim(self, init_candidates_list, tokens, index_space, input_text):
        batch_size = 128
        all_batch_sentences = []
        all_index_lengths = []
        for init_candidates, masked_token_index in zip(init_candidates_list, index_space):
            batch_sentences = [self.tokenizer.convert_tokens_to_string(
                tokens[1:masked_token_index] + [token] + tokens[masked_token_index + 1:-1]) for token in
                init_candidates]
            all_batch_sentences.extend([input_text + '</s></s>' + s for s in batch_sentences])
            all_index_lengths.append(len(init_candidates))

        all_relatedness_scores = []
        start_index = 0
        for i in range(0, len(all_batch_sentences), batch_size):
            batch_sentences = all_batch_sentences[i: i + batch_size]
            encoded_dict = self.relatedness_tokenizer.batch_encode_plus(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt')

            input_ids = encoded_dict['input_ids'].to(self.device)
            attention_masks = encoded_dict['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.relatedness_model(input_ids=input_ids, attention_mask=attention_masks)
                logits = outputs[0]
            probs = torch.softmax(logits, dim=1)
            relatedness_scores = probs[:, 2]
            all_relatedness_scores.extend(relatedness_scores)

        all_relatedness_scores_split = []
        for length in all_index_lengths:
            all_relatedness_scores_split.append(all_relatedness_scores[start_index:start_index + length])
            start_index += length
        return all_relatedness_scores_split

    # generate candidates
    def candidates_gen(self, tokens, index_space, input_text, topk=64, dropout_prob=0.3):
        input_ids_bert = self.tokenizer.convert_tokens_to_ids(tokens)
        new_index_space = []
        masked_text = self.tokenizer.convert_tokens_to_string(tokens)
        # Create a tensor of input IDs
        input_tensor = torch.tensor([input_ids_bert]).to(self.device)

        with torch.no_grad():
            embeddings = self.model.bert.embeddings(input_tensor.repeat(len(index_space), 1))

        dropout = nn.Dropout2d(p=dropout_prob)

        masked_indices = torch.tensor(index_space).to(self.device)
        embeddings[torch.arange(len(index_space)), masked_indices] = dropout(
            embeddings[torch.arange(len(index_space)), masked_indices])
        with torch.no_grad():
            outputs = self.model(inputs_embeds=embeddings)

        all_processed_tokens = []
        for i, masked_token_index in enumerate(index_space):
            predicted_logits = outputs[0][i][masked_token_index]
            # Set the number of top predictions to return
            n = topk
            # Get the top n predicted tokens and their probabilities
            probs = torch.nn.functional.softmax(predicted_logits, dim=-1)
            top_n_probs, top_n_indices = torch.topk(probs, n)
            top_n_tokens = self.tokenizer.convert_ids_to_tokens(top_n_indices.tolist())
            processed_tokens = self.filter_special_candidate(top_n_tokens, tokens, masked_token_index, input_text)

            if tokens[masked_token_index] not in processed_tokens:
                processed_tokens = [tokens[masked_token_index]] + processed_tokens
            all_processed_tokens.append(processed_tokens)
            new_index_space.append(masked_token_index)

        return all_processed_tokens, new_index_space

    # filter candidates with sentence similarity and word similarity
    def filter_candidates(self, init_candidates_list, tokens, index_space, input_text):

        all_sentence_similarity_scores = self.sentence_sim(init_candidates_list, tokens, index_space, input_text)

        all_filtered_candidates = []
        new_index_space = []

        for init_candidates, sentence_similarity_scores, masked_token_index in zip(init_candidates_list,
                                                                                   all_sentence_similarity_scores,
                                                                                   index_space):
            filtered_candidates = []
            for idx, candidate in enumerate(init_candidates):
                global_word_similarity_score = self.global_word_sim(tokens[masked_token_index], candidate)
                word_similarity_score = self.lamda * sentence_similarity_scores[idx] + (
                        1 - self.lamda) * global_word_similarity_score
                if word_similarity_score >= self.tau_word and sentence_similarity_scores[idx] >= self.tau_sent:
                    filtered_candidates.append((candidate, word_similarity_score))

            if len(filtered_candidates) >= 1:
                all_filtered_candidates.append(filtered_candidates)
                new_index_space.append(masked_token_index)
        return all_filtered_candidates, new_index_space

    def filter_candidates_by_sim(self, init_candidates, tokens, index_space, input_text):
        all_sentence_similarity_scores = self.sentence_sim(init_candidates, tokens, index_space, input_text)
        filter_candidates = []
        for init_candidate, sentence_similarity_score, mask_ids in zip(init_candidates, all_sentence_similarity_scores,
                                                                       index_space):
            for idx, candidate in enumerate(init_candidate):
                global_word_similarity_score = self.global_word_sim(tokens[mask_ids], candidate)
                word_similarity_score = self.lamda * global_word_similarity_score[idx] + (
                            1 - self.lamda) * global_word_similarity_score
                if word_similarity_score >= self.tau_word and sentence_similarity_score >= self.tau_sent:
                    filter_candidates.append(candidate, word_similarity_score)

    # get candidates marked as 1
    def get_candidate_encodings(self, tokens, enhanced_candidates, index_space, greenlist_ids, bits_map):
        best_candidates = []
        new_index_space = []

        for init_candidates, masked_token_index in zip(enhanced_candidates, index_space):
            filtered_candidates = []
            original_token = tokens[masked_token_index]
            target_encodes = bits_map[masked_token_index]

            for idx, candidate in enumerate(init_candidates):
                bit = 1 if self.tokenizer.convert_tokens_to_ids(candidate[0]) in greenlist_ids else 0
                if bit == target_encodes:
                    filtered_candidates.append(candidate)

            # Sort the candidates based on their scores
            filtered_candidates = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)

            if len(filtered_candidates) >= 1:
                best_candidates.append(filtered_candidates[0][0])
                new_index_space.append(masked_token_index)

        return best_candidates, new_index_space

    # 水印嵌入
    def watermark_embed(self, text):
        input_text = text
        tokens = self.tokenizer.tokenize(input_text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        masked_tokens = tokens.copy()
        start_index = 1
        end_index = len(tokens) - 1

        index_space = []
        init_candidates = []

        # 语义map：被替换词的id：embedding
        sent_map = {}
        # 分组map 被替换词的id: group
        group_map = {}

        bits_map = {}

        greenlist_ids = self.get_greenlist_ids_bert(text)

        # sent_emb = bias.get_embedding(self.device, self.embedding_model, self.embedding_tokenizer, text)[0]

        text_embedding = bias.get_embedding(self.device, self.embedding_model, self.embedding_tokenizer, input_text)[0]

        valid_index_space, candidate_list = self.get_index_space_list(text, self.tokenizer)

        if len(valid_index_space) == 0:
            return input_text
        if len(valid_index_space) <= len(self.bits):
            target_bits = self.bits[0: len(valid_index_space)]
        else:
            target_bits = self.bits
            k = (len(valid_index_space) + len(target_bits) - 1) // len(target_bits)  # 向上取整需要重复的次数
            target_bits  =  (target_bits * k)[:len(valid_index_space)]

        for masked_token_index in range(start_index + 1, end_index - 1):
            target_count = 0
            target_encodes = target_bits[target_count]

            binary_encoding = 1 if self.tokenizer.convert_tokens_to_ids(
                tokens[masked_token_index]) in greenlist_ids else 0

            if masked_token_index not in valid_index_space:
                continue
            if binary_encoding == target_encodes:
                target_count += 1
                continue

            index_space.append(masked_token_index)
            init_candidates.append(candidate_list[masked_token_index])
            bits_map[masked_token_index] =  target_encodes

            target_count += 1

        if len(index_space) == 0:
            return text
        # init_candidates, new_index_space = self.candidates_gen(tokens, index_space, input_text, 32, 0)
        # if len(new_index_space) == 0:
        #     return text
        enhanced_candidates, new_index_space = self.filter_candidates(init_candidates, tokens, index_space,
                                                                      input_text)

        enhanced_candidates, new_index_space = self.get_candidate_encodings(tokens, enhanced_candidates,
                                                                            new_index_space, greenlist_ids, bits_map)

        for init_candidate, masked_token_index in zip(enhanced_candidates, new_index_space):
            tokens[masked_token_index] = init_candidate
        watermarked_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])
        return watermarked_text, target_bits

    # 入口方法
    def embed(self, ori_text):
        sents = self.sent_tokenize(ori_text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        watermarked_text = ''

        tokens = self.tokenizer.tokenize(ori_text)
        self.ori_tokens = ['[CLS]'] + tokens + ['[SEP]']

        text_embedding = bias.get_embedding(self.device, self.embedding_model, self.embedding_tokenizer, ori_text)[0]

        for i in range(0, num_sents, 1):
            sent_pair = sents[i]
            if len(watermarked_text) == 0:
                watermarked_text = self.watermark_embed(sent_pair)
            else:
                watermarked_text = watermarked_text + " " + self.watermark_embed(sent_pair)
        if len(self.get_encodings(ori_text)) == 0:
            return ''
        return watermarked_text

    def _seed_rng(self, input_ids):
        seed = self.hash_key * input_ids
        seed = seed % (2 ** 32 - 1)
        self.rng.manual_seed(int(seed))

    def get_encodings(self, text):
        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        encodings = []

        ori_tokens = self.tokenizer.tokenize(text)
        ori_tokens = ['[CLS]'] + ori_tokens + ['[SEP]']

        for i in range(0, num_sents, 1):
            sent_pair = sents[i]
            tokens = self.tokenizer.tokenize(sent_pair)

            greenlist_ids = self.get_greenlist_ids_bert(sent_pair)

            for index in range(1, len(tokens) - 1):
                if not self.pos_filter(tokens, index, text):
                    continue
                if self.tokenizer.convert_tokens_to_ids(tokens[index]) in greenlist_ids:
                    encodings.append(1)
                else:
                    encodings.append(0)
        return encodings

    def get_encodings_one(self, text, groups, all_bits):
        # encoding_groups = groups
        input_text = text
        tokens = self.tokenizer.tokenize(input_text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        masked_tokens = tokens.copy()
        start_index = 1
        end_index = len(tokens) - 1

        index_space = []

        greenlist_ids = self.get_greenlist_ids_bert(text)
        sent_emb = bias.get_embedding(self.device, self.embedding_model, self.embedding_tokenizer, input_text)[0]

        for masked_token_index in range(start_index + 1, end_index - 1):
            tok = tokens[masked_token_index]
            # 分组
            if tok in self.stop_words or tok in string.punctuation:
                continue
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            g = self.group_id(tid, sent_emb)
            # 获取标记
            encode = 1 if tid in greenlist_ids else 0
            # 将标记添加到对应的组
            groups[g].append(encode)
            all_bits.append(encode)

        return groups, all_bits

    def get_encodings_det(self, sent, valid_index_space):
        input_text = sent
        tokens = self.tokenizer.tokenize(input_text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        masked_tokens = tokens.copy()
        start_index = 1
        end_index = len(tokens) - 1

        index_space = []
        all_bits =  []

        greenlist_ids = self.get_greenlist_ids_bert(sent)
        for index in valid_index_space:
            tok = tokens[index]
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            encoding = 1 if tid in greenlist_ids else 0
            all_bits.append(encoding)
        return all_bits


    def detect(self, text, alpha=0.05, return_z=False):

        groups = [[] for _ in range(self.num_bits)]
        all_bits = []  # ← NEW: 用于全局 RTW z

        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        result =  []

        for i in range(0, num_sents, 1):
            sent_pair = sents[i]
            valid_index_space = self.get_index_space_list(sent_pair, self.tokenizer)
            bits = self.get_encodings_det(sent_pair, valid_index_space)
            result.append({
                'watermark-text': sent_pair,
                'water-fast-extracted-message': bits
            })

            # groups, all_bits = self.get_encodings_one(sent_pair, groups, all_bits)

        # bits_hat = []
        # z_scores = []

        # # ---------- bit-wise Z ----------
        # for g in groups:
        #     n = len(g)
        #     if n == 0:
        #         bits_hat.append(0)
        #         z_scores.append(0.0)
        #         continue
        #     ones = sum(g)
        #     z = (ones - 0.5 * n) / np.sqrt(n * 0.25)
        #     bits_hat.append(1 if z > norm.ppf(1 - alpha) else 0)
        #     z_scores.append(z)
        #
        # # ---------- global RTW Z ----------
        # n_all = len(all_bits)
        # if n_all == 0:
        #     z_global = 0.0
        # else:
        #     ones_all = sum(all_bits)
        #     z_global = (ones_all - 0.5 * n_all) / np.sqrt(n_all * 0.25)
        #
        # # 误码率（与预设 self.bits 对比）
        # L = min(len(self.bits), len(bits_hat))
        # errors = sum(1 for i in range(L) if self.bits[i] != bits_hat[i])
        # ber = errors / float(L) if L > 0 else 0.0
        #
        # if return_z:
        #     return bits_hat, z_scores, z_global

        return result
