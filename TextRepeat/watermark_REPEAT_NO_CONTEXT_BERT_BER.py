import os
import sys

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import torch
import torch.nn.functional as F
from torch import nn
import hashlib
from scipy.stats import norm
import gensim
import pdb
from transformers import BertForMaskedLM as WoBertForMaskedLM
# from wobert import WoBertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForMaskedLM, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer, BertModel, \
    RobertaModel, T5Tokenizer, T5Model, AutoTokenizer, AutoModel
import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
import Levenshtein
import string
import spacy
# import paddle
from jieba import posseg

# paddle.enable_static()
import re
from utils1.train_watermark_model import TransformModel
from utils1 import bias

import json
import os
import numpy as np
import random

import subprocess

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

random.seed(1230)


# from sentence_transformers import SentenceTransformer


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
    def __init__(self, tau_word=0.75, lamda=0.83, language='English', mode='embed'):
        # current_directory = os.path.dirname(os.path.abspath(__file__))
        # root_directory = os.path.abspath(os.path.join(current_directory, '..', '..'))
        # sys.path.append(root_directory)
        # print(f'模型路径root:{root_directory}')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.mode = mode
        self.tau_word = tau_word
        self.tau_sent = 0.8
        self.lamda = lamda
        self.cn_tag_black_list = set(
            ['', 'x', 'u', 'j', 'k', 'zg', 'y', 'eng', 'uv', 'uj', 'ud', 'nr', 'nrfg', 'nrt', 'nw', 'nz', 'ns', 'nt',
             'm', 'mq', 'r', 'w', 'PER', 'LOC',
             'ORG'])  #set(['','f','u','nr','nw','nz','m','r','p','c','w','PER','LOC','ORG'])
        self.en_tag_white_list = set(
            ['MD', 'NN', 'NNS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'RB', 'RBR', 'RBS', 'JJ', 'JJR',
             'JJS'])
        self.embedding_model_path = '/root/autodl-tmp/bert-large-uncased'
        self.roberta_large_mnli_path = '/root/autodl-tmp/roberta-large-mnli'
        self.bert_base_cased_path = '/root/autodl-tmp/bert-base-cased'
        self.t5_large_path = '/root/autodl-tmp/-t5-large'
        self.nltk_path = '/root/autodl-tmp/nltk'
        self.glove_path = '/root/autodl-tmp/glove/glove-wiki-gigaword-100.gz'
        if language == 'Chinese':
            self.relatedness_tokenizer = 111
            self.relatedness_model = 111
            self.tokenizer = 111  #WoBertTokenizer.from_pretrained("junnyu/wobert_chinese_plus_base")
            self.model = 111
            self.w2v_model = 111
        elif language == 'English':
            self.tokenizer = BertTokenizer.from_pretrained(f'{self.bert_base_cased_path}', resume_download=True)
            self.model = BertForMaskedLM.from_pretrained(f'{self.bert_base_cased_path}', output_hidden_states=True,
                                                         resume_download=True).to(self.device)
            print('===bert-base-cased 完毕===')
            self.relatedness_model = RobertaForSequenceClassification.from_pretrained(f'{self.roberta_large_mnli_path}',
                                                                                      resume_download=True).to(
                self.device)
            self.relatedness_tokenizer = RobertaTokenizer.from_pretrained(f'{self.roberta_large_mnli_path}',
                                                                          resume_download=True)
            print('===roberta-large-mnli 完毕===')
            self.roberta_model = RobertaModel.from_pretrained(f'{self.roberta_large_mnli_path}',
                                                              resume_download=True).to(self.device)
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained(f'{self.roberta_large_mnli_path}',
                                                                      resume_download=True)
            print('===roberta-large完毕===')
            # self.w2v_model = api.load("glove-wiki-gigaword-100")
            self.w2v_model = gensim.models.KeyedVectors.load(self.glove_path)
            print('===glove 完毕===')
            # nltk.download('stopwords')
            nltk.data.path.append(self.nltk_path)
            self.stop_words = set(stopwords.words('english'))
            print('===nltk完毕===')
            self.rng = torch.Generator(device=self.device)
            self.hash_key = 15485863
            self.vocab_size = len(self.tokenizer)
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path, resume_download=True)
            self.embedding_model = BertModel.from_pretrained(self.embedding_model_path, resume_download=True).to(
                self.device)
            transform_model = TransformModel(input_dim=1024, output_dim=300)
            # self.transform_path = "Mark/SRTWM/models/transform_model-bert_large-DISTANCE-CONDITION.pth"
            self.transform_path = 'models/transform_model-bert_large-DISTANCE-CONDITION.pth'
            transform_model.load_state_dict(torch.load(self.transform_path, map_location=self.device))
            self.transform_model = transform_model.to(self.device)
            print('===transform_model完毕===')
            # mapping_file = "SRTWM/utils1/mappings/300-bert-large-uncased.json"
            # if os.path.exists(mapping_file):
            #     with open(mapping_file, 'r') as f:
            #         self.mapping = json.load(f)
            print('===mapping完毕===')
            self.model_name = 'REPEAT_NO_CONTEXT'
            self.ori_tokens = []
            print(f'==={self.device}===')
            print(f'==={self.model_name}===')
            print(f'==={self.transform_path}===')

    def get_greenlist_ids_bert(self, text):
        context_embedding = bias.get_embedding(self.device, self.embedding_model, self.embedding_tokenizer, text)
        with torch.no_grad():
            output = self.transform_model(context_embedding).cpu()[0].detach().numpy()
        similarity_array = bias.scale_vector(output)
        similarity_array = np.tile(similarity_array, (len(self.tokenizer) // 300) + 1)[:len(self.tokenizer)]  # 扩充到词表大小
        similarity_array = torch.from_numpy(-similarity_array)
        indices = torch.nonzero(similarity_array > 0)  # 获取标记为1的位置
        greenlist_ids = indices.view(-1).tolist()
        return greenlist_ids

    def cut(self, ori_text, text_len):
        if self.language == 'Chinese':
            if len(ori_text) > text_len + 5:
                ori_text = ori_text[:text_len + 5]
            if len(ori_text) < text_len - 5:
                return 'Short'
            return ori_text
        elif self.language == 'English':
            tokens = self.tokenizer.tokenize(ori_text)
            if len(tokens) > text_len + 5:
                ori_text = self.tokenizer.convert_tokens_to_string(tokens[:text_len + 5])
            if len(tokens) < text_len - 5:
                return 'Short'
            return ori_text
        else:
            print(f'Unsupported Language:{self.language}')
            raise NotImplementedError

    def sent_tokenize(self, ori_text):
        if self.language == 'Chinese':
            return cut_sent(ori_text)
        elif self.language == 'English':
            return nltk.sent_tokenize(ori_text)

    def pos_filter(self, tokens, masked_token_index, input_text):
        if self.language == 'Chinese':
            pairs = posseg.lcut(input_text)
            pos_dict = {word: pos for word, pos in pairs}
            pos_list_input = [pos for _, pos in pairs]
            pos = pos_dict.get(tokens[masked_token_index], '')
            if pos in self.cn_tag_black_list:
                return False
            else:
                return True
        elif self.language == 'English':
            pos_tags = pos_tag(tokens)
            pos = pos_tags[masked_token_index][1]
            if pos not in self.en_tag_white_list:
                return False
            if is_subword(tokens[masked_token_index]) or is_subword(tokens[masked_token_index + 1]) or (
                    tokens[masked_token_index] in self.stop_words or tokens[masked_token_index] in string.punctuation):
                return False
            return True

    def filter_special_candidate(self, top_n_tokens, tokens, masked_token_index, input_text):
        if self.language == 'English':
            filtered_tokens = [tok for tok in top_n_tokens if
                               tok not in self.stop_words and tok not in string.punctuation and pos_tag([tok])[0][
                                   1] in self.en_tag_white_list and not is_subword(tok)]

            base_word = tokens[masked_token_index]

            processed_tokens = [tok for tok in filtered_tokens if not is_similar(tok, base_word)]
            return processed_tokens
        elif self.language == 'Chinese':
            pairs = posseg.lcut(input_text)
            pos_dict = {word: pos for word, pos in pairs}
            pos_list_input = [pos for _, pos in pairs]
            pos = pos_dict.get(tokens[masked_token_index], '')
            filtered_tokens = []
            for tok in top_n_tokens:
                watermarked_text_segtest = self.tokenizer.convert_tokens_to_string(
                    tokens[1:masked_token_index] + [tok] + tokens[masked_token_index + 1:-1])
                watermarked_text_segtest = re.sub(
                    r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])',
                    '', watermarked_text_segtest)
                pairs_tok = posseg.lcut(watermarked_text_segtest)
                pos_dict_tok = {word: pos for word, pos in pairs_tok}
                flag = pos_dict_tok.get(tok, '')
                if flag not in self.cn_tag_black_list and flag == pos:
                    filtered_tokens.append(tok)
            processed_tokens = filtered_tokens
            return processed_tokens

    # 使用word2vec计算单词之间的相似度
    def global_word_sim(self, word, ori_word):
        try:
            global_score = self.w2v_model.similarity(word, ori_word)
        except KeyError:
            global_score = 0
        return global_score

    # 使用roberta计算句子的语义嵌入相似度
    def sentence_sim(self, init_candidates_list, tokens, index_space, input_text):
        batch_size = 128
        all_batch_sentences = []
        all_index_lengths = []
        for init_candidates, masked_token_index in zip(init_candidates_list, index_space):
            if self.language == 'Chinese':
                batch_sents = [self.tokenizer.convert_tokens_to_string(
                    tokens[1:masked_token_index] + [token] + tokens[masked_token_index + 1:-1]) for token in
                               init_candidates]
                batch_sentences = [re.sub(
                    r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff，。？！、：])|(?<=[\u4e00-\u9fff，。？！、：])\s+(?=[\u4e00-\u9fff])',
                    '', sent) for sent in batch_sents]
                all_batch_sentences.extend([input_text + '[SEP]' + s for s in batch_sentences])
            elif self.language == 'English':
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
            if self.language == 'Chinese':
                relatedness_scores = probs[:, 1]  #.tolist()
            elif self.language == 'English':
                relatedness_scores = probs[:, 2]  #.tolist()
            all_relatedness_scores.extend(relatedness_scores)

        all_relatedness_scores_split = []
        for length in all_index_lengths:
            all_relatedness_scores_split.append(all_relatedness_scores[start_index:start_index + length])
            start_index += length
        return all_relatedness_scores_split

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

    # 筛选出相似度通过阈值的候选词
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

    # 筛选出编码为1的候选词
    def get_candidate_encodings(self, tokens, enhanced_candidates, index_space, greenlist_ids, embedding_bit):
        '''
        @param enhanced_candidates:(候选词，word_sim)
        '''
        best_candidates = []
        new_index_space = []
        bit_index = 0
        for init_candidates, masked_token_index in zip(enhanced_candidates, index_space):
            filtered_candidates = []

            for idx, candidate in enumerate(init_candidates):
                mask_encoding = self.tokenizer.convert_tokens_to_ids(tokens[masked_token_index]) in greenlist_ids
                if masked_token_index - 1 in new_index_space:
                    # self._seed_rng(self.tokenizer.convert_tokens_to_ids(best_candidates[-1]))
                    # greenlist_size = int(self.vocab_size * 0.5)
                    # vocab_permutation = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
                    # greenlist_ids = vocab_permutation[:greenlist_size]

                    candidate_encoding = self.tokenizer.convert_tokens_to_ids(candidate[0]) in greenlist_ids
                    bit = 1 if mask_encoding != candidate_encoding else 0

                    # bit = binary_encoding_function(best_candidates[-1]+candidate[0])
                else:
                    # self._seed_rng(self.tokenizer.convert_tokens_to_ids(tokens[masked_token_index-1]))
                    # greenlist_size = int(self.vocab_size * 0.5)
                    # vocab_permutation = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
                    # greenlist_ids = vocab_permutation[:greenlist_size]
                    candidate_encoding = self.tokenizer.convert_tokens_to_ids(candidate[0]) in greenlist_ids
                    bit = 1 if mask_encoding != candidate_encoding else 0
                    # bit = binary_encoding_function(tokens[masked_token_index-1]+candidate[0])
                if bit == 1:
                    # print(f'======\nrandom_selected_bit=>{random_selected_bit}\n候选词bit=>{self.tokenizer.convert_tokens_to_ids(candidate[0]) in greenlist_ids}\n======')
                    filtered_candidates.append(candidate)

            # Sort the candidates based on their scores
            filtered_candidates = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)

            if len(filtered_candidates) >= 1:
                best_candidates.append(filtered_candidates[0][0])
                new_index_space.append(masked_token_index)

        return best_candidates, new_index_space

    # 为每一个句子添加水印
    def watermark_embed(self, text, embedding_bit):
        # 无嵌入信息，直接返回原文
        if len(embedding_bit) == 0:
            return text
        
        input_text = text
        # tokens为当前处理的句子的分词
        tokens = self.tokenizer.tokenize(input_text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        masked_tokens = tokens.copy()
        start_index = 1
        end_index = len(tokens) - 1

        index_space = []

        greenlist_ids = self.get_greenlist_ids_bert(text)
        for masked_token_index in range(start_index + 1, end_index - 1):
            
            binary_encoding = 1 if self.tokenizer.convert_tokens_to_ids(
                tokens[masked_token_index]) in greenlist_ids else 0
            if not self.pos_filter(tokens, masked_token_index, input_text):
                continue
            if masked_token_index - 1 not in index_space:
                continue
            index_space.append(masked_token_index)

            
        print(f'===index_space:{len(index_space)}===')
        if len(index_space) == 0:
            return text
        init_candidates, new_index_space = self.candidates_gen(tokens, index_space, input_text, 8, 0)
        if len(new_index_space) == 0:
            return text
        enhanced_candidates, new_index_space = self.filter_candidates(init_candidates, tokens, new_index_space,
                                                                      input_text)
        print(f'===new_index_space:{len(new_index_space)}===')
        enhanced_candidates, new_index_space = self.get_candidate_encodings(tokens, enhanced_candidates,
                                                                            new_index_space, greenlist_ids, embedding_bit)
        print(f'===new_index_space:{len(new_index_space)}===')
        for init_candidate, masked_token_index in zip(enhanced_candidates, new_index_space):
            tokens[masked_token_index] = init_candidate
        watermarked_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])
        
        
        return watermarked_text

    def wm_embed(self, text, embedding_bit):
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        embedding_tokens = []
        greenlist_ids = self.get_greenlist_ids_bert(text)
        for index in range(1, len(tokens)-1):
            if self.pos_filter(tokens, index, text):
                embedding_tokens.append({
                    'token':tokens[index],
                    'index':index})
        embedding_tokens = [embedding_tokens[i:i+len(embedding_bit)] for i in range(0, len(embedding_tokens), len(embedding_bit))]

        # 存储全局索引，其中的token需要反转标记
        index_space = []
        for token_split in embedding_tokens:
            for index in range(0, len(token_split)):
                binary_encoding = 1 if self.tokenizer.convert_tokens_to_ids(token_split[index]['token']) in greenlist_ids else 0 
                if binary_encoding == embedding_bit[index]:
                    continue
                index_space.append(token_split[index]['index'])
        
        if len(index_space) == 0:
            return text
        init_candidates, new_index_space = self.candidates_gen(tokens, index_space, text, 8, 0)
        if len(new_index_space) == 0:
            return text
        enhanced_candidates, new_index_space = self.filter_candidates(init_candidates, tokens, new_index_space,
                                                                      text)

        enhanced_candidates, new_index_space = self.get_candidate_encodings(tokens, enhanced_candidates,
                                                                            new_index_space, greenlist_ids, embedding_bit)

        for init_candidate, masked_token_index in zip(enhanced_candidates, new_index_space):
            tokens[masked_token_index] = init_candidate
        watermarked_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])
        
        
        return watermarked_text




    def _seed_rng(self, input_ids):
        seed = self.hash_key * input_ids
        seed = seed % (2 ** 32 - 1)
        self.rng.manual_seed(int(seed))

    def compute_ber(self, pred, gt):
        # print(f'======\nber\ntype(pred)=>{type(pred)}\npred => {pred}\ntype(gt)=>{type(gt)}\ngt => {gt}\n======')
        error_cnt = 0
        cnt = 0
        if len(pred) != len(gt):
            error_cnt += abs(len(gt) - len(pred))
            cnt += abs(len(gt) - len(pred))
            if len(pred) < len(gt):
                gt = gt[:len(pred)]
            else:
                pred = pred[:len(gt)]

        for p, g in zip(pred, gt):
            if p != g:
                error_cnt += 1
            cnt += 1

        return error_cnt, cnt

    def get_encoding_multi(self, text, bit_length):
        sent_pair = text.strip()
        tokens = self.tokenizer.tokenize(sent_pair)

        greenlist_ids = self.get_greenlist_ids_bert(sent_pair)
        encodings = []
        for index in range(1, len(tokens) - 1):
            if not self.pos_filter(tokens, index, text):
                continue
            if self.tokenizer.convert_tokens_to_ids(tokens[index]) in greenlist_ids:
                encodings.append(1)
            else:
                encodings.append(0)
        return [encodings[i:i+bit_length] for i in range(0, len(encodings), bit_length)], encodings
    
    def watermark_detect_bert_multi(self, text, embedding_bit):
        if len(embedding_bit)==0:
            return 1, []
        extracted_encodings, encodings = self.get_encoding_multi(text, len(embedding_bit))
        errors = 0
        if len(extracted_encodings) == 0:
            return 0, []
        for encoding in extracted_encodings:
            if len(encoding) < len(embedding_bit):
                compare = embedding_bit[:len(encoding)]
                if encoding != compare:
                    errors += 1
                continue
            if encoding != embedding_bit:
                errors += 1
        return errors, extracted_encodings
        

