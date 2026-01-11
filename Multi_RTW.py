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


def cut_sent(text):
    text = re.sub(r'\n|\t|    ', ' ', text)
    sent_text = nltk.sent_tokenize(text)
    return sent_text


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

        # 默认的 group 比特（向后兼容原实现）
        # 在多比特场景下，这个向量表示每个分组希望达到的目标比特值，
        # 可以在调用 embed() 前通过映射函数由 payload 比特自动生成。
        self.bits = [1, 0, 1, 0]
        self.num_bits = len(self.bits)

        # 多比特 payload 的长度（可以根据需要修改）
        # 这里默认 payload_len 与 num_bits 一致，后续可以通过修改 H 来支持更灵活的映射。
        self.payload_len = self.num_bits

        # 线性映射矩阵 H: group 比特 g -> payload 比特 b = H @ g (mod 2)
        # 先给一个单位阵做占位，表示“每个 group 对应一个 payload 位”。
        # 若要使用更复杂的映射，可自行在外部替换 self.H。
        self.H = np.eye(self.payload_len, self.num_bits, dtype=np.int8)

        # 最近一次检测得到的 payload 比特（便于外部读取）
        self.last_payload_bits = None

    def group_id(self, token_id, sent_embedding):
        h = hashlib.sha256(
            (str(token_id) + str(self.hash_key)).encode()
        ).hexdigest()
        return int(h, 16) % self.num_bits

    def compute_payload_from_group_bits(self, group_bits):
        """
        给定按分组得到的比特向量 group_bits（长度 = self.num_bits），
        通过线性映射矩阵 H 计算 payload 比特：
            b = H @ g (mod 2)
        默认 H 为单位阵，可在外部自定义为更复杂的 {0,1} 矩阵。
        """
        g = np.array(group_bits, dtype=np.int8)
        H = self.H.astype(np.int8)
        # (payload_len, num_bits) @ (num_bits,) -> (payload_len,)
        b = (H @ g) % 2
        return b.tolist()

    def compute_group_bits_from_payload(self, payload_bits):
        """
        给定想要嵌入的 payload 比特（长度 = self.payload_len），
        计算一组目标分组比特 g_target（长度 = self.num_bits）。
        当前默认实现：前 payload_len 个分组直接等于 payload，比特位数不足的填 0。
        如果你在外部把 self.H 改成非单位阵，可以在这里实现 GF(2) 线性求解：
            寻找 g 使得 H @ g = payload_bits (mod 2)。
        """
        # 保证是 0/1 向量
        payload_bits = [int(b) & 1 for b in payload_bits]
        g_target = [0] * self.num_bits
        L = min(len(payload_bits), self.payload_len, self.num_bits)
        for i in range(L):
            g_target[i] = payload_bits[i]
        return g_target

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

    def sent_tokenize(self, text):
        sentences = cut_sent(text)
        # sentences = re.split(r'[。！？\?]', text)
        return sentences

    def en_pos_filter(self, tokens, index):
        tagged_tokens = pos_tag(tokens)
        # print(tagged_tokens)
        selected_tagged_tokens = [tagged_tokens[index]]
        for word, tag in selected_tagged_tokens:
            if tag in self.en_tag_white_list:
                return True
        return False

    # def cn_pos_filter(self, tokens, index):
    #     from pyltp import Segmentor, Postagger
    #     segmentor = Segmentor()
    #     segmentor.load('/root/autodl-tmp/ltp_data_v3.4.0/cws.model')
    #     postagger = Postagger()
    #     postagger.load('/root/autodl-tmp/ltp_data_v3.4.0/pos.model')
    #     pos_tags = list(postagger.postag(tokens))
    #     pos = pos_tags[index]
    #     if pos in self.cn_tag_black_list:
    #         return False
    #     else:
    #         return True

    def pos_filter(self, tokens, index, text):
        # 判断中英文
        text = re.sub(r'\s+', ' ', text)
        word_pattern = re.compile(r'[a-zA-Z]+')
        words = word_pattern.findall(text)
        # 如果有英文字符，使用英文
        if words:
            return self.en_pos_filter(tokens, index)
        else:
            # 如果没有中文字符，使用英文
            return self.en_pos_filter(tokens, index)

    def word_sent_similarity(self, original_sentence, candidate_sentence):
        # Tokenize and encode the original and candidate sentences
        encoded_dict = self.roberta_tokenizer.batch_encode_plus(
            [original_sentence, candidate_sentence],
            add_special_tokens=True,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        # Get the BERT outputs for the original and candidate sentences
        with torch.no_grad():
            outputs = self.roberta_model(**encoded_dict)

        # Extract the CLS token's representation (for sentence embedding)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]

        # Compute cosine similarity between the sentence embeddings
        cosine_similarity = torch.nn.functional.cosine_similarity(
            sentence_embeddings[0].unsqueeze(0),
            sentence_embeddings[1].unsqueeze(0)
        ).item()

        return cosine_similarity

    def candidates_gen(self, tokens, index_space, input_text, batch_size, select_stratege_num=0):
        sent_embedding = bias.get_embedding(self.device, self.embedding_model, self.embedding_tokenizer, input_text)[0]
        original_texts = self.ori_tokens
        coun = {}
        counter_fa = {}
        init_failed_candidates = {}
        init_candidates_list = []

        # Compute the similarity between current text and ori text

        if select_stratege_num == 0:
            original_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])
            # original_text = input_text

        elif select_stratege_num == 1:
            original_text = self.tokenizer.convert_tokens_to_string(original_texts[1:-1])
        original_embedding = bias.get_embedding(
            self.device,
            self.embedding_model,
            self.embedding_tokenizer,
            original_text
        )[0]
        # Using BERT for fill-masking # Generate and rank candidates with NLI-based coherence
        for i in range(len(index_space)):
            masked_index = index_space[i]
            token_length = len(tokens)
            if self.tokenizer.convert_tokens_to_string([tokens[masked_index - 1]]) in ['.', '!', '?', ',', ':',
                                                                                       ';', "'", '"', '-', ')', ']',
                                                                                       '}',
                                                                                       ' ']:
                continue

            if self.tokenizer.convert_tokens_to_string([tokens[masked_index + 1]]) in ['.', '!', '?', ',', ':', ';',
                                                                                       "'", '"', '-', ')', ']', '}',
                                                                                       ' ']:
                continue

            masked_tokens = tokens.copy()
            masked_tokens[masked_index] = self.tokenizer.mask_token
            masked_tokenized_text = self.tokenizer.convert_tokens_to_string(masked_tokens)

            inputs = self.tokenizer(masked_tokenized_text, return_tensors='pt').to(self.device)

            # self._seed_rng(inputs['input_ids'])
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits

            masked_index_modified = masked_index
            # masked_index_modified= masked_index-1+len([self.tokenizer.cls_token])
            topk = torch.topk(predictions[0, masked_index_modified], 50)

            candidates = []
            for idx in range(topk.indices.shape[0]):
                candidate_ids = topk.indices[idx]
                candidate_token = self.tokenizer.convert_ids_to_tokens(candidate_ids.tolist())[0]
                masked_tokens[masked_index] = candidate_token
                candidate_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
                # tokenss = original_text.split(" ")
                # candidate_token_similarity = self.w2v_similarity(original_text, candidate_token, tokenss[masked_index - 1])
                # 用当前被 mask 的 BERT token 还原出一个“原始词”字符串
                original_word_for_w2v = self.tokenizer.convert_tokens_to_string([tokens[masked_index]]).strip(
                    string.punctuation)

                candidate_token_similarity = self.w2v_similarity(
                    original_text,
                    candidate_token,
                    original_word_for_w2v
                )
                candidates.append((candidate_token, candidate_text, candidate_token_similarity))

            init_candidates = sorted(set(candidates), key=lambda x: x[2], reverse=True)
            init_candidates_list.append(init_candidates[:40])

        #  print(initial_beam_subword_candidates)

        return init_candidates_list, index_space

    def filter_candidates(self, init_candidates_list, tokens, index_space, input_text):
        self.w2v_model.fill_norms()
        filtered_candidates_list = []
        new_index_space = []

        for init_candidates, masked_token_index in zip(init_candidates_list, index_space):
            filtered_candidates = []
            original_word = tokens[masked_token_index]
            original_word = original_word.strip(string.punctuation)

            if original_word in self.stop_words or original_word in string.punctuation:
                continue

            candidates = init_candidates[:]

            for candidate, candidate_text, candidate_token_similarity in candidates:
                candidate_word = candidate.strip(string.punctuation)

                if candidate_word in self.stop_words or candidate_word in string.punctuation:
                    continue

                # 词向量相似 / 语义相似
                semantic_similarity = self.word_sent_similarity(input_text, candidate_text)

                # 用 gensim 的 similarity 接口替代 cosine_similarities
                if original_word in self.w2v_model and candidate_word in self.w2v_model:
                    word_similarity = float(self.w2v_model.similarity(original_word, candidate_word))
                else:
                    word_similarity = 0.0

                total_similarity = self.lamda * semantic_similarity + (1 - self.lamda) * word_similarity

                filtered_candidates.append((candidate, total_similarity))

            filtered_candidates = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)
            filtered_candidates_list.append(filtered_candidates[:20])
            new_index_space.append(masked_token_index)

        return filtered_candidates_list, new_index_space

    def w2v_similarity(self, original_sentence, candidate_word, original_word):
        original_tokens = original_sentence.split()
        original_context_words = [
            word for word in original_tokens
            if word not in [original_word, candidate_word]
               and word not in self.stop_words
               and word not in string.punctuation
        ]

        most_frequent_context_word = None
        max_word_frequency = 0
        for word in original_context_words:
            word_frequency = original_tokens.count(word)
            if word_frequency > max_word_frequency:
                max_word_frequency = word_frequency
                most_frequent_context_word = word

        if most_frequent_context_word is None:
            return 0.0

        # 关键改动在这里：直接用 similarity 接口，而不是 cosine_similarities
        if most_frequent_context_word in self.w2v_model and candidate_word in self.w2v_model:
            similarity = float(self.w2v_model.similarity(most_frequent_context_word, candidate_word))
        else:
            similarity = 0.0
        return similarity

    def get_candidate_encodings(self, tokens, enhanced_candidates, index_space, greenlist_ids, sentence_map, group_map):
        best_candidates = []
        new_index_space = []

        for init_candidates, masked_token_index in zip(enhanced_candidates, index_space):
            filtered_candidates = []
            original_token = tokens[masked_token_index]
            if self.tokenizer.convert_tokens_to_ids(original_token) in greenlist_ids:
                target_encodes = 0
            else:
                target_encodes = 1

            for idx, candidate in enumerate(init_candidates):
                bit = 1 if self.tokenizer.convert_tokens_to_ids(candidate[0]) in greenlist_ids else 0
                # 要求候选词的比特正好是目标比特，并且候选词和原词属于同一个分组
                if bit == target_encodes and self.group_id(self.tokenizer.convert_tokens_to_ids(candidate[0]),
                                                           sentence_map[masked_token_index]) == group_map[
                    masked_token_index]:
                    filtered_candidates.append(candidate)

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
        sent_map = {}
        group_map = {}

        greenlist_ids = self.get_greenlist_ids_bert(text)
        text_embedding = bias.get_embedding(self.device, self.embedding_model, self.embedding_tokenizer, input_text)[0]

        for masked_token_index in range(start_index + 1, end_index - 1):
            if not self.pos_filter(tokens, masked_token_index, input_text):
                continue

            if tokens[masked_token_index] in self.stop_words or tokens[masked_token_index] in string.punctuation:
                continue

            token_id = self.tokenizer.convert_tokens_to_ids(tokens[masked_token_index])
            group = self.group_id(token_id, text_embedding)
            binary_encoding = 1 if token_id in greenlist_ids else 0

            # 若所在组对应比特与当前标记一致，则跳过，否则加入候选位置
            target_encodes = self.bits[group]
            if binary_encoding == target_encodes and masked_token_index - 1 not in index_space:
                continue
            if not self.pos_filter(tokens, masked_token_index, input_text):
                continue
            index_space.append(masked_token_index)
            sent_map[masked_token_index] = text_embedding
            group_map[masked_token_index] = group

        if len(index_space) == 0:
            return text
        init_candidates, new_index_space = self.candidates_gen(tokens, index_space, input_text, 32, 0)
        if len(new_index_space) == 0:
            return text
        enhanced_candidates, new_index_space = self.filter_candidates(init_candidates, tokens, new_index_space,
                                                                      input_text)

        enhanced_candidates, new_index_space = self.get_candidate_encodings(tokens, enhanced_candidates,
                                                                            new_index_space, greenlist_ids, sent_map,
                                                                            group_map)

        for init_candidate, masked_token_index in zip(enhanced_candidates, new_index_space):
            tokens[masked_token_index] = init_candidate
        watermarked_text = self.tokenizer.convert_tokens_to_string(tokens[1:-1])
        return watermarked_text

    # 入口方法
    def embed(self, ori_text, payload_bits=None):
        """对整段文本进行水印嵌入。

        参数
        ----
        ori_text : str
            原始待水印文本。
        payload_bits : list[int] or None
            想要嵌入的 payload 比特串（长度建议为 self.payload_len）。
            若为 None，则沿用当前 self.bits 作为目标分组比特（兼容原逻辑）。
        """
        z
        # 如果提供了 payload，比特先通过映射得到目标分组比特向量 g_target
        if payload_bits is not None:
            g_target = self.compute_group_bits_from_payload(payload_bits)
            # 将当前目标 group 比特更新为 g_target
            self.bits = g_target

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
        """
        对单句进行编码：根据 RTW 标记和分组函数，收集每个分组的比特。
        """
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        greenlist_ids = self.get_greenlist_ids_bert(text)
        sent_emb = bias.get_embedding(
            self.device,
            self.embedding_model,
            self.embedding_tokenizer,
            text
        )[0]

        for idx in range(1, len(tokens) - 1):
            tok = tokens[idx]
            if tok in self.stop_words or tok in string.punctuation:
                continue
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            # 分组
            g = self.group_id(tid, sent_emb)
            bit = 1 if tid in greenlist_ids else 0

            groups[g].append(bit)
            all_bits.append(bit)

        return groups, all_bits

    def detect(self, text, alpha=0.05, return_z=False):

        groups = [[] for _ in range(self.num_bits)]
        all_bits = []  # ← NEW: 用于全局 RTW z

        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        for i in range(0, num_sents, 1):
            sent_pair = sents[i]
            groups, all_bits = self.get_encodings_one(sent_pair, groups, all_bits)

        bits_hat = []
        z_scores = []

        # ---------- bit-wise Z ----------
        for g in groups:
            n = len(g)
            if n == 0:
                bits_hat.append(0)
                z_scores.append(0.0)
                continue
            ones = sum(g)
            z = (ones - 0.5 * n) / np.sqrt(n * 0.25)
            bits_hat.append(1 if z > norm.ppf(1 - alpha) else 0)
            z_scores.append(z)

        # ---------- global RTW Z ----------
        n_all = len(all_bits)
        if n_all == 0:
            z_global = 0.0
        else:
            ones_all = sum(all_bits)
            z_global = (ones_all - 0.5 * n_all) / np.sqrt(n_all * 0.25)

        # 误码率（与预设 self.bits 对比）
        L = min(len(self.bits), len(bits_hat))
        errors = sum(1 for i in range(L) if self.bits[i] != bits_hat[i])
        ber = errors / float(L) if L > 0 else 0.0

        # 通过映射矩阵 H 将 group 比特映射为 payload 比特，并缓存下来
        try:
            self.last_payload_bits = self.compute_payload_from_group_bits(bits_hat)
        except Exception:
            # 映射失败时不影响原有检测流程，置为空
            self.last_payload_bits = None

        if return_z:
            return bits_hat, z_scores, z_global

        return bits_hat, z_global, ber