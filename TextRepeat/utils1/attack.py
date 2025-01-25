import textattack
from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapMaskedLM,WordInsertionMaskedLM,WordDeletion
import transformers
import torch
import numpy as np
import random
from tqdm import tqdm
def substitution(sentences,attack_percentage=0.1):
    # 初始化一个使用 WordSwapMaskedLM 的 Augmenter
    transformation = WordSwapMaskedLM(method="bae", masked_language_model=r"D:\model\bert-base-uncased",max_candidates=10,
                    min_confidence=0.3, window_size=None)
    augmenter = Augmenter(transformation=transformation,pct_words_to_swap=attack_percentage,fast_augment=True)

    # 定义你想要攻击的句子
    pbar = tqdm(total=len(sentences))
    output = []
    for sentence in sentences:
        # 执行攻击
        # print(sentence)
        augmented_texts = augmenter.augment(sentence)

    # 打印结果
        for augmented_text in augmented_texts:
            output.append(augmented_text)
            break
        pbar.update(1)
        # print(sentence)


    return output

def insertion(sentences, attack_percentage=0.05):

    # 初始化 WordInsertionMaskedLM，可以指定使用的模型，这里使用默认的 'bert-base-uncased'
    shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained("D:\model\distilbert\distilroberta-base")
    shared_tokenizer = transformers.AutoTokenizer.from_pretrained("D:\model\distilbert\distilroberta-base")
    transformation = WordInsertionMaskedLM(masked_language_model=shared_masked_lm,
                tokenizer=shared_tokenizer,
                max_candidates=50,
                min_confidence=0.0,)

    # 创建 Augmenter 对象，使用上面创建的 transformation
    augmenter = Augmenter(transformation=transformation, transformations_per_example=5,pct_words_to_swap=attack_percentage, fast_augment=True)

    # 定义要增强的文本
    # sentence = "i am a boy"

    pbar = tqdm(total=len(sentences))
    output = []
    for sentence in sentences:
        # 执行攻击
        augmented_texts = augmenter.augment(sentence)

    # 打印结果
        for augmented_text in augmented_texts:
            output.append(augmented_text)
            break
        pbar.update(1)
    return output


def deletion(sentences, attack_percentage=0.05):
    transformation = WordDeletion()

    augmenter = Augmenter(transformation=transformation, transformations_per_example=1,
                          pct_words_to_swap=attack_percentage, fast_augment=True)

    # 定义要增强的文本
    # sentence = "i am a boy"
    pbar = tqdm(total=len(sentences))
    output = []
    for sentence in sentences:
        # 执行攻击
        augmented_texts = augmenter.augment(sentence)

    # 打印结果
        for augmented_text in augmented_texts:
            output.append(augmented_text)
            break
        pbar.update(1)
    return output


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True





if __name__ == "__main__":

    # 设置随机数种子
    setup_seed(1230)
    sentences = ["['There are many different best seller lists that are published by various organizations, and the New York Times is just one of them. The New York Times best seller list is a weekly list that ranks the best-selling books in the United States based on sales train_data from a number of different retailers. The list is published in the New York Times newspaper and is widely considered to be one of the most influential best seller lists in the book industry. \\nIt\\'s important to note that the New York Times best seller list is not the only best seller list out there, and there are many other lists that rank the top-selling books in different categories or in different countries. So it\\'s possible that a book could be a best seller on one list but not on another. \\nAdditionally, the term \"best seller\" is often used more broadly to refer to any book that is selling well, regardless of whether it is on a specific best seller list or not. So it\\'s possible that you may hear about a book being a \"best seller\" even if it is not specifically ranked as a number one best seller on the New York Times list or any other list.']",
                 "[ \" Salt is utilized on roads to assist melt ice and snow and improve traction during the winter months . When it ' s chill outside , water can freeze on the roads and making them very slippery , which can be deadly for cars and people .Salt assists to melt the ice and snow by lowering the freezing point of water , which means that it can help keep the roads clear and safe to travel on . \\ nThere are other possibilities for freezing ice and snow on roads , such as using chemicals like calcium chloride or magnesium chloride , or use hydraulic methods like plows or gravel .However , salt is often the most efficient and affordable option for numerous populations , particularly when it ' s using in combination with other methods . \\ nIt ' s important to note that while salt can be beneficial for make roads safer during the winter , it can also have negative impacts on the environment and on the cars themselves .Salt can cause corrosion on metal surfaces , notably vehicles , and it can also harm crops and animals if it washes into surrounding waterways . However , despite these possible downsides , many communities remain to use salt as a way to keep roads clear and safe during the winter . \" ]",
                 "Salt is used on roads to help melt ice and snow and improve traction during the winter months.",]
    print(substitution(sentences))