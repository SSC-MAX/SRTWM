import random
import os
import json
import argparse
from transformers import BertTokenizer

def generate_mapping(size=30000, dimension=300):
    return [random.randint(0, dimension-1) for _ in range(size)]


def main():
   
    tokenizer = BertTokenizer.from_pretrained('Mark/models/bert-large-uncased')
    result_path = 'Mark/SRTWM/utils1/mappings/300-bert-large-uncased.json'

    # parser = argparse.ArgumentParser(description='Generate mappings.')
    # parser.add_argument('--length', type=int, required=True, help='Length of the LLM tokenizer')
    # parser.add_argument('--output_dir', type=str, required=True, help='Output file path')
    # args = parser.parse_args()

    mapping = generate_mapping(len(tokenizer), 300)
    
    output_path = os.path.join(result_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    

if __name__ == '__main__':
    main()
    print('===完成===')