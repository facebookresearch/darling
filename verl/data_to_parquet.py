import re
import os
import datasets
import sys

prompt_key=sys.argv[2] 
def map_fn(example):
    
    # <|start_header_id|>user<|end_header_id|>\n\n[Format your response using markdown. Use headings, subheadings, bullet points, and bold to organize the information]\nWhat is the differences between Third Way and Third Position?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    # remove the header and footer tokens
    example[f'{prompt_key}'] = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>\n\n', '', example[f'{prompt_key}'])
    example[f'{prompt_key}'] = re.sub(r'<\|eot_id\|>', '', example[f'{prompt_key}'])

    content = example[f'{prompt_key}']

    example['prompt'] = [
        {'role': 'user', 'content': content},
    ]
    

    return example
    

dataset = datasets.load_dataset("json", data_files=sys.argv[1])

dataset = dataset['train']
dataset = dataset.map(map_fn) 

output_dir = sys.argv[1].replace('.jsonl', '.parquet')
dataset.to_parquet(output_dir)
