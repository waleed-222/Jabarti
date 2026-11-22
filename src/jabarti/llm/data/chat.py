from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

PAD_PLACEHOLDER = 0
DEFAULT_SYSTEM = {
    "ar": "انت مساعد مفيد يجيب عن الاسئلة حول مصر",
    "en": "You are a helpful assistant answering questions about Egypt"
}

ARABIC_LETTERS = range(0x0600, 0x0700)

def system_for(text):
    for char in text:
        if ord(char) in ARABIC_LETTERS:
            return DEFAULT_SYSTEM["ar"]

    return DEFAULT_SYSTEM["en"]


def format_chat(tokenizer, question, answer=None, system=None):
    
    prompt_ids = [tokenizer.BOS]
    
    if system:
        prompt_ids += [tokenizer.SYS] + tokenizer.encode(system)
        
    prompt_ids += [tokenizer.USER] + tokenizer.encode(question)
    
    
    prompt_ids += [tokenizer.ASST]
    
    if answer is None:
        return prompt_ids, None
    
    answer_ids = tokenizer.encode(answer) + [tokenizer.EOS]
    full_prompt = prompt_ids + answer_ids
    
    inputs = full_prompt[: -1]
    targets = full_prompt[1:]
    
    first_graded = len(prompt_ids) - 1
    labels = [PAD_PLACEHOLDER] * first_graded + targets[first_graded:]
    
    return inputs, labels
    
    
class ChataDataset(Dataset):
    
    def __init__(self, questions, answers, tokenizer, config, languages=None, system=None):
        
        self.examples = []
        if languages is None:
            languages  = [None] * len(questions)
            
        for question, answer, language in tqdm(
            zip(questions, answers, languages), total=len(questions),
            desc="tokenizing"
        ):
            if language is None:
                language = "en"
                
            row_system = system if system is not None else DEFAULT_SYSTEM[language]
                
            inputs, labels = format_chat(tokenizer=tokenizer, question=question, answer=answer, system=row_system)
            if len(inputs) > config.max_seq_len:
                continue
            
            self.examples.append((inputs,labels))
    
    
    @classmethod
    
    def from_parquet(cls, path, tokenizer, config, limit=None, **kwargs):
        frame = pd.read_parquet(path,  columns=["question", "answer", "language"])
        if limit is not None:
            frame = frame.head(limit)
            
            
        return cls(
            questions= frame["question"].to_list(),
            answers = frame["answer"].to_list(),
            languages = frame["language"].to_list(),
            tokenizer= tokenizer,
            config= config,
            
            **kwargs
        )
        
    def  __len__(self):
        return len(self.examples)
    
    
    def __getitem__(self, index)  :
        
        inputs, labels =  self.examples[index]
        
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        
def chat_collate_fn(examples, pad_id=0):
    
    longest = max(len(examples["input_ids"])for example in examples)
    
    input_ids = torch.full(
        (len(examples), longest), pad_id, dtype=torch.long
    )
    targets = torch.full(
            (len(examples), longest), pad_id, dtype=torch.long
        )
    
    for ix, example in enumerate(examples):
        
        length = len(example["input_ids"])
        
        input_ids[ix:length] = example["input_ids"]
        targets[ix,:length] = example["labels"]
        
    return {
        "input_ids": input_ids,
        "targets": targets
    }