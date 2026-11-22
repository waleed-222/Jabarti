from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cleaning import build_normalizer


# Configuration

CURRENT_WD = Path(__file__).parent
print(f"Current_wd:{CURRENT_WD}")

CLEAN_PATHS = [
    CURRENT_WD / "output" / "tokenizer_corpus.parquet",
]

OUTPUT_PATH = CURRENT_WD/ "tokenizer.json"

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2

SPEPCIAL_TOKENS = [ "[PAD]", "[UNK]", "[BOS]", "[SEP]", "[EOS]",
                   "[AR]", "[EN]","[SYS]", "[USR]", "[ASST]"
                   ]

# Assemble the tokenizer pipeline

def build_tokenizer() -> Tokenizer:
    
    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]",  # unknown token
        )
    )
    
    tokenizer.normalizer = build_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
    tokenizer.decoder =  decoders.Metaspace(
        prepend_scheme="always"
    )
    return tokenizer
    
# Train BPE on the corpus

def corpus_iterator(df, batch_size=1000):
    
    text = df["text"].to_list()
    for i in range(0, len(text), batch_size):
        yield text[i:i + batch_size]
        
        
def train(tokenizer, df):
    
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPEPCIAL_TOKENS,
        show_progress=True
    )
    
    tokenizer.train_from_iterator(corpus_iterator(df), trainer=trainer, length=len(df))
    
    return tokenizer



def main():
    
    print("Step1: Load cleaned corpus")
    frames= []
    for path in CLEAN_PATHS:
        frame = pd.read_parquet(path)
        print(f"Loaded {path.name} :{len(frame)}")
        frames.append(frame)
        
    df = pd.concat(frames, ignore_index=True)
    
    print("Step2: Build tokenizer + Train BPE")
    
    tokenizer = build_tokenizer()
    tokenizer = train(tokenizer, df)
    
    print("sanity check: encoding/decoding")
    examples = [
    "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّةُ عَاصِمَتُهَا الْقَاهِرَةُ",
    "أَحْمَدُ إِبْرَاهِيمُ آدَمُ وُلِدَ فِي مِصْرَ",
    "The United States and the United Nations signed the agreement",
    "machine learning models require large amounts of training data",
    ]
    
    for text in examples:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids)
        print(f"Input: {text}")
        print(f"Encoded: {encoding.tokens}")
        print(f"Decoded: {decoded}")
        print("=" * 40)
        
        tokenizer.save(str(OUTPUT_PATH))
        
if __name__ == "__main__":
    main()