import pandas as pd
from transformers import BertTokenizer, T5Tokenizer

# 加载数据集
csv_path = 'processed_all_data.csv'
df = pd.read_csv(csv_path)
descriptions = df['description'].values

# 初始化分词器
scibert_tokenizer = BertTokenizer.from_pretrained('./scibert_scivocab_uncased')
molt5_tokenizer = T5Tokenizer.from_pretrained('molt5-small-caption2smiles')

# 使用Scibert分词
scibert_tokens = [scibert_tokenizer.tokenize(desc) for desc in descriptions]

# 使用MolT5分词
molt5_tokens = [molt5_tokenizer.tokenize(desc) for desc in descriptions]

# 分析结果
def analyze_tokens(tokens):
    unique_tokens = set()
    for token_list in tokens:
        unique_tokens.update(token_list)
    return len(unique_tokens), unique_tokens

scibert_vocab_size, scibert_vocab = analyze_tokens(scibert_tokens)
molt5_vocab_size, molt5_vocab = analyze_tokens(molt5_tokens)

print(f"SciBERT vocabulary size: {scibert_vocab_size}")
print(f"MolT5 vocabulary size: {molt5_vocab_size}")

# 比较词汇表
common_vocab = scibert_vocab.intersection(molt5_vocab)
print(f"Common vocabulary size: {len(common_vocab)}")