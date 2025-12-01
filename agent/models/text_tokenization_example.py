"""
文本分词和ID转换快速示例
"""
import torch
import torch.nn as nn

# ============ 简单词表示例 ============
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3,
            "i": 4, "love": 5, "ai": 7, "hello": 15, "world": 16,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        """将句子转换为ID列表"""
        tokens = text.lower().split()
        ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        return ids
    
    def decode(self, ids):
        """将ID列表转换回句子"""
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        return " ".join(tokens)

# ============ 使用示例 ============
print("=" * 60)
print("句子 → ID → Embedding 完整流程")
print("=" * 60)

# 1. 创建分词器
tokenizer = SimpleTokenizer()
print(f"\n词表: {tokenizer.vocab}")

# 2. 测试句子
sentence = "I love AI"
print(f"\n原始句子: '{sentence}'")

# 3. 转换为ID
ids = tokenizer.encode(sentence)
print(f"ID序列: {ids}")

# 4. 转换为tensor
input_ids = torch.tensor([ids])  # [1, 3]
print(f"Tensor: {input_ids}, 形状: {input_ids.shape}")

# 5. 创建Embedding层
vocab_size = len(tokenizer.vocab)
embedding_dim = 512
embedding = nn.Embedding(vocab_size, embedding_dim)
print(f"\nEmbedding配置: vocab_size={vocab_size}, dim={embedding_dim}")

# 6. 获取词向量
word_vectors = embedding(input_ids)  # [1, 3, 512]
print(f"词向量形状: {word_vectors.shape}")

# 7. 查看每个词的向量
print(f"\n每个词对应的向量:")
for i, (token, id) in enumerate(zip(sentence.lower().split(), ids)):
    vec = word_vectors[0, i, :]
    print(f"  '{token}' (ID={id}): 向量维度={vec.shape[0]}, "
          f"前5个值={vec[:5].detach().numpy().round(3).tolist()}")

print("\n" + "=" * 60)
print("关键点:")
print("  1. 分词器将句子转换为ID列表")
print("  2. nn.Embedding根据ID查找对应的向量")
print("  3. 每个ID对应词表中的一行向量")
print("  4. 这些向量在训练中会被更新学习")
print("=" * 60)
