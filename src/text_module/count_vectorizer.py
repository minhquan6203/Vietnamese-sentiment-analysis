import torch
from torch import nn
class CountVectorizer:
    def __init__(self,config ,vocab):
        self.vocab = vocab
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.proj = nn.Linear(len(vocab), config["text_embedding"]['d_model'])
        
    def transform(self, sentence):
        count_vectors = []
        word_counts = torch.zeros(len(self.vocab))
        for word in sentence.split():
            if word in self.vocab:
                word_idx = self.word_to_idx[word]
                word_counts[word_idx] += 1
            else:
                # Xử lý trường hợp từ không xuất hiện trong vocab
                unknown_idx = self.word_to_idx.get("unknown", None)
                if unknown_idx is not None:
                    word_counts[unknown_idx] += 1
        
        count_vectors.append(word_counts)
    
        return self.proj(torch.stack(count_vectors))
