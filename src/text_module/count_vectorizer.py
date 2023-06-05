import torch

class CountVectorizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        
    def transform(self, documents):
        count_vectors = []
        for document in documents: 
            word_counts = torch.zeros(len(self.vocab))
            for word in document.split():
                if word in self.vocab:
                    word_idx = self.word_to_idx[word]
                    word_counts[word_idx] += 1
            
            count_vectors.append(word_counts)
        