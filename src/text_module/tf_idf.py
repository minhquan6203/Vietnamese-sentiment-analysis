import torch
from torch import nn

class IDFVectorizer(nn.Module):
    def __init__(self, config, vocab, word_count):
        super(IDFVectorizer, self).__init__()
        self.vocab = vocab
        self.word_count = word_count
        self.idf_vector = self.compute_idf_vector()
        self.proj = nn.Linear(len(vocab), config["text_embedding"]["d_model"])
    
    def compute_idf_vector(self):
        idf_vector = torch.zeros(len(self.vocab))
        
        for i, word in enumerate(self.vocab):
            if word in self.word_count:
                idf_value = torch.log(torch.tensor(len(self.word_count) / self.word_count[word]))
                idf_vector[i] = idf_value
        
        return idf_vector
    
    def forward(self, texts):
        idf_vectors = []
        
        for text in texts:
            words = text.lower().split()
            idf_vector = torch.zeros(len(self.vocab))
            
            for word in words:
                if word in self.vocab:
                    word_index = self.vocab.index(word)
                else:
                    word_index = self.vocab.index("unknown")
                
                idf_vector[word_index] = self.idf_vector[word_index]
            
            idf_vectors.append(idf_vector)
        
        idf_vectors = torch.stack(idf_vectors, dim=0)  # Xếp các idf_vector thành một tensor
        idf_vectors = idf_vectors.to(self.proj.weight.device)  # Chuyển đổi sang cùng device với self.proj
        
        return self.proj(idf_vectors).unsqueeze(1)
