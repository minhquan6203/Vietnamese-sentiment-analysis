import torch
import torch.nn as nn
from mask.masking import generate_padding_mask
from data_utils.vocab import Vocab
class IDFVectorizer(nn.Module):
    def __init__(self, config):
        super(IDFVectorizer, self).__init__()
        self.vocab = Vocab(config)
        self.all_word, self.word_count = self.vocab.all_word()
        self.idf_vector = self.compute_idf_vector()
        self.proj = nn.Linear(self.vocab.vocab_size(), config["text_embedding"]["d_model"])
    
    def compute_idf_vector(self):
        idf_vector = torch.zeros(self.vocab.vocab_size())
        for i, word in enumerate(self.all_word):
            if word in self.word_count:
                idf_value = torch.log(torch.tensor(len(self.word_count) / self.word_count[word]))
                idf_vector[i] = idf_value
        return idf_vector
    
    def compute_tf_vector(self, input_text):
        tf_vector = torch.zeros(self.vocab.vocab_size())
        total_words = len(input_text.split())
        
        for word in input_text.split():
            tf_vector[self.vocab.word_to_idx.get(word,self.vocab.word_to_idx['[UNK]'])] += 1
        return tf_vector / total_words
    
    def forward(self, input_texts):
        tf_idf_vectors = []
        for input_text in input_texts:
            tf_vector = self.compute_tf_vector(input_text)
            tf_idf_vectors.append(tf_vector*self.idf_vector)
        tf_idf_vectors = torch.stack(tf_idf_vectors, dim=0)
        embedding = self.proj(tf_idf_vectors.to(self.proj.weight.device)).unsqueeze(1)
        padding_mask = generate_padding_mask(embedding, padding_idx=0)
        return embedding, padding_mask



# import torch
# import torch.nn as nn
# from mask.masking import generate_padding_mask
# from data_utils.vocab import Vocab

# import math
# from collections import Counter
# from typing import List, Dict

# class IDFVectorizer(nn.Module):
#     def __init__(self, config: Dict):
#         super(IDFVectorizer, self).__init__()
#         self.max_len=config['tokenizer']['max_length']
#         self.proj = nn.Linear(1, config["text_embedding"]["d_model"])

#     def _compute_term_frequencies(self, documents):
#         term_freqs = []
#         vocabulary = set()
#         for document in documents:
#             term_freq = Counter(document.split())
#             term_freqs.append(term_freq)
#             vocabulary.update(term_freq.keys())
#         return term_freqs,vocabulary

#     def _compute_document_frequencies(self, term_freqs, vocabulary):
#         doc_freqs={}
#         for term in vocabulary:
#             doc_freqs[term] = sum(1 for term_freq in term_freqs if term in term_freq)
#         return doc_freqs

#     def pad_list(self, list: List, max_len: int, value):
#         pad_value_list = [value] * (max_len - len(list))
#         list.extend(pad_value_list)
#         return list

#     def forward(self, documents) -> List[Dict[str, float]]:
#         term_freqs, vocabulary = self._compute_term_frequencies(documents)
#         doc_freqs = self._compute_document_frequencies(term_freqs, vocabulary)

#         tfidf_vectors = []
#         num_documents = len(documents)

#         for term_freq in term_freqs:
#             tfidf_vector = []
#             for term, freq in term_freq.items():
#                 tf = freq / sum(term_freq.values())
#                 idf = math.log(num_documents / (1 + doc_freqs[term]))
#                 tf_idf_term = tf * idf
#                 tfidf_vector.append(tf_idf_term)
#             tfidf_vector=tfidf_vector[:self.max_len]
#             tfidf_vector=self.pad_list(tfidf_vector,self.max_len,0.)
#             tfidf_vectors.append(torch.tensor(tfidf_vector))
#         tfidf_vectors = torch.stack(tfidf_vectors, dim=0).unsqueeze(2)
#         embedding = self.proj(tfidf_vectors.to(self.proj.weight.device))
#         padding_mask = generate_padding_mask(embedding, padding_idx=0)
#         return embedding, padding_mask