import torch

class IDFEncoder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.idf = {}

    def _calculate_idf(self, documents):
        num_documents = len(documents)
        word_counts = {}
        for doc in documents:
            words = set(doc.split())
            for word in words:
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1

        for word in self.vocab:
            doc_freq = word_counts.get(word, 0)
            self.idf[word] = torch.log(torch.tensor(num_documents / (doc_freq + 1)))

    def encode_text(self, text):
        vector = torch.zeros(len(self.vocab))
        words = text.split()
        word_counts = {}
        for word in words:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

        total_words = len(words)
        for i, word in enumerate(self.vocab):
            tf = word_counts.get(word, 0) / total_words
            vector[i] = tf * self.idf[word]

        return vector