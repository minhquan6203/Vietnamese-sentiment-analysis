from model.svm_model import createSVM_Model,SVM_Model
from model.transformer_model import createTrans_Model,Trans_Model
from model.transformer_svm_model import createTrans_SVM_Model,Trans_SVM_Model
from model.roberta_seq_classify import createRoberta_Model,Roberta_Model
from model.bert_cnn import createTextCNN_Model,TextCNN_Model
from text_module.text_embedding import Text_Embedding
from text_module.count_vectorizer import CountVectorizer
from text_module.tf_idf import IDFVectorizer
from data_utils.vocab import create_vocab

def build_model(config, answer_space):
    if config['model']['type_model']=='svm':
        return createSVM_Model(config, answer_space)
    if config['model']['type_model']=='trans':
        return createTrans_Model(config, answer_space)
    if config['model']['type_model']=='trans_svm':
        return createTrans_SVM_Model(config, answer_space)
    if config['model']['type_model']=='roberta':
        return createRoberta_Model(config, answer_space)
    if config['model']['type_model']=='bert_cnn':
        return createTextCNN_Model(config,answer_space)
    
def get_model(config, num_labels):
    if config['model']['type_model']=='svm':
        return SVM_Model(config, num_labels)
    if config['model']['type_model']=='trans':
        return Trans_Model(config, num_labels)
    if config['model']['type_model']=='trans_svm':
        return Trans_SVM_Model(config, num_labels)
    if config['model']['type_model']=='roberta':
        return Roberta_Model(config, num_labels)
    if config['model']['type_model']=='bert_cnn':
        return TextCNN_Model(config,num_labels)

def build_text_embbeding(config):
    vocab,word_count=create_vocab(config)
    if config['text_embedding']['type']=='pretrained':
        return Text_Embedding(config)
    if config['text_embedding']['type']=='count_vector':
        return CountVectorizer(config,vocab)
    if config['text_embedding']['type']=='tf_idf':
        return IDFVectorizer(config,vocab,word_count)
