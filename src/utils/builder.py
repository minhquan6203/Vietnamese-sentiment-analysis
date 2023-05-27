from model.svm_model import createSVM_Model
from model.transformer_model import createTrans_Model
from model.transformer_svm_model import createTrans_SVM_Model

def build_model(config, answer_space):
    if config['model']['type_model']=='svm':
        return createSVM_Model(config, answer_space)
    if config['model']['type_model']=='trans':
        return createTrans_Model(config, answer_space)
    if config['model']['type_model']=='trans_svm':
        return createTrans_SVM_Model(config, answer_space)