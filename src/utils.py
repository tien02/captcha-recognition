import torch
from src import config

class Converter:
    vocab = None

    @staticmethod
    def getVocab():
        if Converter.vocab is None:
            Converter.vocab = VoCab(config.VOCAB)
        return Converter.vocab
    

class VoCab:
    def __init__(self, vocab:str):
        self.vocab = vocab
        self.label_to_idx = {}
        self.idx_to_label = {}

        idx = 1
        for char in self.vocab:
            self.label_to_idx[char] = idx
            self.idx_to_label[idx] = char

            idx += 1
        
        self.num_class = len(self.label_to_idx)
    
    def encode(self, text:str):
        return [self.label_to_idx.get(t, 0) for t in text.strip()]
    
    def decode(self, logits):
        return "".join([self.idx_to_label.get(i, "<blank>") for i in logits])
    

def decoding_pred(pred_logits: torch.tensor, target_logits: torch.tensor=None, decoder: VoCab = Converter.getVocab()) -> list[str]:
    '''
    Decode prediction to text
    Return list of predicted text
    '''
    def txt_decode(logits):
        pred = torch.unique_consecutive(logits)
        pred = torch.masked_select(pred, pred.ne(0))
        pred = decoder.decode(pred.detach().cpu().numpy())
        return pred
    
    if target_logits is None:
        pred_txt_lst = []
        
        for pred in pred_logits:
            pred_txt = txt_decode(pred)
            pred_txt_lst.append(pred_txt)

        return pred_txt_lst
    else:
        pred_txt_lst = []
        target_txt_lst = []

        for pred, target in zip(pred_logits, target_logits):
            pred_txt = txt_decode(pred)
            target_txt = decoder.decode(target.detach().cpu().numpy())

            pred_txt_lst.append(pred_txt)
            target_txt_lst.append(target_txt)

        return {
            'prediciton': pred_txt_lst,
            'target': target_txt_lst,
        }