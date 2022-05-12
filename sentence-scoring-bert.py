import tensorflow as tf
import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
import math
import numpy as np
import random
from torch.nn import functional as F
import sys
import random

def load_dataset(location='datasets/prodrop-train.txt'):
    print('Downloading dataset...')
    dataset = []
    with open(location) as file:
        for line in file:
            dataset.append(line)
    return dataset

def get_bert_tokens(orig_tokens, tokenizer):
    """
    Given a list of sentences, return a list of those sentences in BERT tokens,
    and a list mapping between the indices of each sentence, where
    bert_tokens_map[i][j] tells us where in the list bert_tokens[i] to find the
    start of the word in sentence_list[i][j]
    The input orig_tokens should be a list of lists, where each element is a word.
    """
    bert_tokens = []
    orig_to_bert_map = []
    bert_to_orig_map = []
    for i, sentence in enumerate(orig_tokens):
        sentence_bert_tokens = []
        sentence_map_otb = []
        sentence_map_bto = []
        sentence_bert_tokens.append("[CLS]")
        for orig_idx, orig_token in enumerate(sentence):
            sentence_map_otb.append(len(sentence_bert_tokens))
            tokenized = tokenizer.tokenize(orig_token)
            for bert_token in tokenized:
                sentence_map_bto.append(orig_idx)
            sentence_bert_tokens.extend(tokenizer.tokenize(orig_token))
        sentence_map_otb.append(len(sentence_bert_tokens))
        sentence_bert_tokens = sentence_bert_tokens[:511]
        sentence_bert_tokens.append("[SEP]")
        bert_tokens.append(sentence_bert_tokens)
        orig_to_bert_map.append(sentence_map_otb)
        bert_to_orig_map.append(sentence_map_bto)
    bert_ids = [tokenizer.convert_tokens_to_ids(b) for b in bert_tokens]
    return bert_tokens, bert_ids, orig_to_bert_map, bert_to_orig_map

def get_score(sentence, tokenizer, model, k):
    #get logits for sentence
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens = True, max_length = 500,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',)
    output = model(**encoded_dict)
    logits = F.softmax(output.logits, dim = -1)
    logits = logits.squeeze().detach().numpy()
    #connect logits to sentence
    encodings = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
    #checks if the sentence is big enough to be counted
    score = 0
    if len(encodings)<=k:
        return score
    indices = random.sample(range(0, len(encodings)-1), k)
    for i in indices:
        score+=logits[i][encodings[i]]
    return score


def main():
    trainset_location = sys.argv[1] 
    k = sys.argv[2]
    n = sys.argv[3]
    device = torch.device("cpu")
    print("loading dataset...")
    sentences = load_dataset(location=trainset_location)
    tokenizer_mbert = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
    tokenizer_beto = BertTokenizer.from_pretrained('pytorch/', do_lower_case=True)

    model_mbert = BertForMaskedLM.from_pretrained("bert-base-multilingual-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,)
    model_mbert.eval()
    model_beto = BertForMaskedLM.from_pretrained("pytorch/",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,)
    model_beto.eval()
    average_score_mbert = 0
    average_score_beto = 0
    count = 0
    print("starting experiment...")
    count = 0
    for sentence in sentences:
        if count>n:
            break
        score_beto = get_score(sentence, tokenizer_beto, model_beto, k)
        average_score_beto+=score_beto
        score_mbert = get_score(sentence, tokenizer_mbert, model_mbert, k)
        average_score_mbert+=score_mbert
        if score_beto!=0:
            count+=1

    print("average beto loss: ", average_score_beto/count)
    print("average mbert loss: ", average_score_mbert/count)

if __name__ == "__main__":
    main()