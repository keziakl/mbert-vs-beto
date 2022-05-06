import tensorflow as tf
import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
import math
import numpy as np
import random
from torch.nn import functional as F
import sys

def load_dataset(location='datasets/prodrop-train.txt'):
    print('Downloading dataset...')
    dataset = []
    with open(location) as file:
        for line in file:
            dataset.append(line)
    return dataset

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
    for i in range(k):
        score+=logits[i][encodings[i]]
    return score


def main():
    trainset_location = sys.argv[1] 
    k = sys.argv[2]
    n = sys.argv[3]
    device = torch.device("cpu")
    print("loading dataset...")
    sentences = load_dataset(location=trainset_location)
    minutes = n*23/60
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
        if count%10==0:
            minutes-=3.833
            print("Doing iteration number ", count, " estimated remaining minutes: ", minutes)
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