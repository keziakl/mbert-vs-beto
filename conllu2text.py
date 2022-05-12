from conllu import parse_incr
from io import open
import sys

def get_token_info(token, tokenlist):
    token_info = {}
    token_info["role"] = None
    token_info["verb_word"] = ""
    token_info["verb_idx"] = -1
    token_info["subject_word"] = ""
    token_info["object_word"] = ""
    if not (token["upostag"] == "NOUN" or token["upostag"] == "PROPN"):
        return token_info

    head_id = token['head']
    head_list = tokenlist.filter(id=head_id)
    head_pos = None
    if len(head_list) > 0:
        head_token = head_list[0]
        if head_token["upostag"] == "VERB":
            head_pos = "verb"
            token_info["verb_word"] = head_token["lemma"]
            token_info["verb_idx"] = int(head_token["id"]) - 1
        elif head_token["upostag"] == "AUX":
            head_pos = "aux"
            token_info["verb_word"] = head_token["lemma"]
            token_info["verb_idx"] = int(head_token["id"]) - 1
        else:
            return token_info

    if "nsubj" in token['deprel']:
        token_info["subject_word"] = token['form']
        has_object = False
        has_expletive_sibling = False
        # 'deps' field is often empty in treebanks, have to look through
        # the whole sentence to find if there is any object of the head
        # verb of this subject (this would determine if it's an A or an S)
        for obj_token in tokenlist:
            if obj_token['head'] == head_id:
                if "obj" in obj_token['deprel']:
                    has_object = True
                    token_info["object_word"] = obj_token["form"]
                if obj_token['deprel'] == "expl":
                    has_expletive_sibling = True
        if has_expletive_sibling:
            token_info["role"] = "S-expletive"
        elif has_object:
            token_info["role"] = "A"
        else:
            token_info["role"] = "S"
        if "pass" in token['deprel']:
            token_info["role"] += "-passive"
    elif "obj" in token['deprel']:
        token_info["role"] = "O"
        token_info["object_word"] = token['form']
        for subj_token in tokenlist:
            if subj_token['head'] == head_id:
                if "subj" in subj_token['deprel']:
                    token_info["subject_word"] = subj_token['form']
    if head_pos == "aux" and token_info["role"] is not None:
        token_info["role"] += "-aux"
    return token_info

#only 2 sentence types may be inputted here: prodrop and haspro
def is_prodrop_sentence(sentence, sentence_type):
    root_idx = -1
    '''
    t = sentence[0]
    for key in t:
        print(key, '->', t[key])
    '''
    verb_indices = []
    #find all verb indices in sentence
    for token in sentence:
        if token['upostag'] == 'VERB':
            verb_indices.append(int(token['id'])-1)
    #find root (verb) indx (and get rid of phrases)
    for token in sentence:
        if token['deprel'] == 'root' and token['upostag'] != 'VERB':
            return False, verb_indices
        #get rid of "haber" sentences
        if token['deprel'] == 'root' and token['lemma'] == 'haber':
            return False,  verb_indices
        if token['deprel'] == 'root' and token['upostag'] == 'VERB':
            root_idx = token["id"]
    #get rid of special passive voice sentences
    for token in sentence:
        if token['deprel'] == 'iobj' and token["head"] == root_idx:
            return False, verb_indices
    #get rid of sentences with noun and pronoun subjects
    if sentence_type=='prodrop':
        for token in sentence:
            if (token['deprel'] == 'nsubj' or token['deprel'] == 'nsubj:pass') and token["head"] == root_idx:
                return False, verb_indices
    elif 'haspro':
        for token in sentence:
            if (token['deprel'] == 'nsubj' or token['deprel'] == 'nsubj:pass') and token["upostag"] == 'PRON' and token["head"] == root_idx:
                return True, verb_indices
    if sentence_type=='prodrop':   
        return True, verb_indices
    #the 'haspro' case
    return False, verb_indices

'''
Runs the parser for the specified sentence type.
'haspro': parsing for sentences with a pronoun subject
'prodrop': parsing for sentences with no subject (phonetically, of any type)
'hassubj': parsing for sentences that have any (pronoun or other noun) subject
'''
def opener(sentence_type, file_name="UD_Spanish-GSD/es_gsd-ud-train.conllu", w2w='datasets/non-prodrop-train.txt'):
    data_file = open(file_name, "r", encoding="utf-8")
    count=0
    prodrop_sentences = []
    sentence_verb_indices = []
    for sentence in parse_incr(data_file):
        is_prodrop, verb_indices = is_prodrop_sentence(sentence, sentence_type) 
        if is_prodrop and sentence_type=='haspro':
            count+=1
            prodrop_sentences.append(sentence.metadata['text'])
            sentence_verb_indices.append(verb_indices)
        if is_prodrop and sentence_type=='prodrop':
            count+=1
            prodrop_sentences.append(sentence.metadata['text'])
            sentence_verb_indices.append(verb_indices)
        if  sentence_type == 'hassubj' and not is_prodrop:
            count+=1
            prodrop_sentences.append(sentence.metadata['text'])
            sentence_verb_indices.append(verb_indices)
    print("num sentences:", count)
    print(prodrop_sentences[16:21])
    with open(w2w, 'w') as f:
        count = 0
        for sent, indices in zip(prodrop_sentences, sentence_verb_indices):
            count+=1
            f.write(sent + "\n")
            for x in indices:
                f.write(str(x) + " ")
            if count != (len(prodrop_sentences)-1):
                f.write("\n")

def main():
    #There are 3 required command line arguments:
    sentence_type = sys.argv[1] #"haspro", "prodrop", "hassubj"
    file_name = sys.argv[2]
    w2w = sys.argv[3]
    opener(sentence_type, file_name, w2w)

if __name__ == "__main__":
    main()