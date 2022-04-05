import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import pandas as pd

project_path = ''

import pickle
with open(project_path+'sorted_graph_new_2.pickle', 'rb') as handle:
    sorted_graph = pickle.load(handle)

"""

SOURCE: https://github.com/commonsense/conceptnet-numberbatch/blob/master/text_to_uri.py

"""
import wordfreq
import re


# English-specific stopword handling
STOPWORDS = ['the', 'a', 'an']
DROP_FIRST = ['to']
DOUBLE_DIGIT_RE = re.compile(r'[0-9][0-9]')
DIGIT_RE = re.compile(r'[0-9]')


def standardized_uri(term, language='en'):

    if not (term.startswith('/') and term.count('/') >= 2):
        term = _standardized_concept_uri(language, term)
    return replace_numbers(term)


def english_filter(tokens):

    non_stopwords = [token for token in tokens if token not in STOPWORDS]
    while non_stopwords and non_stopwords[0] in DROP_FIRST:
        non_stopwords = non_stopwords[1:]
    if non_stopwords:
        return non_stopwords
    else:
        return tokens


def replace_numbers(s):

    if DOUBLE_DIGIT_RE.search(s):
        return DIGIT_RE.sub('#', s)
    else:
        return s


def _standardized_concept_uri(language, term):
    if language == 'en':
        token_filter = english_filter
    else:
        token_filter = None
    language = language.lower()
    norm_text = _standardized_text(term, token_filter)
    #return '/c/{}/{}'.format(language, norm_text)
    return norm_text


def _standardized_text(text, token_filter):
    tokens = simple_tokenize(text.replace('_', ' '))
    if token_filter is not None:
        tokens = token_filter(tokens)
    return '_'.join(tokens)


def simple_tokenize(text):

    return wordfreq.tokenize(text, 'xx')


import spacy
nlp = spacy.load('en_core_web_sm')

def get_tokens_from_text(text,graph):
  #doc1 = nlp("Robert Downey Jr. is looking at buying a huge U.K. startup is for $1 billion. what's the purpose?")
  tokens = []

  temp_text = standardized_uri(text).lower()
  if(temp_text in graph.keys()):
      tokens.append(temp_text)
      return tokens

  doc1 = nlp(text)

  skip = -1
  
  # Token and Tag
  for i,token in enumerate(doc1):
    if(i<=skip):
      continue
    #token = doc1[i]
    #print(doc1[i])
    #token = token.lower()
    #print(token, token.tag_)
    if(token.tag_.startswith('NN')):
      isDone = False
      temp_i = i+1
      temp_token = token
      max_token = token
      while(temp_i<len(doc1) and doc1[temp_i].tag_.startswith('NN')):
        temp_token = standardized_uri(str(temp_token)+' '+str(doc1[temp_i]))
        #print(temp_token)
        if(temp_token.lower() in graph.keys()):
          #tokens.append(temp_token.lower())
          skip = temp_i
          max_token = temp_token
          isDone = True
          temp_i+=1
        else:
          break
      
      if(isDone):
        tokens.append(max_token.lower())
        continue


      if(str(token).lower() in graph.keys()):
        tokens.append(str(token).lower())
      elif(token.tag_.endswith('S')):
        singular = token.lemma_
        if(str(singular).lower() in graph.keys()):
          tokens.append(str(singular).lower())

    if(token.tag_.startswith('JJ')):
      if(str(token).lower() in graph.keys()):
        tokens.append(str(token).lower())
    
    if((token.tag_.startswith('VB')) and (not token.pos_.startswith('AUX'))):
      #token = token.lemma_
      isDone = False
      if(i+1<len(doc1) and doc1[i+1].tag_ == 'IN'):
        temp_token = standardized_uri(str(token)+' '+str(doc1[i+1]))
        #print(temp_token)
        if(temp_token.lower() in graph.keys()):
          tokens.append(temp_token.lower())
          skip = i+1
          isDone = True
      
      elif(i+1<len(doc1) and doc1[i+1].tag_.startswith('NN')):
        temp_token = standardized_uri(str(token)+' '+str(doc1[i+1]))
        #print(temp_token)
        if(temp_token.lower() in graph.keys()):
          tokens.append(temp_token.lower())
          skip = i+1
          isDone = True
      
      elif(i+1<len(doc1) and doc1[i+1].tag_.startswith('R')):
        temp_token = standardized_uri(str(token)+' '+str(doc1[i+1]))
        #print(temp_token)
        if(temp_token.lower() in graph.keys()):
          tokens.append(temp_token.lower())
          skip = i+1
          isDone = True

      
      if((not isDone) and (str(token).lower() in graph.keys())):
        tokens.append(str(token).lower())

      elif(len(token.tag_)>2):
        base = token.lemma_
        if(str(base).lower() in graph.keys()):
          tokens.append(str(base).lower())

    #print(token, token.tag_)
  #print(tokens)
  return tokens

relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas/formof',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/hasa',
    'relatedto/derivedfrom/similarto/synonym/etymologicallyderivedfrom/etymologicallyrelatedto',
    'usedfor',
    'receivesaction',
]


embeddings_index = dict()

f = open('data/numberbatch/numberbatch-en.txt')

for line in f:
    values = line.split()
    word = values[0].split('/')[-1]
    coefs = np.array(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))



from transformers import BertForMaskedLM, RobertaForSequenceClassification, GPT2Tokenizer, RobertaForMaskedLM, RobertaTokenizer, GPT2ForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("RobertaTokenizer")

class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    #
    def __init__(self, config):
        super().__init__(config)
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs


lm_model = RobertaForMaskedLMwithLoss.from_pretrained("RobertaForMaskedLM").to(device)




import numpy as np

def process_one_example(one_story, full_story, sorted_graph, lm_model, relation_groups, embeddings_index, tokenizer, example_id=-1):

  temp_list = str(one_story).split('<SEP>')
  assert(len(temp_list)==4) ######################################
  one_story_list = []
  for temp in temp_list:
    if(len(temp)>0):
      one_story_list.append(temp.strip().lower())
  #print(one_story_list)
  tokens_one_story = []

  full_text = str(full_story)

  for sent in one_story_list:
    context_nodes = list(set(get_tokens_from_text(sent,sorted_graph)))
    tokens_one_story.append(context_nodes)

  #print(tokens_one_story)

  rel_list = {}
  all_nodes = []
  main_nodes = []
  extra_nodes = []

  for j in range(1, len(tokens_one_story)):
    for t1 in tokens_one_story[0]:
      for t2 in tokens_one_story[j]:
        if(t1==t2):
          continue
        if(str(t1)+'_x_'+str(t2) in rel_list.keys()):
          continue
        t1_extra_nodes = set(sorted_graph[t1].keys())
        t2_extra_nodes = set(sorted_graph[t2].keys())

        t12_extra_nodes = list(t1_extra_nodes & t2_extra_nodes)
        rel_list[str(t1)+'_x_'+str(t2)] = t12_extra_nodes
        main_nodes.append(t1)
        main_nodes.append(t2)
        all_nodes.append(t1)
        all_nodes.append(t2)
        all_nodes+= t12_extra_nodes
        extra_nodes+= t12_extra_nodes


  main_nodes = list(set(main_nodes))
  all_nodes = list(set(all_nodes))
  extra_nodes = list(set(extra_nodes))
  #print(rel_list)
  #print(len(all_nodes))
  ##################################################

 ''' temp_sents = []
  _scores = []
  for t in all_nodes:
    ss = '{} {}'.format(full_text, ' '.join(t.split('_')))
    #print(ss)
    #temp_sents.append(ss)

    #print(datasets['train'][0]['choice1'])
    #print(datasets['train'][0]['choice2'])

    encoded = tokenizer(ss,padding=True, truncation=True, return_tensors='pt', add_special_tokens=True).to(device)
    #print(tokenizer.batch_decode(encoded['input_ids'],skip_special_tokens=False))

    
    lm_model.eval()
    with torch.no_grad():
      outputs = lm_model(encoded['input_ids'], encoded['attention_mask'], masked_lm_labels=encoded['input_ids'])
      loss = outputs[0] #[B, ]
      #_scores = list(loss.detach().cpu().numpy())
      _scores.append(loss.detach().cpu().numpy())

  node_scores = []
  for t,s in zip(all_nodes,_scores):
    node_scores.append([t,s[0]])

  sorted_node_scores = sorted(node_scores, key=lambda x: float(x[1]))
  #print(sorted_node_scores)

  #############################################################################

  top_k = min(100, len(sorted_node_scores))
  top_nodes = []
  for i in range(top_k):
    top_nodes.append(sorted_node_scores[i][0])'''


  '''if(len(all_nodes)>70):
    print('GREATER THAN 70')'''
  top_nodes = all_nodes[:min(70,len(all_nodes))] #############################

  concept_nodes = []
  rel_edges = {}
  for m1 in main_nodes:
    if(m1 not in top_nodes):
      continue
    for m2 in main_nodes:
      if((m1==m2) or (m2 not in top_nodes)):
        continue
      keyword = str(m1)+'_x_'+str(m2)

      if(m2 in sorted_graph[m1].keys()):
        if(keyword not in rel_edges.keys()):
          rel_edges[keyword] = []
        rel_edges[keyword].append(['direct_rel', sorted_graph[m1][m2][0], sorted_graph[m2][m1][0]])
        concept_nodes.append(m1)
        concept_nodes.append(m2)

      if(keyword not in rel_list.keys()):
        continue
      for interim in rel_list[keyword]:
        if(interim not in top_nodes):
          continue
        if(keyword not in rel_edges.keys()):
          rel_edges[keyword] = []

        if((sorted_graph[m1][interim][0].lower().startswith('externalurl')) or 
           (sorted_graph[interim][m2][0].lower().startswith('externalurl'))):
          continue

        rel_edges[keyword].append([interim, sorted_graph[m1][interim][0], sorted_graph[interim][m2][0]])
        concept_nodes.append(m1)
        concept_nodes.append(m2)
        concept_nodes.append(interim)

  #print(rel_edges)
  ###############################################################################
  #print(concept_nodes)
  concept_set = list(set(concept_nodes))


  concept2id = {k:i for i,k in enumerate(concept_set,len(tokens_one_story))}
  #print(concept2id)
  id2concept = {v:k for k,v in concept2id.items()}


  concept_edges = []
  concept_edge_types = []

  for sent_j in range(1, len(tokens_one_story),1):
    concept_edges.append([0, sent_j])
    concept_edge_types.append(0)

  for sent_i, tokens in enumerate(tokens_one_story):
    for concept in concept_set:
      if(concept in tokens):
        concept_edges.append([sent_i, concept2id[concept]])
        concept_edge_types.append(1)

  #print(sent2id)
  #################################################################################
  merged_relations = {}
  rel2id = {}
  id2rel = {}
  ID = 1 ############################################
  for r in relation_groups:
    ID+=1

    rel_name = r.split('/')[0]
    rels = r.split('/')
    for rel in rels:
      merged_relations[rel] = rel_name
    rel2id[rel_name] = ID
    id2rel[ID] = rel_name
  rel2id['unk'] = ID+1
  id2rel[ID+1] = 'unk'

  #print(merged_relations)
  #print(rel2id)
  #################################################################
  #concept_edges = []
  #concept_edge_types = []

  for key in rel_edges.keys():
    t1,t2 = key.split('_x_')
    for interim_list in rel_edges[key]:
      interim, r1, r2 = interim_list
      rel1 = r1.lower().split('_inv')[0]
      rel2 = r2.lower().split('_inv')[0]
      if(interim=='direct_rel'):
        concept_edges.append([concept2id[t1], concept2id[t2]])
        try:
          concept_edge_types.append(rel2id[merged_relations[rel1]])
        except:
          concept_edge_types.append(rel2id['unk'])

      else:
        concept_edges.append([concept2id[t1], concept2id[interim]])
        try:
          concept_edge_types.append(rel2id[merged_relations[rel1]])
        except:
          concept_edge_types.append(rel2id['unk'])
        concept_edges.append([concept2id[interim], concept2id[t2]])
        try:
          concept_edge_types.append(rel2id[merged_relations[rel2]])
        except:
          concept_edge_types.append(rel2id['unk'])

  #print(concept_edges)
  #print(concept_edge_type)
  ###############################################################################
  term_X = []
  for ind in id2concept.keys():
    try:
      temp = embeddings_index[id2concept[ind]]
      temp2 = np.concatenate((temp,np.random.rand(768-300)),axis=0)
      #print(temp2.shape)
      term_X.append(temp2)

    except:
      #print('missing ', id2concept[ind])
      term_X.append(np.random.rand(768))
    
  term_X = np.array(term_X, dtype='float32')
  #print(term_X.shape)
    
  #######################################################################
  #######################################################################
  encoded = tokenizer(one_story_list, max_length = 100, padding='max_length', truncation=True, return_tensors='pt',)
  encoded_full = tokenizer(full_story, max_length = 512, padding='max_length', truncation=True, return_tensors='pt',)
  #print(encoded)

  example = {}
  example['input_ids'] = encoded['input_ids']
  #example['token_type_ids'] = encoded['token_type_ids']
  example['attention_mask'] = encoded['attention_mask']
  example['full_input_ids'] = encoded_full['input_ids']
  #example['full_token_type_ids'] = encoded_full['token_type_ids']
  example['full_attention_mask'] = encoded_full['attention_mask']
  example['term_X'] = term_X
  example['concept_edges'] = concept_edges
  example['concept_edge_types'] = concept_edge_types
  example['concept2id'] = concept2id
  example['id2concept'] = id2concept
  example['rel2id'] = rel2id
  example['id2rel'] = id2rel

  return example


df_train = pd.read_csv('data/Datasets/Texts/all_dataset_Machine_Sep_Train_1.csv')
df_val = pd.read_csv('data/Datasets/Texts/all_dataset_Machine_Sep_Val_1.csv')
df_test = pd.read_csv('data/Datasets/Texts/all_dataset_Machine_Sep_Test_1.csv')


import pickle
from tqdm import tqdm


stories = df_train['text_sep'].tolist()
full_stories = df_train['text'].tolist()


all_example = {}
all_example['input_ids'] = []
#all_example['token_type_ids'] = []
all_example['attention_mask'] = []
all_example['full_input_ids'] = []
#all_example['token_type_ids'] = []
all_example['full_attention_mask'] = []
all_example['term_X'] = []
all_example['concept_edges'] = []
all_example['concept_edge_types'] = []
all_example['concept2id'] = []
all_example['id2concept'] = []
all_example['rel2id'] = []
all_example['id2rel'] = []

ind = 0
for one_story, full_story in tqdm(zip(stories, full_stories)):

  example = process_one_example(one_story, full_story, sorted_graph, lm_model, relation_groups, embeddings_index, tokenizer)

  ind += 1
  for k,v in example.items():
    all_example[k].append(v)



with open('Pickles/Dataset_Graph/Relevant/all_examples_Roberta_with_Full_Machine_Train_1.pickle','wb') as f:
  pickle.dump(all_example,f)



import pickle
from tqdm import tqdm


stories = df_val['text_sep'].tolist()
full_stories = df_val['text'].tolist()


all_example = {}
all_example['input_ids'] = []
#all_example['token_type_ids'] = []
all_example['attention_mask'] = []
all_example['full_input_ids'] = []
#all_example['token_type_ids'] = []
all_example['full_attention_mask'] = []
all_example['term_X'] = []
all_example['concept_edges'] = []
all_example['concept_edge_types'] = []
all_example['concept2id'] = []
all_example['id2concept'] = []
all_example['rel2id'] = []
all_example['id2rel'] = []

ind = 0
for one_story, full_story in tqdm(zip(stories, full_stories)):

  example = process_one_example(one_story, full_story, sorted_graph, lm_model, relation_groups, embeddings_index, tokenizer)

  ind += 1
  for k,v in example.items():
    all_example[k].append(v)



with open('Pickles/Dataset_Graph/Relevant/all_examples_Roberta_with_Full_Machine_Val_1.pickle','wb') as f:
  pickle.dump(all_example,f)



import pickle
from tqdm import tqdm


stories = df_test['text_sep'].tolist()
full_stories = df_test['text'].tolist()


all_example = {}
all_example['input_ids'] = []
#all_example['token_type_ids'] = []
all_example['attention_mask'] = []
all_example['full_input_ids'] = []
#all_example['token_type_ids'] = []
all_example['full_attention_mask'] = []
all_example['term_X'] = []
all_example['concept_edges'] = []
all_example['concept_edge_types'] = []
all_example['concept2id'] = []
all_example['id2concept'] = []
all_example['rel2id'] = []
all_example['id2rel'] = []

ind = 0
for one_story, full_story in tqdm(zip(stories, full_stories)):

  example = process_one_example(one_story, full_story, sorted_graph, lm_model, relation_groups, embeddings_index, tokenizer)

  ind += 1
  for k,v in example.items():
    all_example[k].append(v)



with open('Pickles/Dataset_Graph/Relevant/all_examples_Roberta_with_Full_Machine_Test_1.pickle','wb') as f:
  pickle.dump(all_example,f)





import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        #self.temp = None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return idx,item

    def __len__(self):
        return len(self.labels)



import pickle
with open('Pickles/Dataset_Graph/Relevant/all_examples_Roberta_with_Full_Machine_Train_1.pickle','rb') as f:
  train_examples = pickle.load(f)

print(len(train_examples['attention_mask']))

labels = df_train['labels'].tolist()
print(len(labels))

encoded = {}
encoded['input_ids'] = train_examples['full_input_ids']
encoded['attention_mask'] = train_examples['full_attention_mask']

#labels = [0,1,1,0,1]
dataset = CustomDataset(encoded,labels)

import pickle

with open('Pickles/Dataset_Graph/Relevant/all_dataset_Roberta_with_Full_Machine_Train_1.pickle','wb') as f:
  pickle.dump(dataset, f)



import pickle
with open('Pickles/Dataset_Graph/Relevant/all_examples_Roberta_with_Full_Machine_Val_1.pickle','rb') as f:
  val_examples = pickle.load(f)

print(len(val_examples['attention_mask']))

labels = df_val['labels'].tolist()
print(len(labels))

encoded = {}
encoded['input_ids'] = val_examples['full_input_ids']
encoded['attention_mask'] = val_examples['full_attention_mask']

#labels = [0,1,1,0,1]
dataset = CustomDataset(encoded,labels)

import pickle

with open('Pickles/Dataset_Graph/Relevant/all_dataset_Roberta_with_Full_Machine_Val_1.pickle','wb') as f:
  pickle.dump(dataset, f)



import pickle
with open('Pickles/Dataset_Graph/Relevant/all_examples_Roberta_with_Full_Machine_Test_1.pickle','rb') as f:
  test_examples = pickle.load(f)

print(len(test_examples['attention_mask']))

labels = df_test['labels'].tolist()
print(len(labels))

encoded = {}
encoded['input_ids'] = test_examples['full_input_ids']
encoded['attention_mask'] = test_examples['full_attention_mask']

#labels = [0,1,1,0,1]
dataset = CustomDataset(encoded,labels)

import pickle

with open('Pickles/Dataset_Graph/Relevant/all_dataset_Roberta_with_Full_Machine_Test_1.pickle','wb') as f:
  pickle.dump(dataset, f)



print('DONE !!!!!!!!!!!!!!!!!!')





    
