import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import pandas as pd

from transformers import RobertaPreTrainedModel,RobertaModel
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from torch_geometric.nn import GCNConv, global_max_pool, max_pool_neighbor_x, TransformerConv, RGCNConv, GlobalAttention
from torch_geometric.data import Data

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from tqdm import tqdm


from torch import tensor
from torch.optim import AdamW
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.data import DataLoader as geom_DataLoader

import warnings
warnings.filterwarnings("ignore")



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs): ###############################
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        return x




class RobertaForBoundMC(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        return logits




class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
      super().__init__()
      self.rgcn1 = RGCNConv(in_channels, hidden_channels, 19)
      self.conv1 = TransformerConv(hidden_channels, hidden_channels, dropout = dropout)
      self.conv2 = TransformerConv(hidden_channels, hidden_channels, dropout = dropout)
      self.lin1 = nn.Linear(in_channels, hidden_channels)
      self.lin2 = nn.Linear(2*hidden_channels, out_channels)
      self.dropout = dropout
      self.sequential1 = nn.Sequential(
          nn.Linear(hidden_channels,1),
          nn.ReLU()
        )
      self.sequential2 = nn.Sequential(
          nn.Dropout(self.dropout),
          nn.Linear(hidden_channels,hidden_channels)
        )
      


      self.global_max_pool = global_max_pool
      self.global_attn = GlobalAttention(self.sequential1,self.sequential2)


    def forward(self, data, output_lm):
      x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
      x = self.rgcn1(x, edge_index, edge_type)
      x = F.elu(x)
      x = self.conv1(x, edge_index)
      x = F.elu(x)
      x = self.conv2(x, edge_index)
      x_global_attn = self.global_attn(x,batch = data.batch)
      logits = self.lin1(output_lm)
      logits = F.dropout(logits, self.dropout, training=self.training)
      new_x = torch.cat((logits, x_global_attn), dim=1)
      new_x = self.lin2(new_x)
      new_x = F.log_softmax(new_x, dim=1)

      return new_x



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.temp = None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return idx, item

    def __len__(self):
        return len(self.labels)



with open('Pickles/Dataset_Graph/all_dataset_with_Full_Machine_Train_1.pickle','rb') as f:
  train_dataset = pickle.load(f)

with open('Pickles/Dataset_Graph/all_dataset_with_Full_Machine_Val_1.pickle','rb') as f:
  val_dataset = pickle.load(f)

with open('Pickles/Dataset_Graph/all_dataset_with_Full_Machine_Test_1.pickle','rb') as f:
  test_dataset = pickle.load(f)

with open('Pickles/Dataset_Graph/all_examples_with_Full_Machine_Train_1.pickle','rb') as f:
  train_example = pickle.load(f)

with open('Pickles/Dataset_Graph/all_examples_with_Full_Machine_Val_1.pickle','rb') as f:
  val_example = pickle.load(f)

with open('Pickles/Dataset_Graph/all_examples_with_Full_Machine_Test_1.pickle','rb') as f:
  test_example = pickle.load(f)




def train_model(lm_model, gnn_model, optimizer, idx, batch, all_example, grad_accu, total_grad_accu, device):  
  #lm_model.train()
  gnn_model.train()

  ###########################################################################################
  lm_model.eval()

  sep_input_ids = torch.LongTensor(all_example['input_ids'][idx[0]]).to(device)
  sep_attention_mask = torch.LongTensor(all_example['attention_mask'][idx[0]]).to(device)

  with torch.no_grad():
    output_sep = lm_model(input_ids = sep_input_ids, attention_mask = sep_attention_mask)
  #print(output_sep.size())
  output_sep = output_sep.unsqueeze(0)
  output_sep_list = output_sep.detach().clone()
  
  for ind in idx[1:]:
    sep_input_ids = torch.LongTensor(all_example['input_ids'][ind]).to(device)
    sep_attention_mask = torch.LongTensor(all_example['attention_mask'][ind]).to(device)
    #print(x.size())
    ########### collect concepts file from memory based on idx
    #print(len(all_example['term_X'][0]))

    with torch.no_grad():
      output_sep = lm_model(input_ids = sep_input_ids, attention_mask = sep_attention_mask)
    output_sep = output_sep.detach().clone()
    output_sep = output_sep.unsqueeze(0)
    output_sep_list = torch.cat((output_sep_list,output_sep),dim=0)
  
  lm_model.train()
  #print(output_sep_list.size())
  ###########################################################################################


  b_i = batch['input_ids'].squeeze(1).to(device)
  b_a = batch['attention_mask'].squeeze(1).to(device)
  labels = batch['labels'].to(device)


  output = lm_model(input_ids = b_i, attention_mask = b_a)


  data_list = []
  #print(output_copy.size())
  #print('ok_lm')

  for ind, b in zip(idx, output_sep_list):
    x = b
    #print(x.size())
    ########### collect concepts file from memory based on idx
    #print(len(all_example['term_X'][0]))
    term_X = torch.FloatTensor(all_example['term_X'][ind]).to(device)
    new_x = torch.cat((x,term_X),dim=0)
    edge = torch.LongTensor(all_example['concept_edges'][ind]).T
    #print(edge.size())
    row, col = edge
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge = torch.stack([row, col], dim=0)

    edge_type = torch.LongTensor(all_example['concept_edge_types'][ind]).to(device)
    edge_type = torch.cat((edge_type,edge_type), dim=0)
    #print(edge.size())
    #print(new_x.size())
    #print(edge.size())
    data_list.append(Data(x=new_x, edge_index=edge, edge_type = edge_type))

  #print(data_list)
  dl = geom_DataLoader(data_list,batch_size=len(batch['input_ids']))
  batch_ind = None
  for d in dl:
    batch_list = d.batch.tolist()
    #print(batch_list)
    batch_ind = [batch_list.index(bl) for bl in list(dict.fromkeys(batch_list))]
    #print(batch_ind)
    d = d.to(device)
    x_ = gnn_model(d, output)
    #pass

  

  #print(x_.size())
  #out = model(data)
  #print(data.num_edges)
  dummy_labels = batch['labels'].to(device)
  
  loss = F.nll_loss(x_, dummy_labels)
  loss_val = loss.item()
  loss = loss/total_grad_accu
  loss.backward()

  #print(dict(lm_model.named_parameters())['classifier.out_proj.weight'].grad)

  if(grad_accu==True):
    #print('optimizer.step()')
    optimizer.step()
    optimizer.zero_grad()

  
  '''print('Transformer: ')
  print(dict(lm_model.named_parameters())['transformer.ln_f.bias'][0])
  print('GCN: ')
  print(dict(gnn_model.named_parameters())['conv1.lin_query.bias'][0])'''
  
  

  return loss_val



def eval_model(lm_model, gnn_model, dl_val, idx, batch, all_example, device):

  lm_model.eval()
  gnn_model.eval()

  stat = {}
  val_loss = 0
  val_acc = 0.0
  val_f1 = 0.0

  print('Evaluation Starts:')
  for idx, batch in dl_val:
    ###########################################################################################
    lm_model.eval()

    sep_input_ids = torch.LongTensor(all_example['input_ids'][idx[0]]).to(device)
    sep_attention_mask = torch.LongTensor(all_example['attention_mask'][idx[0]]).to(device)
    #print(x.size())
    ########### collect concepts file from memory based on idx
    #print(len(all_example['term_X'][0]))
    #print(sep_attention_mask.size())
    #print(batch['input_ids'].squeeze(1).size())

    with torch.no_grad():
      output_sep = lm_model(input_ids = sep_input_ids, attention_mask = sep_attention_mask)
    #print(output_sep.size())
    output_sep = output_sep.unsqueeze(0)
    output_sep_list = output_sep.detach().clone()
    
    for ind in idx[1:]:
      sep_input_ids = torch.LongTensor(all_example['input_ids'][ind]).to(device)
      sep_attention_mask = torch.LongTensor(all_example['attention_mask'][ind]).to(device)
      #print(x.size())
      ########### collect concepts file from memory based on idx
      #print(len(all_example['term_X'][0]))

      with torch.no_grad():
        output_sep = lm_model(input_ids = sep_input_ids, attention_mask = sep_attention_mask)
      output_sep = output_sep.detach().clone()
      output_sep = output_sep.unsqueeze(0)
      output_sep_list = torch.cat((output_sep_list,output_sep),dim=0)
    
    #print(output_sep_list.size())
    ###########################################################################################


    b_i = batch['input_ids'].squeeze(1).to(device)
    b_a = batch['attention_mask'].squeeze(1).to(device)
    labels = batch['labels'].to(device)


    with torch.no_grad():
      output = lm_model(input_ids = b_i, attention_mask = b_a)
    


    data_list = []
    #print(output_copy.size())
    #print('ok_lm')

    for ind, b in zip(idx, output_sep_list):
      x = b
      #print(x.size())
      ########### collect concepts file from memory based on idx
      #print(len(all_example['term_X'][0]))
      term_X = torch.FloatTensor(all_example['term_X'][ind]).to(device)
      new_x = torch.cat((x,term_X),dim=0)
      edge = torch.LongTensor(all_example['concept_edges'][ind]).T
      #print(edge.size())
      row, col = edge
      row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
      edge = torch.stack([row, col], dim=0)

      edge_type = torch.LongTensor(all_example['concept_edge_types'][ind]).to(device)
      edge_type = torch.cat((edge_type,edge_type), dim=0)
      #print(edge.size())
      #print(new_x.size())
      #print(edge.size())
      data_list.append(Data(x=new_x, edge_index=edge, edge_type = edge_type))

    #print(data_list)
    dl = geom_DataLoader(data_list,batch_size=len(batch['input_ids']))
    batch_ind = None
    for d in dl:
      batch_list = d.batch.tolist()
      #print(batch_list)
      batch_ind = [batch_list.index(bl) for bl in list(dict.fromkeys(batch_list))]
      #print(batch_ind)
      d = d.to(device)
      with torch.no_grad():
        x_ = gnn_model(d, output)
      #pass

    
    #out = model(data)
    #print(data.num_edges)
    labels = batch['labels'].to(device)

    loss = F.nll_loss(x_, labels)
    #loss = F.nll_loss(output_lm, labels)

    val_loss += loss.item()

    
    #print(x_)
    logits = x_.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = np.argmax(logits, axis=-1)


    accuracy = accuracy_score(labels, predictions)
    val_acc += accuracy
    f1 = f1_score(labels, predictions, average='macro')
    val_f1 += f1



    '''print('Transformer: ')
    print(dict(lm_model.named_parameters())['roberta.pooler.dense.bias'][0])
    print('GCN: ')
    print(dict(gnn_model.named_parameters())['conv1.bias'][0])'''

  stat['val_loss'] = val_loss/len(dl_val)
  stat['val_acc'] = val_acc/len(dl_val)
  stat['val_f1'] = val_f1/len(dl_val)

  return stat

  
  

def print_stat(stat_dic):
  print('Train Loss: '+str(stat_dic['train_loss']))
  print('Val Loss: '+str(stat_dic['val_loss']))
  print('Val Acc: '+str(stat_dic['val_acc']))
  print('Val F1: '+str(stat_dic['val_f1']))





batch_size = 16
num_segment = 4
num_labels = 3 

EVAL_STEP = 400
GRAD_ACCU = 1


dl_train = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
dl_val = DataLoader(val_dataset,batch_size=batch_size)
dl_test = DataLoader(test_dataset,batch_size=batch_size)



pretrained = 'roberta-base'
model = RobertaForBoundMC.from_pretrained(pretrained, num_labels=128).to(device) #################

gnn = GNN(in_channels=768, hidden_channels=128, out_channels=num_labels, dropout=0.45).to(device)

optimizer = AdamW(list(model.parameters())+list(gnn.parameters()),lr=5e-6)
#optimizer = AdamW(model3.parameters(),lr=5e-5) #####################################################
#scheduler = ExponentialLR(optimizer, gamma=0.1,)
#total_steps = len(dl_lm)*10
#scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=total_steps, lr_end=1e-4)


grad_accu = 0
eval_step = 0
optimizer.zero_grad()
Best_acc = -1
Best_epoch = -1
Best_step = -1
#print('Training Start')
for epoch in range(0,20):
  ############# batching goes here ###################
  #eval_step = 0
  train_loss = 0
  train_batch_loss = 0
  for idx, batch in tqdm(dl_train):
    #print(idx)
    grad_accu = grad_accu+1
    if(grad_accu!=GRAD_ACCU):
      train_batch_loss += train_model(model, gnn, optimizer, idx, batch, 
                                      train_example, False, GRAD_ACCU, device)
    else:
      train_batch_loss += train_model(model, gnn, optimizer, idx, batch, 
                                      train_example, True, GRAD_ACCU, device)
      grad_accu = 0

    train_loss += train_batch_loss
    #print('Batch Loss: ', train_batch_loss)
    eval_step+=1
    if(eval_step%EVAL_STEP==0):
      stat = eval_model(model, gnn, dl_val, idx, batch, val_example, device)
      stat['train_loss'] = train_batch_loss/eval_step
      print_stat(stat)
      #eval_step = 0
      train_batch_loss = 0
      #scheduler.step() #########################################

      PATH = "Saved_Models/Training_Custom/state_dict_epoch_"+str(epoch)+"_step_"+str(eval_step)+".pt"
      torch.save({
            'epoch': epoch,
            'step': eval_step,
            'lm_state_dict': model.state_dict(),
            'gnn_state_dict': gnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
      if(stat['val_acc']>Best_acc):
         Best_acc = stat['val_acc']
         Best_epoch = epoch
         Best_step = eval_step

         
  
  print('***End of Epoch: '+str(epoch) + ' ***')
  '''stat = eval_model(model3, gcn, gcn2, dl_val, idx, batch, num_choice, val_example, device)
  stat['train_loss'] = train_loss/len(dl_lm)
  print_stat(stat)
  eval_step = 0'''
  #scheduler.step()




print('Best Epoch: '+str(Best_epoch))
print('Best Step: '+str(Best_step))
print('Best Acc: '+str(Best_acc))
print('************ BEST EVAL ************************')

pretrained = 'roberta-base'
model = RobertaForBoundMC.from_pretrained(pretrained, num_labels=128).to(device) #################

gnn = GNN(in_channels=768, hidden_channels=128, out_channels=num_labels, dropout=0.45).to(device)

optimizer = AdamW(list(model.parameters())+list(gnn.parameters()),lr=5e-6)

PATH = "Saved_Models/Training_Custom/state_dict_epoch_"+str(Best_epoch)+"_step_"+str(Best_step)+".pt"
checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint['lm_state_dict'])
gnn.load_state_dict(checkpoint['gnn_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

stat = eval_model(model, gnn, dl_test, idx, batch, test_example, device)
stat['train_loss'] = None
print_stat(stat)
    



    



    




















