#!/usr/bin/env python
# coding: utf-8

# # Importing Library

# In[1]:


import torch
from torch import nn
import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field,BucketIterator

import spacy
import random
import re 
import time
import math


# In[2]:


torch.__version__, torchtext.__version__


# ## Spacy Model

# Spacy is an open-source software python library used in advanced natural language processing and machine learning. It will be used to build information extraction, natural language understanding systems, and to pre-process text for deep learning.
# 
# Word tokens are the basic units of text involved in any NLPlabeling task. The first step, when processing text, is to split it into tokens.

# In[3]:


#!python -m spacy download en_core_web_sm
#!python -m spacy download de_core_news_sm


# In[4]:


spacy_english = spacy.load("en_core_web_sm")
spacy_german = spacy.load("de_core_news_sm")


# In[5]:


def tokenise_of_german(text):
    return [tokenise_str.text for tokenise_str in spacy_german.tokenizer(text)]


# In[6]:


def tokenise_of_english(text):
    return [tokenise_str.text for tokenise_str in spacy_english.tokenizer(text)]


# # torch.leagacy.data.Fields

# In[7]:


SRC = Field(sequential=True,
           init_token='<sos>',
           eos_token='<eos>',
            lower=True,
            tokenize=tokenise_of_german
           )


# In[8]:


TRG = Field(sequential=True,
           init_token='<sos>',
           eos_token='<eos>',
           lower=True,
           tokenize=tokenise_of_english)


# # Importing Multi30k Datasets

# In[9]:


train_data, valid_data, test_data = Multi30k.splits(exts=('.de','.en'), fields=(SRC,TRG))


# All Source sentences with there targets.
# This is raw data just taken out of Multi30k and we pass to spacy for tokenisation process and below is what happens to data.

# In[10]:


for index in range(len(train_data)):
    print("Index: ",index)
    print("German Source Sentence: ",train_data[index].src)
    print("English Source Sentence: ",train_data[index].trg)
    if index==4:
        break
    print("====================================================================")


# In[11]:


# Priting number of data in training,valid and testing loader.
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")


# ## Buidling up Vocabulary 

# Next, we'll build the vocabulary for the source and target languages. The vocabulary is used to associate each unique token with an index (an integer). The vocabularies of the source and target languages are distinct.
# 
# Using the min_freq argument, we only allow tokens that appear at least 2 times to appear in our vocabulary. Tokens that appear only once are converted into an <unk> (unknown) token.
# 
# It is important to note that our vocabulary should only be built from the training set and not the validation/test set. This prevents "information leakage" into our model, giving us artifically inflated validation/test scores.

# Paramters for build_vocab() ============>
# 
# Parameters:	
# counter – collections.Counter object holding the frequencies of each value found in the data.<br>
# max_size – The maximum size of the vocabulary, or None for no maximum. Default: None.<br>
# min_freq – The minimum frequency needed to include a token in the vocabulary. Values less than 1 will be set to 1. Default: 1.<br>
# specials – The list of special tokens (e.g., padding or eos) that will be prepended to the vocabulary in addition to an <unk> token. Default: [‘<pad>’]<br>
# vectors – One of either the available pretrained vectors or custom pretrained vectors (see Vocab.load_vectors); or a list of aforementioned vectors<br>
# unk_init (callback) – by default, initialize out-of-vocabulary word vectors to zero vectors; can be any function that takes in a Tensor and returns a Tensor of the same size. Default: torch.Tensor.zero_<br>
# vectors_cache – directory for cached vectors. Default: ‘.vector_cache’<br>
# specials_first – Whether to add special tokens into the vocabulary at first. If it is False, they are added into the vocabulary at last. Default: True.<br>

# In[12]:


SRC.build_vocab(train_data,min_freq=2)
TRG.build_vocab(train_data, min_freq=2)


# Below I Have converted into dictionary form. Open and see the dictionary and see data stored into there respective keys of dictionary.

# In[13]:


dict_of_vocab_TRG = vars(TRG.vocab)
dict_of_vocab_TRG.keys()


# In[14]:


dict_of_vocab_SRC = vars(SRC.vocab)
dict_of_vocab_SRC.keys()


# ## Bucket Iterator 
# This Iterrator used for batching purpose and it Defines an iterator that batches examples of similar lengths together.
# Minimizes amount of padding needed while producing freshly shuffled batches for each new epoch. See pool for the bucketing procedure used.

# Parameters :	
# dataset – The Dataset object to load Examples from.<br>
# batch_size – Batch size.<br>
# batch_size_fn – Function of three arguments (new example to add, current count of examples in the batch, and current effective batch size) that returns the new effective batch size resulting from adding that example to a batch. This is useful for dynamic batching, where this function would add to the current effective batch size the number of tokens in the new example.<br>
# sort_key – A key to use for sorting examples in order to batch together examples with similar lengths and minimize padding. The sort_key provided to the Iterator constructor overrides the sort_key attribute of the Dataset, or defers to it if None.<br>
# train – Whether the iterator represents a train set.<br>
# repeat – Whether to repeat the iterator for multiple epochs. Default: False.<br>
# shuffle – Whether to shuffle examples between epochs.<br>
# sort – Whether to sort examples according to self.sort_key. Note that shuffle and sort default to train and (not train).<br>
# sort_within_batch – Whether to sort (in descending order according to self.sort_key) within each batch. If None, defaults to self.sort. If self.sort is True and this is False, the batch is left in the original (ascending) sorted order.<br>
# device (str or torch.device) – A string or instance of torch.device specifying which device the Variables are going to be created on. If left as default, the tensors will be created on cpu. Default: None.<br>

# In[15]:


device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_sizes=(BATCH_SIZE,BATCH_SIZE,BATCH_SIZE),
    device=device,
    shuffle=True,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src)
)


# In[16]:


for index, data in enumerate(train_iterator.data()):
    print(index)
    print(vars(data))
    if index==4:
        break;
    print("=============================================================================")


# # Building Sequence to Sequence Model

# ## Encoder Class

# First, the encoder, a 2 layer LSTM. The paper we are implementing uses a 4-layer LSTM, but in the interest of training time we cut this down to 2-layers. The concept of multi-layer RNNs is easy to expand from 2 to 4 layers. 
# 
# For a multi-layer RNN, the input sentence, $X$, after being embedded goes into the first (bottom) layer of the RNN and hidden states, $H=\{h_1, h_2, ..., h_T\}$, output by this layer are used as inputs to the RNN in the layer above. Thus, representing each layer with a superscript, the hidden states in the first layer are given by:
# 
# $$h_t^1 = \text{EncoderRNN}^1(e(x_t), h_{t-1}^1)$$
# 
# The hidden states in the second layer are given by:
# 
# $$h_t^2 = \text{EncoderRNN}^2(h_t^1, h_{t-1}^2)$$
# 
# Using a multi-layer RNN also means we'll also need an initial hidden state as input per layer, $h_0^l$, and we will also output a context vector per layer, $z^l$.
# 
# Without going into too much detail about LSTMs (see [this](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) blog post to learn more about them), all we need to know is that they're a type of RNN which instead of just taking in a hidden state and returning a new hidden state per time-step, also take in and return a *cell state*, $c_t$, per time-step.
# 
# $$\begin{align*}
# h_t &= \text{RNN}(e(x_t), h_{t-1})\\
# (h_t, c_t) &= \text{LSTM}(e(x_t), h_{t-1}, c_{t-1})
# \end{align*}$$
# 
# We can just think of $c_t$ as another type of hidden state. Similar to $h_0^l$, $c_0^l$ will be initialized to a tensor of all zeros. Also, our context vector will now be both the final hidden state and the final cell state, i.e. $z^l = (h_T^l, c_T^l)$.
# 
# Extending our multi-layer equations to LSTMs, we get:
# 
# $$\begin{align*}
# (h_t^1, c_t^1) &= \text{EncoderLSTM}^1(e(x_t), (h_{t-1}^1, c_{t-1}^1))\\
# (h_t^2, c_t^2) &= \text{EncoderLSTM}^2(h_t^1, (h_{t-1}^2, c_{t-1}^2))
# \end{align*}$$
# 
# Note how only our hidden state from the first layer is passed as input to the second layer, and not the cell state.
# 
# So our encoder looks something like this: 
# ![](images/encoder.png)
# 
# In the `forward` method, we pass in the source sentence, $X$, which is converted into dense vectors using the `embedding` layer, and then dropout is applied. These embeddings are then passed into the RNN. As we pass a whole sequence to the RNN, it will automatically do the recurrent calculation of the hidden states over the whole sequence for us! Notice that we do not pass an initial hidden or cell state to the RNN. This is because, as noted in the [documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM), that if no hidden/cell state is passed to the RNN, it will automatically create an initial hidden/cell state as a tensor of all zeros. 
# 
# The RNN returns: `outputs` (the top-layer hidden state for each time-step), `hidden` (the final hidden state for each layer, $h_T$, stacked on top of each other) and `cell` (the final cell state for each layer, $c_T$, stacked on top of each other).
# 
# As we only need the final hidden and cell states (to make our context vector), `forward` only returns `hidden` and `cell`. 
# 
# The sizes of each of the tensors is left as comments in the code. In this implementation `n_directions` will always be 1, however note that bidirectional RNNs (covered in tutorial 3) will have `n_directions` as 2.

# In[17]:


from torch.nn import Embedding,LSTM,Linear


# In[18]:


class Encoder(torch.nn.Module):
    def __init__(self,input_dim, embeddind_dim, hidden_dim, num_layers, droput_value):
        super(Encoder,self).__init__();
        
        # Attributes
        self.input_dim = input_dim;
        self.embeddind_dim = embeddind_dim;
        self.hidden_dim = hidden_dim;
        self.num_layers = num_layers;
        self.droput_value = droput_value;
        
        # Layers Object
        self.embedded = Embedding(input_dim,embeddind_dim);
        self.lstm = LSTM(embeddind_dim,hidden_dim,num_layers,dropout=droput_value)
        
    def forward(self,source):
        # source.shape => [seq_len, batch_size]  
        # source is 2d array where column wise sentences are present and batch_size is number of sentences in a set.
        
        output, (hidden,cell) = self.lstm(self.embedded(source)) #self.embedded.shape=>[seq_len, batch_size,embedding_size]
        return hidden, cell


# ## Decoder Class

# Next, we'll build our decoder, which will also be a 2-layer (4 in the paper) LSTM.
# 
# ![](images/decoder.png)
# 
# The `Decoder` class does a single step of decoding, i.e. it ouputs single token per time-step. The first layer will receive a hidden and cell state from the previous time-step, $(s_{t-1}^1, c_{t-1}^1)$, and feeds it through the LSTM with the current embedded token, $y_t$, to produce a new hidden and cell state, $(s_t^1, c_t^1)$. The subsequent layers will use the hidden state from the layer below, $s_t^{l-1}$, and the previous hidden and cell states from their layer, $(s_{t-1}^l, c_{t-1}^l)$. This provides equations very similar to those in the encoder.
# 
# $$\begin{align*}
# (s_t^1, c_t^1) = \text{DecoderLSTM}^1(d(y_t), (s_{t-1}^1, c_{t-1}^1))\\
# (s_t^2, c_t^2) = \text{DecoderLSTM}^2(s_t^1, (s_{t-1}^2, c_{t-1}^2))
# \end{align*}$$
# 
# Remember that the initial hidden and cell states to our decoder are our context vectors, which are the final hidden and cell states of our encoder from the same layer, i.e. $(s_0^l,c_0^l)=z^l=(h_T^l,c_T^l)$.
# 
# We then pass the hidden state from the top layer of the RNN, $s_t^L$, through a linear layer, $f$, to make a prediction of what the next token in the target (output) sequence should be, $\hat{y}_{t+1}$. 
# 
# $$\hat{y}_{t+1} = f(s_t^L)$$
# 
# The arguments and initialization are similar to the `Encoder` class, except we now have an `output_dim` which is the size of the vocabulary for the output/target. There is also the addition of the `Linear` layer, used to make the predictions from the top layer hidden state.
# 
# Within the `forward` method, we accept a batch of input tokens, previous hidden states and previous cell states. As we are only decoding one token at a time, the input tokens will always have a sequence length of 1. We `unsqueeze` the input tokens to add a sentence length dimension of 1. Then, similar to the encoder, we pass through an embedding layer and apply dropout. This batch of embedded tokens is then passed into the RNN with the previous hidden and cell states. This produces an `output` (hidden state from the top layer of the RNN), a new `hidden` state (one for each layer, stacked on top of each other) and a new `cell` state (also one per layer, stacked on top of each other). We then pass the `output` (after getting rid of the sentence length dimension) through the linear layer to receive our `prediction`. We then return the `prediction`, the new `hidden` state and the new `cell` state.
# 

# In[19]:


class Decoder(torch.nn.Module):
    def __init__(self,output_dim, embedding_dim, hidden_dim, num_layers, dropout_value):
        super(Decoder,self).__init__();
        
        # Attributes
        self.output_dim = output_dim;
        self.embedding_dim = embedding_dim;
        self.hidden_dim = hidden_dim;
        self.num_layers = num_layers;
        self.dropout_value = dropout_value;
        
        # Layers
        self.embedded = Embedding(output_dim,embedding_dim);
        self.lstm = LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout_value);
        
        # Linear layer applied to LSTM outputs for havimg most highest proability of predictions(after Softmax to it.)
        self.fullyConnectedLayer = Linear(hidden_dim,output_dim);
        
    def forward(self,target, hidden, cell):
        #target.shape => [batch_size]
        # This target is just one word at a time but for complete batch_size or we can say Single word for same seq_len
        # of Batch Size.
        
        # Making 1d to 2d array to pass to Embedding layer at dimension 0.
        target = target.unsqueeze(0) 
        # target.shape => [1,batch_size]
        
        out_embedded = self.embedded(target)
        # out_embedded.shape =>[1,batch_size, embedding_size]
        
        outputs,(hidden,cell) = self.lstm(out_embedded)
        # outputs.shape => [1, batch_size, hidden_size]
        # hidden.shape => [1*1, batch_size, hidden_size]
        # cell.shape => [1*1, batch_size, hidden_size]    : num_layers=d=1
        
        outputs = outputs.squeeze(0)
        #outputs.shape =>[batch_size, hidden_size]
        
        predictions = self.fullyConnectedLayer(outputs)
        #predictions =>[batch_size, output_dim]
        
        return predictions,hidden,cell


# ## Sequence to Sequence Class 

# ![](images/seq2seq.png)
# Our `forward` method takes the source sentence, target sentence and a teacher-forcing ratio. The teacher forcing ratio is used when training our model. When decoding, at each time-step we will predict what the next token in the target sequence will be from the previous tokens decoded, $\hat{y}_{t+1}=f(s_t^L)$. With probability equal to the teaching forcing ratio (`teacher_forcing_ratio`) we will use the actual ground-truth next token in the sequence as the input to the decoder during the next time-step. However, with probability `1 - teacher_forcing_ratio`, we will use the token that the model predicted as the next input to the model, even if it doesn't match the actual next token in the sequence.  
# 
# The first thing we do in the `forward` method is to create an `outputs` tensor that will store all of our predictions, $\hat{Y}$.
# 
# We then feed the input/source sentence, `src`, into the encoder and receive out final hidden and cell states.
# 
# The first input to the decoder is the start of sequence (`<sos>`) token. As our `trg` tensor already has the `<sos>` token appended (all the way back when we defined the `init_token` in our `TRG` field) we get our $y_1$ by slicing into it. We know how long our target sentences should be (`max_len`), so we loop that many times. The last token input into the decoder is the one **before** the `<eos>` token - the `<eos>` token is never input into the decoder. 
# 
# During each iteration of the loop, we:
# - pass the input, previous hidden and previous cell states ($y_t, s_{t-1}, c_{t-1}$) into the decoder
# - receive a prediction, next hidden state and next cell state ($\hat{y}_{t+1}, s_{t}, c_{t}$) from the decoder
# - place our prediction, $\hat{y}_{t+1}$/`output` in our tensor of predictions, $\hat{Y}$/`outputs`
# - decide if we are going to "teacher force" or not
#     - if we do, the next `input` is the ground-truth next token in the sequence, $y_{t+1}$/`trg[t]`
#     - if we don't, the next `input` is the predicted next token in the sequence, $\hat{y}_{t+1}$/`top1`, which we get by doing an `argmax` over the output tensor
#     
# Once we've made all of our predictions, we return our tensor full of predictions, $\hat{Y}$/`outputs`.
# 
# **Note**: our decoder loop starts at 1, not 0. This means the 0th element of our `outputs` tensor remains all zeros. So our `trg` and `outputs` look something like:
# 
# $$\begin{align*}
# \text{trg} = [<sos>, &y_1, y_2, y_3, <eos>]\\
# \text{outputs} = [0, &\hat{y}_1, \hat{y}_2, \hat{y}_3, <eos>]
# \end{align*}$$
# 
# Later on when we calculate the loss, we cut off the first element of each tensor to get:
# 
# $$\begin{align*}
# \text{trg} = [&y_1, y_2, y_3, <eos>]\\
# \text{outputs} = [&\hat{y}_1, \hat{y}_2, \hat{y}_3, <eos>]
# \end{align*}$$

# In[20]:


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device,vocab_size_of_trg):
        super(Seq2Seq,self).__init__();
        self.encoder = encoder;
        self.decoder = decoder;
        self.device = device;
        self.vocab_size_of_trg = vocab_size_of_trg;
        
    def forward(self,src,trg,teacher_forece_ratio=0.5):
        #src.shape => (seq_len, batch_size)
        #trg.shape => (seq_len, batch_size)
        
        trg_len = trg.shape[0];
        batch_size = trg.shape[1];
        trg_vocal_size = self.vocab_size_of_trg;
        
        outputs_of_zeros=torch.zeros(trg_len,batch_size,trg_vocal_size).to(self.device)
        # Its 3d array where we have predictions(in terms of proablity) for all vocab words.
        # trg_len corresponds to all words in that selected batch of senetences.
        
        # Sending source to encoder
        hidden ,cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        for t in range(trg_len):
            outputs,hidden,cell = self.decoder(input,hidden,cell);
            
            #place predictions in a tensor holding predictions for each token
            outputs_of_zeros[t] = outputs
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forece_ratio
            
            #get the highest predicted token from our predictions
            top1 = outputs.argmax(1)
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs_of_zeros


# ## Hyperparameters, Parameters and Variable Declarations

# In[21]:


Input_dim = len(SRC.vocab)
Output_dim = len(TRG.vocab)

hidden_size_of_Encoder = hidden_size_of_Decoder = 512
embedding_size = 256

num_layers = 2
dropout=0.5
EPOCHS=10

enc = Encoder(Input_dim,embedding_size,hidden_size_of_Encoder,num_layers,dropout).to(device)
dec = Decoder(Output_dim,embedding_size, hidden_size_of_Decoder, num_layers, dropout).to(device)

model = Seq2Seq(enc,dec,device,Output_dim).to(device)


# In[22]:


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)


# In[23]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# # Training Loop

# In[24]:


# Optimizer Declaration
from torch.optim import Adam
optimiser = Adam(model.parameters())


# In[25]:


# Loss Functions Declaration
from torch.nn import CrossEntropyLoss
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
loss = CrossEntropyLoss(ignore_index=TRG_PAD_IDX) # Ignoring all indexes that have padding token.


# Next, we'll define our training loop. 
# 
# First, we'll set the model into "training mode" with `model.train()`. This will turn on dropout (and batch normalization, which we aren't using) and then iterate through our data iterator.
# 
# As stated before, our decoder loop starts at 1, not 0. This means the 0th element of our `outputs` tensor remains all zeros. So our `trg` and `outputs` look something like:
# 
# $$\begin{align*}
# \text{trg} = [<sos>, &y_1, y_2, y_3, <eos>]\\
# \text{outputs} = [0, &\hat{y}_1, \hat{y}_2, \hat{y}_3, <eos>]
# \end{align*}$$
# 
# Here, when we calculate the loss, we cut off the first element of each tensor to get:
# 
# $$\begin{align*}
# \text{trg} = [&y_1, y_2, y_3, <eos>]\\
# \text{outputs} = [&\hat{y}_1, \hat{y}_2, \hat{y}_3, <eos>]
# \end{align*}$$
# 
# At each iteration:
# - get the source and target sentences from the batch, $X$ and $Y$
# - zero the gradients calculated from the last batch
# - feed the source and target into the model to get the output, $\hat{Y}$
# - as the loss function only works on 2d inputs with 1d targets we need to flatten each of them with `.view`
#     - we slice off the first column of the output and target tensors as mentioned above
# - calculate the gradients with `loss.backward()`
# - clip the gradients to prevent them from exploding (a common issue in RNNs)
# - update the parameters of our model by doing an optimizer step
# - sum the loss value to a running total
# 
# Finally, we return the loss that is averaged over all batches.

# In[26]:


## Training Function

def Train(model, training_data, optimizer, criterion, clip_value,OPTIM_ENCODER):
    model.train()
    epoch_loss=0;
    
    for i, batch_wise_data in enumerate(training_data):
        src = batch_wise_data.src
        trg = batch_wise_data.trg
        
        optimizer.zero_grad()
        OPTIM_ENCODER.zero_grad()
        
        output = model(src,trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
            
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg) 
        loss.backward()   
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  
        optimizer.step()
        OPTIM_ENCODER.step()
        
        epoch_loss += loss.item()
        
        return epoch_loss/len(training_data)


# In[27]:


def evaluate(model,data,criterion):
    epoch_loss=0
    model.eval()
    
    with torch.no_grad():
        for i , batch_wise_data in enumerate(data):
            src = batch_wise_data.src;
            trg = batch_wise_data.trg;
            
            output=model(src,trg,0) #teacher_ratio_force=0
            
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1] # That is output_dim.
            
            # Exculding <sos> tokens of all batches.
            output = output[1:].view(-1,output_dim);
            trg = trg[1:].view(-1);
            
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            
            loss = criterion(output,trg)
            
            epoch_loss +=loss.item()
    return epoch_loss/len(data)


# In[28]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[29]:


CLIP = 1

best_valid_loss = float('inf')

OPTIM_ENCODER = Adam(enc.parameters())
OPTIM_DECODER = Adam(dec.parameters())
for epoch in range(EPOCHS):
    start_time = time.time()
    
    train_loss = Train(model,train_iterator, OPTIM_DECODER, loss,CLIP, OPTIM_ENCODER);
    valid_loss = evaluate(model, valid_iterator, loss);
    
    end_time = time.time();
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model1.pt')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


# In[30]:


'''model.load_state_dict(torch.load('tut1-model.pt'))

test_loss = evaluate(model, test_iterator, loss)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')'''


# In[ ]:




