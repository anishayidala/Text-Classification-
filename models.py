#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sentiment_data import *
from evaluator import *

from collections import Counter
import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional 


# In[9]:



######################################
# IMPLEMENT THE SENTIMENT CLASSIFIER #
######################################

class FeedForwardNeuralNetClassifier(nn.Module):
    """
    The Feed-Forward Neural Net sentiment classifier.
    """
    def __init__(self, n_classes, vocab_size, emb_dim, n_hidden_units,vocab,gl_path=None):
        """
        In the __init__ function, you will define modules in FFNN.
        :param n_classes: number of classes in this classification problem
        :param vocab_size: size of vocabulary
        :param emb_dim: dimension of the embedding vectors
        :param n_hidden_units: dimension of the hidden units
        """
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.vocab = vocab
        self.gl_path = gl_path
       
        # TODO: create a randomly initialized embedding matrix, and set padding_idx as 0
        # PAD's embedding will not be trained and by default is initialized as zero
        if self.gl_path:
            gl_dict = read_glove(self.gl_path)
            em_matrix = np.zeros((self.vocab_size, self.emb_dim))
            for i , word in enumerate(vocab):
                em_matrix[i] = gl_dict[word] if gl_dict.get(word) is not None else np.zeros(self.emb_dim)
            self.word_embeddings = nn.Embedding(self.vocab_size , self.emb_dim)
            self.word_embeddings.load_state_dict({'weight':torch.from_numpy(em_matrix)})
        
        else:
            self.word_embeddings = nn.Embedding(num_embeddings = vocab_size,embedding_dim=self.emb_dim,padding_idx = 0) # replace None with the correct implementation

        # TODO: implement the FFNN architecture
        # when you build the FFNN model, you will need specify the embedding size using self.emb_dim, the hidden size using self.n_hidden_units,
        # and the output class size using self.n_classes 
        self.nn1 = nn.Linear(self.emb_dim,n_hidden_units)
        self.relu = nn.ReLU()
        self.nn2 = nn.Linear(self.n_hidden_units,self.n_classes)


    def forward(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward function, which defines how FFNN should work when given a batch of inputs and their actual sent lengths (i.e., before PAD)
        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return the logits outputs of FFNN (i.e., the unnormalized hidden units before softmax)
        """
        # TODO: implement the forward function, which returns the logits
        emb = self.word_embeddings(batch_inputs)
        avg = torch.mean(emb,1)
        tr_emb = self.nn1(avg)
        act = self.relu(tr_emb)
        logit = self.nn2(act)
        return logit
        #raise Exception("Not Implemented!")
        
    
    def batch_predict(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> List[int]:
        """
        Make predictions for a batch of inputs. This function may directly invoke `forward` (which passes the input through FFNN and returns the output logits)

        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return: a list of predicted classes for this batch of data, either 0 for negative class or 1 for positive class
        """
        # TODO: implement the prediction function, which could reuse the forward function 
        # but should return a list of predicted labels
        logits = self.forward(batch_inputs , batch_lengths)
        _softmax = nn.Softmax()
        return torch.argmax(_softmax(logits), 1)
        #raise Exception("Not Implemented!")



# In[10]:


class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.num_words = 0
        self.vocab = []
    def add_word(self,word):
        if word not in self.word2index:
            self.vocab.append(word) 
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1
        return self.vocab 
  


# In[11]:


def train_feedforward_neural_net(
    args,
    train_exs: List[SentimentExample], 
    dev_exs: List[SentimentExample]) -> FeedForwardNeuralNetClassifier:
    """
    Main entry point for your modifications. Trains and returns a FFNN model (whose architecture is configured based on args)

    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # TODO: read in all training examples and create a vocabulary (a List-type object called `vocab`)
    x = Vocabulary()
    for i in range(len(train_exs)):
        for word in train_exs[i].words:
            vocab = x.add_word(word) # replace None with the correct implementation
    
    # add PAD and UNK as the first two tokens
    # DO NOT CHANGE, PAD must go first and UNK next (as their indices have been hard-coded in several places)
    vocab = ["PAD", "UNK"] + vocab
    print("Vocab size:", len(vocab))
    # write vocab to an external file, so the vocab can be reloaded to index the test set
    with open("data/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    # indexing the training/dev examples
    indexing_sentiment_examples(train_exs, vocabulary=vocab, UNK_idx=1)
    indexing_sentiment_examples(dev_exs, vocabulary=vocab, UNK_idx=1)
    
    # TODO: create the FFNN classifier
    model = FeedForwardNeuralNetClassifier(n_classes=2,vocab_size=len(vocab),emb_dim=args.emb_dim,n_hidden_units=args.n_hidden_units, vocab=vocab , gl_path = args.glove_path) 
    # replace None with the correct implementation
    
    # TODO: define an Adam optimizer, using default config
    optimizer = optim.Adam(model.parameters()) 
    # replace None with the correct implementation

    # create a batch iterator for the training data
    batch_iterator = SentimentExampleBatchIterator(train_exs, batch_size=args.batch_size, PAD_idx=0, shuffle=True)

    # training
    best_epoch = -1
    best_acc = -1
    for epoch in range(args.n_epochs):
        print("Epoch %i" % epoch)

        batch_iterator.refresh() # initiate a new iterator for this epoch

        model.train() # turn on the "training mode"
        batch_loss = 0.0
        batch_example_count = 0
        batch_data = batch_iterator.get_next_batch()
        while batch_data is not None:
            batch_inputs, batch_lengths, batch_labels = batch_data
            # TODO: clean up the gradients for this batch
            
            optimizer.zero_grad()
            
            # TODO: call the model to get the logits
            logits = model.forward(batch_inputs,batch_lengths)

            # TODO: calculate the loss (let's name it Floss`, so the follow-up lines could collect the stats)
            Floss = nn.CrossEntropyLoss()
            loss = Floss(logits,batch_labels)

            # record the loss and number of examples, so we could report some stats
            batch_example_count += len(batch_labels)
            batch_loss += loss.item() * len(batch_labels)

            # TODO: backpropagation (backward and step)
            loss.backward()
            optimizer.step()
            # get another batch
            batch_data = batch_iterator.get_next_batch()

        print("Avg loss: %.5f" % (batch_loss / batch_example_count))

        # evaluate on dev set
        model.eval() # turn on the "evaluation mode"
        acc, _, _, _ = evaluate(model, dev_exs, return_metrics=True)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            print("Secure a new best accuracy %.3f in epoch %d!" % (best_acc, best_epoch))
            
            # save the current best model parameters
            print("Save the best model checkpoint as `best_model.ckpt`!")
            torch.save(model.state_dict(), "best_model.ckpt")
        print("-" * 10)

    # load back the best checkpoint on dev set
    model.load_state_dict(torch.load("best_model.ckpt"))
    
    model.eval() # switch to the evaluation mode
    return model

if __name__ == "__main__":
    model = FeedForwardNeuralNetClassifier(n_classes=2, vocab_size=100, emb_dim=50, n_hidden_units=20)    
    print(model)


# In[ ]:




