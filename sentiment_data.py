#!/usr/bin/env python
# coding: utf-8

# In[64]:


from typing import List, Dict
import random
import numpy as np
import torch


# In[65]:


class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
        word_indices (List[int]): list of word indices in the vocab, which will generated by the `indexing_sentiment_examples` method
    """
    def __init__(self, words, label):
        self.words = words
        self.label = label
        self.word_indices = None # the word indices in vocab

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()
 


# In[67]:


def indexing_sentiment_examples(exs: List[SentimentExample], vocabulary: List[str], UNK_idx: int):
    """
    Indexing words in each SentimentExample based on a given vocabulary. This method will directly modify the `word_indices` attribute of each ex.
    :param exs: a list of SentimentExample objects
    :param vocabulary: the vocabulary, which should be a list of words
    :param UNK_idx: the index of UNK token in the vocabulary
    """
    for ex in exs:
        ex.word_indices = [vocabulary.index(word) if word in vocabulary 
                           else UNK_idx for word in ex.words]


# In[68]:


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
        """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples. Note that all words have been lowercased.

    :param infile: file to read from
    :return: a list of SentimentExamples parsed from the file
    """
        f = open(infile)
        exs = []
        for line in f:
            if len(line.strip()) > 0:
                fields = line.split("\t")
            if len(fields) != 2:
                fields = line.split()
                label = 0 if "0" in fields[0] else 1
                sent = " ".join(fields[1:])
            else:
                # Slightly more robust to reading bad output than int(fields[0])
                label = 0 if "0" in fields[0] else 1
                sent = fields[1]
            sent = sent.lower() # lowercasing
            tokenized_cleaned_sent = list(filter(lambda x: x != '', sent.rstrip().split(" ")))
            exs.append(SentimentExample(tokenized_cleaned_sent, label))
        f.close()
        return exs


# In[70]:


def read_blind_sst_examples(infile: str) -> List[SentimentExample]:
    """
    Reads the blind SST test set, which just consists of unlabeled sentences. Note that all words have been lowercased.
    :param infile: path to the file to read
    :return: list of tokenized sentences (list of list of strings)
    """
    f = open(infile, encoding='utf-8')
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            words = line.lower().split(" ")
            exs.append(SentimentExample(words, label=-1)) # pseudo label -1
    return exs


# In[71]:


def write_sentiment_examples(exs: List[SentimentExample], outfile: str):
    """
    Writes sentiment examples to an output file with one example per line, the predicted label followed by the example.
    Note that what gets written out is tokenized.
    :param exs: the list of SentimentExamples to write
    :param outfile: out path
    :return: None
    """
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([word for word in ex.words]))
    o.close()


# In[72]:
def read_glove(infile: str):
    gl_dict = {}
    f = open(infile,encoding = 'utf-8')
    for word in f:
        buff = word.split()
        gl_dict[buff[0]] = np.asarray(buff[1:],"float32")
    f.close()
    return gl_dict


def pad(it:List[int],n:int,p:int):
    for i in range(n):
        it.append(p)
    return it


# In[74]:


class SentimentExampleBatchIterator:
    """
    A batch iterator which will produce the next batch indexed data.

    Attributes:
        data: a list of SentimentExample objects, which is the source data input
        batch_size: an integer number indicating the number of examples in each batch
        PAD_idx: the index of PAD in the vocabulary
        shuffle: whether to shuffle the data (should set to True only for training)
    """
    def __init__(self, data: List[SentimentExample], batch_size: int, PAD_idx: int, shuffle: bool=True):
        self.data = data
        self.batch_size = batch_size
        self.PAD_idx = PAD_idx
        self.shuffle = shuffle
        
        self._indices = None
        self._cur_idx = None
        
    def refresh(self):
        self._indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self._indices)
        self._cur_idx = 0
    def get_next_batch(self):
        if self._cur_idx < len(self.data): # loop over the dataset
            st_idx = self._cur_idx
            if self._cur_idx + self.batch_size > len(self.data) - 1:
                ed_idx = len(self.data)
            else:
                ed_idx = self._cur_idx + self.batch_size
            self._cur_idx = ed_idx # update
            # retrieve a batch of SentimentExample data
            batch_exs = [self.data[self._indices[_idx]] for _idx in range(st_idx, ed_idx)]
            """
            The next a few lines collect batch_inputs (np.array of int type), batch_lengths (np.array of int type), and batch_labels (np.array of int type) from batch_exs.
            For example, when we have a vocab of ["PAD", "UNK", "i", "feel", "very", "happy", "sad", "this", "is", "interesting"], 
            then given 
            batch_exs = [
                SentimentExample(words=["i", "feel", "very" "happy"], label=1),
                SentimentExample(words=["i", "feel", "very", "sad"], label=0),
                SentimentExample(words=["this", "is", "interesting"], label=1)
            ],
            the follow lines of code should generate
            batch_inputs = 
            [[2, 3, 4, 5], # where 2, 3, 4, 5 are the indices of word "i", "feel", "very", "happy", respectively
             [2, 3, 4, 6], # 0 is PAD_idx
             [7, 8, 9, 0]],
            batch_lengths = [4, 4, 3],
            batch_labels = [1, 0, 1].

            Tip: the indexed SentimentExample object already has the indices saved in `word_indices`; 
            what you need to do is to get them into one matrix (batch_inputs) and add PAD when necessary.
            """
            # TODO: implement batch_inputs, batch_lengths, and batch_labels
            batch_inputs = []
            batch_labels = []
            batch_lengths = []
            for exs in batch_exs:
                batch_lengths.append(len(exs.word_indices))
            max_length = max(batch_lengths)
            for exs in batch_exs:
                if(len(exs.word_indices)<max_length):
                    batch_inputs.append(pad(exs.word_indices,max_length-len(exs.word_indices),self.PAD_idx)) 
                else:
                    batch_inputs.append(exs.word_indices)
                batch_labels.append(exs.label)
             # the next line converts them to torch.Tensor objects so they could be used by follow-up code
            return (torch.tensor(batch_inputs), torch.tensor(batch_lengths), torch.tensor(batch_labels))
        else:
            return None          


# In[ ]:




