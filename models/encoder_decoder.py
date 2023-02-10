import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import operator
from queue import PriorityQueue

import itertools
counter = itertools.count()
import numpy as np

import utils
from utils import device

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
            features = features.reshape(features.size(0), -1)
            features = self.bn(self.linear(features))
        return features

class BeamSearchNode(object):                                
    def __init__(self, hiddenstate, previousNode, wordId, embedding, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.state = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId.data.cpu().numpy().item() 
        self.embedding = embedding
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        """ Use length normalization (e.g. Wu et al, 2016)"""
        return self.logp / float(self.leng - 1 + 1e-6)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab,
                 max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # number of captions to return in beam
        self.embed = nn.Embedding(self.vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.max_seq_length = max_seq_length

    def update(self, new_vocab):
        """
        Update new vocabulary and related parameters in Decoder
        """
        self.vocab = new_vocab
        self.vocab_size = len(self.vocab)

    def forward(self, features, captions, lengths, unpack = False):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True,
                                      enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        if unpack :
            hiddens, out_lengths = pad_packed_sequence(hiddens, batch_first=True)
            outputs = self.linear(hiddens)
        else :
            out_lengths = packed.batch_sizes
            outputs = self.linear(hiddens[0])
        return outputs, out_lengths

    def decoder_step(self, inputs, states) :
        hiddens, states = self.lstm(inputs, states)
        outputs = self.linear(hiddens.squeeze(1))
        return outputs, states

    def greedy_sample(self, features, states=None, maskEnd=False):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)

        # initialize some tensors if masking is necessary
        if maskEnd :
            batch_size = features.size()[0]
            bad_out = torch.zeros(batch_size, dtype=torch.long).to(device)
            zeros = torch.zeros(batch_size, dtype=torch.long).to(device)

        for i in range(self.max_seq_length):
            outputs, states = self.decoder_step(inputs, states)
            _, predicted = outputs.max(1)               # (batch_size)
            inputs = self.embed(predicted).unsqueeze(1) # (batch_size, embed_size)

            # Mask out stop token and everything after
            if maskEnd :
                bad_out = (bad_out.eq(1) | predicted.eq(2))
                sampled_ids.append(torch.where(bad_out, zeros, predicted))
            else :
                sampled_ids.append(predicted)

        sampled_ids = torch.stack(sampled_ids, 1)        # (batch_size, max_seq_length)
        return sampled_ids

    def beam_sample(self, features, topk, states = None) :
        '''
        :param features: pass encoder outputs
        :param states: initial hidden state
        :return: decoded_batch
        '''

        beam_width = 50
        decoded_batch = []
        inputs = features.unsqueeze(1)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # initialize queue with embedding 
        start_token = torch.LongTensor([self.vocab.word2idx['<start>']]).to(device)
        EOS_token = self.vocab.word2idx['<end>']
        nodes = PriorityQueue()
        count = next(counter)
        nodes.put((0, count, BeamSearchNode(states, None, start_token, inputs, 0, 1)))

        while True:
            # fetch the best node & embed it
            score, _, n = nodes.get()
            inputs = n.embedding
            states = n.state

            # if the best next one is EOS, terminate w/ that node (containing history)
            # don't allow empty utterances
            if n.wordid == EOS_token and n.prevNode != None and n.leng > 3:
                endnodes.append((score, n)) 
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
                
            # otherwise decode for one step using decoder
            output, states = self.decoder_step(inputs, states)
            log_prob, indexes = torch.topk(output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(states, n, decoded_t, self.embed(decoded_t),
                                      n.logp + log_p, n.leng + 1)
                # Priority queue retrieves lowest valued entries first, so we need to flip sign
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                count = next(counter)
                nodes.put((score, count, nn))

        # in the event that you weren't able to get endnodes,
        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        scores = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []

            # back trace, omitting first node w/ embedding
            # (prevents double start token)
            while n.prevNode != None :
                utterance.append(n.wordid)
                n = n.prevNode

            utterance = utterance[::-1]
            utterances.append(torch.LongTensor(utterance))
            # Flip sign back around
            scores.append(-score)
        return torch.nn.utils.rnn.pad_sequence(utterances, batch_first = True), torch.tensor(scores).to(device)

