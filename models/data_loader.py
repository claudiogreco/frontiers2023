# suppress allennlp logging at level of INFO or below
import logging
logging.disable(logging.INFO)

import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import itertools
import utils
import spacy

from utils import coco, data_dir, data_type, device, nlp

from itertools import combinations, chain, product
from nltk.tree import Tree
from random import shuffle
import copy

class ReducedDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, image, captions, vocab):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            image: target image that caption was generated for
            captions: set of reduced captions to pair with image
            vocab: vocabulary 
        """
        self.image_path = image
        self.captions = captions
        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption = self.captions[index]
        image = utils.load_image(self.image_path, for_batching=True)
        id_caption = torch.Tensor(utils.words_to_ids(caption, self.vocab))
        return image, id_caption

    def __len__(self):
        return len(self.captions)

class ReducedHistoryDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, agent, history, method, vocab):
        self.vocab = vocab
        self.possible_pairs = []

        # initialize history with original captions for each image
        if hasattr(agent, 'orig_captions') :
            for img, caption in agent.orig_captions :
                self.possible_pairs.append((img, caption))

        
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        image, caption = self.possible_pairs[index]
        id_caption = torch.Tensor(utils.words_to_ids(caption, self.vocab))
        real_image = utils.load_image(image, for_batching=True)
        return real_image, id_caption

    def __len__(self):
        return len(self.possible_pairs)

def powersets(caption):
    "powersets([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = caption.split()
    if len(s) < 20:  # introduce hack to get around long human utterances
        ps = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
        return list(map(lambda v : ' '.join(v), ps[1:]))
    else:
        return NPs(caption)

def span(lst):
    yield [lst]
    for i in range(1, len(lst)):
        for x in span(lst[i:]):
            yield [lst[:i]] + x

def ordered_subsets(caption):
    s = caption.split()
    ps = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    assert len(ps)>0
    consecutive_subsets = []
    for subset in ps:
        is_consecutive = True
        if len(subset)>1:
            for i in range(0, len(subset)-1):
                if s.index(subset[i+1]) - s.index(subset[i]) != 1:
                    is_consecutive = False
                    break
        if is_consecutive: consecutive_subsets.append(subset)
    return np.unique(list(map(lambda v : ' '.join(v), consecutive_subsets[1:])))

def NPs(caption):
    dataset = []
    doc = nlp(caption)
    for word in doc:
        if word.dep_ in ('xcomp', 'ccomp', 'pcomp', 'acomp'):
            subtree_span = doc[word.left_edge.i : word.right_edge.i + 1]
            print('clause ancestors', [t.text for t in word.ancestors])
            dataset.append(' '.join([t.text for t in subtree_span]))
        elif word.dep_ in ('ROOT') :
            left_subtree = [doc[w.left_edge.i : w.i + 1] for w in word.lefts if w.dep_ != 'aux']
            right_subtree = [doc[w.i : w.right_edge.i + 1] for w in word.rights]
            for l in itertools.product(left_subtree, right_subtree) :
                dataset.append(' '.join([l[0].text, word.text, l[1].text]))
                dataset.append(' '.join([word.text, l[1].text]))
                dataset.append(' '.join([l[0].text, word.text]))
                dataset.append(' '.join([t.text for t in l[0].subtree] + [word.text]))
        # note: this is a failed attempt to extract local prepositional phrases
        # e.g. 'the dog with a frisbee in his mouth' -> 'the dog with a frisbee'
        # elif word.pos_ in ('ADP') and word.dep_ != 'prt':
        #     span = ([t.text for t in word.lefts]
        #             + [word.text] +
        #             [t.text for t in word.rights])
        #     dataset.append(' '.join([a.text for a in word.ancestors][:1] + span))
        #     dataset.append(' '.join([a.text for a in word.ancestors][:1] +
        # [a.text for a in word.subtree]))
    noun_chunks = [n.text for n in doc.noun_chunks if not n.root.is_stop] + \
                  [n.root.text for n in doc.noun_chunks if not n.root.is_stop]
    dataset = np.unique(dataset + [caption] + noun_chunks)

    # add original one if it's not already there
    return list(dataset)# + [caption] if caption not in dataset else dataset

def build_dataset(caption, method):
    """
    Generates an artificial dataset from a given caption;
    Currently using random dropout
    :param caption: a string
           method: 'powerset' (all powersets of orig caption) or 'phrases' (just NPs)
    :return: a list of captions (strings)
    """

    # Construct sub-phrases from caption
    if method == 'powerset':
        ds = powersets(caption)
    elif method == 'ordered_subset':
        ds = ordered_subsets(caption)
    elif method == 'NP':
        ds = NPs(caption)
    elif method == 'none' :
        ds = [caption]
    else:
        raise NameError('unspecified method')
    return ds

def get_history_loader(agent, history, vocab, shuffle, num_workers) :
    """
    Returns torch.utils.data.DataLoader for custom coco dataset.
    """
    # initialize in Dataset class
    method = agent.dataset_type
    reduced_dataset = ReducedHistoryDataset(agent, history, method, vocab)
    batch_size = min(len(reduced_dataset), 16)
    data_loader = torch.utils.data.DataLoader(dataset=reduced_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=utils.collate_fn,
                                              drop_last = False)
    return iter(data_loader)
    
def get_reduction_loader(image, vocab, batch_size, caption, method,
                         shuffle, num_workers):
    """
    Returns torch.utils.data.DataLoader for custom coco dataset.
    """
    # build dataset
    ds = build_dataset(caption, method)

    # initialize in Dataset class
    reduced_dataset = ReducedDataset(image, ds, vocab)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=reduced_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=utils.collate_fn,
                                              drop_last = False)
    return data_loader


