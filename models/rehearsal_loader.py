import torch.utils.data as data

import torch
import torch.utils.data as data

import numpy as np
import utils

class Rehearsals(data.Dataset):
    def __init__(self, vocab):
        self.vocab = vocab

    def __getitem__(self, i):
        rehearsal = utils.choose_diff_images(1, self.vocab, cats_to_avoid=[])[0]
        path, _, caption = rehearsal
        image = utils.load_image(path, for_batching=True)
        return image, torch.Tensor(caption)

    def __len__(self):
        return len(utils.coco.getImgIds())

def get_rehearsal_loader(num_rehearsals, vocab,
                         shuffle=True, num_workers=1):
    return iter(torch.utils.data.DataLoader(
        dataset=Rehearsals(vocab),
        batch_size=num_rehearsals,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
        drop_last=True
    ))

