import numpy as np

import torch
import torch.utils.data as data

from utils import coco
import utils

import json

with open("data/preprocess/coco_contexts_hard.json", "r") as read_file:
    coco_contexts_hard = json.load(read_file)
    possible_ctx_ids_hard = [ctx['cluster_ids'] for ctx in coco_contexts_hard]

with open("data/preprocess/coco_contexts_easy.json", "r") as read_file:
    coco_contexts_easy = json.load(read_file)
    possible_ctx_ids_easy = [ctx['cluster_ids'] for ctx in coco_contexts_easy]

with open("data/preprocess/adaptation_2_unknown_2_known.json", "r") as read_file:
    adaptation_similar_random_contexts = json.load(read_file)
    adaptation_similar_random_contexts_cluster_ids2contexts = {ctx['cluster_ids']: ctx for ctx in
                                                                 adaptation_similar_random_contexts}
    possible_ctx_ids_adaptation_similar_random_contexts = [ctx['cluster_ids'] for ctx in adaptation_similar_random_contexts]

class Contexts(data.Dataset):
    def __init__(self, ctx_type, ctx_size, num_samples, tag=None, unk_obj=None, sequential=False):
        """
        Args:
            ctx_type: (far, close, challenge)
            ctx_size: number of images in context
            num_samples: number of contexts we will return
            tag : optionally specify a particular cluster (for 'challenge' type)
                  or coco category (for 'same' type)
        """
        self.ctx_type = ctx_type
        self.ctx_size = ctx_size
        self.num_samples = num_samples
        self.tag = tag
        self.unk_obj = unk_obj
        self.sequential = sequential
        self.current_context = 0

        if self.ctx_type == "challenge":
            if num_samples is None:
                self.num_samples = len(possible_ctx_ids_hard)
        elif self.ctx_type == "easy":
            if num_samples is None:
                self.num_samples = len(possible_ctx_ids_easy)
        elif self.ctx_type == "adaptation_similar_random_contexts":
            if num_samples is None:
                self.num_samples = len(possible_ctx_ids_adaptation_similar_random_contexts)

        print("Number of samples: {}".format(self.num_samples))

        if self.sequential:
            print("The dataset is sequential.")

        assert(isinstance(tag, str) if ctx_type == 'same' else True)
        assert(tag >= 0 and tag < 100 if tag and ctx_type == 'challenge' else True)

    def sample_img_from_tag(self, img_tag):
        """
        Samples img (and meta-data) from provided coco category tag
        """
        cat_id = coco.getCatIds(catNms=img_tag)
        img_id = np.random.choice(coco.getImgIds(catIds=cat_id), 1)
        img_path = utils.get_img_path(coco.loadImgs(int(img_id))[0]['file_name'])
        return utils.load_image(img_path), img_path, img_tag

    def get_ctx_from_tag(self, ctx_tag):
        """
        Retrieves requested context from coco_contexts.json or coco_contexts_easy.json
        """
        if self.ctx_type == "challenge":
            coco_contexts = coco_contexts_hard
        elif self.ctx_type == "easy":
            coco_contexts = coco_contexts_easy
        elif self.ctx_type == "adaptation_similar_random_contexts":
            coco_contexts = adaptation_similar_random_contexts

        if self.unk_obj is not None:
            print("Retrieving contexts for: {}".format(self.unk_obj))
            coco_contexts = [x for x in coco_contexts if x["unknown_label"] == self.unk_obj]
            for i in range(len(coco_contexts)):
                coco_contexts[i]["cluster_ids"] = i

        ctx = next(filter(lambda x: x['cluster_ids'] == ctx_tag, coco_contexts))
        filenames = ctx['neighbor_names'][:self.ctx_size]
        paths = [utils.get_img_path(name) for name in filenames]
        tags = ['custom' + str(ctx_tag) for i in range(self.ctx_size)]
        imgs = [utils.load_image(path) for path in paths]
        return imgs, paths, tags, ctx

    def sample_context(self):
        """
        Samples from possible contexts according to class-level criteria
        """
        if self.ctx_type == 'easy':
            tag = np.random.choice(possible_ctx_ids_easy) if self.tag is None else self.tag
            imgs,paths,tags, ctx = self.get_ctx_from_tag(tag)
            if self.ctx_size > len(paths):
                tags_pool = np.random.choice(possible_ctx_ids_easy, int(np.ceil((self.ctx_size-5)/5)))
                for tag in tags_pool:
                    imgs2, paths2, tags2, ctx = self.get_ctx_from_tag(tag)
                    imgs.extend(imgs2)
                    paths.extend(paths2)
                    tags.extend(tags2)
                imgs = torch.cat(imgs[:self.ctx_size])
                paths = paths[:self.ctx_size]
                tags = tags[:self.ctx_size]
            return imgs, paths, tags, ctx
        elif self.ctx_type == 'challenge':
            tag = np.random.choice(possible_ctx_ids_hard) if self.tag is None else self.tag
            imgs,paths,tags, ctx = self.get_ctx_from_tag(tag)
            if self.ctx_size > len(paths):
                tags_pool = np.random.choice(possible_ctx_ids_hard, int(np.ceil((self.ctx_size-5)/5)))
                for tag in tags_pool:
                    imgs2, paths2, tags2 = self.get_ctx_from_tag(tag)
                    imgs.extend(imgs2)
                    paths.extend(paths2)
                    tags.extend(tags2)
                imgs = torch.cat(imgs[:self.ctx_size])
                paths = paths[:self.ctx_size]
                tags = tags[:self.ctx_size]
            return imgs, paths, tags, ctx
        elif self.ctx_type == 'adaptation_similar_random_contexts':
            if self.sequential:
                tag = self.current_context
                self.current_context += 1
                if self.current_context == self.num_samples:
                    self.current_context = 0
            else:
                tag = np.random.choice(possible_ctx_ids_adaptation_similar_random_contexts) if self.tag is None else self.tag
            imgs,paths,tags, ctx = self.get_ctx_from_tag(tag)
            if self.ctx_size > len(paths):
                tags_pool = np.random.choice(possible_ctx_ids_adaptation_similar_random_contexts, int(np.ceil((self.ctx_size - 5) / 5)))
                for tag in tags_pool:
                    imgs2, paths2, tags2 = self.get_ctx_from_tag(tag)
                    imgs.extend(imgs2)
                    paths.extend(paths2)
                    tags.extend(tags2)
                imgs = torch.cat(imgs[:self.ctx_size])
                paths = paths[:self.ctx_size]
                tags = tags[:self.ctx_size]
            return imgs, paths, tags, ctx

        elif self.ctx_type == 'close':
            tag = self.tag if self.tag else np.random.choice(utils.get_cat_names(), 1)
            return [self.sample_img_from_tag(img_tag) for img_tag
                    in [tag for i in range(self.ctx_size)]]
        elif self.ctx_type == 'far':
            all_cat_names = utils.get_cat_names()
            return [self.sample_img_from_tag(img_tag) for img_tag
                    in np.random.choice(all_cat_names, self.ctx_size, replace=True)]
        else :
            raise Exception('unknown ctx_type: {}'.format(self.ctx_type))

    def __getitem__(self, i):
        """
        Sample a context according to criteria in class-level
        """
        #context = self.sample_context()
        #imgs, img_dirs, tags = zip(*context)
        imgs, img_dirs, tags, ctx = self.sample_context()
        return imgs, list(img_dirs), list(tags), ctx

    def __len__(self):
        return self.num_samplesnum_samples


def get_context_loader(ctx_type, ctx_size, num_samples, tag=None, unk_obj=None, sequential=False):
    contexts = Contexts(ctx_type, ctx_size, num_samples, tag=tag, unk_obj=unk_obj, sequential=sequential)
    return iter(contexts), contexts.num_samples

def get_context_loader_by_filename(filename, ctx_size, num_samples, tag=None, sequential=False):
    return iter(Contexts(filename, ctx_size, num_samples, tag=tag, sequential=sequential))
