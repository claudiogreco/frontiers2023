# python experiments/communicative_efficiency.py --num_samples 10 --num_images 10 --num_reductions 10
import sys
sys.path.append("../models/")

import argparse
import os
from model import AdaptiveAgent

import utils
from utils import Writer, Vocabulary
from utils import coco, device
import numpy as np
import context_loader
import random
from random import shuffle
import torch

import json
with open("../data/model-as-listener/listenerLesionInput.json", "r") as read_file:
    human_input = json.load(read_file)
    gameids = set([ctx['gameid'] for ctx in human_input])

class LossWriter(Writer):
    def __init__(self, a, save_file):
        super().__init__(self, save_file)
        self.debug = a.debug
        if not self.debug:
            self.init_output_csv([
            ['i_iter', 'loss_terms', 'trial_num', 'step', 'loss_term', 'loss_term_weight', 'loss',
             'scaled_loss']
            ])
    def writerow(self, i_iter, loss_terms, trial_num, step, loss_term, loss_term_weight, loss, scaled_loss):
        row = [i_iter, loss_terms, trial_num, step, loss_term, loss_term_weight, loss, scaled_loss]
        if not self.debug:
            super().writerow(row)

class EfficiencyWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.num_steps = a.num_steps
        self.debug = a.debug
        if not self.debug :
            self.init_output_csv([
                [ 'context_type', 'i_iter', 'loss', 'ds_type', 
                  'gameid', 'trial_num', 'rep_num', 'target',
                  'caption', 'scores', 'num_words', 'learning_rate', 'num_steps',
                  'target_score', 'cap_score', 'correct']
            ])

    def writerow(self, ctx, game, caption, scores, num_words,
                 targetScore, cap_score, correct) :
        row = [ctx['context_type'], ctx['sample_num'], ctx['loss'], ctx['ds_type'], 
               game['gameid'], game['trialNum'], game['repNum'],
               game['targetImg'], caption, scores, num_words, self.learning_rate,
               self.num_steps, targetScore, cap_score, correct]
        print('writing row')
        print(row)
        if not self.debug :
            super().writerow(row)

def get_context_loader(context_info):
    return context_loader.get_context_loader(
        ctx_type = context_info['context_type'],
        ctx_size=context_info['context_size'],
        num_samples = context_info['num_samples'],
        tag = context_info['context_id']
    )

def construct_context_grid(args) :
    print('constructing grid...')

    # Loop through, sample contexts w/ desired nesting properties
    grid = []
    if args.num_samples >= len(gameids) :
        args.num_samples = len(gameids)
        
    sampled_gameids = np.random.choice(list(gameids), args.num_samples, replace = False)
       
    for i, gameid in enumerate(sampled_gameids) :
        speaker_data = list(filter(lambda x : x['gameid'] == gameid, human_input))
        
        context_info = dict(
            context_type = 'challenge',
            context_size = args.context_size,
            num_samples = 1,
            context_id = speaker_data[0]['context_id'],
            speaker_data = speaker_data,
            gameid = gameid
        )

        # call context loader
        imgs, img_dirs, img_tags = next(get_context_loader(context_info))
        for loss in ['SCE+SKL+SR+LCE', 'SCE+SKL+LCE', 'SCE+SKL+SR', 'fixed'] :
            for augmentation in ['NP', 'none'] :
                grid.append(dict(
                    context_info,
                    imgs = imgs,
                    dirs = img_dirs,
                    cats = img_tags,
                    sample_num = i,
                    loss = loss,
                    handleOOV = False,
                    ds_type = augmentation
                ))
    return grid

def main_memory(args):
    path = '../data/model_output/listener_lesions.csv'
    writer = EfficiencyWriter(args, path)

    # init listener model
    listener = AdaptiveAgent(args)
    listener.reduction_history_window = 'complete'
    grid = construct_context_grid(args)

    for ctx in grid:
        print("\n------gameid: {}, sample_num: {}, loss: {}, handle-oov: {}"
              .format(ctx['gameid'], ctx['sample_num'], ctx['loss'], ctx['handleOOV']))

        # reset speaker & listener to pretrained settings
        listener.reset_to_initialization(ctx['dirs'])  # setting context
        listener.loss = ctx['loss']
        listener.dataset_type = ctx['ds_type']
        listener.history = []  
        for datum in ctx['speaker_data'] :
            rep_num = datum['repNum']
            trial_num = datum['trialNum']
            target = utils.get_img_path(datum['targetImg'])
            raw_cap = datum['msg']
            listener.trial_num = trial_num
            listener.sample_num = ctx['sample_num']

            # Set up for new round
            print('\nround {}, target {}, msg {}'.format(
                rep_num, utils.get_id_from_path(target), raw_cap
            ))
            listener.set_image(target)

            id_cap = (listener.process_human_caption(raw_cap) if ctx['handleOOV']
                      else utils.words_to_ids(raw_cap, listener.vocab))
            scores = listener.L0_score(np.expand_dims(id_cap, axis=0), ctx['dirs'])
            cap_score = listener.S0_score(utils.load_image(target).to(device),
                                          torch.tensor([id_cap]).to(device),
                                          len(id_cap))

            # Write out
            scores = scores.data.cpu().numpy()[0]
            target_idx = listener.context.index(listener.raw_image)
            target_score = scores[target_idx]
            best = list(scores).index(max(scores))
            correct = best == target_idx
            if args.debug: print(list(map(lambda x: np.exp(x), scores)))
            if args.debug: print('{}, model says: {}, target actually: {}'.format(correct, best, target_idx))
            if args.debug: print('accuracy in real game: {}'.format(datum['correct']))

            writer.writerow(ctx, datum, raw_cap, scores, len(raw_cap),
                            target_score, cap_score, correct)

            # Update models as relevant
            if ctx['loss'] != 'fixed' :
                listener.update_model(trial_num, raw_cap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_dir = '../data/preprocess/'
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--encoder_path', type=str, default=f'{data_dir}/encoder-5-3000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default=f'{data_dir}/decoder-5-3000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default=f'{data_dir}/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='f{data_dir}/val2014', help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10, help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--num_workers', type=int, default=0)

    # Expt-specific parameters
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--context_sim_metric', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda', type=int, default=0)

    # Important hyper-params
    parser.add_argument('--reduction_history_window', type=str, default='complete',
                        help='how far back to look')
    parser.add_argument('--ds_type', type=str, default='NP',
                        help='type of dataset')
    parser.add_argument('--loss', type=str, default='SCE')
    parser.add_argument('--speaker_KL_weight', type=float, default=.5)
    parser.add_argument('--speaker_CE_weight', type=float, default=1)
    parser.add_argument('--speaker_rehearsal_weight', type=float, default=.3)
    parser.add_argument('--listener_KL_weight', type=float, default=.5)
    parser.add_argument('--listener_rehearsal_weight', type=float, default=.5)
    parser.add_argument('--listener_CE_weight', type=float, default=.1)
    parser.add_argument('--num_rehearsals', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_steps', type=int, default=6, help='number of steps to take')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print(args)
    main_memory(args)
