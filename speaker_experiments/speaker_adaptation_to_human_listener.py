import sys
sys.path.append("../models/")

import argparse
import os
from model import AdaptiveAgent
import utils
from utils import Writer, Vocabulary
from utils import coco, choose_similar_images, choose_diff_images
import numpy as np
import context_loader
import random
from random import shuffle
import torch

import json
with open("../data/model-as-speaker/speakerLesionInput.json", "r") as read_file:
    human_input = json.load(read_file)
    gameids = set([ctx['gameid'] for ctx in human_input])

class EfficiencyWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.num_steps = a.num_steps
        self.debug = a.debug
        if not self.debug :
            self.init_output_csv([[
                'loss', 'context_type', 'ds_type', 'cost_weight', 'speaker_model',
                'sample_num', 'gameid', 'trial_num', 'rep_num', 'target', 'context',
                'caption', 'num_words', 'learning_rate', 'num_steps'
            ]])

    def writerow(self, ctx, datum, round_num, target, caption, num_words) :
        row = [ctx['loss'], ctx['context_type'], ctx['ds_type'], ctx['cost_weight'], ctx['speaker_model'],
               ctx['sample_num'], datum['gameid'], datum['trialNum'], datum['repNum'], target, ctx['dirs'],
               caption, num_words, self.learning_rate, self.num_steps]
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
    sampled_gameids = np.random.choice(list(gameids), args.num_samples,
                                       replace = False)
    for i, gameid in enumerate(sampled_gameids) :
        speaker_data = list(filter(lambda x : x['gameid'] == gameid, human_input))
        context_info = dict(
            context_type = 'easy',
            context_size = args.context_size,
            num_samples = 1,
            context_id = speaker_data[0]['context_id'],
            speaker_data = speaker_data,
            gameid = gameid
        )

        # call context loader
        imgs, img_dirs, img_tags = next(get_context_loader(context_info))
        for cost_weight in [0.5, 0.75, 1, 5, 10] :
            for speaker_model in ['S0_with_cost'] :
                grid.append(dict(
                        context_info,
                        imgs = imgs,
                        dirs = img_dirs,
                        cats = img_tags,
                        sample_num = i,
                        loss = 'SCE+SKL+SR+LCE',
                        speaker_model = speaker_model,
                        cost_weight = cost_weight,
                        ds_type = 'none'
                    ))
        for ds_type in ['NP', 'none'] :
            grid.append(dict(
                context_info,
                imgs = imgs,
                dirs = img_dirs,
                cats = img_tags,
                sample_num = i,
                loss = 'SCE+SKL+SR+LCE',
                speaker_model = 'S0',
                cost_weight = 0,
                ds_type = ds_type
            ))
    return grid

def main(args):
    path = '../data/model_output/speaker_lesions.csv'
    writer = EfficiencyWriter(args, path)
    speaker = AdaptiveAgent(args)
    grid = construct_context_grid(args)

    for ctx in grid:
        print("\n------gameid: {}, sample_num: {}, loss: {}, ds_type: {}, speaker_model: {}, cost_weight: {}"
            .format(ctx['gameid'], ctx['sample_num'], ctx['loss'], ctx['ds_type'],
                    ctx['speaker_model'], ctx['cost_weight']))

        speaker.loss = ctx['loss']
        speaker.reset_to_initialization(ctx['dirs'])

        speaker.dataset_type = ctx['ds_type']
        speaker.context_type = ctx['context_type']
        speaker.cost_weight = ctx['cost_weight']
        # simulate round-robin style by looping through targets in random order
        for datum in ctx['speaker_data'] :
            rep_num = datum['repNum']
            trial_num = datum['trialNum']
            target = utils.get_img_path(datum['targetImg'])
            print(target)
            speaker.trial_num = trial_num
            speaker.sample_num = ctx['sample_num']
            speaker.set_image(target)
            
            cap = speaker.generate_utterance(ctx['speaker_model'], as_string = True)
            cap = utils.ids_to_words(utils.words_to_ids(cap, speaker.vocab), speaker.vocab)

            if cap[:7] == '<start>' :
                cap = cap[8:-6]

            print('\nround {}, target {}, msg {}'.format(
                rep_num, utils.get_id_from_path(target), cap
            ))
            
            if datum['correct'] == True :
                print('training')
                speaker.update_model(trial_num, cap)

            writer.writerow(ctx, datum, trial_num, target, cap, len(cap))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_dir = '../data/preprocess'
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--encoder_path', type=str, default=f'{data_dir}/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default=f'{data_dir}/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default=f'{data_dir}/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default=f'{data_dir}/val2014', help='directory for resized images')

    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10, help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--num_workers', type=int, default=0)

    # Expt-specific parameters
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--use_feedback', type=bool, default=False)
    parser.add_argument('--context_sim_metric', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--debug', action='store_true')

    # Important hyper-params
    parser.add_argument('--num_reductions', type=int, default=8, help='# times to reduce')
    parser.add_argument('--ds_type', type=str, default='powerset', help='type of dataset')
    parser.add_argument('--loss', type=str, default='SCE')
    parser.add_argument('--speaker_KL_weight', type=float, default=.5)
    parser.add_argument('--speaker_CE_weight', type=float, default=1)
    parser.add_argument('--speaker_rehearsal_weight', type=float, default=1)
    parser.add_argument('--listener_KL_weight', type=float, default=.5)
    parser.add_argument('--listener_CE_weight', type=float, default=.5)
    parser.add_argument('--listener_rehearsal_weight', type=float, default=1)
    parser.add_argument('--reduction_history_window', type=str, default='complete')
    parser.add_argument('--num_rehearsals', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_steps', type=int, default=8, help='number of steps to take')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print(args)
    main(args)
