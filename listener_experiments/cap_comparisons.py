import sys
sys.path.append("../models/")

import argparse
import os
from model import AdaptiveAgent
from encoder_decoder import DecoderRNN
from utils import Writer, Vocabulary
from utils import coco, choose_similar_images, choose_diff_images, device
import utils
import time
import csv
import numpy as np
import context_loader
import torchvision.transforms as transforms
import torch
import copy

import json
with open("../data/human-human/modelInput.json", "r") as read_file:
    human_input = json.load(read_file)
    gameids = set([ctx['gameid'] for ctx in human_input])

class CapCompWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.num_steps = a.num_steps
        self.batch_size = a.batch_size
        self.debug = a.debug
        if not self.debug :
            self.init_output_csv([[
                'i_iter', 'loss', 'gameid', 'trial_num', 'target_img',
                'control_caption', 'model_caption', 'human_caption',
                'control_score', 'first_score', 'last_score',
            ]])

    def writerow(self, ctx, trial_num, target_img, control_cap, model_cap, human_cap,
                 control_score, first_score, last_score):
        row = [
            ctx['sample_num'], ctx['loss'], ctx['gameid'], trial_num, target_img,
            control_cap, model_cap, human_cap,
            control_score, first_score, last_score
        ]

        if not self.debug :
            super().writerow(row)

def check_weight_eq(m1, mcopy):
    for key in m1.state_dict().keys():
        assert(torch.all(torch.eq(m1.state_dict()[key], mcopy[key])))

def get_context_loader(context_info):
    return context_loader.get_context_loader(
        ctx_type = context_info['context_type'],
        ctx_size=context_info['context_size'],
        num_samples = context_info['num_samples'],
        tag = context_info['context_id']
    )

def construct_context_grid(args) :
    print('constructing grid...')

    # Choose targets
    grid = []
    if args.num_samples > len(gameids) :
        args.num_samples = len(gameids)

    sampled_gameids = np.random.choice(list(gameids), args.num_samples,
                                       replace = False)
#    sampled_gameids = ['5463-bb90cd20-cc2c-428c-9775-d62716accacf'] # amazon/pokadot
    # Loop through, sample contexts w/ desired nesting properties
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

        # Call context loader for targets
        imgs, img_dirs, img_tags = next(get_context_loader(context_info))

        # Choose control
        _, control_dir, _ = utils.sample_img()
        print(control_dir)
        for loss in ['SCE+SKL+SR+LCE', 'SCE+SR+LCE', 'SCE+SKL+LCE', 'SCE+LCE',
                     'SCE+SKL', 'SCE'] :
            grid.append(dict(
                context_info,
                imgs = imgs,
                target_dirs = img_dirs,
                sample_num = i,
                loss = loss,
                control_dir = control_dir,
            ))
    return grid

def get_eval_captions(args, ctx):
    speaker = AdaptiveAgent(args)
    speaker.loss = ctx['loss']
    speaker_decoders = []
    target_caps = []

    # generate control caption & save round 0 speaker model
    print(ctx['control_dir'])
    speaker_decoders.append(copy.deepcopy(speaker.decoder))

    speaker.reset_to_initialization(ctx['target_dirs'])

    speaker.set_context([ctx['control_dir']])
    speaker.set_image(ctx['control_dir'])
    orig_control_cap = speaker.generate_utterance('S0')
    
    # generate initial target captions
    speaker.set_context(ctx['target_dirs'])
    for t, target_img_dir in enumerate(ctx['target_dirs']):
        speaker.set_image(target_img_dir)
        target_caps.append({'gen_cap' : speaker.generate_utterance('S0'),
                            'target' : target_img_dir,
                            'rep_num' : 'pre'})

    # adapt model and get generated cap at each point
    for trial in ctx['speaker_data'] :
        target = utils.get_img_path(trial['targetImg'])
        target_idx = speaker.context.index(target)
        speaker.set_image(target)
        speaker.update_model(trial['trialNum'], trial['msg'])
        target_caps.append({'gen_cap' : speaker.generate_utterance('S0'),
                            'target' : target,
                            'rep_num' : trial['repNum']})
        speaker_decoders.append(copy.deepcopy(speaker.decoder))

    for t, target_img_dir in enumerate(ctx['target_dirs']):
        speaker.set_image(target_img_dir)
        target_caps.append({'gen_cap' : speaker.generate_utterance('S0'),
                            'target' : target_img_dir,
                            'rep_num' : 'post'})
    return target_caps, orig_control_cap, speaker_decoders

def retrieve_eval_caps (target_caps, target) :
    first = [cap['gen_cap'] for cap in target_caps
             if cap['target'] == target and cap['rep_num'] == 'pre']
    last = [cap['gen_cap'] for cap in target_caps
             if cap['target'] == target and cap['rep_num'] == 'post']
    return first[0], last[0]

def retrieve_curr_cap (target_caps, target, rep_num) :
    return [cap['gen_cap'] for cap in target_caps
            if cap['target'] == target and cap['rep_num'] == rep_num][0]

def get_score(listener, img, cap) :
    cap_ids = utils.words_to_ids(cap, listener.vocab)
    return float(listener.S0_score(
        utils.load_image(img).to(device),
        torch.tensor([cap_ids]).to(device),
        len(cap_ids)
    ))

def main_memory(args):
    path = '../data/model_output/cap_comparison.csv'
    writer = CapCompWriter(args, path)
    categories = coco.loadCats(coco.getCatIds())
    listener = AdaptiveAgent(args)
    grid = construct_context_grid(args)
    for ctx in grid:
        print("\nsample_num: {}, loss: {}, getting captions..."
              .format(ctx['sample_num'], ctx['loss']))
        target_caps, control_cap, speaker_decoders = get_eval_captions(args, ctx)
        print(target_caps)
        print("no. speaker decoders: ", len(speaker_decoders))
        for trial in ctx['speaker_data'] :
            print("\n evaluating for trial : ", trial)
            # retrieve model state and vars for this round
            speaker_decoder = speaker_decoders[trial['trialNum']]

            # update decoder weights for this trial
            current_vocab = copy.deepcopy(speaker_decoder.vocab)
            listener.decoder = DecoderRNN(256, 512, current_vocab).to(device)
            listener.decoder.load_state_dict(speaker_decoder.state_dict())

            target_img = utils.get_img_path(trial['targetImg'])
            control_img = ctx['control_dir']

            # we have all the captions but let's just score first and last
            first_cap, last_cap = retrieve_eval_caps(target_caps, target_img)
            curr_cap = retrieve_curr_cap(target_caps, target_img, trial['repNum'])

            control_score = get_score(listener, control_img, control_cap)
            first_score = get_score(listener, target_img, first_cap)
            last_score = get_score(listener, target_img, last_cap)
            human_cap = trial['msg']

            if args.debug: print("\ncontrol: {}\n, target: {}\n, reduced target: {}"
                                 .format(control_string, first_string, last_string))
            if args.debug: print("control {}, target {}, reduced target {}"
                                 .format(control_score, first_score, last_score))
            # save scores
            writer.writerow(ctx, trial['trialNum'], target_img,
                            control_cap, curr_cap, human_cap,
                            control_score, first_score, last_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_dir = '../data/preprocess/'
    parser.add_argument('--encoder_path', type=str, default='f{data_dir}/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='f{data_dir}/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='f{data_dir}/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='f{data_dir}/val2014/', help='directory for resized images')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10, help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--num_workers', type=int, default=0)

    # Expt-specific parameters
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--context_sim_metric', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=60)
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
