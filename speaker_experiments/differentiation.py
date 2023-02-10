# python experiments/communicative_efficiency.py --num_samples 10 --num_images 10 --num_reductions 10
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


class EfficiencyWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.num_steps = a.num_steps
        self.debug = a.debug
        if not self.debug :
            self.init_output_csv([[
                'speaker_model', 'speaker_loss', 'listener_loss', 'context_type',
                'i_iter', 'round_num', 'target', 'context',
                'caption', 'scores', 'num_words', 'learning_rate',
                'num_steps', 'targetScore'
            ]])

    def writerow(self, ctx, round_num, target, caption,
                 scores, num_words, targetScore) :
        row = [ctx['speaker_model'], ctx['speaker_loss'], ctx['listener_loss'], ctx['context_type'], 
               ctx['sample_num'], round_num, target, ctx['dirs'],
               caption, scores.data.cpu().numpy(), num_words, self.learning_rate,
        self.num_steps, targetScore]
        if not self.debug :
            print(row)
            super().writerow(row)

def get_context_loader(context_info):
    return context_loader.get_context_loader(
        ctx_type = context_info['context_type'],
        ctx_size=context_info['context_size'],
        num_samples = context_info['num_samples']
    )

def construct_expt_grid(args) :
   # Loop through, sample contexts w/ desired nesting properties
   grid = []
   for context_type in ['challenge'] : # 'far', 'close', 
       context_info = dict(
           context_type = context_type,
           context_size = args.context_size,
           num_samples = args.num_samples
       )

       # call context loader
       for sample_num in range(args.num_samples) :
           imgs, img_dirs, img_tags = next(get_context_loader(context_info))
           for speaker_loss in ['SCE+SKL+SR+LCE', 'SCE+SKL+SR'] : # 'fixed', 'SCE', 'SCE+SKL', 'LCE', 
               for speaker_model in ['S0', 'S1'] : # 'fixed', 'tied_to_speaker', 'SCE', 'LCE', 
                   grid.append(dict(
                       context_info,
                       speaker_model = speaker_model,
                       speaker_loss = speaker_loss,
                       listener_loss = 'SCE+SKL+SR+LCE',
                       dirs = img_dirs,
                       cats = img_tags,
                       sample_num = sample_num
                   ))
   return grid
    
def main_memory(args):
    path = '../data/model_output/prag_speaker.csv'
    writer = EfficiencyWriter(args, path)

    # init separate speaker/listener models
    speaker = AdaptiveAgent(args)
    listener = AdaptiveAgent(args)
    grid = construct_expt_grid(args)
    utt_store = {}

    for ctx in grid:
        print("\ntype: {}, speaker loss: {}, listener loss: {}, speaker model: {}"
              .format(ctx['context_type'], ctx['speaker_loss'], ctx['listener_loss'], ctx['speaker_model']))

        speaker.loss = ctx['speaker_loss']
        speaker.reset_to_initialization(ctx['dirs'])
        listener.loss = ctx['listener_loss']
        listener.reset_to_initialization(ctx['dirs'])

        # update round-robin style by looping through targets in random order
        for round_num in range(1, args.num_reductions) :
            targets = random.sample(ctx['dirs'], len(ctx['dirs']))
            for target in targets :
                print('round {}, target {}'.format(round_num, target))

                # Set up for new round
                cap_key = "{}-{}-{}-{}-{}".format(ctx['speaker_loss'], ctx['speaker_model'], ctx['sample_num'], 
                                               target, round_num)
                target_idx = ctx['dirs'].index(target)
                speaker.set_image(target)
                listener.set_image(target)

                # Generate caption and update if this is first time
                if cap_key in utt_store :
                    cap = utt_store[cap_key]
                    str_cap = utils.ids_to_words(cap, speaker.vocab)
                    print('found {} in utt_store!'.format(str_cap))
                else :
                    cap = np.array(speaker.generate_utterance(ctx['speaker_model'], as_string = False))
                    print('regular: ', speaker.generate_utterance('S0'))
                    print('prag: ', speaker.generate_utterance('S1'))
                    str_cap = utils.ids_to_words(cap, speaker.vocab)
                    utt_store[cap_key] = cap
#                    print('adding {} to utt_store'.format(cap_key))
                    if ctx['speaker_loss'] != 'fixed' :
                        speaker.update_model(round_num, str_cap)

                # evaluate caption & update listener models as relevent
                scores = listener.L0_score(np.expand_dims(cap, axis=0), ctx['dirs'])
                if not ctx['listener_loss'] in ['fixed', 'tied_to_speaker']:
                    listener.update_model(round_num, str_cap)
                elif ctx['listener_loss'] == 'tied_to_speaker':
                    listener.decoder.load_state_dict(speaker.decoder.state_dict())
                    
                # Write out
                writer.writerow(ctx, round_num, target, str_cap, scores, len(cap), 
                                scores[0][target_idx].data.cpu().numpy())

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
    parser.add_argument('--context_sim_metric', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_reductions', type=int, default = 5)
    
    # Important hyper-params
    parser.add_argument('--num_steps', type=int, default=6,
                        help='number of steps to take in each epoch')
    parser.add_argument('--reduction_history_window', type=str, default='complete',
                        help='how far back to look')
    parser.add_argument('--ds_type', type=str, default = 'NP')
    parser.add_argument('--num_rehearsals', type=int, default = 50)
    parser.add_argument('--loss', type=str, default='SCE+SKL+SR+LCE')
    parser.add_argument('--speaker_KL_weight', type=float, default=.5)
    parser.add_argument('--speaker_CE_weight', type=float, default=1)
    parser.add_argument('--speaker_rehearsal_weight', type=float, default=.3)   
    parser.add_argument('--listener_KL_weight', type=float, default=.5)
    parser.add_argument('--listener_CE_weight', type=float, default= 1)
    parser.add_argument('--listener_rehearsal_weight', type=float, default=.5) 
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    args = parser.parse_args()

    print(args)
    main_memory(args)
