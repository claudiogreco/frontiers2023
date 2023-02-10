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
with open("../data/human-human-easy/humanHumanEasyContext.json", "r") as read_file:
    human_input = json.load(read_file)
    gameids = set([ctx['gameid'] for ctx in human_input])

class EfficiencyWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.num_steps = a.num_steps
        self.debug = a.debug
        if not self.debug :
            self.init_output_csv([
                [ 'context_type', 'i_iter', 'loss', 'ds_type', 'handleOOV',
                  'train_gameid', 'test_gameid', 'train_contextid', 'test_contextid',
                  'trial_num', 'rep_num', 'target',
                  'caption', 'scores', 'num_words', 'learning_rate', 'num_steps',
                  'target_score', 'cap_score', 'correct']
            ])

    def writerow(self, ctx, game, caption, scores, num_words, targetScore, cap_score, correct, test_context_info) :
        row = [ctx['train_context_info']['context_type'], ctx['sample_num'], ctx['loss'], ctx['ds_type'], ctx['handleOOV'],
               ctx['train_context_info']['gameid'], test_context_info['gameid'],
               ctx['train_context_info']['context_id'], test_context_info['context_id'],
               game['trialNum'], game['repNum'],
               game['targetImg'], caption, scores, num_words, self.learning_rate, self.num_steps, targetScore, cap_score, correct]
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
    if args.num_samples > len(gameids) :
        args.num_samples = len(gameids)
    train_gameids = np.random.choice(list(gameids), args.num_samples, replace = False)
    print('building contexts for ', len(train_gameids))
    for i,train_gameid in enumerate(train_gameids):
        test_gameids = [gameid for gameid in gameids if gameid != train_gameid]
        assert train_gameid not in list(test_gameids)
        assert len(gameids) == (len(test_gameids) + 1)
        train_speaker_data = list(filter(lambda x : x['gameid'] == train_gameid, human_input))
        train_context_info = dict(
            context_type = 'easy',
            context_size = args.context_size,
            num_samples = 1,
            context_id = train_speaker_data[0]['context_id'],
            speaker_data = train_speaker_data,
            gameid = train_gameid
        )
        # call context loader
        train_imgs, train_img_dirs, train_img_tags = next(get_context_loader(train_context_info))
        test_context_infos = []
        test_dirs = []
        for test_gameid in test_gameids:
            test_speaker_data = list(filter(lambda x : x['gameid'] == test_gameid, human_input))
            test_context_info = dict(
                context_type = 'easy',
                context_size = args.context_size,
                num_samples = 1,
                context_id = test_speaker_data[0]['context_id'],
                speaker_data = test_speaker_data,
                gameid = test_gameid
            )
            if test_context_info['context_id'] != train_context_info['context_id'] :
                test_context_infos.append(test_context_info)
                test_imgs, test_img_dirs, test_img_tags = next(get_context_loader(test_context_info))
                test_dirs.append(test_img_dirs)
                        
        for loss in ['SCE+SKL+SR+LCE', 'SCE+SR+LCE', 'fixed'] :
            grid.append(dict(
                train_context_info = train_context_info,
                test_context_infos = test_context_infos,
                train_imgs = train_imgs,
                train_dirs = train_img_dirs,
                train_cats = train_img_tags,
                test_dirs = test_dirs,
                sample_num = i,
                loss = loss,
                handleOOV = False,
                ds_type = 'NP'
            ))
    return grid

def main(args):
    path = '../data/model_output/listener_cat_forgetting.csv'
    writer = EfficiencyWriter(args, path)
    # init listener model 
    listener = AdaptiveAgent(args)
    listener.reduction_history_window = 'complete'
    grid = construct_context_grid(args) 
    
    for ctx in grid:
        print("\n------train gameid: {}, sample_num: {}, loss: {}"
              .format(ctx['train_context_info']['gameid'],
                      ctx['sample_num'], ctx['loss']))
        
        # train: reduce with human 
        # reset speaker and listener to pretrained setting
        listener.reset_to_initialization(ctx['train_dirs'])  # setting context
        listener.loss = ctx['loss']
        listener.dataset_type = ctx['ds_type']
        listener.history = []  # TODO: redundant ? 
        train_target = None
        for datum in ctx['train_context_info']['speaker_data']:
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
            scores = listener.L0_score(np.expand_dims(id_cap, axis=0), ctx['train_dirs'])
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
            
            # Update models as relevant
            if ctx['loss'] != 'fixed' :
                listener.update_model(trial_num, raw_cap)
                
            if rep_num == 5 :
                writer.writerow(ctx, datum, raw_cap, scores, len(raw_cap),
                                target_score, cap_score, correct, {'gameid' : None, 'context_id' : None})

            # test on new human 
            if args.debug: print("\nTESTING!")
            
        for j, test_context_info in enumerate(ctx['test_context_infos']):
            print("\ntest context: {}".format(j))
            listener.set_context(ctx['test_dirs'][j])  # set context to test dirs, BUT don't reset weights
            listener.history = []
            # TODO: should we reset vocab? 
            #for datum in ctx['test_context_info']['speaker_data']:
            for datum in test_context_info['speaker_data']:
                rep_num = datum['repNum']
                if rep_num > 0: 
                    break
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
                scores = listener.L0_score(np.expand_dims(id_cap, axis=0), ctx['train_dirs'])
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
                                target_score, cap_score, correct, test_context_info)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_dir = '../data/preprocess/'
    parser.add_argument('--device', type='str', default='cpu', help='device')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--encoder_path', type=str, default='f{data_dir}/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='f{data_dir}/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='f{data_dir}/vocab.pkl', help='path for vocabulary wrapper')
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
    main(args)
    
