import tornado.ioloop
import tornado.web

import argparse
import numpy as np
import time
import sys
import csv
import json
import os
import gc
import torch
import utils
import shutil
import copy

from model import AdaptiveAgent
from utils import Writer, Vocabulary, words_to_ids, get_img_path
from tornado.escape import json_decode
from data_loader import build_dataset

def initialize_agent() :
    # Add a bunch of in this kind of clumsy way pasted from model expts...
    parser = argparse.ArgumentParser()
    data_dir = '../data/preprocess'
    parser.add_argument('--checkpoint', type=str, default = True)
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--encoder_path', type=str,
                        default=f'{data_dir}/encoder-5-3000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str,
                        default=f'{data_dir}/decoder-5-3000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--model_path', type=str,
                        default='.' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default=f'{data_dir}/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default=f'{data_dir}/val2014',
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10,
                        help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda', type=int, default=0)

    # Important hyperparams
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
    parser.add_argument('--listener_CE_weight', type=float, default=.1)
    parser.add_argument('--listener_rehearsal_weight', type=float, default=.5) 
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0005)

    args = parser.parse_args()
    return AdaptiveAgent(args)

# this global agent is intended to fix massive latency in initializing agent
# for the first time, but note that it's a really bad idea in general
# any asyncronous process will cause blurring/interference between different games
shared_agent = initialize_agent()
print('agent initialized with loss' + shared_agent.loss)

game_histories = {}
orig_captions = {}

def load_agent(request_args) :
    # re-init shared agent with args
    round_num = request_args['roundNum']
    gameid = request_args['gameid']

    shared_agent.gameid = gameid
    shared_agent.round_num = round_num
    
    # initialize history with initial caps
    if gameid not in game_histories:
        game_histories[gameid] = []

    # load pre-trained model on first round, otherwise load previous saved weights
    if round_num == 0 :
        shutil.copy(shared_agent.model_path + '/decoder-5-3000.pkl',
                    shared_agent.model_path + '/decoder-{}.ckpt'.format(gameid))
    decoder_path =  shared_agent.model_path + '/decoder-{}.ckpt'.format(gameid)

    shared_agent.reset_to_initialization(request_args['context'],
                                         alt_decoder=torch.load(decoder_path))
    shared_agent.set_image(request_args['target'])
    shared_agent.history = copy.deepcopy(game_histories[shared_agent.gameid])
    if gameid not in orig_captions :
        orig_captions[gameid] =  [
            (img, shared_agent.S0_sample(img, beam_sample = True))
            for img in request_args['context']
        ]
    shared_agent.orig_captions = copy.deepcopy(orig_captions[shared_agent.gameid])

    return shared_agent

class ActionHandler(tornado.web.RequestHandler):
    def prepare(self):
        """ this is a RequestHandler class method; called on packet"""
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.json_args = json.loads(self.request.body)
            self.json_args['target'] = get_img_path(self.json_args['target'])
            # Sort because apparently the updating depends on context order?
            self.json_args['context'] = sorted([get_img_path(img) for img
                                                in self.json_args['context']])
        else:
            self.json_args = None

    def speak(self):
        """ sends a caption for the specified image"""
        if self.json_args['roundNum'] > 0 :
            if self.json_args['prevCorrect'] :
                # if correct, update based on results from previous round
                self.update(self.json_args['gameid'])
            else :
                # otherwise, *remove from history (so doesn't get hit by rehearsal)
                game_histories[self.json_args['gameid']].pop()
            
        # pick utterance
        agent = load_agent(self.json_args)
        cap = agent.generate_utterance('S0', as_string = True)
        cap = utils.ids_to_words(utils.words_to_ids(cap, agent.vocab), agent.vocab)
       
        if cap[:7] == '<start>' :
            cap = cap[8:-6]

        # update history for next round and return response
        self.extend_history(agent, self.json_args['target'], cap)
        self.write(cap)
        
    def listen(self) :
        # update model based on results of previous round...
        if self.json_args['roundNum'] > 0 :
            self.update(self.json_args['gameid'])

        agent = load_agent(self.json_args)
        
        # pick response 
        id_cap = words_to_ids(self.json_args['caption'], agent.vocab)
        scores = agent.L0_score(np.expand_dims(id_cap, axis=0), agent.context)    
        scores = scores.data.cpu().numpy()[0]
        best = list(scores).index(max(scores))

        # update history for next round and return response
        self.extend_history(agent, self.json_args['target'], self.json_args['caption'])
        self.write(agent.context[best].split('/')[-1])

    def extend_history(self, agent, image, caption) :
        new_round = {
            'gameid' : self.json_args['gameid'],
            'roundNum' : self.json_args['roundNum'],
            'target': image, 
            'context' : self.json_args['context'],
            'cap': caption
        }
        game_histories[agent.gameid].append(new_round)

    def update(self, gameid) :
        # remove prev_round for updating (so it's the history *at the prev round*)
        prev_round = game_histories[gameid].pop()
        prev_agent = load_agent(prev_round)
        prev_agent.update_model(prev_round['roundNum'], prev_round['cap'])

        # add it back to history for future rounds
        game_histories[gameid].append(prev_round)

        # precompute history captions so we don't have to do it again on every step
        for reduced_cap in build_dataset(prev_round['cap'], prev_agent.dataset_type) :
            orig_captions[prev_agent.gameid].append((prev_round['target'], reduced_cap))
        
    def post(self):
        print('received request from gameid=', self.json_args['gameid'])
        print(self.json_args)
        action = self.json_args['action']
        if action == 'speak' :
            self.speak()
        elif action == 'listen' :
            self.listen()
        elif action == 'update' :
            self.update(load_agent(self.json_args))
        else :
            raise Exception('unknown action', args.action)
        
def make_app():
    return tornado.web.Application([
        (r"/request_model_action", ActionHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(5005)
    print('starting loop')
    tornado.ioloop.IOLoop.current().start()
