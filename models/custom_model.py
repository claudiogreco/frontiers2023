import torch
import torch.nn as nn
import torch.nn.functional as F


import pickle
import numpy as np
import copy

import utils
from utils import device, Vocabulary
from encoder_decoder import EncoderCNN, DecoderRNN
import operator


class AdaptiveAgent():
    init_encoder_params = None
    init_decoder_params = None

    def __init__(self, args):
        self.gameid = args.gameid if 'gameid' in args else 'testing'
        self.checkpoint = args.checkpoint if 'checkpoint' in args else False
        self.encoder_path = args.listener_encoder_path
        self.decoder_path = args.listener_decoder_path
        if not self.init_decoder_params or not self.init_encoder_params:
            if torch.cuda.is_available():
                self.init_encoder_params = torch.load(self.encoder_path)
                self.init_decoder_params = torch.load(self.decoder_path)
            else:
                self.init_encoder_params = torch.load(self.encoder_path, map_location=torch.device("cpu"))
                self.init_decoder_params = torch.load(self.decoder_path, map_location=torch.device("cpu"))
        self.device = device
        self.topk = 20
        self.cost_weight = 0.1
        self.vocab_path = args.vocab_path
        self.context = None
        self.context_type = None

        with open(args.vocab_path, 'rb') as f:
            # Load vocabulary wrapper
            self.vocab = pickle.load(f)
            Vocabulary()

        # Initialize encoder with pre-trained params
        self.generating_encoder = EncoderCNN(256).to(device).eval()
        self.generating_encoder.load_state_dict(self.init_encoder_params)

        # Initialize decoder with pre-trained params, allocating tensors for oov words
        self.decoder = DecoderRNN(256, 512, self.vocab).to(device)
        self.orig_decoder = DecoderRNN(256, 512, self.vocab).to(device)

        # load the model with this initialization
        self.orig_decoder.load_state_dict(self.init_decoder_params)
        self.decoder.load_state_dict(self.init_decoder_params)
        self.debug = args.debug

    def reset_to_initialization(self, context, alt_decoder=None):
        """
        helper for running many iterations without having to re-initialize the class
        """

        # Explicitly reset the *dimensionality* of the tensors we extended w/ oov
        self.decoder.embed.weight = copy.deepcopy(
            nn.Parameter(self.orig_decoder.embed.weight))
        self.decoder.linear.weight = copy.deepcopy(
            nn.Parameter(self.orig_decoder.linear.weight))
        self.decoder.linear.bias = copy.deepcopy(
            nn.Parameter(self.orig_decoder.linear.bias))

        # Reload original vocab
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Reset to the original weights
        self.decoder.load_state_dict(
            copy.deepcopy(self.orig_decoder.state_dict()))

        # Set up target & context if passed
        self.set_context(context)

        # Initialize history w/ "pseudo-counts" of initial captions
        self.history = []
        self.orig_captions = [(img, self.S0_sample(img, beam_sample=True))
                              for img in self.context]

    def set_context(self, context):
        self.context = context

    def set_image(self, raw_image):
        assert raw_image in self.context, 'target must be in context'
        self.raw_image = raw_image

        # load image as-is (without training transforms)
        if self.raw_image:
            image = utils.load_image(self.raw_image)
            self.image_tensor = image.to(device)

    # TODO: come up with better system for handling the need for representing
    # captions as vocab indices in some places (e.g. model input) and strings in others
    def iterate_round(self, round_num, prev_caption, as_string=True):
        caption = (prev_caption if as_string else
                   utils.ids_to_words(prev_caption, self.vocab))
        self.update_model(round_num, caption)
        return self.generate_utterance(as_string)

    # images shape: context_size x 3 x 244 x 244
    # utt shape: batch_size x max_length
    # lengths shape: batch_size
    # output shape: batch_size x context_size
    def S0_score(self, images, utt, lengths, use_old_decoder=False):
        # transforming features, utt, lengths, target into long form
        features = self.generating_encoder(images)
        features_long = features.repeat(len(utt), 1)

        # TODO: dangerous to do list comprehension over tensor; specify dimension
        utt_long = torch.cat([u.repeat(images.shape[0], 1) for u in utt])
        lengths_long = np.repeat(lengths, images.shape[0])

        if use_old_decoder:
            outputs, sizes = self.orig_decoder(
                features_long, utt_long, lengths_long, unpack=True)
        else:
            outputs, sizes = self.decoder(
                features_long, utt_long, lengths_long, unpack=True)

        # log prob should be close to 0 (i.e. probs should add up to 1)
        assert torch.abs(torch.sum(torch.logsumexp(F.log_softmax(outputs, dim=1),
                                                   dim=1))) < 0.01

        # CrossEntropy flips sign, but we want to keep the raw (log) likelihood
        # of each word, so we flip it back
        losses = []
        for i in range(outputs.shape[0]):
            ce_loss = -1 * F.cross_entropy(outputs[i], utt_long[i], reduction="none",
                                           ignore_index=0)
            losses.append(ce_loss)
        padded_scores = torch.stack(losses, 0)

        # Then we sum up log likelihoods to get overall likelihood of utt
        score = torch.sum(padded_scores, dim=1)

        # Finally, reshape score into batch_size x context_size matrix
        score_reshaped = score.reshape(utt.shape[0], images.shape[0])
        return score_reshaped

    def L0_score(self, utt, context, temp=1):
        """
        images shape: context_size x 3 x 244 x 244
        utt shape: batch_size x max_length
        output shape: batch_size x context_size
        """
        # transform context to tensor of context_size x 3 x 244 x 244
        if type(self.context[0]) == str:
            context_images = utils.load_images(self.context).to(device)
        else:
            context_images = self.context.to(device)

        # computing lengths of unpadded sequences
        if not torch.is_tensor(utt):
            utt = torch.Tensor(utt).long().to(device)
        else:
            utt = utt.to(device)

        # counting nonzero elems for each batch elem
        lengths = (utt.shape[1] - (utt == 0).sum(dim=1)).tolist()
        # compute S0 score as a batch, use softmax temp...
        # (temp < 1 flattens so forces
        S0_scores = self.S0_score(context_images, utt, lengths)

        # compute sum of S0_scores per batch
        total = torch.logsumexp(S0_scores, dim=1)  # shape = [batch_size]
        total = total.reshape((total.shape[0], 1))  # shape = [1, batch_size]
        L0_scores = S0_scores - total
        # this should be close to 0 (i.e. listener probabilities should add up to 1...)
        assert torch.abs(torch.sum(torch.logsumexp(L0_scores, dim=1))) <= 0.01
        return L0_scores

    def S0_sample(self, image, as_string=True, use_old_decoder=False,
                  beam_sample=True):
        """
        generate greedy utterance for image in isolation

        Args: as_string: boolean indicating whether to convert to readable string
        """
        image_tensor = utils.load_image(image) if type(image) == str else image
        feature = self.generating_encoder(image_tensor.to(device))

        # (1, max_seq_length) -> (max_seq_length)
        decoder = self.orig_decoder if use_old_decoder else self.decoder
        sampled_ids, sampled_score = decoder.beam_sample(feature, 1) if beam_sample \
            else decoder.greedy_sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()
#        print('sampled score of top utt', sampled_score)
        # Truncate at '<end>' token
        out = []
        for word_id in sampled_ids:
            out.append(word_id)
            if (self.vocab.idx2word[word_id] == '<end>'
                    or self.vocab.idx2word[word_id] == '<pad>'):
                break
        return utils.ids_to_words(out, self.vocab) if as_string else out

    def generate_utterance(self, speaker_type, as_string=True):
        """
        generate utterance for image

        Args: 
          speaker_type: 'S0', 'S0_with_cost', or 'S1'
          as_string: boolean indicating whether to convert to readable string
        """

        if speaker_type == 'S0':
            # print('beam search would say:', self.S0_sample(self.image_tensor, as_string, beam_sample=True))
            # print('greedy search would say:', self.S0_sample(self.image_tensor, as_string, beam_sample=False))
            return self.S0_sample(self.image_tensor, as_string, beam_sample=True)

        results = []
        feature = self.generating_encoder(self.image_tensor)
        topk_batch, topk_scores = self.decoder.beam_sample(feature, self.topk)
        topk_batch_strings = []
        lengths = []
        for caption in topk_batch.cpu().numpy():
            out = []
            for word_id in caption:
                out.append(word_id)
                if self.vocab.idx2word[word_id] == '<end>':
                    lengths.append(len(out))
                    topk_batch_strings.append(utils.ids_to_words(out, self.vocab)
                                              if as_string else out)
                    break
        uttCost = torch.tensor(lengths).to(
            device).float().unsqueeze(1) * self.cost_weight
        target_idx = torch.LongTensor(
            [self.context.index(self.raw_image)]).to(device)
        if speaker_type == 'S0_with_cost':
            utility = topk_scores.unsqueeze(1) - uttCost
        elif speaker_type == 'S1':
            listener_scores = self.L0_score(topk_batch, self.context)
            informativity = (torch.index_select(
                listener_scores, 1, target_idx))
            utility = informativity - uttCost
        else:
            raise Exception('unknown speaker_type', speaker_type)
        reranked = sorted(zip(list(utility.data.cpu().numpy()),
                              topk_batch_strings),
                          key=operator.itemgetter(0))[::-1]
#        print(reranked)
#        print('top S0 sample', self.S0_sample(self.image_tensor))
        # if as_string else reutils.ids_to_words(reranked[0][1], self.vocab)
        return reranked[0][1]

    def handle_oov(self, word):
        """
        Checks a caption for OOV words. If found,
        - adds to the agent's vocabulary
        - adds to embedding blah blah
        """
        # initialize embedding to closest word
        closest_word = utils.find_invocab_word(
            word, self.history, self.vocab.word2idx)
        print('closest word', closest_word)
        if closest_word in self.vocab.word2idx:
            closest_word_id = self.vocab.word2idx[closest_word]
            embed_weight = self.decoder.embed(torch.unsqueeze(
                torch.tensor(closest_word_id), 0).to(device))
        else:
            embed_weight = torch.randn(1, self.decoder.embed_size).to(device)

        # extend embed weight
        self.decoder.embed.weight = nn.Parameter(
            torch.cat((self.decoder.embed.weight, embed_weight)))

        # extend linear weight
        linear_weight_zeros = torch.randn(
            1, self.decoder.hidden_size).to(device)
        self.decoder.linear.weight = nn.Parameter(
            torch.cat((self.decoder.linear.weight, linear_weight_zeros)))

        # extend linear bias
        linear_bias_zeros = torch.randn(1,).to(device)
        self.decoder.linear.bias = nn.Parameter(
            torch.cat((self.decoder.linear.bias, linear_bias_zeros)))

        # add word to agent's vocab & propogate changes to other
        # objects that rely on vocab
        self.vocab.add_word(word)
        print('updated vocab with new word',
              word, ' size now ', len(self.vocab))
        self.decoder.update(self.vocab)

    def process_human_caption(self, caption):
        """
        Wrapper function that processes raw human caption into format AI
        can work with. Includes OOV handling.
        caption: raw caption (string)
        """

        # add oovs to vocab before converting to ids
        tokens = utils.tokenize(caption)
        for word in tokens:
            if not word in self.vocab.word2idx:  # if oov
                self.handle_oov(word)

        return utils.words_to_ids(caption, self.vocab)
