import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import operator

import utils
from utils import device, Writer
import torch.nn.functional as F
from rehearsal_loader import get_rehearsal_loader
from context_loader import get_context_loader
from data_loader import get_history_loader


class CombinedLoss() :
    def __init__(self, agent) :
        self.agent = agent
        self.device = agent.device
        # Extract some properties we'll use
        self.vocab = agent.vocab
        self.num_rehearsals = agent.num_rehearsals
        self.context = agent.context
        self.speaker_KL_weight = agent.speaker_KL_weight
        self.speaker_CE_weight = agent.speaker_CE_weight
        self.speaker_rehearsal_weight = agent.speaker_rehearsal_weight
        self.listener_CE_weight = agent.listener_CE_weight
        self.listener_KL_weight = agent.listener_KL_weight
        self.listener_rehearsal_weight = agent.listener_rehearsal_weight

    def compute(self, batch, steps):
        total = 0
        loss_terms = self.agent.loss.split('+')
        for term in loss_terms:
            if term == 'SCE':
                loss = self.speaker_CE_term(batch)
                weight = self.speaker_CE_weight
            elif term == 'SKL':
                loss = self.speaker_KL_term()
                weight = self.speaker_KL_weight
                #print(term, loss, weight)
            elif term == 'SR':
                loss = self.speaker_rehearsal_term(batch)
                weight = self.speaker_rehearsal_weight
                #print(term, loss, weight)
            elif term == 'LCE':
                loss = self.listener_CE_term(batch)
                weight = self.listener_CE_weight
            elif term == 'LKL':
                loss = self.listener_KL_term(batch)
                weight = self.listener_KL_weight
            elif term == 'LR':
                loss = self.listener_rehearsal_term(batch)
                weight = self.listener_rehearsal_weight
            else:
                raise Exception("loss term {} not recognized".format(term))
            scaled_loss = weight * loss
            total += scaled_loss
            if self.agent.loss_writer:
                self.agent.loss_writer.writerow(self.agent.sample_num,
                                                self.agent.loss,
                                                self.agent.trial_num,
                                                steps,
                                                term,
                                                weight,
                                                loss.data.cpu().numpy(),
                                                scaled_loss.data.cpu().numpy())
        return total

    def speaker_CE_term(self, batch) :
        images, captions, lengths = batch
        captions = captions.to(self.device)        
        criterion = nn.CrossEntropyLoss()

        image_feats = (self.agent.generating_encoder(images.to(self.device))
                       .to(self.device))
        target_probs, sizes = self.agent.decoder(image_feats, captions, lengths)
        
        target_probs.to(self.device)
        target_truth = pack_padded_sequence(
            captions, lengths, batch_first=True, enforce_sorted=False
        )[0].long().to(self.device)
        
        return criterion(target_probs, target_truth)

    def speaker_KL_term(self) :
        # sample rehearsals
        KL_reg = nn.KLDivLoss(reduction ='batchmean')
        try:
            r_images, r_captions, r_lengths = next(self.agent.loader)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.agent.loader = get_rehearsal_loader(self.num_rehearsals, self.vocab)
            r_images, r_captions, r_lengths = next(self.agent.loader)

        r_images = r_images.to(self.device)
        r_feats = self.agent.generating_encoder(r_images)

        # Compute MAP utterance under p (note we have to mask the garbage at end)
        p_caption = self.agent.orig_decoder.greedy_sample(r_feats, maskEnd = True)
        p_lengths = p_caption.size()[1] - (p_caption == 0).sum(dim=1)
        p_out, _ = self.agent.orig_decoder(r_feats, p_caption, p_lengths)
        q_out, _ = self.agent.decoder(r_feats, p_caption, p_lengths)
        q_out_no_oov = q_out[:,:p_out.size()[1]] 
        KL_term1 = KL_reg(F.log_softmax(p_out, dim = 1),
                          F.softmax(q_out_no_oov, dim = 1))
        return KL_term1

    def speaker_rehearsal_term(self, batch):
        # If there's no memory, we can't regularize to it...
        if not self.agent.history :
            return torch.tensor(0.0)

        # Make previous captions, images compatible with pytorch
        criterion = nn.CrossEntropyLoss()
        imgs_lens_caps = []
        if self.agent.reduction_history_window != 'complete' :
            history = self.agent.history[-self.agent.reduction_history_window:]
        else :
            history = self.agent.history

        batch = next(get_history_loader(self.agent, history, self.vocab, shuffle=True,
                                        num_workers=self.agent.num_workers))

        images, captions, lengths = batch
              
        # Use prev captions as target_truths
        target_truths = pack_padded_sequence(
            captions, lengths, batch_first=True, enforce_sorted=False
        )[0].long().to(self.device)

        # Compute target_outputs
        target_feats = (self.agent.generating_encoder(images.to(self.device))
                        .to(self.device))
        input_captions = captions.long().to(self.device)
        target_outputs, _ = self.agent.decoder(target_feats, input_captions, lengths)
        target_outputs.to(self.device)
        return criterion(target_outputs, target_truths)

    def listener_CE_term(self, batch, use_likelihood_ratios=False):
        images, captions, lengths = batch

        criterion = nn.NLLLoss()
        captions = captions.long() 
        target_img_idx = self.context.index(self.agent.raw_image)
        target_outputs = self.agent.L0_score(captions, self.context).to(self.device)
        if use_likelihood_ratios :
            dist_idx = torch.tensor(np.delete(range(len(self.context)),
                                                    target_img_idx, 0)).to(device)
            dist = torch.index_select(target_outputs, 1, dist_idx).to(device)
            target_score = torch.index_select(
                target_outputs, 1, torch.tensor([target_img_idx]).to(device)
            )
            return torch.sub(dist, target_score.reshape(dist.shape[0],1)).sum()
        else :
            target_truth = (torch.Tensor([target_img_idx] * captions.shape[0])
                            .long().to(self.device))
            return criterion(target_outputs, target_truth)

#     def listener_KL_term(self, batch):
#         KL_reg = nn.KLDivLoss(reduction ='batchmean')
#         images, captions, lengths = batch
#         captions = captions.numpy()
#         #print("captions", captions)

#         #generate the distribution for the images for the old and new models
#         p_out = self.agent.L0_score(captions, self.agent.context)
#         self.agent.is_old_decoder = True

#         q_out = self.agent.L0_score(captions, self.agent.context)
#         self.agent.is_old_decoder = False

#         #compute KL term
#         KL_term = KL_reg(p_out, torch.exp(q_out))
#         return KL_term


#     def listener_KL_term2(self, batch):
#         #sample rehearsal contexts + include the target in one of the contexts
#         KL_reg = nn.KLDivLoss(reduction ='batchmean')
#         r_imgs, r_img_dirs, r_tags = next(get_context_loader(ctx_type='challenge', ctx_size=self.num_rehearsals, \
#                                             num_samples=1))
#         assert len(r_imgs) == len(r_img_dirs) == len(r_tags) == self.num_rehearsals

#         #generate and padd utt for randomly selected target images in a context
#         human_lens_caps = []
#         for img_dir in r_img_dirs:
#             human_cap = utils.get_human_caps(img_dir, self.vocab)
#             human_lens_caps.append((len(human_cap), human_cap))
#         human_lens_caps.sort(key=operator.itemgetter(0), reverse=True)
#         human_lens, human_caps = zip(*human_lens_caps)
#         padded_human_caps = torch.nn.utils.rnn.pad_sequence(human_caps, batch_first=True)

#         #generate the distribution for the images for the old and new models
#         p_out = self.agent.L0_score(padded_human_caps.float(), r_imgs)
#         self.agent.is_old_decoder = True
#         q_out = self.agent.L0_score(padded_human_caps.float(), r_imgs)
#         #print("qout", q_out)
#         self.agent.is_old_decoder = False

#         #compute KL term
#         KL_term1 = KL_reg(p_out, torch.exp(q_out))
#         return KL_term1

#     def listener_rehearsal_term(self, batch):
#         # If there's no memory, we can't regularize to it...
#         if not self.agent.history :
#             return torch.tensor(0.0)

#         # Make previous captions, images compatible with pytorch
#         criterion = nn.CrossEntropyLoss()
#         imgs_lens_caps = []
#         if self.agent.reduction_history_window != 'complete' :
#             history = self.agent.history[-self.agent.reduction_history_window:]
#         else :
#             history = self.agent.history

#         batch = next(get_history_loader(self.agent, history, self.vocab, shuffle=True,
#                                         num_workers=self.agent.num_workers))
#         images, captions, lengths = batch
        
#         # Compute target_outputs
#         target_feats = (self.agent.generating_encoder(images.to(self.device))
#                         .to(self.device))
#         input_captions = captions.long().to(self.device)
#         target_outputs, _ = self.agent.decoder(target_feats, input_captions, lengths,
#                                                self.agent.history)
#         target_outputs.to(self.device)
#         return criterion(target_outputs, target_truths)
