import sys
sys.path.append("models/")

import argparse
import json
import random
import time
from collections import defaultdict

import context_loader
import numpy as np
import torch
import torch.nn as nn
import utils
from custom_model import AdaptiveAgent
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *

from recurrent_rsa.bayesian_agents.joint_rsa import RSA
from recurrent_rsa.recursion_schemes.recursion_schemes import ana_beam
from recurrent_rsa.rsa_utils.numpy_functions import (make_initial_prior,
                                                     uniform_vector)


class EfficiencyWriter(Writer):
    def __init__(self, a, save_file):
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.debug = a.debug
        if not self.debug:
            self.init_output_csv([[
                "speaker_loss", "listener_loss", "context_type",
                "i_iter", "round_num", "target",
                "context", "caption", "scores",
                "num_words",  "learning_rate",  "targetScore",
                "correct", "target_memory", "unknown_label",
                "context_id", "row_type"
            ]])

    def writerow(self, ctx, round_num, target, caption,
                 num_words, scores, targetScore, correct,
                 target_memory, unk_obj, context_id, row_type):
        row = [ctx["speaker_loss"], ctx["listener_loss"], ctx["context_type"],
               ctx["sample_num"], round_num, target,
               ctx["dirs"], caption, scores,
               num_words, self.learning_rate, targetScore,
               correct, target_memory, unk_obj,
               context_id, row_type]
        if not self.debug:
            super().writerow(row)


def get_context_loader(context_info):
    return context_loader.get_context_loader(
        ctx_type=context_info["context_type"],
        ctx_size=context_info["context_size"],
        num_samples=context_info["num_samples"],
        unk_obj=context_info["unk_obj"],
        sequential=args.sequential
    )


def construct_expt_grid(args):
    grid = []
    for context_type in [args.context_type]:
        context_info = dict(
            context_type=context_type,
            context_size=args.context_size,
            num_samples=None,
            unk_obj=None
        )

        ctx_loader, args.num_samples = get_context_loader(context_info)

        for sample_num in range(args.num_samples):
            adaptation_ctx = next(ctx_loader)
            imgs, img_dirs, img_tags, ctx = adaptation_ctx
            for speaker_loss in [args.speaker_loss_type]:
                for listener_loss in ["fixed"]:
                    context_info["all_info"] = ctx
                    grid.append(dict(
                        context_info,
                        speaker_loss=speaker_loss,
                        listener_loss=listener_loss,
                        dirs=img_dirs,
                        cats=img_tags,
                        sample_num=sample_num
                    ))
    return grid


def main(args):
    with open("data/preprocess/annotations/instances_val2014.json") as in_file:
        instances = json.load(in_file)

    file_name2flickr_url = {
        x["file_name"]: x["flickr_url"] for x in instances["images"]
    }

    path = args.report_path
    writer = EfficiencyWriter(args, path)
    grid = construct_expt_grid(args)
    listener = AdaptiveAgent(args)
    criterion = nn.CrossEntropyLoss()
    speaker_model = RSA(seg_type="char", tf=False)

    start_time = time.time()

    for ctx_index, ctx in enumerate(grid):
        print(f"\nContext: {ctx_index + 1}/{len(grid)}")

        print("Unknown label: {}".format(grid[ctx_index]["all_info"]["unknown_label"]))

        if args.speaker_reset_after == "context":
            print("Resetting speaker...")
            speaker_model.initialize_speakers(["coco"])
        elif args.speaker_reset_after == "domain":
            if ctx_index > 0 and grid[ctx_index]["all_info"]["unknown_label"] != grid[ctx_index - 1]["all_info"]["unknown_label"]:
                print("Resetting speaker...")
                speaker_model.initialize_speakers(["coco"])
        else:
            raise RuntimeError("Reset method not supported!")

        ctx_accuracies = defaultdict(list)
        ctx_lengths = defaultdict(list)

        listener.loss = ctx["listener_loss"]
        listener.reset_to_initialization(ctx["dirs"])
        listener.context_type = ctx["context_type"]

        interaction_memory = defaultdict(list)

        optimizer = torch.optim.Adam(
            speaker_model.initial_speakers[0].parameters(),
            lr=args.learning_rate
        )

        for round_num in range(1, args.num_repetitions + 1):
            targets = random.sample(ctx["dirs"], len(ctx["dirs"]))
            print([x.split("/")[-1] for x in targets])
            print([file_name2flickr_url[x.split("/")[-1]] for x in targets])

            for target_index, target in enumerate(targets):
                print(f"Round {round_num}")

                target_idx = ctx["dirs"].index(target)
                print(f"Target: {ctx['dirs'][target_idx].split('/')[-1]}")

                if ctx["dirs"][target_idx] not in interaction_memory[target_idx]:
                    interaction_memory[target_idx].append(
                        ctx["dirs"][target_idx])

                listener.set_image(target)

                urls = interaction_memory[target_idx]
                print(f"Memory: {[x.split('/')[-1] for x in urls]}")

                rat = [100.0]
                number_of_images = len(urls)
                initial_image_prior = uniform_vector(number_of_images)
                initial_rationality_prior = uniform_vector(1)
                initial_speaker_prior = uniform_vector(1)
                initial_world_prior = make_initial_prior(
                    initial_image_prior,
                    initial_rationality_prior,
                    initial_speaker_prior
                )

                speaker_model.speaker_prior.set_features(
                    images=urls, tf=False, rationalities=rat)
                speaker_model.initial_speakers[0].set_features(
                    images=urls, tf=False, rationalities=rat)
                depth = 0 if "rsa" not in ctx["speaker_loss"] else (
                    1 if len(urls) > 1 else 0)
                print(f"Depth: {depth}")
                pragmatic_caption = ana_beam(
                    speaker_model,
                    target=0,
                    depth=depth,
                    speaker_rationality=0,
                    speaker=0,
                    start_from=list(""),
                    initial_world_prior=initial_world_prior,
                    beam_width=5
                )

                raw_cap = pragmatic_caption[0][0].replace(
                    "^", "").replace("&", "").replace("$", "")
                id_cap = utils.words_to_ids(raw_cap, listener.vocab)
                scores = listener.L0_score(
                    np.expand_dims(id_cap, axis=0), ctx["dirs"])

                scores = scores.data.cpu().numpy()[0]
                target_score = scores[target_idx]
                listener_idx = list(scores).index(max(scores))
                correct = listener_idx == target_idx

                print(f"Caption: {raw_cap}")
                print(f"Correct: {correct}")
                print(f"Listener selected: {ctx['dirs'][listener_idx]}")
                print()

                writer.writerow(
                    ctx,
                    round_num,
                    target,
                    raw_cap,
                    len(id_cap),
                    scores,
                    target_score,
                    correct,
                    urls,
                    ctx["all_info"]["unknown_label"],
                    ctx["all_info"]["context_id"],
                    "adaptation"
                )

                if ctx["speaker_loss"] in ["rsa_likelihood_reset", "rsa_likelihood", "likelihood"] and correct:
                    features = speaker_model.initial_speakers[0].features[0]
                    captions = torch.LongTensor(
                            [
                                speaker_model.initial_speakers[0].seg2idx[c]
                                for c in pragmatic_caption[0][0]
                            ]
                        ).cuda()
                    captions=captions.unsqueeze(0)
                    print("Increasing likelihood..\n.")
                    decoder=speaker_model.initial_speakers[0].decoder
                    lengths=torch.LongTensor([captions.shape[1]])
                    out=decoder.forward(features, captions, lengths)
                    targets=pack_padded_sequence(
                        captions, lengths, batch_first = True)[0]
                    loss=criterion(out, targets)
                    decoder.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if ctx["speaker_loss"] == "rsa_likelihood_reset":
                        print("Resetting memory...\n")
                        if len(interaction_memory[target_idx]) > 1:
                            interaction_memory[target_idx]=[
                                ctx["dirs"][target_idx]]

                if "rsa" in ctx["speaker_loss"]:
                    if ctx["dirs"][listener_idx] not in interaction_memory[target_idx]:
                        print(f"Adding {ctx['dirs'][listener_idx]} to memory")
                        interaction_memory[target_idx].append(
                            ctx["dirs"][listener_idx])

                ctx_accuracies[target.split("/")[-1]].append(correct)
                ctx_lengths[target.split("/")[-1]].append(len(id_cap))

        current_time=time.time()
        elapsed_time=current_time - start_time
        time_left = args.num_samples * elapsed_time / \
            (ctx_index + 1) - elapsed_time
        print(f"Time left (minutes): {time_left / 60}")
        print(ctx_accuracies)
        print(np.mean(np.array(list(ctx_accuracies.values()), dtype=int), axis=0))
        print(np.mean(np.array(list(ctx_lengths.values()), dtype=int), axis=0))
        print()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_path",
        type = str,
        default = "data/preprocess/listener/total_listener_encoder-5-3000.pkl",
        help = "path of the listener encoder"
    )
    parser.add_argument(
        "--decoder_path",
        type = str,
        default = "data/preprocess/listener/total_listener_decoder-5-3000.pkl",
        help = "path of the listener decoder"
    )
    parser.add_argument(
        "--vocab_path",
        type = str,
        default = "data/preprocess/listener/vocab.pkl",
        help = "path of the listener vocabulary"
    )
    parser.add_argument(
        "--context_type",
        type = str,
        default = "adaptation_similar_random_contexts",
        help = "type of the contexts"
    )
    parser.add_argument(
        "--sequential",
        type = int,
        default = True,
        help = "specifies whether the contexts must be read sequentially"
    )
    parser.add_argument(
        "--context_size",
        type = int,
        default = 4,
        help = "number of images to use for each context"
    )
    parser.add_argument(
        "--speaker_loss_type",
        type = str,
        default = "fixed",
        help = "speaker adaptation method"
    )
    parser.add_argument(
        "--speaker_reset_after",
        type = str,
        default = "context",
        help = "specifies whether to reset the speaker after each context or domain"
    )
    parser.add_argument(
        "--report_path",
        type = str,
        default = "report.csv",
        help = "filename of the generated report"
    )
    parser.add_argument(
        "--debug",
        action = "store_true",
        help = "specifies whether to run in debug mode"
    )
    parser.add_argument(
        "--num_repetitions",
        type = int,
        default = 11,
        help = "number of repetitions per target"
    )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 0.0003
    )
    args = parser.parse_args()

    print(args)
    main(args)
