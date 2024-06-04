#!/usr/local/bin/python3

import sys

sys.path.append('./open_source_model/LLaVA-main')  # Replace it with your llava path
sys.path.append('./open_source_model/InstructBlip')  # Replace it with your instructblip path
sys.path.append('./open_source_model/mPLUG-Owl-main/mPLUG-Owl2')  # Replace it with your mplug-owl2 path

import os
import math
import torch
import argparse
from tqdm import tqdm


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def main():
    parser = argparse.ArgumentParser(description="Command line interface for preparation.")
    parser.add_argument("--model_type", default="LLaVA", type=str, help="Type of model")
    parser.add_argument("--model_base", default=None, type=str, help="Base of model")
    parser.add_argument("--model_path", type=str, help="Path of model")
    parser.add_argument("--save_to", type=str, default="./preparation", help="Path to save")
    parser.add_argument("--top_tokens_num", type=int, default=10, help="Number of top tokens for each neuron")
    parser.add_argument("--bz", type=int, default=100, help="Batch size")

    args = parser.parse_args()

    args.model_type = args.model_type.lower()

    if args.model_type == 'llava':
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base,
                                                                               model_name)
        print('Successfully load.')

        # save tokenizer
        if not os.path.exists(f'{args.save_to}/llava_13b_tokenizer'):
            torch.save(tokenizer, f'{args.save_to}/llava_13b_tokenizer')

        # save unembedding matrix
        if not os.path.exists(f'{args.save_to}/llava_13b_unembedding'):
            torch.save(model.lm_head.weight.data.T.cpu(), f'{args.save_to}/llava_13b_unembedding')

        vocab_size = tokenizer.vocab_size

        unembedding = model.lm_head.weight.data.T.cuda()[:, :vocab_size]

        # save layer projection and top tokens for each neuron
        create_dir(f'{args.save_to}/llava_13b_layer_projection')
        create_dir(f'{args.save_to}/llava_13b_mlp_out')

        total_top_words = []
        for n in range(model.config.num_hidden_layers):
            print(f'layer: {n}')
            w = model.model.layers[n].mlp.down_proj.weight.data.T.cuda()
            torch.save(w.cpu(), f'{args.save_to}/llava_13b_mlp_out/layer_{n}')
            prob = torch.matmul(w.float(), unembedding.float()).half()
            del w
            torch.cuda.empty_cache()
            torch.save(prob.cpu(), f'{args.save_to}/llava_13b_layer_projection/layer_{n}')
            bz_num = math.ceil(prob.shape[0] / args.bz)
            top_words = []
            for i in tqdm(range(bz_num)):
                if i == bz_num - 1:
                    top_token = prob[i * args.bz:, :].sort(descending=True, dim=1).indices[:, :args.top_tokens_num]
                else:
                    top_token = prob[i * args.bz:(i + 1) * args.bz, :].sort(descending=True, dim=1).indices[:,
                                :args.top_tokens_num]
                top_word = []
                for j, token in enumerate(top_token):
                    top_word.append(tokenizer.batch_decode(token.unsqueeze(-1), skip_special_tokens=True))
                top_words.extend(top_word)
            print(top_words[:10])
            total_top_words.append(top_words)
        torch.save(total_top_words, f'{args.save_to}/llava_13b_neuron_tokens')

    elif args.model_type == 'instructblip':
        from modeling_instructblip import InstructBlipForConditionalGeneration
        from processing_instructblip import InstructBlipProcessor

        model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path, low_cpu_mem_usage=True,
                                                                     device_map="auto")
        processor = InstructBlipProcessor.from_pretrained(args.model_path, low_cpu_mem_usage=True, device_map="auto")
        config = model.config
        language_model = model.language_model
        del model
        torch.cuda.empty_cache()
        tokenizer = processor.tokenizer

        print('Successfully load.')

        # save tokenizer
        if not os.path.exists(f'{args.save_to}/instructblip_7b_tokenizer'):
            torch.save(tokenizer, f'{args.save_to}/instructblip_7b_tokenizer')

        # save unembedding matrix
        if not os.path.exists(f'{args.save_to}/instructblip_7b_unembedding'):
            torch.save(language_model.lm_head.weight.data.T.cpu(), f'{args.save_to}/instructblip_7b_unembedding')

        vocab_size = tokenizer.vocab_size

        unembedding = language_model.lm_head.weight.data.T[:, :vocab_size].cuda()

        # save layer projection and top tokens for each neuron
        create_dir(f'{args.save_to}/instructblip_7b_layer_projection')
        create_dir(f'{args.save_to}/instructblip_7b_mlp_out')

        total_top_words = []
        for n in range(config.text_config.num_hidden_layers):
            print(f'layer: {n}')
            w = language_model.model.layers[n].mlp.down_proj.weight.data.T.cuda()
            torch.save(w.cpu(), f'{args.save_to}/instructblip_7b_mlp_out/layer_{n}')
            prob = torch.matmul(w.float(), unembedding.float()).half()
            torch.save(prob.cpu(), f'{args.save_to}/instructblip_7b_layer_projection/layer_{n}')
            del w
            torch.cuda.empty_cache()
            bz_num = math.ceil(prob.shape[0] / args.bz)
            top_words = []
            for i in tqdm(range(bz_num)):
                if i == bz_num - 1:
                    top_token = prob[i * args.bz:, :].sort(descending=True, dim=1).indices[:, :args.top_tokens_num]
                else:
                    top_token = prob[i * args.bz:(i + 1) * args.bz, :].sort(descending=True, dim=1).indices[:,
                                :args.top_tokens_num]
                top_word = []
                for j, token in enumerate(top_token):
                    top_word.append(tokenizer.batch_decode(token.unsqueeze(-1), skip_special_tokens=True))
                top_words.extend(top_word)
            print(top_words[:10])
            total_top_words.append(top_words)
            del prob
            torch.cuda.empty_cache()
        torch.save(total_top_words, f'{args.save_to}/instructblip_7b_neuron_tokens')

    elif args.model_type == 'mplug-owl2':
        from mplug_owl2.mm_utils import get_model_name_from_path
        from mplug_owl2.model.builder import load_pretrained_model
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base,
                                                                               model_name)
        print('Successfully load.')

        # save tokenizer
        if not os.path.exists(f'{args.save_to}/mplug_owl2_7b_tokenizer'):
            torch.save(tokenizer, f'{args.save_to}/mplug_owl2_7b_tokenizer')

        # save unembedding matrix
        if not os.path.exists(f'{args.save_to}/mplug_owl2_7b_unembedding'):
            torch.save(model.lm_head.weight.data.T.cpu(), f'{args.save_to}/mplug_owl2_7b_unembedding')

        vocab_size = tokenizer.vocab_size

        unembedding = model.lm_head.weight.data.T.cuda()[:, :vocab_size]

        # save layer projection and top tokens for each neuron
        create_dir(f'{args.save_to}/mplug_owl2_7b_layer_projection')
        create_dir(f'{args.save_to}/mplug_owl2_7b_mlp_out')

        total_top_words = []
        for n in range(model.config.num_hidden_layers):
            print(f'layer: {n}')
            w = model.model.layers[n].mlp.down_proj.weight.data.T.cuda()
            torch.save(w.cpu(), f'{args.save_to}/mplug_owl2_7b_mlp_out/layer_{n}')
            prob = torch.matmul(w.float(), unembedding.float()).half()
            del w
            torch.cuda.empty_cache()
            torch.save(prob.cpu(), f'{args.save_to}/mplug_owl2_7b_layer_projection/layer_{n}')
            bz_num = math.ceil(prob.shape[0] / args.bz)
            top_words = []
            for i in tqdm(range(bz_num)):
                if i == bz_num - 1:
                    top_token = prob[i * args.bz:, :].sort(descending=True, dim=1).indices[:, :args.top_tokens_num]
                else:
                    top_token = prob[i * args.bz:(i + 1) * args.bz, :].sort(descending=True, dim=1).indices[:,
                                :args.top_tokens_num]
                top_word = []
                for j, token in enumerate(top_token):
                    top_word.append(tokenizer.batch_decode(token.unsqueeze(-1), skip_special_tokens=True))
                top_words.extend(top_word)
            print(top_words[:10])
            total_top_words.append(top_words)
        torch.save(total_top_words, f'{args.save_to}/mplug_owl2_7b_neuron_tokens')
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
