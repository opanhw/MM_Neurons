#!/usr/local/bin/python3

import sys

sys.path.append('./open_source_model/LLaVA-main')  # Replace it with your llava path
sys.path.append('./open_source_model/InstructBlip')  # Replace it with your instructblip path
sys.path.append('./open_source_model/mPLUG-Owl-main/mPLUG-Owl2')  # Replace it with your mplug-owl2 path

import os
import cv2
import log
import json
import copy
import torch
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from corenlp_client import CoreNLP
from torch.utils.data import DataLoader
from transformers import StoppingCriteria, AutoTokenizer


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def bilinear_interpolation(img, height, width):
    """
	bilinear interpolation
	:param img: original image
	:param height: image height after scaling
	:param width: image width after scaling
	:return: image after scaling
	"""
    # height and width of original image
    ori_height, ori_width = img.shape[:2]
    # image after scaling
    dst = np.zeros((height, width, img.shape[2]))
    # scaling factor
    scale_x = float(ori_width) / width
    scale_y = float(ori_height) / height

    for i in range(height):
        for j in range(width):
            x = min(int(j * scale_x), ori_width - 2)
            y = min(int(i * scale_y), ori_height - 2)
            u = j * scale_x - x
            v = i * scale_y - y
            dst[i, j] = (1 - u) * (1 - v) * img[y, x] + \
                        u * (1 - v) * img[y, x + 1] + \
                        (1 - u) * v * img[y + 1, x] + \
                        u * v * img[y + 1, x + 1]
    return dst


class trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.annotator = CoreNLP()
        self.getModel()
        logger = log.get_logger(args.save_to + "/log")
        self.logger = logger
        logger.info(args)

    def getModel(self):
        # Getting the required model
        if self.args.model_type == "llava":
            from llava.model import *
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
                DEFAULT_IM_END_TOKEN

            # load llava
            self.model_name = get_model_name_from_path(self.args.model_path)
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.args.model_path,
                                                                                                  self.args.model_base,
                                                                                                  self.model_name)
            print('Successfully load.')

            # get query
            qs = self.args.query
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if 'llama-2' in self.model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in self.model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in self.model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if self.args.conv_mode is not None and conv_mode != self.args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                    conv_mode, self.args.conv_mode, self.args.conv_mode))
            else:
                self.args.conv_mode = conv_mode

            self.conv = conv_templates[self.args.conv_mode].copy()
            self.conv.append_message(self.conv.roles[0], qs)
            self.conv.append_message(self.conv.roles[1], None)
            prompt = self.conv.get_prompt()

            # add an instruction string for better generation
            if self.args.add_prompt:
                prompt += 'An image of'

            # get input ids
            self.input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                   return_tensors='pt').unsqueeze(0).cuda()

            # image token length of llava is 256
            self.image_token_len = 256


        elif self.args.model_type == "instructblip":
            from modeling_instructblip import InstructBlipForConditionalGeneration
            from processing_instructblip import InstructBlipProcessor

            self.model = InstructBlipForConditionalGeneration.from_pretrained(self.args.model_path,
                                                                              low_cpu_mem_usage=True, device_map="auto")
            self.processor = InstructBlipProcessor.from_pretrained(self.args.model_path, low_cpu_mem_usage=True,
                                                                   device_map="auto")
            self.tokenizer = self.processor.tokenizer

            self.prompt = self.args.query
            if self.args.add_prompt:
                self.prompt += ' An image of'

            self.image_token_len = 32

        elif self.args.model_type == "mplug-owl2":
            from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from mplug_owl2.conversation import conv_templates, SeparatorStyle
            from mplug_owl2.model.builder import load_pretrained_model
            from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
                KeywordsStoppingCriteria

            self.model_name = get_model_name_from_path(self.args.model_path)
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.args.model_path,
                                                                                                  self.args.model_base,
                                                                                                  self.model_name)
            print('Successfully load.')

            inp = DEFAULT_IMAGE_TOKEN + self.args.query

            self.conv = conv_templates["mplug_owl2"].copy()
            self.conv.append_message(self.conv.roles[0], inp)
            self.conv.append_message(self.conv.roles[1], None)
            prompt = self.conv.get_prompt()

            # add an instruction string for better generation
            if self.args.add_prompt:
                prompt += ' An image of'

            print(prompt)

            # get input ids
            self.input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                   return_tensors='pt').unsqueeze(0).cuda()

            self.image_token_len = 65

    def process_batch(self, batch):
        if "sbu" in self.args.task_type:
            image = Image.fromarray(np.uint8(batch['image'][0].numpy().transpose((1, 2, 0)) * 255))
            label = None
        else:
            raise NotImplementedError

        return image, label

    def perturbation(self, eval_data):
        test_dataloader = DataLoader(eval_data, batch_size=self.args.bz, num_workers=1, pin_memory=True)
        cnt = 0
        self.model.eval()

        for data in tqdm(test_dataloader):
            cnt += 1
            if cnt < self.args.start:
                continue

            print(f'cnt: {cnt}')

            noun_paths = glob(f'{self.args.save_to}/{cnt}/noun_*')

            with open(f'{noun_paths[0]}/data.json') as f:
                original_outputs = json.load(f)['caption']

            image, _ = self.process_batch(data)

            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

            # randomly select neurons
            _random = torch.randperm(self.model.config.intermediate_size * self.model.config.num_hidden_layers)[
                           :self.args.top_neurons]
            random_mm_neurons = [[x // self.model.config.intermediate_size, x % self.model.config.intermediate_size] for
                                 x in _random]

            # modify model state dict
            state_dict = self.model.state_dict()
            original_w = []

            for i, mm_neuron in enumerate(random_mm_neurons):
                w = state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'].T
                original_w.append(copy.deepcopy(w[mm_neuron[1]]))
                w[mm_neuron[1]] += torch.randn(w.shape[1]).to(w.device) * self.args.sigma
                state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'] = w.T

            self.model.load_state_dict(state_dict)

            with torch.no_grad():
                output_ids = self.model.generate(
                    self.input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    max_new_tokens=50,
                    stopping_criteria=[stopping_criteria])

            input_token_len = self.input_ids.shape[1]
            print(output_ids[:, input_token_len:])
            n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            random_outputs = outputs.strip()

            print(f'random outputs: {random_outputs}')

            # reset model state dict
            for i, mm_neuron in enumerate(random_mm_neurons):
                w = state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'].T
                w[mm_neuron[1]] = original_w[i]
                state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'] = w.T

            self.model.load_state_dict(state_dict)

            for noun_path in noun_paths:
                print(f'noun: {noun_path.split("_")[-1]}')

                mm_neurons = torch.load(f'{noun_path}/mm_neurons')[:self.args.top_neurons]

                # modify model state dict
                state_dict = self.model.state_dict()
                original_w = []

                for i, mm_neuron in enumerate(mm_neurons):
                    w = state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'].T
                    original_w.append(copy.deepcopy(w[mm_neuron[1]]))
                    w[mm_neuron[1]] += torch.randn(w.shape[1]).to(w.device) * self.args.sigma
                    state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'] = w.T

                self.model.load_state_dict(state_dict)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        self.input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        max_new_tokens=50,
                        stopping_criteria=[stopping_criteria])

                input_token_len = self.input_ids.shape[1]
                print(output_ids[:, input_token_len:])
                n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                print(f'original outputs: {original_outputs}')
                print(f'perturb outputs: {outputs}')

                results = {'original outputs': original_outputs, 'perturb outputs': outputs,
                           'random outputs': random_outputs}

                with open(f'{noun_path}/perturb.json', 'w+', encoding='utf-8') as f:
                    f.write(json.dumps(results, indent=4, ensure_ascii=False))

                # reset model state dict
                for i, mm_neuron in enumerate(mm_neurons):
                    w = state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'].T
                    w[mm_neuron[1]] = original_w[i]
                    state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'] = w.T

                self.model.load_state_dict(state_dict)

            if 0 < self.args.max_num <= cnt - self.args.start + 1 or self.args.end <= cnt:
                break

    def find_mm_neurons(self, eval_data):
        test_dataloader = DataLoader(eval_data, batch_size=self.args.bz, num_workers=1, pin_memory=True)
        cnt = 0
        self.model.eval()

        if self.args.model_type == 'llava':
            total_top_words = torch.load(f'{self.args.preparation_path}/llava_13b_neuron_tokens')
            projection = [torch.load(f'{self.args.preparation_path}/llava_13b_layer_projection/layer_{layer}').cpu() for
                          layer in tqdm(range(self.model.config.num_hidden_layers))]
        elif self.args.model_type == 'instructblip':
            total_top_words = torch.load(f'{self.args.preparation_path}/instructblip_7b_neuron_tokens')
            projection = [
                torch.load(f'{self.args.preparation_path}/instructblip_7b_layer_projection/layer_{layer}').cpu() for
                layer in
                tqdm(range(self.model.config.num_hidden_layers))]
        elif self.args.model_type == 'mplug-owl2':
            total_top_words = torch.load(f'{self.args.preparation_path}/mplug_owl2_7b_neuron_tokens')
            projection = [torch.load(f'{self.args.preparation_path}/mplug_owl2_7b_layer_projection/layer_{layer}').cpu()
                          for layer in
                          tqdm(range(self.model.config.num_hidden_layers))]

        import time
        start = time.perf_counter()

        for data in tqdm(test_dataloader):
            cnt += 1
            if cnt < self.args.start:
                continue

            create_dir(f'{self.args.save_to}/{cnt}')

            if self.args.cal_activations:
                self.model.set_get_activations(self.patch_start, self.patch_end)

            image, _ = self.process_batch(data)

            image.save(f'{self.args.save_to}/{cnt}/original_image.png')

            if self.args.model_type == 'llava':
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                img_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel'][0]

                stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

            elif self.args.model_type == 'instructblip':
                image_tensor = self.processor.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                img_tensor = self.processor.image_processor.preprocess(image, return_tensors='pt')['pixel'][0]

                stop_str = ""

            elif self.args.model_type == 'mplug-owl2':
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                img_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel'][0]

                stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

            if self.args.shuffle:
                # shuffle the input sequence of image patches
                self.model.set_shuffle()
                idx = torch.randperm(self.image_token_len)
                print(idx)
                torch.save(idx.cpu(), f'{self.args.save_to}/{cnt}/idx')
                idx_dict = {i: x.item() for i, x in enumerate(idx)}
                idx_restore_dict = {x.item(): i for i, x in enumerate(idx)}
                idx_restore = torch.tensor([idx_restore_dict[x] for x in range(self.image_token_len)])
                self.model.set_idx(idx)

            with torch.no_grad():
                if self.args.model_type == 'llava':
                    output_ids = self.model.generate(
                        self.input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        max_new_tokens=50,
                        stopping_criteria=[stopping_criteria])

                elif self.args.model_type == "instructblip":
                    self.inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
                    for k, v in self.inputs.items():
                        self.inputs[k] = v.cuda()

                    self.input_ids = self.inputs['input_ids']

                    output_ids = self.model.generate(
                        **self.inputs,
                        max_new_tokens=50,
                    )

                elif self.args.model_type == 'mplug-owl2':
                    output_ids = self.model.generate(
                        self.input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        max_new_tokens=50,
                        stopping_criteria=[stopping_criteria])

                input_token_len = self.input_ids.shape[1]
                print(output_ids[:, input_token_len:])
                n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

                output_tokens = output_ids[:, input_token_len:]

                outputs = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if self.args.model_type == 'llava' or self.args.model_type == 'mplug-owl2':
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()
                print(outputs)

                # exit(0)

                output_tokens = output_tokens[0].cpu().tolist()
                print(output_tokens)

            if self.args.cal_activations:
                last_activation = self.model.get_activations()
                # print(last_activation)
                last_activation = torch.stack([torch.stack(x) for x in last_activation])

                print(last_activation.shape)

            tokenized_text = self.annotator.tokenize(outputs.replace('.', ' '))[0]
            pos_tag = self.annotator.pos_tag(outputs.replace('.', ' '))[0]
            nouns = list(set(word for word, pos in zip(tokenized_text, pos_tag) if pos.startswith('N')))
            nouns.sort()

            print(nouns)

            noun_tokens = [x[0] for x in self.tokenizer.batch_encode_plus(nouns, add_special_tokens=False)['input_ids']]
            noun_pos = []
            new_nouns = []

            for i, x in enumerate(noun_tokens):
                if x in output_tokens:
                    noun_pos.append(output_tokens.index(x))
                    new_nouns.append(nouns[i])

            nouns = new_nouns

            with open(f'{self.args.save_to}/{cnt}/nouns.json', 'w+', encoding='utf-8') as f:
                f.write(json.dumps(nouns, ensure_ascii=False))

            torch.save(last_activation[noun_pos].cpu(), f'{self.args.save_to}/{cnt}/last_activation')

            for j, (token, noun, pos) in enumerate(zip(noun_tokens, nouns, noun_pos)):
                save_dir = f'{self.args.save_to}/{cnt}/noun_{noun}'
                create_dir(save_dir)

                # calculate scores of neurons
                scores = torch.zeros((last_activation.shape[1], last_activation.shape[2])).half().cuda()
                for layer in tqdm(range(last_activation.shape[1])):
                    act = last_activation[pos, layer]
                    prob = act.cuda() * projection[layer][:, token].flatten().cuda()
                    scores[layer] += prob.flatten()

                print(noun)

                top_index = scores.flatten().sort(descending=True, dim=0).indices[
                            :10000]  # save top-10000 multi-modal neurons
                top_pos = [(x.item() // scores.shape[1], x.item() % scores.shape[1]) for x in top_index]

                results = {'caption': outputs, 'top_neurons': []}

                # find multi-modal neurons
                mm_neurons = []
                for i, pos in enumerate(top_pos):
                    mm_neurons.append([pos[0], pos[1]])
                    if i < 10:
                        print(
                            f'Neuron: L{pos[0]}.u{pos[1]} \tScore: {round(scores[pos[0]][pos[1]].item(), 4)}\tTop 10 tokens: {total_top_words[pos[0]][pos[1]][:10]}')
                        results['top_neurons'].append({'neuron': pos, 'tokens': total_top_words[pos[0]][pos[1]][:10],
                                                       'score': scores[pos[0]][pos[1]].item()})

                with open(f'{save_dir}/data.json', 'w+', encoding='utf-8') as f:
                    f.write(json.dumps(results, indent=4, ensure_ascii=False))

                torch.save(mm_neurons, f'{save_dir}/mm_neurons')

                torch.save(img_tensor, f'{save_dir}/image_tensor')
                image = Image.fromarray(np.uint8(img_tensor.numpy())).convert('RGBA')
                image.save(f'{save_dir}/image.png')

                # draw heatmap and binary mask
                for mean_num in [1, 10, 50, 100, 500, 1000]:
                    mean_activation = torch.stack(
                        [activations[x[0], :, x[1]].reshape(16, 16, 1) for x in mm_neurons[:mean_num]]).mean(dim=0).numpy()

                    new_activation = bilinear_interpolation(mean_activation, img_tensor.shape[0], img_tensor.shape[1])

                    # plot heatmap
                    new_activation_ = (new_activation - new_activation.min()) / (new_activation.max() - new_activation.min())
                    heatmap = cv2.applyColorMap(np.uint8(255 * new_activation_), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = Image.fromarray(heatmap).convert('RGBA')
                    heatmap.putalpha(int(0.5 * 255))
                    new_image = Image.alpha_composite(image, heatmap).convert('RGB')
                    new_image.save(f'{save_dir}/heatmap_average_{mean_num}.png')

                    # plot binary mask
                    thres = np.percentile(new_activation, 95)
                    pos = np.where(new_activation > thres)
                    mask = Image.new('RGBA', (img_tensor.shape[0], img_tensor.shape[1]), (0, 0, 0, 200))
                    mask_pixels = mask.load()
                    for i, j in zip(pos[0], pos[1]):
                        mask_pixels[j, i] = (255, 255, 255, 0)
                    new_image = Image.alpha_composite(image, mask).convert('RGB')
                    new_image.save(f'{save_dir}/mask_average_{mean_num}.png')

                for i in range(5):
                    mm_neuron = mm_neurons[i]
                    torch.save(activations[mm_neuron[0], :, mm_neuron[1]].float().half(), f'{save_dir}/activation_L{mm_neuron[0]}_u{mm_neuron[1]}')
                    activation = activations[mm_neuron[0], :, mm_neuron[1]].reshape(16, 16, 1).numpy()
                    new_activation = bilinear_interpolation(activation, img_tensor.shape[0], img_tensor.shape[1])

                    # plot heatmap
                    new_activation_ = (new_activation - new_activation.min()) / (
                                new_activation.max() - new_activation.min())
                    heatmap = cv2.applyColorMap(np.uint8(255 * new_activation_), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = Image.fromarray(heatmap).convert('RGBA')
                    heatmap.putalpha(int(0.5 * 255))
                    new_image = Image.alpha_composite(image, heatmap).convert('RGB')
                    new_image.save(f'{save_dir}/heatmap_L{mm_neuron[0]}_u{mm_neuron[1]}.png')

                    # plot binary mask
                    thres = np.percentile(new_activation, 95)
                    pos = np.where(new_activation > thres)
                    mask = Image.new('RGBA', (img_tensor.shape[0], img_tensor.shape[1]), (0, 0, 0, 200))
                    mask_pixels = mask.load()
                    for i, j in zip(pos[0], pos[1]):
                        mask_pixels[j, i] = (255, 255, 255, 0)
                    new_image = Image.alpha_composite(image, mask).convert('RGB')
                    new_image.save(f'{save_dir}/mask_L{mm_neuron[0]}_u{mm_neuron[1]}.png')

            if 0 < self.args.max_num <= cnt - self.args.start + 1 or self.args.end <= cnt:
                break

        end = time.perf_counter()
        print(f"avg time: {(end - start) / cnt}")
