# Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers
This repository is the official implementation of the paper ["Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers"](https://arxiv.org/abs/2311.07470v2) (Findings of ACL 2024).

## Requirements

To get started, please clone this repository and install packages as:

```bash
git clone https://github.com/opanhw/MM_Neurons.git
conda create -n MM_Neurons python=3.9
...
pip install -r requirements.txt
```

## Model Preparation

We evaluate three widely used Multi-modal Large Language Models: [LLaVA](https://github.com/haotian-liu/LLaVA), [InstructBLIP](https://github.com/salesforce/lavis) and [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl). Model weights can be downloaded from the following links:

- [LLaVA](https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview)
- [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b)
- [mPLUG-Owl2](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b)

**Please note that we recommend using an earlier version of LLaVA ([link](https://github.com/haotian-liu/LLaVA/archive/refs/tags/v1.0.2.tar.gz)), as the method of calling certain classes and the parameters in the latest version of LLaVA may not be consistent with our code.**

Since we need to obtain and modify neuron activations, we have made some modifications to the model source code. Please replace `LLaVA/llava/model/language_model/llava_llama.py` in your LLaVA project path with `open_source_model/LLaVA/llava_llama.py` and replace `mPLUG-Owl/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py` in your mPLUG-Owl2 project path with `open_source_model/mPLUG-Owl2/modeling_mplug_owl2.py` .

You should run `src/preparation.py` for some preparation work before finding neurons. A sample usage command is:

```bash
python src/preparation.py --model_type LLaVA --model_path YOUR_LLAVA_MODEL_PATH
```

## Dataset

- SBU Captions Dataset

See the dataset details in [this page](https://www.cs.rice.edu/~vo9/sbucaptions/).

You should download the [json file](https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions-all.tar.gz) which contains image urls + captions.

## Finding Multi-Modal Neurons

You can use `src/find_mm_neurons.py` to find multi-modal neurons. A sample usage command is:

```bash
python src/find_mm_neurons.py --model_type LLaVA --model_path YOUR_LLAVA_MODEL_PATH --save_to ./results --task_type sbu --data_path ./datasets/sbu --add_prompt --cal_activations
```

`src/find_mm_neurons.py` contains the following arguments.

- `model_type`: we support (case-insensitive)
  - `LLaVA`
  - `InstructBLIP`
  - `mPLUG-Owl2`
- `model_path`: path to the model you choose
- `save_to`: path to save the results
- `task_type`: name of the dataset, we only support `sbu` now
- `data_path`: path to the dataset you choose
- `query`: prompt of the model, we use `Describe the image in few words.` in all experiments
- `max_num`: maximum number of the samples, default is `1000`
- `start`: start number of the samples, default is `0`
- `end`: end number of the samples, default is `1000`
- `add_prompt`: whether to add a prefix "An image of", store true
- `cal_activations`: whether to return activations of neurons, store true
- `shuffle`: whether to shuffle the input sequence of image tokens, store true

## Acknowledgements

Some codes are built upon [Skill-Neuron](https://github.com/THU-KEG/Skill-Neuron).

Thanks [@wonderful9462](https://github.com/wonderful9462) for the assistance in this work.

## Citation

If you find this code useful, please kindly cite our work as:
```bibtex
@inproceedings{pan-etal-2024-finding,
    title = "Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers",
    author = "Pan, Haowen  and
      Cao, Yixin  and
      Wang, Xiaozhi  and
      Yang, Xun  and
      Wang, Meng",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.60",
    pages = "1012--1037",
}
```
