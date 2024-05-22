# Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers
This repository is the official implementation of the paper ["Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers"](https://arxiv.org/abs/2311.07470)

## News
* 16/05/2024: The paper has been accepted to the Findings of ACL 2024! Code will be released soon.

## Model Preparation

We evaluate three widely used Multi-modal Large Language Models: [LLaVA](https://github.com/haotian-liu/LLaVA), [InstructBLIP](https://github.com/salesforce/lavis) and [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl). Model weights can be downloaded from the following links:

- [LLaVA](https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview)
- [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b)
- [mPLUG-Owl2](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b)

Since we need to obtain and modify neuron activations, we have made some modifications to the model source code. Please replace `LLaVA/llava/model/language_model/llava_llama.py` in your LLaVA project path with `open_source_model/LLaVA/llava_llama.py` and replace `mPLUG-Owl/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py` in your mPLUG-Owl2 project path with `open_source_model/mPLUG-Owl2/modeling_mplug_owl2.py` .

## Dataset

- SBU Captions Dataset

See the dataset details in [this page](https://www.cs.rice.edu/~vo9/sbucaptions/).

## Acknowledgements

Some codes are built upon [Skill-Neuron](https://github.com/THU-KEG/Skill-Neuron).

Thanks [@Wonderful9462](https://github.com/wonderful9462) for the assistance in this work.

## Citation

If you find this code useful, please kindly cite our work as:
```bibtex
@misc{pan2023finding,
      title={Finding and Editing Multi-Modal Neurons in Pre-Trained Transformer}, 
      author={Haowen Pan and Yixin Cao and Xiaozhi Wang and Xun Yang},
      year={2023},
      eprint={2311.07470},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
