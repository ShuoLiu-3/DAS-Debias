import os
import json

import numpy as np
import torch
from tqdm import tqdm

from evaluation.evaluate import Evaluate
from utils.globals import *
from baukit import Trace, TraceDict

import logging

logger = logging.getLogger(name='logger')

# [Strategy 1 New] 定义一组中性 Prompts，用于计算“基准”激活值
NEUTRAL_PROMPTS = [
    "The person is performing an action.",
    "Someone is doing something.",
    "They are located in a place.",
    "The individual described is here.",
    "It happened yesterday.",
    "This is a description of a person.",
    "People are walking around.",
    "The group was gathered together.",
    "An event took place recently.",
    "The human being is thinking."
]

# 文件：trace.py
CATEGORY_NEUTRAL_PROMPTS = {
    # "Religion": [
    #     "A person attends a religious service.",
    #     "They hold a belief system.",
    #     "The group gathered for a ceremony.",
    #     "Someone is reading a scripture.",
    #     "A person is praying quietly."
    # ],
    # "Physical_appearance": [
    #     "A person has a specific height.",
    #     "The individual has a certain hair color.",
    #     "They are describing their appearance.",
    #     "Someone looks like this.",
    #     "A person with physical features."
    # ],
    # "SES": [  # 社会地位
    #     "A person is working at a job.",
    #     "They live in a residence.",
    #     "Someone earns a salary.",
    #     "The individual has a profession.",
    #     "They are managing their finances."
    # ],
    "default": [  # 对于其他表现尚可的类别，沿用通用 Prompt
        "The person is performing an action.",
        "Someone is doing something.",
        "They are located in a place.",
        "The individual described is here.",
        "It happened yesterday."
    ]
}


def get_neutral_prompts(category):
    # 如果类别在字典中，返回特定的；否则返回默认的
    return CATEGORY_NEUTRAL_PROMPTS.get(category, CATEGORY_NEUTRAL_PROMPTS["default"])


def get_llama_activations_bau(model, tok, prompt):
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    # [Fix Bug]: 自动修复 "addmm_impl_cpu_ not implemented for Half"
    # 获取当前模型参数所在的设备
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")  # Fallback

    # 如果模型在 CPU 上，但数据类型是 float16 (Half)，必须移动到 GPU，否则会报错
    if device.type == 'cpu' and torch.cuda.is_available():
        # 检查是否是 float16
        is_fp16 = False
        try:
            if model.dtype == torch.float16:
                is_fp16 = True
        except:
            pass  # 有些包装模型可能没有 .dtype 属性

        if is_fp16:
            logger.warning(f"Detected model on CPU with float16. Moving to CUDA to avoid Runtime Error...")
            model.to("cuda")
            device = torch.device("cuda")

    # 确保 input_ids 移动到正确的 device
    input_ids = tok.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        with TraceDict(model, HEADS + MLPS) as ret:
            output = model(input_ids, output_hidden_states=True)

        logits = output.logits.float()
        probabilities = logits.softmax(dim=2)[:, -1, :].squeeze()

        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, probabilities.tolist()


def preprocess(text: str, sentence_suffix: str):
    if text.count("[") != 2:
        return None, None
    bias_concept = text.split("[")[1].split("]")[0]
    clean_sentence = text.replace(f"[{bias_concept}]", bias_concept.lower()).split(f"[")[0].strip()
    return clean_sentence, bias_concept.lower()


class EvaluateTrace(Evaluate):

    def __init__(self, model, tok, test_file, task, protect_attr='gender'):
        super().__init__(model, tok, test_file, task)

        assert self.task == "trace", f"Task class mismatch:, expected 'trace', got '{self.task}' instead"
        self.protect_attr = protect_attr
        self.load_attr(protect_attr)

        self.results = {}
        self.partial_results = {}
        for item in self.attr_words:
            self.results[f"prob_{item}"] = 0.
            self.results[f"predicted_{item}"] = 0.
            self.partial_results[f"prob_{item}"] = []
        self.partial_results[f"predicted_token"] = []
        self.partial_results["prompt"] = []

        self.load_data()

    def load_attr(self, protect_attr):
        def load_attr_words(protect_attr, _label):
            attr_words_base_dir = "../biased_knowledge"
            if protect_attr in ['Age', 'Disability_status', 'Gender_identity',
                                'Nationality', 'Physical_appearance',
                                'Race_ethnicity', 'SES',
                                'Sexual_orientation', 'Religion']:
                file = os.path.join(attr_words_base_dir, protect_attr, "group.json")
                with open(file, 'r') as f:
                    data = json.load(f)
                    attr_words = sorted(list(data.keys()))
                    return attr_words, data
            else:
                sub_dir = os.path.join(attr_words_base_dir, protect_attr, _label + ".txt")
                with open(sub_dir, 'r') as f:
                    lines = f.readlines()
                    word_list = [line.strip() for line in lines if line.strip() != ""]
                    word_list = list(set(word_list))
                    f.close()
                    return word_list

        self.attr_words = []
        self.attr_words_map = {}

        if protect_attr in ['Age', 'Disability_status', 'Gender_identity',
                            'Nationality', 'Physical_appearance',
                            'Race_ethnicity', 'SES',
                            'Sexual_orientation', 'Religion']:
            self.attr_words, self.attr_words_map = load_attr_words(protect_attr, None)
            self.sentence_suffix = f" [{protect_attr}] group."

        self.attr_words_tokens_map = {}
        for item in self.attr_words:
            self.attr_words_tokens_map[item] = [self.tok.encode(token)[0] for token in self.attr_words_map[item]]
            self.attr_words_tokens_map[item] = list(set(self.attr_words_tokens_map[item]))
        # logger.info(f"attr_words: {self.attr_words}")
        # logger.info(f"attr_words_map: {self.attr_words_map}")
        # logger.info(f"attr_words_tokens_map: {self.attr_words_tokens_map}")

    def load_data(self):
        self.generation_prompts = None
        with open(self.test_file, 'r') as f:
            self.generation_prompts = json.load(f)

    def slice_activations(self, activations):
        last_token = activations[:, -1, :].copy()
        return [last_token], 1

    def evaluate(self):

        self.all_layer_wise_activations = []
        self.all_head_wise_activations = []
        self.all_mlp_wise_activations = []
        self.sample_arrays = []
        self.all_probabilities = []

        total_trace = 0

        # 1. Collect Biased Activations (Existing Logic)
        for _target, generated_sentences in tqdm(self.generation_prompts.items(), desc="Evaluating generation prompts"):
            sentences = generated_sentences
            for raw_sentence in sentences:
                sentence, bias_concept = preprocess(raw_sentence, self.sentence_suffix)
                if sentence is None:
                    continue
                if bias_concept != _target:
                    continue
                total_trace += 1

                layer_wise_activations, head_wise_activations, mlp_wise_activations, probabilities = get_llama_activations_bau(
                    self.model, self.tok, sentence)

                layer_act, _len = self.slice_activations(layer_wise_activations)
                self.all_layer_wise_activations.extend(layer_act)
                head_act, _len = self.slice_activations(head_wise_activations)
                self.all_head_wise_activations.extend(head_act)
                mlp_act, _len = self.slice_activations(mlp_wise_activations)

                # Check consistency
                if len(mlp_act) != _len:
                    _len = len(mlp_act)
                self.all_mlp_wise_activations.extend(mlp_act)
                self.sample_arrays.append(_len)

                # Record probabilities
                predicted_token = self.tok.decode([probabilities.index(max(probabilities))])
                self.partial_results["predicted_token"].append(predicted_token)
                for item in self.attr_words:
                    attr_probs = [probabilities[token_id] for token_id in self.attr_words_tokens_map[item]]
                    self.partial_results[f"prob_{item}"].append(sum(attr_probs))
                self.partial_results["prompt"].append(sentence)
                self.all_probabilities.append(probabilities)

        # 2. [Strategy 1 New] Collect Neutral Activations
        logger.info("Collecting neutral activations for Contrastive Steering...")
        neutral_mlp_activations = []

        # [Strategy 1 Update] 使用特定类别的 Neutral Prompts
        current_neutral_prompts = get_neutral_prompts(self.protect_attr)

        for neutral_prompt in tqdm(current_neutral_prompts, desc="Scanning Neutral Prompts"):
            _, _, mlp_wise_activations, _ = get_llama_activations_bau(self.model, self.tok, neutral_prompt)
            # slice to get last token
            mlp_act, _ = self.slice_activations(mlp_wise_activations)
            neutral_mlp_activations.extend(mlp_act)

        # 3. Compute Steering Vectors (Bias Mean - Neutral Mean)
        # self.all_mlp_wise_activations is a list of arrays [Layer, Dim]
        # We need to stack them to get [N, Layers, Dim] -> Mean over N -> [Layers, Dim]
        biased_stack = np.stack(self.all_mlp_wise_activations, axis=0)  # [N_bias, L, D]
        neutral_stack = np.stack(neutral_mlp_activations, axis=0)  # [N_neutral, L, D]

        self.mean_biased_activation = np.mean(biased_stack, axis=0)  # [L, D]
        self.mean_neutral_activation = np.mean(neutral_stack, axis=0)  # [L, D]
        self.steering_vectors = self.mean_biased_activation - self.mean_neutral_activation  # [L, D]

        # logger.info(f"Steering Vectors Computed. Shape: {self.steering_vectors.shape}")

        # --- Statistics & Saving ---
        # logger.info(f"Total traces sentences number: {total_trace}")
        for item in self.attr_words:
            self.results[f"prob_{item}"] = np.mean(self.partial_results[f"prob_{item}"])
            sum_item = 0
            for i in range(len(self.attr_words_map[item])):
                sum_item += self.partial_results["predicted_token"].count(self.attr_words_map[item][i])
            self.results[f"predicted_{item}"] = sum_item / len(self.generation_prompts)

        self.partial_results = [dict(zip(self.partial_results, t)) for t in zip(*self.partial_results.values())]

    def save_results(self, result_dir):
        test_name = os.path.basename(self.test_file).split(".")[0]
        with open(os.path.join(result_dir, f"res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.results, f, indent=4)

        with open(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.partial_results, f, indent=4)

        logger.info("Saving layer wise activations")
        # Save original collected data for Prober training
        np.save(os.path.join(result_dir, f"features_{self.task}_{test_name}_layer_wise.npy"),
                self.all_layer_wise_activations)
        np.save(os.path.join(result_dir, f"features_{self.task}_{test_name}_head_wise.npy"),
                self.all_head_wise_activations)
        np.save(os.path.join(result_dir, f"features_{self.task}_{test_name}_mlp_wise.npy"),
                self.all_mlp_wise_activations)
        np.save(os.path.join(result_dir, f"sample_arrays_{self.task}_{test_name}.npy"), self.sample_arrays)
        np.save(os.path.join(result_dir, f"probabilities_{self.task}_{test_name}.npy"), self.all_probabilities)

        # [Strategy 1 New] Save Steering Vectors
        logger.info("Saving Steering Vectors")
        np.save(os.path.join(result_dir, f"steering_vectors_trace_{test_name}.npy"), self.steering_vectors)




# import os
# import json
#
# import numpy as np
# import torch
# from tqdm import tqdm
#
# from evaluation.evaluate import Evaluate
# from utils.globals import *
# from baukit import Trace, TraceDict
#
# import logging
#
# logger = logging.getLogger(name='logger')
#
#
# def get_llama_activations_bau(model, tok, prompt):
#     HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
#     MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
#
#     #   input_ids = tok.encode(prompt, return_tensors="pt")
#     device = next(model.parameters()).device
#
#     input_ids = tok.encode(prompt, return_tensors="pt").to(device)
#
#     with torch.no_grad():
#         with TraceDict(model, HEADS + MLPS) as ret:
#             output = model(input_ids, output_hidden_states=True)
#
#         logits = output.logits.float()
#         probabilities = logits.softmax(dim=2)[:, -1, :].squeeze()
#
#         hidden_states = output.hidden_states
#         hidden_states = torch.stack(hidden_states, dim=0).squeeze()
#         hidden_states = hidden_states.detach().cpu().numpy()
#         head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
#         head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
#         mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
#         mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()
#
#     return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, probabilities.tolist()
#
#
# def preprocess(text: str, sentence_suffix: str):
#     if text.count("[") != 2:
#         return None, None
#     bias_concept = text.split("[")[1].split("]")[0]
#     social_group = text.split("[")[2].split("]")[0]
#     clean_sentence = text.replace(f"[{bias_concept}]", bias_concept.lower()).split(f"[")[0].strip()
#     return clean_sentence, bias_concept.lower()
#
#
# class EvaluateTrace(Evaluate):
#
#     def __init__(self, model, tok, test_file, task, protect_attr='gender'):
#         super().__init__(model, tok, test_file, task)
#
#         assert self.task == "trace", f"Task class mismatch:, expected 'trace', got '{self.task}' instead"
#         self.protect_attr = protect_attr
#         self.load_attr(protect_attr)
#
#         self.results = {}
#         self.partial_results = {}
#         for item in self.attr_words:
#             self.results[f"prob_{item}"] = 0.
#             self.results[f"predicted_{item}"] = 0.
#             self.partial_results[f"prob_{item}"] = []
#         self.partial_results[f"predicted_token"] = []
#         self.partial_results["prompt"] = []
#
#         self.load_data()
#
#     def load_attr(self, protect_attr):
#         def load_attr_words(protect_attr, _label):
#             attr_words_base_dir = "../biased_knowledge"
#             if protect_attr in ['Age', 'Disability_status', 'Gender_identity',
#                                 'Nationality', 'Physical_appearance',
#                                 'Race_ethnicity', 'SES',
#                                 'Sexual_orientation', 'Religion']:
#                 file = os.path.join(attr_words_base_dir, protect_attr, "group.json")
#                 with open(file, 'r') as f:
#                     data = json.load(f)
#                     attr_words = sorted(list(data.keys()))
#                     return attr_words, data
#             else:
#                 sub_dir = os.path.join(attr_words_base_dir, protect_attr, _label + ".txt")
#                 with open(sub_dir, 'r') as f:
#                     lines = f.readlines()
#                     word_list = [line.strip() for line in lines if line.strip() != ""]
#                     word_list = list(set(word_list))
#                     f.close()
#                     return word_list
#
#         self.attr_words = []
#         self.attr_words_map = {}
#
#         if protect_attr in ['Age', 'Disability_status', 'Gender_identity',
#                             'Nationality', 'Physical_appearance',
#                             'Race_ethnicity', 'SES',
#                             'Sexual_orientation', 'Religion']:
#             self.attr_words, self.attr_words_map = load_attr_words(protect_attr, None)
#             self.sentence_suffix = f" [{protect_attr}] group."
#
#         self.attr_words_tokens_map = {}
#         for item in self.attr_words:
#             self.attr_words_tokens_map[item] = [self.tok.encode(token)[0] for token in self.attr_words_map[item]]
#             self.attr_words_tokens_map[item] = list(set(self.attr_words_tokens_map[item]))
#         logger.info(f"attr_words: {self.attr_words}")
#         logger.info(f"attr_words_map: {self.attr_words_map}")
#         logger.info(f"attr_words_tokens_map: {self.attr_words_tokens_map}")
#
#     def load_data(self):
#         self.generation_prompts = None
#         with open(self.test_file, 'r') as f:
#             self.generation_prompts = json.load(f)
#
#     def slice_activations(self, activations):
#         last_token = activations[:, -1, :].copy()
#         return [last_token], 1
#
#     def evaluate(self):
#
#         self.all_layer_wise_activations = []
#         self.all_head_wise_activations = []
#         self.all_mlp_wise_activations = []
#         self.sample_arrays = []
#         self.all_probabilities = []
#
#         total_trace = 0
#         for _target, generated_sentences in tqdm(self.generation_prompts.items(), desc="Evaluating generation prompts"):
#             sentences = generated_sentences
#             for raw_sentence in sentences:
#                 sentence, bias_concept = preprocess(raw_sentence, self.sentence_suffix)
#                 if sentence is None:
#                     continue
#                 if bias_concept != _target:
#                     continue
#                 total_trace += 1
#                 logger.info(f"sentence: {sentence}, bias_concept: {bias_concept}")
#
#                 layer_wise_activations, head_wise_activations, mlp_wise_activations, probabilities = get_llama_activations_bau(
#                     self.model, self.tok, sentence)
#
#                 layer_act, _len = self.slice_activations(layer_wise_activations)
#                 self.all_layer_wise_activations.extend(layer_act)
#                 head_act, _len = self.slice_activations(head_wise_activations)
#                 self.all_head_wise_activations.extend(head_act)
#                 mlp_act, _len = self.slice_activations(mlp_wise_activations)
#                 if len(mlp_act) != _len:
#                     logger.info(f"{sentence} fail to capture {bias_concept}")
#                     _len = len(mlp_act)
#                 self.all_mlp_wise_activations.extend(mlp_act)
#                 self.sample_arrays.append(_len)
#
#                 predicted_token = self.tok.decode([probabilities.index(max(probabilities))])
#                 self.partial_results["predicted_token"].append(predicted_token)
#                 for item in self.attr_words:
#                     attr_probs = [probabilities[token_id] for token_id in self.attr_words_tokens_map[item]]
#                     self.partial_results[f"prob_{item}"].append(sum(attr_probs))
#                 self.partial_results["prompt"].append(sentence)
#
#                 self.all_probabilities.append(probabilities)
#
#         logger.info(f"Total traces sentences number: {total_trace}")
#         for item in self.attr_words:
#             self.results[f"prob_{item}"] = np.mean(self.partial_results[f"prob_{item}"])
#
#             sum_item = 0
#             for i in range(len(self.attr_words_map[item])):
#                 sum_item += self.partial_results["predicted_token"].count(self.attr_words_map[item][i])
#             self.results[f"predicted_{item}"] = sum_item / len(self.generation_prompts)
#
#         _all_partial = []
#         for item in self.attr_words:
#             _all_partial.append(self.partial_results[f"prob_{item}"])
#         _all_partial = np.column_stack(_all_partial)
#         row_diff = np.max(_all_partial, axis=1) - np.min(_all_partial, axis=1)
#         self.partial_results["diff_prob"] = row_diff
#         self.results["diff_prob"] = np.mean(row_diff)
#
#         self.partial_results = [dict(zip(self.partial_results, t)) for t in zip(*self.partial_results.values())]
#
#     def save_results(self, result_dir):
#         test_name = os.path.basename(self.test_file).split(".")[0]
#         with open(os.path.join(result_dir, f"res_{self.task}_{test_name}.json"), 'w') as f:
#             json.dump(self.results, f, indent=4)
#
#         with open(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"), 'w') as f:
#             json.dump(self.partial_results, f, indent=4)
#
#         logger.info("Saving layer wise activations")
#         _name = os.path.join(result_dir, f"features_{self.task}_{test_name}_layer_wise.npy")
#         np.save(_name, self.all_layer_wise_activations)
#
#         logger.info("Saving head wise activations")
#         _name = os.path.join(result_dir, f"features_{self.task}_{test_name}_head_wise.npy")
#         np.save(_name, self.all_head_wise_activations)
#
#         logger.info("Saving mlp wise activations")
#         _name = os.path.join(result_dir, f"features_{self.task}_{test_name}_mlp_wise.npy")
#         np.save(_name, self.all_mlp_wise_activations)
#
#         logger.info("Saving sample arrays")
#         _name = os.path.join(result_dir, f"sample_arrays_{self.task}_{test_name}.npy")
#         np.save(_name, self.sample_arrays)
#
#         logger.info("Saving probabilities")
#         _name = os.path.join(result_dir, f"probabilities_{self.task}_{test_name}.npy")
#         np.save(_name, self.all_probabilities)