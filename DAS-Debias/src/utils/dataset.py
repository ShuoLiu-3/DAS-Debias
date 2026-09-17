from collections import Counter
import json
import os
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import logging
logger = logging.getLogger("logger")

class TraceDataset(Dataset):
    def __init__(self, trace_dir, test_file, tokenizer, num_layers, num_heads, class_num=10, 
                 val_ratio=0.2, protect_attr='gender'):
        test_name = os.path.basename(test_file).split(".")[0]
        mlp_wise = os.path.join(trace_dir, f'features_trace_{test_name}_mlp_wise.npy')
        self.mlp_wise_activations = np.load(mlp_wise)
        
        # logger.info(f"mlp_wise_activations shape: {self.mlp_wise_activations.shape}")
        
        probabilities = os.path.join(trace_dir, f'probabilities_trace_{test_name}.npy')
        self.all_probabilities = np.load(probabilities)
        
        sample_file = os.path.join(trace_dir, f"sample_arrays_trace_{test_name}.npy")
        self.sample_arrays = np.load(sample_file)

        self.tokenizer = tokenizer
        self.get_attr_words(protect_attr=protect_attr, tokenizer=tokenizer)
        self.labels, self.label_map = self.deal_probabilities(class_num=class_num)
        
        self.separated_head_wise_activations, self.separated_labels = self.get_separated_activations(self.labels, self.sample_arrays, self.mlp_wise_activations)
        
        assert len(self.separated_labels) == len(self.labels), f"{len(self.separated_labels)} != {len(self.labels)}"
        assert len(self.separated_head_wise_activations) == len(self.labels), f"{len(self.separated_head_wise_activations)} != {len(self.labels)}"
        train_idxs = np.arange(len(self.labels))
        # logger.info(f"len(train_idxs): {len(train_idxs)}; len(self.labels): {len(self.labels)}")


        self.train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-val_ratio)), replace=False)
        self.val_set_idxs = np.array([x for x in train_idxs if x not in self.train_set_idxs])
        train_labels = self.labels[self.train_set_idxs]
        if len(self.val_set_idxs)>0:
            val_labels = self.labels[self.val_set_idxs]
        else:
            val_labels = []
        # logger.info(f"Train set label distribution: {Counter(np.argmax(train_labels, axis=1))}")
        if len(val_labels)>0:
            logger.info(f"Val set label distribution: {Counter(np.argmax(val_labels, axis=1))}")
        # logger.info(f"label_map is {self.label_map}")

        
    def get_attr_words(self, protect_attr, tokenizer):
        
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
                sub_dir = os.path.join(attr_words_base_dir, protect_attr, _label+".txt")
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
            self.attr_words_tokens_map[item] = [tokenizer.encode(token)[0] for token in self.attr_words_map[item]]
            self.attr_words_tokens_map[item] = list(set(self.attr_words_tokens_map[item]))
            
        # logger.info(f"attr_words: {self.attr_words}")
        # logger.info(f"attr_words_map: {self.attr_words_map}")
        # logger.info(f"attr_words_tokens_map: {self.attr_words_tokens_map}")

        self.label_map = {i:word for i,word in enumerate(self.attr_words)}

    def get_all_train(self):
        all_X_train = np.concatenate([self.separated_head_wise_activations[i] for i in self.train_set_idxs], axis = 0)
        return all_X_train
    def get_train_val_set(self):
        all_X_train = np.concatenate([self.separated_head_wise_activations[i] for i in self.train_set_idxs], axis = 0)
        all_X_val = np.concatenate([self.separated_head_wise_activations[i] for i in self.val_set_idxs], axis = 0)
                
        temp = [self.separated_labels[i] for i in self.train_set_idxs]
        y_train = np.concatenate(temp, axis = 0)
        y_val = np.concatenate([self.separated_labels[i] for i in self.val_set_idxs], axis = 0)
        # logger.info(f"all_X_train shape: {all_X_train.shape}, all_X_val shape: {all_X_val.shape}, y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")
        return all_X_train, all_X_val, y_train, y_val

    
    def deal_probabilities(self, class_num=10):
        
        candidates = []
        for probability in self.all_probabilities:
            top_token_ids = np.argsort(probability)[-class_num:][::-1]
            candidates.append(top_token_ids)
        flattened_candidates = [token_id for sublist in candidates for token_id in sublist]
        token_id_counter = Counter(flattened_candidates)
        sorted_token_ids = token_id_counter.most_common()
        # logger.info(f"Top tokens: {sorted_token_ids[:2*class_num]}")

        other_token_ids = []

        def check(token_id):
            for key in self.attr_words_tokens_map:
                if token_id in self.attr_words_tokens_map[key]:
                    return True
            return False


        for token_id, freq in sorted_token_ids:
            if check(token_id):
                continue
            if len(other_token_ids) < class_num - len(self.attr_words):
                other_token_ids.append(token_id)
            if len(other_token_ids) >= class_num - len(self.attr_words):
                break
        other_tokens = [self.tokenizer.decode([token_id]) for token_id in other_token_ids]
        # logger.info(f"Other token ids: {other_token_ids}")
        # logger.info(f"Other tokens: {other_tokens}")
        
        
        for label_id in range(len(self.attr_words), class_num):
            self.label_map[label_id] = other_tokens[label_id-len(self.attr_words)]


        labels = []
        porb_records = []
        for probability in self.all_probabilities:
            prob_list = []
            for item in self.attr_words:
                _attr_probs = [probability[token_id] for token_id in self.attr_words_tokens_map[item]]
                attr_prob = sum(_attr_probs)
                prob_list.append(attr_prob)

            other_probs = [probability[token_id] for token_id in other_token_ids]
            
            prob_list.extend(other_probs)
            total_prob = sum(prob_list)

            label = prob_list / total_prob
            labels.append(label)
            porb_records.append(total_prob)

        counts, bin_edges = np.histogram(porb_records, bins=5)
        # logger.info(f"Counts in each bin: {counts}")
        # logger.info(f"Bin edges: {bin_edges}")
        return np.array(labels), self.label_map
    

    def get_separated_activations(self, labels, sample_arrays, mlp_wise_activations):
        separated_labels = []
        idxs_to_split_at = np.cumsum(sample_arrays)   
        # logger.info(f"idxs_to_split_at: {idxs_to_split_at}; {len(idxs_to_split_at)}")
        full_labels = []
        for i in range(len(sample_arrays)):
            full_labels.extend([labels[i]] * sample_arrays[i])
 
        for i in range(len(sample_arrays)):
            if i == 0:
                separated_labels.append(full_labels[:idxs_to_split_at[i]])
            else:
                separated_labels.append(full_labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
        separated_head_wise_activations = np.split(mlp_wise_activations, idxs_to_split_at)
        separated_head_wise_activations.pop(-1)
        
        return separated_head_wise_activations, separated_labels     

    def __len__(self):
        return len(self.labels)

    
