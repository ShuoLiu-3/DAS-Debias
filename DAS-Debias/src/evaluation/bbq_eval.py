from functools import partial
import json
import os

import torch
from tqdm import tqdm
from evaluation.bbq import BBQScenario, BBQMetric
from evaluation.evaluate import Evaluate
from evaluation.bbq import Instance
from utils.globals import *
from baukit import Trace, TraceDict

import logging
logger = logging.getLogger(name='logger')




numbers_to_letters = {
    0: "A",
    1: "B",
    2: "C",
}

DEBIAS_PROMPTS = {
    "Age": "Note that the answer does not rely on age stereotypes.\n",
    "Religion": "Note that the answer does not rely on religion stereotypes.\n",
    "Disability_status": "Note that the answer does not rely on disability stereotypes.\n",
    "Gender_identity": "Note that the answer does not rely on gender stereotypes.\n",
    "Nationality": "Note that the answer does not rely on nationality stereotypes.\n",
    "Physical_appearance": "Note that the answer does not rely on physical appearance stereotypes.\n",
    "Race_ethnicity": "Note that the answer does not rely on race/ethnicity stereotypes.\n",
    "SES": "Note that the answer does not rely on socioeconomic status stereotypes.\n",
    "Sexual_orientation": "Note that the answer does not rely on sexual orientation stereotypes.\n",
}

class EvaluateBBQ(Evaluate):

    def __init__(self, model, tok, test_file, task, interventions={}, intervention_fn=None,
                 protect_attr='gender', category="Religion", 
                 debias_method="interven"):
        super().__init__(model, tok, test_file, task, interventions, intervention_fn)

        self.bbq_data_dir = "../data/BBQ"
        self.debias_method = debias_method
        assert self.task == "bbq", f"Task class mismatch:, expected 'bbq', got '{self.task}' instead"

        self.category = category
        self.results = {}
        self.partial_results = {
            "A": [],
            "B": [],
            "C": [],
            "questions": [],
            "choices": [],
            "label": [],
            "pred": [],
            "stereotype_context": [],
            "is_negative": [],
            "is_ambigous": [],
            "target": [],
            "non_target":[],
        }

        self.device = next(self.model.parameters()).device

        self.load_data()
        
    def load_data(self):

        bbqscenario = BBQScenario(subject=self.category)
        self.instances = bbqscenario.get_instances(self.bbq_data_dir)
    def evaluate(self):
        for instance in tqdm(self.instances, desc=f"Evaluating BBQ {self.category} instances"):
            instance: Instance
            prompt = instance.input.text
            
            raw_choices = instance.choices
            reference = instance.references[0]
            choices = []
            for i, choice in enumerate(raw_choices):
                choices.append(f"{choice}\n")
            prompt = f"Context: {prompt}\nChoices:\n {choices[0]}\n {choices[1]}\n {choices[2]}"
            selection_probs = []
            
            input_prompt = prompt + "\nAnswer:\n"
            
            for number, choice in enumerate(choices):
                input_prompt = prompt + "\nAnswer:\n" + choice
                
                
                if self.debias_method == "interven":
                    probs_q = self.get_prediction_probability_interven(input_prompt, choice=choice)
                elif self.debias_method == "DePrompt":
                    input_prompt = DEBIAS_PROMPTS[self.category] + input_prompt
                    probs_q = self.get_prediction_probability(input_prompt, choice)
                else:
                    probs_q = self.get_prediction_probability(input_prompt, choice)
                
                mean_log_p = torch.mean(torch.log(probs_q)).item()

                selection_probs.append((number, mean_log_p))
                self.partial_results[f"{numbers_to_letters[number]}"].append(mean_log_p)

            selection_probs = sorted(selection_probs, key=lambda x: x[1], reverse=True)
            predicted_answer = numbers_to_letters[selection_probs[0][0]]
            instance.pred_answer = predicted_answer

            self.partial_results["choices"].append(choices)
            self.partial_results["questions"].append(prompt)
            self.partial_results["label"].append(instance.references[0].tags[-4])
            self.partial_results["pred"].append(instance.pred_answer)
            self.partial_results["stereotype_context"].append(reference.tags[-7])
            self.partial_results["is_negative"].append(reference.tags[-6])
            self.partial_results["is_ambigous"].append(reference.tags[-5])
            self.partial_results["target"].append(reference.tags[-3])
            self.partial_results["non_target"].append(reference.tags[-2])
        bbqmetric = BBQMetric()
        stat = bbqmetric.evaluate_instances(self.instances)
        
        logger.info(f"{json.dumps(stat, indent=4)}")
        self.results = stat

        self.partial_results = [dict(zip(self.partial_results, t)) for t in zip(*self.partial_results.values())]


    def get_prediction_probability(self, input_prompt, choice):
        input_prompt_tokens = self.tok.encode(input_prompt, return_tensors="pt").to(self.model.device)
        choice_tokens = self.tok.encode(choice, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        choice_len = choice_tokens.shape[1]
        logits = self.model(input_prompt_tokens).logits
        probs_q = torch.softmax(logits[:,-choice_len-1:-1,:].float(), dim=-1)
        probs_q = probs_q[0, list(range(choice_len)), choice_tokens[0]]
        return probs_q

    def get_prediction_probability_interven(self, input_prompt, choice):
        input_prompt_tokens = self.tok.encode(input_prompt, return_tensors="pt").to(self.model.device)
        choice_tokens = self.tok.encode(choice, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        choice_len = choice_tokens.shape[1]
        
        start_index = input_prompt_tokens.shape[1] - choice_len
        end_index = input_prompt_tokens.shape[1] + 1
        last_indexs = [(start_index, end_index)]

        def id(head_output, layer_name): 
            return head_output
        if self.interventions == {}: 
            layers_to_intervene = []
            intervene = id
        else: 
            layers_to_intervene = list(self.interventions.keys())
            intervene = partial(self.intervention_fn, last_indexs=last_indexs)
        
        with torch.no_grad():
            with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret:
                logits = self.model(input_prompt_tokens).logits
                probs_q = torch.softmax(logits[:,-choice_len-1:-1,:].float(), dim=-1)
                probs_q = probs_q[0, list(range(choice_len)), choice_tokens[0]]
        return probs_q
    
    def save_results(self, result_dir):
        test_name = self.category
        with open(os.path.join(result_dir, f"res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.results, f, indent=4)

        with open(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.partial_results, f, indent=4)