from baukit import Trace, TraceDict
from functools import partial
import os
import json
from abc import abstractmethod
import torch

class Evaluate:

    def __init__(self, model, tok, test_file, task, interventions={}, intervention_fn=None):
        self.model = model
        self.tok = tok
        self.test_file = test_file
        self.task = task
        self.interventions = interventions
        self.intervention_fn = intervention_fn
        self.results = {}
        self.partial_results = {}

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def get_prediction_probability(self, prompt):

        device = next(self.model.parameters()).device

        input_ids = self.tok.encode(prompt, return_tensors="pt").to(device)

        logits = self.model(input_ids)[0].float()
        
        probabilities = logits.softmax(dim=2)[:,-1,:].squeeze()
        return probabilities.tolist()

    def get_prediction_probability_interven(self, prompt):
        device = next(self.model.parameters()).device
        input_ids = self.tok.encode(prompt, return_tensors="pt").to(device)

        def id(head_output, layer_name): 
            return head_output

        if self.interventions == {}: 
            layers_to_intervene = []
            intervene = id
        else: 
            layers_to_intervene = list(self.interventions.keys())
            intervene = partial(self.intervention_fn, start_edit_location='lt')

        with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret: 
            logits = self.model(input_ids)[0].float()
            
        probabilities = logits.softmax(dim=2)[:,-1,:].squeeze()

        return probabilities.tolist()

    def save_results(self, result_dir):
        test_name = os.path.basename(self.test_file).split(".")[0]
        with open(os.path.join(result_dir, f"res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.results, f, indent=4)

        with open(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.partial_results, f, indent=4)

