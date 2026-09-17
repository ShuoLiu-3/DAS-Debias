from functools import partial
import json
import os
import torch
from math import log, prod

from tqdm import tqdm

from evaluation import Evaluate
from baukit import Trace, TraceDict

class EvaluateMMLU(Evaluate):
    def __init__(self, model, tok, test_file, task, interventions={}, intervention_fn=None):

        super().__init__(model,tok, test_file, task, interventions, intervention_fn)
        assert task == "mmlu", f"Task class mismatch:, expected 'mmlu', got '{task}' instead"

        self.load_data()
        self.partial_results = {}
    def load_data(self):
        with open(self.test_file) as f:
            self.dataset = json.load(f)
        

    # def evaluate(self):
    #     def score(a, n, norm):
    #         logx = lambda x: log(x) if x > 0 else float('-inf')
    #         return prod(a) / len(a), prod(a) / norm, sum([logx(x) for x in a]) - sum([logx(x) for x in n]), prod(a),  prod(a)**(1/len(a))
    #
    #     def id(head_output, layer_name):
    #         return head_output
    #
    #     if self.interventions == {}:
    #         layers_to_intervene = []
    #         intervene = id
    #     else:
    #         layers_to_intervene = list(self.interventions.keys())
    #         intervene = partial(self.intervention_fn, start_edit_location='lt')
    #
    #     partial_results = []
    #     for d in tqdm(self.dataset, desc="Evaluating QA prompts"):
    #         prompt = d['question']
    #         norm = "Answer:"
    #         prompt_tokens = self.tok.encode(prompt, return_tensors="pt")
    #         q_start_edit_location = len(prompt_tokens[0])
    #         norm_tokens = self.tok.encode(norm, return_tensors="pt")
    #         n_start_edit_location = len(norm_tokens[0])
    #
    #
    #         ans = []
    #         correct = d['answerKey']
    #         labels = d['choices']['label']
    #         texts = d['choices']['text']
    #         subject = d['subject']
    #         for label, text in zip(labels, texts):
    #             choice_tokens = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
    #             choice_len = choice_tokens.shape[1]
    #
    #             start_index = q_start_edit_location
    #             end_index = q_start_edit_location + choice_len + 1
    #             last_indexs = [(start_index, end_index)]
    #
    #             if self.interventions == {}:
    #                 intervene = id
    #             else:
    #                 intervene = partial(self.intervention_fn, last_indexs=last_indexs)
    #
    #             with torch.no_grad():
    #                 with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret:
    #                     probs_q = torch.softmax(self.model.forward(torch.cat((prompt_tokens, choice_tokens), 1)).logits[:,-choice_len-1:-1,:].float(), dim=-1)
    #                     probs_q = probs_q[0, list(range(choice_len)), choice_tokens[0]]
    #
    #             start_index = n_start_edit_location
    #             end_index = n_start_edit_location + choice_len + 1
    #             last_indexs = [(start_index, end_index)]
    #             if self.interventions == {}:
    #                 intervene = id
    #             else:
    #                 intervene = partial(self.intervention_fn, last_indexs=last_indexs)
    #
    #             with torch.no_grad():
    #                 with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret:
    #                     probs_n = torch.softmax(self.model.forward(torch.cat((norm_tokens, choice_tokens), 1)).logits[:,-choice_len-1:-1,:].float(), dim=-1)
    #                     probs_n = probs_n[0, list(range(choice_len)), choice_tokens[0]]
    #
    #             answer_scores = score(probs_q.tolist(), probs_n.tolist(), len(text))
    #             ans.append((label, answer_scores))
    #
    #
    #         partial_results.append((
    #             sorted(ans, key=lambda x: x[1][0])[-1][0] == correct,
    #             sorted(ans, key=lambda x: x[1][1])[-1][0] == correct,
    #             sorted(ans, key=lambda x: x[1][2])[-1][0] == correct,
    #             sorted(ans, key=lambda x: x[1][3])[-1][0] == correct,
    #             sorted(ans, key=lambda x: x[1][4])[-1][0] == correct
    #         ))
    #         if subject not in self.partial_results:
    #             self.partial_results[subject] = []
    #         self.partial_results[subject].append((
    #             sorted(ans, key=lambda x: x[1][0])[-1][0] == correct,
    #             sorted(ans, key=lambda x: x[1][1])[-1][0] == correct,
    #             sorted(ans, key=lambda x: x[1][2])[-1][0] == correct,
    #             sorted(ans, key=lambda x: x[1][3])[-1][0] == correct,
    #             sorted(ans, key=lambda x: x[1][4])[-1][0] == correct
    #         ))
    #
    #     self.results = {
    #         'per_token_prob_root': sum([x[4] for x in partial_results])/len(partial_results),
    #         'per_token_prob': sum([x[0] for x in partial_results])/len(partial_results),
    #         'per_char_prob': sum([x[1] for x in partial_results])/len(partial_results),
    #         'normed_prob': sum([x[2] for x in partial_results])/len(partial_results),
    #         'unnormed_prob': sum([x[3] for x in partial_results])/len(partial_results),
    #     }

    def evaluate(self):
        def score(a, n, norm):
            logx = lambda x: log(x) if x > 0 else float('-inf')
            return prod(a) / len(a), prod(a) / norm, sum([logx(x) for x in a]) - sum([logx(x) for x in n]), prod(
                a), prod(a) ** (1 / len(a))

        def id(head_output, layer_name):
            return head_output

        if self.interventions == {}:
            layers_to_intervene = []
            intervene = id
        else:
            layers_to_intervene = list(self.interventions.keys())
            intervene = partial(self.intervention_fn, start_edit_location='lt')

        partial_results = []
        for d in tqdm(self.dataset, desc="Evaluating QA prompts"):
            prompt = d['question']
            norm = "Answer:"

            # 【修复点 1】：将 prompt_tokens 放到模型所在的 GPU 上
            prompt_tokens = self.tok.encode(prompt, return_tensors="pt").to(self.model.device)
            q_start_edit_location = len(prompt_tokens[0])

            # 【修复点 2】：将 norm_tokens 放到模型所在的 GPU 上
            norm_tokens = self.tok.encode(norm, return_tensors="pt").to(self.model.device)
            n_start_edit_location = len(norm_tokens[0])

            ans = []
            correct = d['answerKey']
            labels = d['choices']['label']
            texts = d['choices']['text']
            subject = d['subject']

            for label, text in zip(labels, texts):
                # 【修复点 3】：将 choice_tokens 放到模型所在的 GPU 上
                choice_tokens = self.tok.encode(text, add_special_tokens=False, return_tensors="pt").to(
                    self.model.device)
                choice_len = choice_tokens.shape[1]

                start_index = q_start_edit_location
                end_index = q_start_edit_location + choice_len + 1
                last_indexs = [(start_index, end_index)]

                if self.interventions == {}:
                    intervene = id
                else:
                    intervene = partial(self.intervention_fn, last_indexs=last_indexs)

                with torch.no_grad():
                    with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret:
                        probs_q = torch.softmax(
                            self.model.forward(torch.cat((prompt_tokens, choice_tokens), 1)).logits[:,
                            -choice_len - 1:-1, :].float(), dim=-1)
                        probs_q = probs_q[0, list(range(choice_len)), choice_tokens[0]]

                start_index = n_start_edit_location
                end_index = n_start_edit_location + choice_len + 1
                last_indexs = [(start_index, end_index)]
                if self.interventions == {}:
                    intervene = id
                else:
                    intervene = partial(self.intervention_fn, last_indexs=last_indexs)

                with torch.no_grad():
                    with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret:
                        probs_n = torch.softmax(
                            self.model.forward(torch.cat((norm_tokens, choice_tokens), 1)).logits[:, -choice_len - 1:-1,
                            :].float(), dim=-1)
                        probs_n = probs_n[0, list(range(choice_len)), choice_tokens[0]]

                answer_scores = score(probs_q.tolist(), probs_n.tolist(), len(text))
                ans.append((label, answer_scores))

            partial_results.append((
                sorted(ans, key=lambda x: x[1][0])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][1])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][2])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][3])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][4])[-1][0] == correct
            ))
            if subject not in self.partial_results:
                self.partial_results[subject] = []
            self.partial_results[subject].append((
                sorted(ans, key=lambda x: x[1][0])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][1])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][2])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][3])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][4])[-1][0] == correct
            ))

        self.results = {
            'per_token_prob_root': sum([x[4] for x in partial_results]) / len(partial_results),
            'per_token_prob': sum([x[0] for x in partial_results]) / len(partial_results),
            'per_char_prob': sum([x[1] for x in partial_results]) / len(partial_results),
            'normed_prob': sum([x[2] for x in partial_results]) / len(partial_results),
            'unnormed_prob': sum([x[3] for x in partial_results]) / len(partial_results),
        }
    
    def calculate_sub_acc(self, eval_file):
        subcategories = {
            "abstract_algebra": ["math"],
            "anatomy": ["health"],
            "astronomy": ["physics"],
            "business_ethics": ["business"],
            "clinical_knowledge": ["health"],
            "college_biology": ["biology"],
            "college_chemistry": ["chemistry"],
            "college_computer_science": ["computer science"],
            "college_mathematics": ["math"],
            "college_medicine": ["health"],
            "college_physics": ["physics"],
            "computer_security": ["computer science"],
            "conceptual_physics": ["physics"],
            "econometrics": ["economics"],
            "electrical_engineering": ["engineering"],
            "elementary_mathematics": ["math"],
            "formal_logic": ["philosophy"],
            "global_facts": ["other"],
            "high_school_biology": ["biology"],
            "high_school_chemistry": ["chemistry"],
            "high_school_computer_science": ["computer science"],
            "high_school_european_history": ["history"],
            "high_school_geography": ["geography"],
            "high_school_government_and_politics": ["politics"],
            "high_school_macroeconomics": ["economics"],
            "high_school_mathematics": ["math"],
            "high_school_microeconomics": ["economics"],
            "high_school_physics": ["physics"],
            "high_school_psychology": ["psychology"],
            "high_school_statistics": ["math"],
            "high_school_us_history": ["history"],
            "high_school_world_history": ["history"],
            "human_aging": ["health"],
            "human_sexuality": ["culture"],
            "international_law": ["law"],
            "jurisprudence": ["law"],
            "logical_fallacies": ["philosophy"],
            "machine_learning": ["computer science"],
            "management": ["business"],
            "marketing": ["business"],
            "medical_genetics": ["health"],
            "miscellaneous": ["other"],
            "moral_disputes": ["philosophy"],
            "moral_scenarios": ["philosophy"],
            "nutrition": ["health"],
            "philosophy": ["philosophy"],
            "prehistory": ["history"],
            "professional_accounting": ["other"],
            "professional_law": ["law"],
            "professional_medicine": ["health"],
            "professional_psychology": ["psychology"],
            "public_relations": ["politics"],
            "security_studies": ["politics"],
            "sociology": ["culture"],
            "us_foreign_policy": ["politics"],
            "virology": ["health"],
            "world_religions": ["philosophy"],
        }

        categories = {
            "humanities": ["history", "philosophy", "law"],
            "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
            "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
            "other (business, health, misc.)": ["other", "business", "health"],
        }

        def find_large_category(eval_cate):
            subcategory = subcategories[eval_cate][0]
            for category, _subcategories in categories.items():
                if subcategory in _subcategories:
                    return category
            return "other"

        def cal_mmlu_sub_score(eval_file):
            with open(eval_file, 'r') as f:
                data = json.load(f)

            acc_keys = {}
            total_results_per_category = {category: [0] * len(data[next(iter(data))][0]) for category in categories}
            correct_results_per_category = {category: [0] * len(data[next(iter(data))][0]) for category in categories}
            data_total_results = 0
            data_total_correct_results = [0] * len(data[next(iter(data))][0])
            print(data_total_correct_results)

            for key, results in data.items():
                num_evaluations = len(results[0])
                accuracy_per_evaluation = [0] * num_evaluations
                for result in results:
                    for i in range(num_evaluations):
                        if result[i]:
                            accuracy_per_evaluation[i] += 1

                total_results = len(results)
                accuracy_rates = [count / total_results for count in accuracy_per_evaluation]
                acc_keys[key] = accuracy_rates
                
                data_total_results+=total_results
                for i in range(num_evaluations):
                    data_total_correct_results[i] += accuracy_per_evaluation[i]
                
                large_category = find_large_category(key)

                for i in range(num_evaluations):
                    total_results_per_category[large_category][i] += total_results
                    correct_results_per_category[large_category][i] += accuracy_per_evaluation[i]
                    
            category_accuracy_rates = {}
            for category in categories:
                category_accuracy_rates[category] = [
                    correct / total if total > 0 else 0
                    for correct, total in zip(correct_results_per_category[category], total_results_per_category[category])
                ]
            
            data_accuracy_rates = [
                    correct / data_total_results if data_total_results > 0 else 0
                    for correct in data_total_correct_results
                ]
            
            result_dict = {
                
            }
            eval_map = {
                4: 'per_token_prob_root',
                0: 'per_token_prob',
                1: 'per_char_prob',
                2: 'normed_prob',
                3: 'unnormed_prob',
            }
            result_dict['dataset'] = {}
            for i, accuracy in enumerate(data_accuracy_rates):
                result_dict['dataset'][eval_map[i]] = accuracy

            for category, accuracies in category_accuracy_rates.items():
                result_dict[category] = {}
                for i, accuracy in enumerate(accuracies):
                    result_dict[category][eval_map[i]] = accuracy

            json.dump(result_dict, open(eval_file.replace(".json", "_sub.json"), 'w'), indent=4)

        cal_mmlu_sub_score(eval_file)
    
    def save_results(self, result_dir):
        test_name = os.path.basename(self.test_file).split(".")[0]
        with open(os.path.join(result_dir, f"res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.results, f, indent=4)

        with open(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.partial_results, f, indent=4)
            
        self.calculate_sub_acc(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"))
