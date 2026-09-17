import copy
import json
import time
import torch
import pickle
import os

from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from utils.globals import RESULTS_DIR, HF_NAMES, plot_line
from utils.dataset import TraceDataset
from evaluation import Evaluate, EvaluateBBQ, EvaluateBiasAsker, EvaluateTrace, EvaluateMMLU
from utils.logger import setup_logger
from evaluation.stereoset_eval import EvaluateStereoSet
from utils.model import DNN, NeutralizationAdapter, evaluate_dnn, train_dnn, plot_confusion_matrix

from utils.attack import pgd_attack_general


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--exp_suffix', type=str, default='')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--trace_file', type=str, default='')
    parser.add_argument('--protect_attr', type=str, default='gender')

    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size',
                        default=0.2)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--class_num', type=int, default=10)
    # parser.add_argument('--num_mlps', type=int, default=9)

    parser.add_argument('--loss_type', type=str, default='KL', help='loss type')
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--bbq_cate', type=str)
    parser.add_argument('--eval_dataset_name', type=str)
    parser.add_argument('--noise_factor', type=float, default=5)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--eps_iter', type=int, default=4)  ### noise_factor/eps_iter
    parser.add_argument('--eval_task', type=str, help="just for eval")
    parser.add_argument('--num_mlps', type=int, default=12)

    # [Tuning] 调高默认阈值，减少对模糊样本的误伤
    parser.add_argument('--gating_threshold', type=float, default=0.05,
                        help='门控阈值：当 KL 散度大于此值时触发中和。建议 0.03-0.1')

    # Adapter / Distillation Args
    parser.add_argument('--use_adapter', action='store_true', default=True, help='是否使用轻量级 Adapter 替代 PGD 进行推理')
    parser.add_argument('--distill_epochs', type=int, default=50, help='Adapter 蒸馏训练轮数')
    parser.add_argument('--adapter_hidden_dim', type=int, default=512, help='Adapter 隐藏层维度')

    # [New Arg] 一致性正则化系数
    parser.add_argument('--consistency_alpha', type=float, default=1.5, help='一致性损失权重，用于保护原始语义')

    # [Strategy 1 New Args] Contrastive Steering
    parser.add_argument('--use_steering', action='store_true', default=True, help='是否启用对比导向向量 (Steering Vectors)')
    parser.add_argument('--steering_alpha', type=float, default=0.3, help='导向向量的强度系数。正值表示减去偏见方向。')

    parser.add_argument(
        "--no_debias",
        action="store_true",
        help="只评估原始 LLaMA，不执行 trace/probe/distillation/debias 干预"
    )


    args = parser.parse_args()
    return args


class Ours:
    def __init__(self, args):
        self.args = args
        self.prepare(args)
        self.load_model(args)

    def prepare(self, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        exp_dir = os.path.join(RESULTS_DIR, "AdvDebias", args.model_name, args.eval_dataset_name,
                               args.protect_attr, args.exp_suffix)
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(exp_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        logger = setup_logger(work_dir=log_dir, logfile_name='log.txt')

        self.logger = logger
        self.exp_dir = exp_dir
        self.log_dir = log_dir

        # self.logger.info(f"args: {args}")

        self.trace_dir = os.path.join(exp_dir, "trace")
        os.makedirs(self.trace_dir, exist_ok=True)
        self.probe_dir = os.path.join(exp_dir, "probe")
        os.makedirs(self.probe_dir, exist_ok=True)
        self.debias_dir = os.path.join(exp_dir, "debias")
        os.makedirs(self.debias_dir, exist_ok=True)

        self.adapter_dir = os.path.join(exp_dir, "adapter")
        os.makedirs(self.adapter_dir, exist_ok=True)

    def load_model(self, args):
        model_name_or_path = HF_NAMES[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, add_bos_token=False)
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "</s>"
        tokenizer.unk_token = "<unk>"
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True,
                                                     torch_dtype=torch.float16, device_map="auto",
                                                     trust_remote_code=True)

        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        # model.to(self.device) # map='auto' handles this

        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads

        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_heads = num_heads

    def load_dataset(self):
        act_dataset = TraceDataset(self.trace_dir, args.trace_file, self.tokenizer, self.num_layers, self.num_heads,
                                   class_num=args.class_num, val_ratio=args.val_ratio, protect_attr=args.protect_attr)
        all_X_train, all_X_val, y_train, y_val = act_dataset.get_train_val_set()

        self.all_X_train = all_X_train
        self.all_X_val = all_X_val
        self.y_train = y_train
        self.y_val = y_val

    def run_trace(self):
        trace_start = time.time()
        # self.logger.info(f"begin to run trace")

        evaluator = EvaluateTrace(self.model, self.tokenizer,
                                  self.args.trace_file, "trace", self.args.protect_attr)

        evaluator.evaluate()
        evaluator.save_results(self.trace_dir)
        # self.logger.info(f"Trace consume time: {time.time() - trace_start}")
        self.trace_time = time.time() - trace_start

    def run_probe(self):
        args = self.args
        probe_start = time.time()
        # self.logger.info(f"begin to run probe")
        # self.logger.info(f"model device: {self.model.device}")

        probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np, all_mlp_cms_np = \
            self.train_dnn(self.all_X_train, self.all_X_val, self.y_train, self.y_val, self.model.device.index,
                           class_num=args.class_num)
        # self.logger.info(f"all_mlp_accs_np: {sorted(all_mlp_accs_np, reverse=True)}")
        # self.logger.info(f"all_mlp_f1s_np: {sorted(all_mlp_f1s_np, reverse=True)}")
        # self.logger.info(f"all_mlp_kls_np: {sorted(all_mlp_kls_np, reverse=False)}")
        # self.logger.info(f"all_mlp_mses_np: {sorted(all_mlp_mses_np, reverse=False)}")
        top_mlps = np.argsort(all_mlp_f1s_np)[::-1][:args.num_mlps]

        plot_line(all_mlp_accs_np, name=f"{self.log_dir}/mlp_accs.png")
        plot_line(all_mlp_f1s_np, name=f"{self.log_dir}/mlp_f1s.png")
        plot_line(all_mlp_kls_np, name=f"{self.log_dir}/mlp_kls.png")
        plot_line(all_mlp_mses_np, name=f"{self.log_dir}/mlp_mses.png")

        for i, mlp in enumerate(all_mlp_cms_np):
            plot_confusion_matrix(mlp, name=f"{self.log_dir}/mlp_layer_{i}_cm.png")
        # self.logger.info(f"Mlps intervened: {sorted(top_mlps)}")

        self.save_probe_results(top_mlps, probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np,
                                self.probe_dir)
        # self.logger.info(f"Probe consume time: {time.time() - probe_start}")
        self.probe_time = time.time() - probe_start

    def train_dnn(self, all_X_train, all_X_val, y_train, y_val, device, class_num=10):
        start_time = time.time()
        all_mlp_accs = []
        all_mlp_f1s = []
        all_mlp_kls = []
        all_mlp_mses = []
        all_mlp_cms = []
        probes = []
        for layer in tqdm(range(self.num_layers), desc="train_probes"):
            X_train = all_X_train[:, layer, :]
            X_val = all_X_val[:, layer, :]
            model, loss_history = train_dnn(X_train, y_train, device=device, input_dim=X_train.shape[-1],
                                            class_num=class_num)
            accuracy, f1, avg_kl_div, avg_mse, cm = evaluate_dnn(model, X_val, y_val, device=device, layer=layer)
            all_mlp_accs.append(accuracy)
            all_mlp_f1s.append(f1)
            all_mlp_kls.append(avg_kl_div)
            all_mlp_mses.append(avg_mse)
            probes.append(model)
            all_mlp_cms.append(cm)
        all_mlp_accs_np = np.array(all_mlp_accs)
        all_mlp_f1s_np = np.array(all_mlp_f1s)
        all_mlp_kls_np = np.array(all_mlp_kls)
        all_mlp_mses_np = np.array(all_mlp_mses)
        all_mlp_cms_np = all_mlp_cms
        return probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np, all_mlp_cms_np

    def save_probe_results(self, top_mlps, probes, all_mlp_accs_np, all_mlp_f1s_np, all_mlp_kls_np, all_mlp_mses_np,
                           save_dir):
        models_dict = {f"model_{i}": model.state_dict() for i, model in enumerate(probes)}
        torch.save(models_dict, os.path.join(save_dir, 'all_dnn_models.pth'))

        information_dict = {"all_mlp_accs_np": all_mlp_accs_np,
                            "all_mlp_f1s_np": all_mlp_f1s_np,
                            "all_mlp_kls_np": all_mlp_kls_np,
                            "all_mlp_mses_np": all_mlp_mses_np,
                            "top_mlps": top_mlps
                            }
        with open(os.path.join(save_dir, 'information_dict.pkl'), 'wb') as f:
            pickle.dump(information_dict, f)

    def load_probe_results(self, save_dir, device, input_dim=4096, class_num=3):
        models_dict = torch.load(os.path.join(save_dir, 'all_dnn_models.pth'), map_location=device)
        probes = []
        for i in range(len(models_dict)):
            model = DNN(input_dim=input_dim, class_num=class_num).to(device)
            model.load_state_dict(models_dict[f"model_{i}"])
            model = model.to(device)
            probes.append(model)
        information_dict = pickle.load(open(os.path.join(save_dir, 'information_dict.pkl'), 'rb'))
        return information_dict["top_mlps"], probes, information_dict["all_mlp_accs_np"], information_dict[
            "all_mlp_f1s_np"], information_dict["all_mlp_kls_np"], information_dict["all_mlp_mses_np"]

    def slice_activations(self, act, last_indexs=[(0, 1)]):
        activations = act.clone().detach()
        last_start, last_end = last_indexs[0]
        last_token = activations[:, last_start:last_end, :]
        return last_token, last_token.shape[1]

    def restore_activations(self, activations, sliced_activations, last_indexs=[(0, 1)]):
        activations_restored = activations.clone().detach()
        last_start, last_end = last_indexs[0]
        activations_restored[:, last_start:last_end, :] = sliced_activations
        return activations_restored

    def get_interventions_dict_dnns_for_mlp(self, top_heads, probes, tuning_activations, noise_factor=3):
        interventions = {}
        for layer in top_heads:
            interventions[f"model.layers.{layer}.mlp"] = []

        for layer in top_heads:
            dnn_model = probes[layer]
            activations = tuning_activations[:, layer, :]
            activations = activations.astype(np.float64)
            l2_norms = np.linalg.norm(activations, axis=1)

            std = np.std(l2_norms)
            noise_level = noise_factor * std
            interventions[f"model.layers.{layer}.mlp"].append((dnn_model, noise_level))
        return interventions

    def run_distillation(self):
        args = self.args
        if not args.use_adapter:
            return

        self.logger.info("Begin Distillation: Transferring PGD neutralization to Adapter")

        _, probes, _, all_mlp_f1s_np, _, _ = self.load_probe_results(
            self.probe_dir, self.model.device,
            input_dim=self.all_X_train.shape[-1], class_num=args.class_num
        )

        candidate_mlps = np.where(all_mlp_f1s_np > 0.4)[0]
        interventions_cfg = self.get_interventions_dict_dnns_for_mlp(
            candidate_mlps, probes, self.all_X_train, noise_factor=args.noise_factor
        )

        adapters = {}
        for layer_idx in candidate_mlps:
            layer_name = f"model.layers.{layer_idx}.mlp"
            # self.logger.info(f"Distilling Adapter for layer: {layer_idx}")

            probe_model = probes[layer_idx]
            noise_level = interventions_cfg[layer_name][0][1]

            X_layer = self.all_X_train[:, layer_idx, :]
            X_tensor = torch.tensor(X_layer, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                logits = probe_model(X_tensor)
                probs = torch.softmax(logits, dim=-1)
                num_classes = probs.size(-1)
                target_dist = torch.ones_like(probs) / num_classes
                risks = torch.nn.functional.kl_div(probs.log(), target_dist, reduction='none').sum(dim=1)

            mask = risks > args.gating_threshold
            if mask.sum() == 0:
                continue

            X_biased = X_tensor[mask].unsqueeze(1)
            X_pgd_input = X_biased.permute(1, 0, 2)

            with torch.enable_grad():
                adv_embeddings, _, _ = pgd_attack_general(
                    model=probe_model,
                    embeddings=X_pgd_input,
                    epsilon=noise_level,
                    eps_iter=noise_level / args.eps_iter,
                    num_iter=args.num_iter,
                    device=self.device,
                    loss_type=args.loss_type
                )

            Y_target = adv_embeddings.squeeze(0).detach()
            X_input = X_biased.squeeze(1).detach()

            adapter = NeutralizationAdapter(input_dim=X_input.shape[-1], hidden_dim=args.adapter_hidden_dim).to(
                self.device)
            optimizer = optim.Adam(adapter.parameters(), lr=1e-3)

            # [Optimization] MSE for fitting, plus Consistency Loss
            criterion_mse = nn.MSELoss()

            dataset = TensorDataset(X_input, Y_target)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            adapter.train()
            for epoch in range(args.distill_epochs):
                total_loss = 0
                for x_batch, y_batch in loader:
                    optimizer.zero_grad()
                    pred = adapter(x_batch)

                    loss_mse = criterion_mse(pred, y_batch)

                    loss_consistency = torch.mean((pred - x_batch) ** 2)

                    loss = loss_mse + args.consistency_alpha * loss_consistency

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            adapters[layer_name] = adapter

        torch.save({k: v.state_dict() for k, v in adapters.items()}, os.path.join(self.adapter_dir, "adapters.pth"))
        # self.logger.info(f"Distillation finished. Adapters saved to {self.adapter_dir}")

    def run_debias(self):
        args = self.args
        debias_start = time.time()
        self.logger.info(
            f"Begin sample-adaptive debiasing (Use Adapter: {args.use_adapter}, Use Steering: {args.use_steering})")

        _, probes, _, all_mlp_f1s_np, _, _ = self.load_probe_results(
            self.probe_dir, self.model.device,
            input_dim=self.all_X_train.shape[-1], class_num=args.class_num
        )

        candidate_mlps = np.where(all_mlp_f1s_np > 0.4)[0]
        interventions = self.get_interventions_dict_dnns_for_mlp(
            candidate_mlps, probes, self.all_X_train, noise_factor=args.noise_factor
        )

        trained_adapters = {}
        if args.use_adapter:
            adapter_path = os.path.join(self.adapter_dir, "adapters.pth")
            if os.path.exists(adapter_path):
                state_dicts = torch.load(adapter_path, map_location=self.device)
                for layer_name, state_dict in state_dicts.items():
                    dim = state_dict['down_proj.weight'].shape[1]
                    adapter = NeutralizationAdapter(input_dim=dim, hidden_dim=args.adapter_hidden_dim).to(self.device)
                    adapter.load_state_dict(state_dict)

                    # [BUGFIX] Convert adapter to float16 to match LLM half-precision activations
                    adapter.to(dtype=torch.float16)

                    adapter.eval()
                    trained_adapters[layer_name] = adapter

        # [Strategy 1] Load Steering Vectors
        steering_vectors = None
        if args.use_steering:
            trace_name = os.path.basename(args.trace_file).split(".")[0]
            vector_path = os.path.join(self.trace_dir, f"steering_vectors_trace_{trace_name}.npy")
            if os.path.exists(vector_path):
                steering_vectors = np.load(vector_path)  # Shape [Layers, Dim]
                # self.logger.info(f"Loaded Steering Vectors from {vector_path}")
            else:
                self.logger.warning(f"Steering vectors not found at {vector_path}, skipping steering.")

        # 运行bbq数据集时候的方法
        # def adv_remove_adaptive(head_output, layer_name, last_indexs=[(0, 1)]):
        #     for dnn_model, noise_level in interventions[layer_name]:
        #         sliced_activation, _ = self.slice_activations(head_output, last_indexs)
        #         device = next(dnn_model.parameters()).device
        #
        #         # [Strategy 1] Activation Steering (Apply before gating or other interventions)
        #         if args.use_steering and steering_vectors is not None:
        #             try:
        #                 layer_idx = int(layer_name.split('.')[2])
        #                 vec = torch.tensor(steering_vectors[layer_idx], dtype=sliced_activation.dtype, device=device)
        #                 steered_activation = sliced_activation - (args.steering_alpha * vec.unsqueeze(0).unsqueeze(0))
        #                 sliced_activation = steered_activation
        #             except Exception as e:
        #                 pass
        #
        #         # Adaptive Gating
        #         gate_input = sliced_activation[:, -1, :].to(dtype=torch.float32, device=device)
        #
        #         with torch.no_grad():
        #             logits = dnn_model(gate_input)
        #             probs = torch.softmax(logits, dim=-1)
        #             num_classes = probs.size(-1)
        #             target_dist = torch.ones_like(probs) / num_classes
        #             risk = torch.nn.functional.kl_div(probs.log(), target_dist, reduction='batchmean')
        #
        #         # [Tuning] Use higher threshold to allow Ambiguous samples to pass without disruption
        #         if risk < args.gating_threshold:
        #             if args.use_steering:
        #                 sliced_activation = sliced_activation.to(dtype=head_output.dtype,
        #                                                          device=head_output.device.index)
        #                 head_output = self.restore_activations(head_output, sliced_activation, last_indexs)
        #             continue
        #
        #         # Intervention (Adapter or PGD)
        #         if args.use_adapter and layer_name in trained_adapters:
        #             inp = sliced_activation.squeeze(1)
        #             out = trained_adapters[layer_name](inp)
        #             adv_output = out.unsqueeze(1)
        #         else:
        #             adv_output, loss, use_iter = pgd_attack_general(
        #                 model=dnn_model,
        #                 embeddings=sliced_activation,
        #                 epsilon=noise_level,
        #                 eps_iter=noise_level / args.eps_iter,
        #                 num_iter=args.num_iter,
        #                 device=device,
        #                 loss_type=args.loss_type
        #             )
        #
        #         sliced_activation = adv_output.to(dtype=head_output.dtype, device=head_output.device.index)
        #         head_output = self.restore_activations(head_output, sliced_activation, last_indexs)
        #
        #     return head_output

        def adv_remove_adaptive(head_output, layer_name, last_indexs=[(0, 1)]):
            for dnn_model, noise_level in interventions[layer_name]:
                sliced_activation, _ = self.slice_activations(head_output, last_indexs)
                device = next(dnn_model.parameters()).device

                # 克隆一份干净的“原始激活值”
                original_activation = sliced_activation.clone()

                # 1. 门控机制评估 (必须喂给探针干净的原始激活值)
                gate_input = original_activation[:, -1, :].to(dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits = dnn_model(gate_input)
                    probs = torch.softmax(logits, dim=-1)
                    num_classes = probs.size(-1)
                    target_dist = torch.ones_like(probs) / num_classes
                    risk = torch.nn.functional.kl_div(probs.log(), target_dist, reduction='batchmean')

                final_activation = original_activation

                # 2. [核心修复]：只有当风险大于阈值时，才允许进行任何干预！
                if risk >= args.gating_threshold:

                    # (A) 微观中和：Adapter 或 PGD
                    if args.use_adapter and layer_name in trained_adapters:
                        inp = original_activation.squeeze(1)
                        out = trained_adapters[layer_name](inp)
                        final_activation = out.unsqueeze(1)
                    else:
                        final_activation, loss, use_iter = pgd_attack_general(
                            model=dnn_model,
                            embeddings=original_activation,
                            epsilon=noise_level,
                            eps_iter=noise_level / args.eps_iter,
                            num_iter=args.num_iter,
                            device=device,
                            loss_type=args.loss_type
                        )

                    if args.use_steering and steering_vectors is not None:
                        try:
                            layer_idx = int(layer_name.split('.')[2])

                            raw_vec = torch.tensor(steering_vectors[layer_idx], dtype=torch.float32, device=device)

                            norm_vec = raw_vec / (torch.norm(raw_vec) + 1e-8)

                            norm_vec = norm_vec.to(dtype=final_activation.dtype)

                            steered_activation = final_activation - (args.steering_alpha * norm_vec.unsqueeze(0).unsqueeze(0))

                            final_activation = steered_activation
                        except Exception as e:
                            # 如果出现维度不对齐等任何意外，安全跳过，保护模型不崩溃
                            print(f"Steering Warning: {e}")
                            pass

                # 3. 写回大模型大脑
                final_activation = final_activation.to(dtype=head_output.dtype, device=head_output.device.index)
                head_output = self.restore_activations(head_output, final_activation, last_indexs)

            return head_output

        self.run_evaluation_on_task(self.model, self.tokenizer,
                                    args.eval_task, args.test_file,
                                    self.debias_dir, interventions, adv_remove_adaptive)

        # self.logger.info(f"Debias consume time: {time.time() - debias_start}")
        self.debias_time = time.time() - debias_start

    def run_evaluation_on_task(self, model, tokenizer, task, test_file, output_dir, interventions={},
                               intervention_fn=None):
        bbq_sample_counts = {
            "Age": 3680,
            "Disability_status": 1556,
            "Gender_identity": 5672,
            "Nationality": 3080,
            "Physical_appearance": 1576,
            "Race_ethnicity": 6880,
            "Religion": 1200,
            "SES": 6864,
            "Sexual_orientation": 864,
        }


        if task == "mmlu":
            evaluator = EvaluateMMLU(model, tokenizer, test_file, task,
                                     interventions, intervention_fn)
        elif task == "bbq":
            evaluator = EvaluateBBQ(model, tokenizer, test_file, task,
                                    interventions, intervention_fn,
                                    protect_attr=self.args.protect_attr,
                                    category=self.args.bbq_cate, debias_method="interven")


        elif task == "biasasker":
            evaluator = EvaluateBiasAsker(model, tokenizer, test_file, task,
                                          interventions, intervention_fn,
                                          protect_attr=self.args.protect_attr,
                                          category=self.args.bbq_cate, debias_method="interven")
        elif task == "stereoset":
            evaluator = EvaluateStereoSet(model, tokenizer, test_file, task,
                                          interventions, intervention_fn)
        else:
            raise ValueError(f"Unknown task {task}")

        evaluator: Evaluate

        self.logger.info(f"================开始评估{task}数据集====================")

        # 记录推理时间
        inference_start = time.time()
        evaluator.evaluate()
        inference_end = time.time()

        total_inference_time = inference_end - inference_start

        total_samples = bbq_sample_counts[self.args.bbq_cate]
        # 计算单条样本的平均推理/干预耗时 (毫秒/条)
        avg_infer_time_ms = (total_inference_time / total_samples) * 1000

        try:
            self.logger.info(f"{task} 总推理时间：{total_inference_time:.2f} s")
            self.logger.info(f"{task} 单样本平均推理时间：{avg_infer_time_ms:.2f} ms/sample")
        except Exception as e:
            pass

        evaluator.save_results(output_dir)

        # 👇【新增核心逻辑：保存时间统计到独立的 JSON】👇
        time_metrics = {
            "task": task,
            "model_name": self.args.model_name,
            "category": self.args.bbq_cate if hasattr(self.args, 'bbq_cate') else "all",
            "train_time_seconds": getattr(self, 'total_train_time', 0.0),  # 探针和Adapter的离线训练时间
            "total_inference_time_seconds": total_inference_time,  # 在该测试集上的总推理时间
            "total_samples": total_samples,
            "avg_infer_time_ms_per_sample": avg_infer_time_ms  # 【核心打榜数据】单样本推理时间
        }

        time_save_path = os.path.join(output_dir, f"time_cost_{task}_{self.args.exp_suffix}.json")
        with open(time_save_path, 'w') as f:
            json.dump(time_metrics, f, indent=4)
        self.logger.info(f"⏱️ 时间统计已保存至: {time_save_path}")


    # def run_pipe(self):
    #     import time
    #     global_start = time.time()
    #
    #     self.run_trace()
    #     self.load_dataset()
    #
    #     # 1. 记录 Train Time (探针训练 + Adapter 蒸馏)
    #     train_start = time.time()
    #     self.run_probe()
    #     if self.args.use_adapter:
    #         self.run_distillation()
    #     train_end = time.time()
    #     self.total_train_time = train_end - train_start  # 保存为类属性，供后续使用
    #
    #     # 2. 运行推理与评估 (Inference / Debias)
    #     self.run_debias()
    #
    #     global_end = time.time()
    #     self.logger.info(f"Train (Probe+Adapter) time: {self.total_train_time:.2f} s")
    #     self.logger.info(f"Total pipeline time: {global_end - global_start:.2f} s")

    def run_pipe(self):
        import time
        global_start = time.time()

        # ============================
        # Original LLaMA Evaluation
        # ============================
        if self.args.no_debias:
            self.logger.info("Running Original LLaMA evaluation without any debiasing intervention.")

            self.run_evaluation_on_task(
                self.model,
                self.tokenizer,
                self.args.eval_task,
                self.args.test_file,
                self.debias_dir,
                interventions={},
                intervention_fn=None
            )

            global_end = time.time()
            self.logger.info(f"Original evaluation total time: {global_end - global_start:.2f} s")
            return

        self.run_trace()
        self.load_dataset()

        train_start = time.time()
        self.run_probe()
        if self.args.use_adapter:
            self.run_distillation()
        train_end = time.time()
        self.total_train_time = train_end - train_start

        self.run_debias()

        global_end = time.time()
        self.logger.info(f"Train (Probe+Adapter) time: {self.total_train_time:.2f} s")
        self.logger.info(f"Total pipeline time: {global_end - global_start:.2f} s")

if __name__ == "__main__":
    args = args_parser()
    start_time = time.time()
    adv = Ours(args)
    adv.run_pipe()
    end_time = time.time()
    adv.logger.info(
        f"Main program Total time taken: {end_time - start_time} seconds; {((end_time - start_time) / 60)} minutes")