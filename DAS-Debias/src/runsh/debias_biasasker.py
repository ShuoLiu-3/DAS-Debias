# from concurrent.futures import ThreadPoolExecutor, as_completed
# import os
# import subprocess
#
# os.chdir('../')
#
# def run_debias(args:dict):
#     # print(args)
#     command = [
#         "python", "main.py",
#         "--exp_suffix", args["exp_suffix"],
#         "--model_name", args["model_name"],
#         "--trace_file", args["trace_file"],
#         "--protect_attr", args["protect_attr"],
#
#         "--class_num", args["class_num"],
#         "--num_mlps", args["num_mlps"],
#         "--val_ratio", args["val_ratio"],
#         "--num_epochs", args["num_epochs"],
#
#         "--loss_type", args["loss_type"],
#         "--test_file", args["test_file"],
#         "--bbq_cate", args["bbq_cate"],
#         "--eval_task", args["eval_task"],
#         "--eval_dataset_name", args["eval_dataset_name"],
#         "--noise_factor", args["noise_factor"],
#         "--num_iter", args["num_iter"],
#         "--eps_iter", args["eps_iter"],
#
#         # [核心修改 1] 传入新增的创新点参数
#         "--steering_alpha", args["steering_alpha"],
#         "--gating_threshold", args["gating_threshold"],
#     ]
#     env = os.environ.copy()  # Get the current environment variables
#     env["CUDA_VISIBLE_DEVICES"] = str("0,1,2,3,4,5,6,7")  # Set the specific CUDA device
#
#     print(command)
#     subprocess.run(command, env=env)
#
#
# if __name__ == "__main__":
#     # Set parameters
#     model_name = "llama2_chat_7B"
#     bbq_cate = "Age"
#
#     class_map = {
#         "Age": 2,
#         "Disability_status": 2,
#         "Gender_identity": 3,
#         "Nationality": 6,
#         "Physical_appearance": 2,
#         "Race_ethnicity": 9,
#         "Religion": 11,
#         "SES": 2,
#         "Sexual_orientation": 5,
#     }
#     categories = [
#         "Age",
#         "Disability_status",
#         "Gender_identity",
#         "Physical_appearance",
#         "Race_ethnicity",
#         "Religion",
#     ]
#     bias_knowledge_dir = "../biased_knowledge/corpus"
#
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         futures = []
#         exp_suffixs = ["neutralizer"]
#
#         noise_factors = {
#             "Age": 4,
#             "Disability_status":3,
#             "Gender_identity":5,
#             "Physical_appearance": 6,
#             "Race_ethnicity":8,
#             "Religion": 7,
#         }
#
#
#         num_iter = 20
#         eps_iter = 15
#
#         for bbq_cate in categories:
#             noise_factor = noise_factors[bbq_cate]
#             protect_attr = bbq_cate
#
#             trace_file = os.path.join(bias_knowledge_dir, f"{bbq_cate}_sentences.json")
#
#             para_templates = {
#                 "exp_suffix": f"exp1",
#                 "model_name": model_name,
#                 "trace_file": trace_file,
#                 "protect_attr": protect_attr,
#
#                 "class_num": f"{class_map[bbq_cate]}",
#                 "num_mlps": f"9",
#                 "val_ratio": "0.2",
#                 "num_epochs": "20",
#
#                 "loss_type": "KL",
#                 "test_file": bbq_cate,
#                 "bbq_cate": bbq_cate,
#                 "eval_task": "biasasker",
#                 "eval_dataset_name": "BiasAsker",
#                 "noise_factor": f"{noise_factor}",
#                 "num_iter":  f"{num_iter}",
#                 "eps_iter": f"{eps_iter}",
#             }
#
#             futures.append(executor.submit(run_debias, para_templates))
#             break
#         # 等待并获取任务结果
#         for future in as_completed(futures):
#             result = future.result()  # 获取每个任务的结果
#             print(result)
#
#
#


from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess

os.chdir('../')


def run_debias(args: dict):
    # print(args)
    command = [
        "python", "main.py",
        "--exp_suffix", args["exp_suffix"],
        "--model_name", args["model_name"],
        "--trace_file", args["trace_file"],
        "--protect_attr", args["protect_attr"],

        "--class_num", args["class_num"],
        "--num_mlps", args["num_mlps"],
        "--val_ratio", args["val_ratio"],
        "--num_epochs", args["num_epochs"],

        "--loss_type", args["loss_type"],
        "--test_file", args["test_file"],
        "--bbq_cate", args["bbq_cate"],
        "--eval_task", args["eval_task"],
        "--eval_dataset_name", args["eval_dataset_name"],
        "--noise_factor", args["noise_factor"],
        "--num_iter", args["num_iter"],
        "--eps_iter", args["eps_iter"],

        # [核心修改 1] 传入新增的创新点参数
        "--steering_alpha", args["steering_alpha"],
        "--gating_threshold", args["gating_threshold"],
    ]
    env = os.environ.copy()  # Get the current environment variables
    env["CUDA_VISIBLE_DEVICES"] = str("0,1,2,3,4,5,6,7")  # Set the specific CUDA device

    print("Running command:", " ".join(command))
    subprocess.run(command, env=env)


if __name__ == "__main__":
    # Set parameters
    model_name = "llama2_chat_7B"
    bias_knowledge_dir = "../biased_knowledge/corpus"

    # [核心修改 2] 放开所有的类别，以便一次性跑完 BiasAsker 的所有数据集
    categories = {
        "Age": 4,
        "Disability_status": 3,
        "Gender_identity": 5,
        "Physical_appearance": 6,
        "Race_ethnicity": 8,
        "Religion": 7,
    }

    # Categories noise factors
    noise_factors = {
        "Age": 6,
        "Disability_status": 6,
        "Gender_identity": 4,
        "Physical_appearance": 3,
        "Race_ethnicity": 4,
        "Religion": 3,
    }

    class_map = {
        # "Age": 4,
        # "Disability_status": 3,
        # "Gender_identity": 5,
        # "Physical_appearance": 6,
        "Race_ethnicity": 8,
        "Religion": 7,
    }

    # [核心修改 3] 为不同类别设置异质化的门控阈值和导向强度
    # 你可以根据跑出来的实际效果随时微调这些超参数
    category_params = {
        "Age": {"alpha": 0.05, "thresh": 0.3},
        "Disability_status": {"alpha": 0.05, "thresh": 0.3},
        # 性别偏见相对容易导致语义翻转，可以把阈值设高一点(0.08)，导向强度设低一点(0.3)
        "Gender_identity": {"alpha": 0.05, "thresh": 0.3},
        # 外貌偏见可能比较顽固，阈值设低一点(0.03)更容易触发，导向强度设高一点(0.8)
        "Physical_appearance": {"alpha": 0.05, "thresh": 0.3},
        "Race_ethnicity": {"alpha": 0.05, "thresh": 0.3},
        "Religion": {"alpha": 0.05, "thresh": 0.3},
    }

    num_iter = 20
    eps_iter = 15

    futures = []
    # 保持原有的多线程并发执行，max_workers 可以根据你的显卡数量调整
    with ThreadPoolExecutor(max_workers=1) as executor:
        for bbq_cate in categories:
            noise_factor = noise_factors[bbq_cate]
            protect_attr = bbq_cate

            # 获取当前类别的特定参数，如果没有匹配到则使用默认值
            current_params = category_params.get(bbq_cate, {"alpha": 0.5, "thresh": 0.05})

            trace_file = os.path.join(bias_knowledge_dir, f"{bbq_cate}_sentences.json")

            para_templates = {
                "exp_suffix": f"exp1",
                "model_name": model_name,
                "trace_file": trace_file,
                "protect_attr": protect_attr,

                "class_num": f"{class_map[bbq_cate]}",
                "num_mlps": f"9",
                "val_ratio": "0.2",
                "num_epochs": "20",

                "loss_type": "KL",
                "test_file": bbq_cate,
                "bbq_cate": bbq_cate,
                "eval_task": "biasasker",
                "eval_dataset_name": "BiasAsker",
                "noise_factor": f"{noise_factor}",
                "num_iter": f"{num_iter}",
                "eps_iter": f"{eps_iter}",

                # [新增] 将超参数写入字典
                "steering_alpha": str(current_params["alpha"]),
                "gating_threshold": str(current_params["thresh"]),
            }

            futures.append(executor.submit(run_debias, para_templates))

        # 等待并捕获可能出现的异常
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Task generated an exception: {exc}')