from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess

os.chdir('../')




def run_debias(args:dict):
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
        # [新增] 添加这两个参数
        # "--use_steering",
        "--steering_alpha", args["steering_alpha"],
        "--gating_threshold", args["gating_threshold"],

    ]
    env = os.environ.copy()  # Get the current environment variables
    env["CUDA_VISIBLE_DEVICES"] = str("0,1,2,3,4,5,6,7")  # Set the specific CUDA device

    print(command)
    subprocess.run(command, env=env)


if __name__ == "__main__":
    # Set parameters
    model_name = "llama2_chat_7B"
    bbq_cate = "Age"

    class_map = {
        "Age": 2,
        "Disability_status": 2,
        "Gender_identity": 3,
        "Nationality": 6,
        "Physical_appearance": 2,
        "Race_ethnicity": 9,
        "Religion": 11,
        "SES": 2,
        "Sexual_orientation": 5,
    }

    categories = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Religion",
        "SES",
        "Sexual_orientation",
    ]
    bias_knowledge_dir = "../biased_knowledge/corpus"

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        exp_suffixs = ["neutralizer"]

        noise_factors = {
            "Age": 4,
            # "Disability_status":3,
            # "Gender_identity":5,
            # "Nationality":4,
            # "Physical_appearance": 6,
            # "Race_ethnicity":8,
            # "Religion": 7,
            # "SES":9,
            # "Sexual_orientation":6,
        }

        category_params = {
            "Age": {"alpha": 0.5, "thresh": 0.05},
            "Disability_status": {"alpha": 0.5, "thresh": 0.05},
            "Gender_identity": {"alpha": 0.5, "thresh": 0.05},
            "Nationality": {"alpha": 0.2, "thresh": 0.10},  # 降低强度
            "Physical_appearance": {"alpha": 0.1, "thresh": 0.15},  # 显著降低强度，提高阈值
            "Race_ethnicity": {"alpha": 0.5, "thresh": 0.05},
            "Religion": {"alpha": 0.0, "thresh": 0.20},  # 建议先关闭 Steering (alpha=0)，仅依靠 Adapter
            "SES": {"alpha": 0.1, "thresh": 0.15},  # 降低强度
            "Sexual_orientation": {"alpha": 0.5, "thresh": 0.05},
        }

        num_iter = 20
        eps_iter = 15

        for bbq_cate in categories:
            for noise_factor in range(3, 10):
                # noise_factor = noise_factors[bbq_cate]
                protect_attr = bbq_cate
                current_params = category_params.get(bbq_cate, {"alpha": 0.5, "thresh": 0.05})
                trace_file = os.path.join(bias_knowledge_dir, f"{bbq_cate}_sentences.json")

                para_templates = {
                    "exp_suffix": f"exp1_noise_factor_{noise_factor}",
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
                    "eval_task": "bbq",
                    "eval_dataset_name": "BBQ",
                    "noise_factor": f"{noise_factor}",
                    "num_iter":  f"{num_iter}",
                    "eps_iter": f"{eps_iter}",
                    "steering_alpha": str(current_params["alpha"]),
                    "gating_threshold": str(current_params["thresh"]),
                }

                futures.append(executor.submit(run_debias, para_templates))
                break
        # 等待并获取任务结果
        for future in as_completed(futures):
            result = future.result()  # 获取每个任务的结果
            print(result)



