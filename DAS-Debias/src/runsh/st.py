from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess

# 切换到项目根目录，以便 main.py 能够正确识别相对路径
os.chdir('../')


def run_debias(args: dict):
    # 构建调用 main.py 的命令行参数
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

        # 你的创新点参数：对比导向向量强度 & 动态门控阈值
        "--steering_alpha", args["steering_alpha"],
        "--gating_threshold", args["gating_threshold"],
    ]

    env = os.environ.copy()
    # 根据你的服务器配置设置 CUDA 设备，可以根据需要调整
    env["CUDA_VISIBLE_DEVICES"] = str("0,1,2,3,4,5,6,7")

    print("Running command:", " ".join(command))
    subprocess.run(command, env=env)


if __name__ == "__main__":
    # --- 全局参数配置 (与你的 debias_bbq.py 保持一致) ---
    model_name = "llama2_chat_7B"
    bias_knowledge_dir = "../biased_knowledge/corpus"  # 请确认这里的路径是否与你的 trace_file 目录一致

    # 指向你刚刚提到的 StereoSet 数据集路径 (注意: 这里相对项目根目录)
    stereoset_test_file = "../data/SteroSet/dev.json"

    # 需要测试的偏见类别
    categories = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Religion",
        "SES",
        "Sexual_orientation"
    ]

    # 不同类别的探测器分类数
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

    # PGD 攻击的噪音缩放因子
    noise_factors = {
        "Age": 4,
        "Disability_status": 8,
        "Gender_identity": 8,
        "Nationality": 8,
        "Physical_appearance": 3,
        "Race_ethnicity": 3,
        "Religion": 12,
        "SES": 6,
        "Sexual_orientation": 8,
    }

    # 每个类别的创新点特定参数 (Alpha & Gating Threshold)
    category_params = {
        "Age": {"alpha": 1.5, "thresh": 0.01},
        "Disability_status": {"alpha": 2, "thresh": 0.1},
        "Gender_identity": {"alpha": 0.8, "thresh": 0.03},
        "Nationality": {"alpha": 2, "thresh": 0.01},  # 降低强度
        "Physical_appearance": {"alpha": 1.0, "thresh": 0.01},  # 显著降低强度，提高阈值
        "Race_ethnicity": {"alpha": 1.0, "thresh": 0.02},
        "Religion": {"alpha": 1.0, "thresh": 0.02},  # 建议先关闭 Steering (alpha=0)，仅依靠 Adapter
        "SES": {"alpha": 1.5, "thresh": 0.01},  # 降低强度
        "Sexual_orientation": {"alpha": 1.5, "thresh": 0.05},
    }

    # PGD 超参数
    num_iter = 20
    eps_iter = 15

    futures = []
    # 使用线程池并发运行
    with ThreadPoolExecutor(max_workers=1) as executor:
        for bbq_cate in categories:
            noise_factor = noise_factors[bbq_cate]
            protect_attr = bbq_cate

            # 获取当前类别的特定参数，如果没有匹配到则使用默认值
            current_params = category_params.get(bbq_cate, {"alpha": 0.5, "thresh": 0.05})

            # 对应的 Trace 特征文件
            trace_file = os.path.join(bias_knowledge_dir, f"{bbq_cate}_sentences.json")

            para_templates = {
                "exp_suffix": f"exp1_noise_factor_{noise_factor}",
                "model_name": model_name,
                "trace_file": trace_file,
                "protect_attr": protect_attr,

                "class_num": f"{class_map[bbq_cate]}",
                "num_mlps": "9",
                "val_ratio": "0.2",
                "num_epochs": "20",

                "loss_type": "KL",
                "test_file": stereoset_test_file,  # 传入 StereoSet 路径
                "bbq_cate": bbq_cate,
                "eval_task": "stereoset",  # [关键] 触发 evaluate_stereoset.py
                "eval_dataset_name": "StereoSet",
                "noise_factor": f"{noise_factor}",
                "num_iter": f"{num_iter}",
                "eps_iter": f"{eps_iter}",
                "steering_alpha": str(current_params["alpha"]),
                "gating_threshold": str(current_params["thresh"]),
            }

            futures.append(executor.submit(run_debias, para_templates))

        # 监控运行状态
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"一个任务执行时发生了异常: {exc}")