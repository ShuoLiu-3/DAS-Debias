from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess

# 切换到项目根目录，保持和 st.py 一致
os.chdir('../')


def run_original(args: dict):
    command = [
        "python", "main.py",

        "--exp_suffix", args["exp_suffix"],
        "--model_name", args["model_name"],

        "--test_file", args["test_file"],
        "--bbq_cate", args["bbq_cate"],
        "--protect_attr", args["protect_attr"],
        "--eval_task", args["eval_task"],
        "--eval_dataset_name", args["eval_dataset_name"],

        # 关键：只跑原始模型，不做任何去偏干预
        "--no_debias",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    print("Running command:", " ".join(command))
    subprocess.run(command, env=env)


if __name__ == "__main__":
    model_name = "llama2_chat_7B"

    # StereoSet 数据路径，保持与你 st.py 一致
    stereoset_test_file = "../data/SteroSet/dev.json"

    # 注意：当前 EvaluateStereoSet 默认会评估整个 dev.json，
    # 不会按照 bbq_cate 自动过滤属性。
    # 所以跑 Original LLaMA 时，实际上跑一次就够。
    categories = [
        "Age",
    ]

    futures = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        for cate in categories:
            para_templates = {
                "exp_suffix": "original_llama_stereoset",
                "model_name": model_name,

                "test_file": stereoset_test_file,
                "bbq_cate": cate,
                "protect_attr": cate,
                "eval_task": "stereoset",
                "eval_dataset_name": "StereoSet",
            }

            futures.append(executor.submit(run_original, para_templates))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"一个任务执行时发生异常: {exc}")