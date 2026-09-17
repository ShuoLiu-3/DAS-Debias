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
    ]
    env = os.environ.copy()  # Get the current environment variables
    env["CUDA_VISIBLE_DEVICES"] = str("0,1,2,3,4,5,6,7")  # Set the specific CUDA device

    print(command)
    subprocess.run(command, env=env)

    
if __name__ == "__main__":
    # Set parameters
    model_name = "llama2_chat_7B"
    # bbq_cate = "Age"
    
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
            "Disability_status":3,
            "Gender_identity":5,
            "Nationality":4,
            "Physical_appearance": 6,
            "Race_ethnicity":8,
            "Religion": 7,
            "SES":9,
            "Sexual_orientation":6,
        }


        num_iter = 20 
        eps_iter = 15 

        for bbq_cate in categories:
            noise_factor = noise_factors[bbq_cate]
            protect_attr = bbq_cate
            
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
                "test_file": f"../data/MMLU/mmlu.json",
                "bbq_cate": bbq_cate,
                "eval_task": "mmlu", 
                "eval_dataset_name": "mmlu",
                "noise_factor": f"{noise_factor}",
                "num_iter":  f"{num_iter}",
                "eps_iter": f"{eps_iter}",
            }
            
            futures.append(executor.submit(run_debias, para_templates))
            break
        # 等待并获取任务结果
        for future in as_completed(futures):
            result = future.result()  # 获取每个任务的结果
            print(result)

    

