RESULTS_DIR = "../results/ori—steroset"

HF_NAMES = {
    'llama2_chat_7B': '/your/path/to/Llama-2-7b-chat-hf',
    'llama2_chat_13B': '/your/path/to/Llama-2-13b-chat-hf',
    'llama3_8B_instruct': '/your/path/to/Meta-Llama-3-8B-Instruct',
    'llama3_1_8B_instruct': '/your/path/to/Meta-Llama-3.1-8B-Instruct',
}

from matplotlib import pyplot as plt
def plot_line(data, name='test.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', color='b')
    plt.title(f'{name.split("/")[-1].split(".")[0]}')
    plt.xlabel('Layer Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(name)