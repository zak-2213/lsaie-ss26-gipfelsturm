import sys
from os import path
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def usage():
    print("Usage: python analyze_log.py <log_file> <log_file_2> ...")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        usage()
    
    log_files = sys.argv[1:]
    
    throughput_per_file = {}
    tokens_per_file = {}
    for log_file in log_files:
        if not path.isfile(log_file):
            print(f"Error: File '{log_file}' does not exist.")
            sys.exit(1)
    
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        regex_throughput = re.compile(r"throughput per GPU \(TFLOP/s/GPU\): ([\d\.]+)")
        regex_tokens = re.compile(r"tokens/sec/GPU: (\d+)")
        
        experiment = path.basename(log_file).replace("gipfel-throughput-32b-50s-4n-", "").replace(".log", "")
        
        throughput_per_file[experiment] = [float(group[1]) for line in lines for group in [regex_throughput.search(line)] if group]
        tokens_per_file[experiment] = [int(group[1]) for line in lines for group in [regex_tokens.search(line)] if group]
    
    print("Throughput numbers:", throughput_per_file)
    print("Tokens per second numbers:", tokens_per_file)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    
    # ============
    # Line plot
    # ============
    
    # for experiment, throughputs in throughput_per_file.items():
    #     axs[0].plot(throughputs, label=experiment)

    # axs[0].set_title("Throughput per GPU (TFLOP/s/GPU)")
    # axs[0].set_xlabel("Time")
    # axs[0].set_ylabel("Throughput (TFLOP/s/GPU)")
    # axs[0].legend()
        
    # for experiment, tokens in tokens_per_file.items():
    #     axs[1].plot(tokens, label=experiment)

    # axs[1].set_title("Tokens per second per GPU")
    # axs[1].set_xlabel("Time")
    # axs[1].set_ylabel("Tokens/sec/GPU")
    # axs[1].legend()

    # ============
    # Bar plot
    # ============
    
    for experiment, throughputs in throughput_per_file.items():
        axs[0].bar(experiment, np.mean(throughputs), yerr=np.std(throughputs), capsize=5)
    
    axs[0].set_title("Average Throughput per GPU (TFLOP/s/GPU)")
    axs[0].set_ylabel("Throughput (TFLOP/s/GPU)")
    axs[0].set_xticks(range(len(throughput_per_file.keys())), labels=throughput_per_file.keys())
    
    for experiment, tokens in tokens_per_file.items():
        axs[1].bar(experiment, np.mean(tokens), yerr=np.std(tokens), capsize=5)

    axs[1].set_title("Average Tokens per second per GPU")
    axs[1].set_ylabel("Tokens/sec/GPU")
    axs[1].set_xticks(range(len(tokens_per_file.keys())), labels=tokens_per_file.keys())
    
    # Store plot
    plt.tight_layout()
    date = datetime.now().strftime("%Y-%m- %d_%H-%M-%S")
    plt.savefig(f"plots/throughput_tokens_{date}.png")
    print(f"Plot saved as 'plots/throughput_tokens_{date}.png'")


if __name__ == "__main__":
    main()
