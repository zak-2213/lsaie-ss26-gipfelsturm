import re
from pathlib import Path

# Folder containing log files
LOG_DIR = "logs"

# Regex for extracting throughput
pattern = re.compile(r"tokens/sec/GPU:\s*(\d+(?:\.\d+)?)\s*\|")

results = {}

for logfile in Path(LOG_DIR).glob("*.log"):
    throughputs = []

    with open(logfile, "r", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                throughputs.append(float(match.group(1)))

    if throughputs:
        results[logfile.name] = {
            "values": throughputs,
            "avg": sum(throughputs) / len(throughputs),
            "max": max(throughputs),
            "min": min(throughputs),
        }

# Print summary
for name, stats in results.items():
    print(f"\n{name}")
    print(f"  Samples : {len(stats['values'])}")
    print(f"  Avg     : {stats['avg']:.2f}")
    print(f"  Max     : {stats['max']:.2f}")
    print(f"  Min     : {stats['min']:.2f}")
