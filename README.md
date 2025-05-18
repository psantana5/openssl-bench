# OpenSSL RSA Performance and Energy Benchmarking

This repository contains a benchmarking script designed to measure the performance and energy consumption of RSA-2048 operations using OpenSSL. The script supports multi-threaded execution and leverages Intel RAPL for precise energy measurements on compatible systems.

## Features

- Automatic detection of logical CPU cores
- RSA-2048 benchmarking using `openssl speed` with multiple threads
- Execution time, average power (W), and total energy consumption (J) measurement via Intel RAPL
- Results exported to CSV format
- Python script for result analysis and visualization

## Requirements

### System

- `bash`
- `openssl` CLI
- `bc` (for floating-point math)
- Intel RAPL support via `/sys/class/powercap/intel-rapl:0` (optional but recommended)

### Python 3.8+

Install Python dependencies with:
```bash
pip install -r requirements.txt
```
Usage
Make the benchmark script executable:
```bash
chmod +x openssl_speed_server.sh
```
Run the benchmark:

``` bash
./openssl_speed_server.sh
```
Results will be stored in:
```bash
./openssl_speed_results/openssl_speed_results.csv
````
Analyze and visualize the results with:
``` python
parse_results.py
```
## Project structure

```
├── openssl_speed_server.sh          # Main benchmarking script
├── parse_results                    # Python analysis and plotting script
├── requirements.txt                 # Python dependencies
└── openssl_speed_results/
    └── openssl_speed_results.csv   # CSV file with benchmarking results
```
Notes
Energy consumption data is only available if the system exposes Intel RAPL to userspace.
