#!/bin/bash

# Directory for results
OUTPUT_DIR="./openssl_speed_results"
mkdir -p "$OUTPUT_DIR"

# Detect number of logical threads
MAX_THREADS=$(nproc --all 2>/dev/null)
if [ -z "$MAX_THREADS" ]; then
  # Fallback for macOS or systems without nproc
  MAX_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null)
fi
if [ -z "$MAX_THREADS" ]; then
  echo "Couldn't detect thread count. Using 4 as default."
  MAX_THREADS=4
fi
echo "Maximum detected threads: $MAX_THREADS"

# Generate scaled thread array: 1,2,4,... up to MAX_THREADS
THREADS=()
t=1
while [ $t -le $MAX_THREADS ]; do
  THREADS+=($t)
  # Double for scaling, or add 4 after reaching 8 to avoid growing too quickly
  if [ $t -lt 8 ]; then
    t=$((t * 2))
  else
    t=$((t + 4))
  fi
done

# Add max threads if not already in the array
if [[ ! " ${THREADS[@]} " =~ " ${MAX_THREADS} " ]]; then
  THREADS+=($MAX_THREADS)
fi

CSV_FILE="$OUTPUT_DIR/openssl_speed_results.csv"
echo "threads,runtime_seconds,power_watts,energy_joules" > "$CSV_FILE"

# Function to check if RAPL interface is available
check_rapl() {
  if [ -e "/sys/class/powercap/intel-rapl:0/energy_uj" ]; then
    return 0  # RAPL available
  else
    echo "Warning: Intel RAPL interface not available. Power measurements will be disabled."
    return 1  # RAPL not available
  fi
}

# Function to read CPU package energy (in microjoules)
# Improved to handle large values and potential errors
read_energy() {
  if [ -e "/sys/class/powercap/intel-rapl:0/energy_uj" ]; then
    local energy_val=$(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null)
    if [ -z "$energy_val" ] || ! [[ "$energy_val" =~ ^[0-9]+$ ]]; then
      echo "Error reading energy value" >&2
      echo 0
    else
      echo "$energy_val"
    fi
  else
    echo 0
  fi
}

# Function to handle energy counter wraparound
# The Intel RAPL counters are typically 32-bit and will wrap around
calculate_energy_diff() {
  local start=$1
  local end=$2
  local max_val=4294967295  # 2^32 - 1
  
  if [ "$start" -gt "$end" ]; then
    # Counter wrapped around
    echo "$(( (max_val - start) + end ))"
  else
    # Normal case
    echo "$(( end - start ))"
  fi
}

run_test() {
  local nthreads=$1
  echo "Running test with $nthreads thread(s)..."
  
  # Check for RAPL support
  local power_supported=false
  check_rapl && power_supported=true
  
  # Record starting energy and time
  local start_energy=0
  if [ "$power_supported" = true ]; then
    start_energy=$(read_energy)
  fi
  local start_time=$(date +%s.%N)
  
  # Run OpenSSL speed test with specified thread count
  openssl speed -multi "$nthreads" rsa2048 > /dev/null 2>&1
  
  # Record ending energy and time
  local end_time=$(date +%s.%N)
  local end_energy=0
  if [ "$power_supported" = true ]; then
    end_energy=$(read_energy)
  fi
  
  # Calculate elapsed time
  local elapsed=$(echo "$end_time - $start_time" | bc)
  
  # Calculate energy consumption and power
  local energy_diff=0
  local energy_joules=0
  local power=0
  
  if [ "$power_supported" = true ] && [ "$start_energy" != "0" ] && [ "$end_energy" != "0" ]; then
    # Handle possible counter wraparound
    energy_diff=$(calculate_energy_diff "$start_energy" "$end_energy")
    energy_joules=$(echo "scale=6; $energy_diff / 1000000" | bc)
    
    # Calculate average power in watts (joules/second)
    if (( $(echo "$elapsed > 0" | bc -l) )); then
      power=$(echo "scale=3; $energy_joules / $elapsed" | bc)
      
      # Sanity check on power value (should be within reasonable range)
      if (( $(echo "$power < 0" | bc -l) )) || (( $(echo "$power > 1000" | bc -l) )); then
        echo "Warning: Suspicious power value detected: $power W" >&2
        echo "  Start energy: $start_energy, End energy: $end_energy" >&2
        echo "  Calculated diff: $energy_diff microjoules" >&2
        power="NaN"  # Mark as invalid
      fi
    fi
  else
    power="NaN"  # Power measurement not available
  fi
  
  # Log results
  echo "$nthreads,$elapsed,$power,$energy_joules" >> "$CSV_FILE"
  echo "Threads: $nthreads | Time: $elapsed s | Power: $power W | Energy: $energy_joules J"
}

# Main loop - run tests for each thread count
for t in "${THREADS[@]}"; do
  run_test $t
done

echo "Tests completed. Results in $CSV_FILE"

# Optional: Generate a simple summary of the results
echo -e "\nSummary of results:"
if command -v column > /dev/null; then
  cat "$CSV_FILE" | column -t -s','
else
  cat "$CSV_FILE"
