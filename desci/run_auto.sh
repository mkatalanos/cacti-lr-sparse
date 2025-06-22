#!/bin/bash

MAX_JOBS=1
SCRIPT="python3 -u automation.py"
LOG_DIR="logs"
RUN_FILE="params.txt"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

total_jobs=$(grep -cve '^\s*$' "$RUN_FILE")  # ignores empty lines
completed_jobs=0
job_id=0
job_count=0

print_progress() {
    echo -ne "Progress: $completed_jobs/$total_jobs\r"
}

while read -r line; do
    ((job_id++))
    log_file="${LOG_DIR}/job_${job_id}.log"
    args=($line)
    (
        echo "Running: ${args[@]}" > "$log_file"
        $SCRIPT "${args[@]}" >> "$log_file" 2>&1
    ) &

    ((job_count++))
    if [[ $job_count -ge $MAX_JOBS ]]; then
        wait -n
        ((job_count--))
        ((completed_jobs++))
        print_progress
    fi
done < "$RUN_FILE"

# Wait for remaining jobs
while [[ $job_count -gt 0 ]]; do
    wait -n
    ((job_count--))
    ((completed_jobs++))
    print_progress
done

echo -e "\nAll jobs completed. Logs are in '$LOG_DIR/'"

