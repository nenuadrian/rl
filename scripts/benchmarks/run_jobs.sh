
# Usage: ./run_jobs.sh [--local]
# If --local is provided, runs jobs locally, emulating sbatch features.
# Otherwise, submits jobs with sbatch as before.

run_local=false
if [[ "$1" == "--local" ]]; then
  run_local=true
fi

for job in jobs/*.sbatch.sh; do
  [ -e "$job" ] || { echo "No job files found in jobs/"; break; }
  if ! $run_local; then
    sbatch "$job"
  else
    # Parse -a (array), -t (time limit)
    array_param=$(grep -E '^#SBATCH +-a ' "$job" | awk '{print $3}')
    time_param=$(grep -E '^#SBATCH +-t ' "$job" | awk '{print $3}')
    # Default: no array
    if [[ -n "$array_param" ]]; then
      # Support formats like 1-3 or 1,2,3
      if [[ "$array_param" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        start=${BASH_REMATCH[1]}
        end=${BASH_REMATCH[2]}
        ids=$(seq $start $end)
      else
        ids=$(echo "$array_param" | tr ',' ' ')
      fi
    else
      ids=0
    fi

    # Parse time limit (e.g., 0-6 means 6 hours, 1-0 means 1 day)
    timeout_secs=""
    if [[ -n "$time_param" ]]; then
      if [[ "$time_param" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        days=${BASH_REMATCH[1]}
        hours=${BASH_REMATCH[2]}
        timeout_secs=$((days*24*3600 + hours*3600))
      elif [[ "$time_param" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        # Format: HH:MM:SS
        hours=${BASH_REMATCH[1]}
        mins=${BASH_REMATCH[2]}
        secs=${BASH_REMATCH[3]}
        timeout_secs=$((hours*3600 + mins*60 + secs))
      fi
    fi

    for id in $ids; do
      echo "Running $job locally with SLURM_ARRAY_TASK_ID=$id (timeout: ${timeout_secs:-none})"
      if [[ -n "$timeout_secs" ]]; then
        SLURM_ARRAY_TASK_ID=$id timeout $timeout_secs bash "$job"
      else
        SLURM_ARRAY_TASK_ID=$id bash "$job"
      fi
    done
  fi
done