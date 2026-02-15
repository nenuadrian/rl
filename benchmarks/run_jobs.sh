for job in jobs/*.sbatch.sh; do
  [ -e "$job" ] || { echo "No job files found in jobs/"; break; }
  sbatch "$job"
done