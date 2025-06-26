#!/bin/bash -l

#SBATCH -A naiss2024-22-631
#SBATCH -J nw2_dino
#SBATCH -t 07:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

# Load necessary modules
ml purge

# ml Python/3.10.4-env-hpc1-gcc-2022a-eb
# ml buildenv-gcc/2022a-eb
# source ~/work/veros_env/bin/activate

ml buildenv-gcc/2022a-eb
ml h5py/3.7.0
source ~/work/veros/myownverosenv/bin/activate

# set number of threads to cpus-per-task
export OMP_NUM_THREADS=1


veros resubmit -i nw2_dino -n 65 -l 315360000  \
         -c "srun --mpi=pmi2 -- veros run /home/x_titmo/work/runs_output/NeverWorld2/Dino_clim_bis/neverworld2_clim_bis.py -b jax --float-type float64 -n 2 4 \
         -s restart_input_filename /home/x_titmo/work/runs_output/NeverWorld2/Dino_clim_bis/dino_clim_100y.restart.h5" \
         --callback "sbatch /home/x_titmo/work/runs_output/NeverWorld2/Dino_clim_bis/veros_batch_neverworld2.sh"
