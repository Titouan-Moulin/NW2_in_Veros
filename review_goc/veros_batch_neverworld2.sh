#!/bin/bash -l

#SBATCH -A naiss2025-22-648
#SBATCH -J nw2_dino
#SBATCH -t 07:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

# Load necessary modules and activate your environment 
ml purge

ml buildenv-gcc/2022a-eb
ml h5py/3.7.0
source ~/work/veros/myownverosenv/bin/activate

# set number of threads to cpus-per-task
export OMP_NUM_THREADS=1
veros resubmit -i nw2_dino -n 100 -l 315360000  \
         -c "srun --mpi=pmi2 -- veros run /home/x_titmo/work/runs_output/Clim_bis/neverworld2_clim_bis.py -b jax --float-type float64 -n 2 4"
         --callback "sbatch /home/x_titmo/work/runs_output/Clim_bis/veros_batch_neverworld2.sh"
