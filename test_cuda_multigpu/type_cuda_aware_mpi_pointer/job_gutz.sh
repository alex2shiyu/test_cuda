#!/bin/bash

#SBATCH -J gpu-test
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH -N 1
#SBATCH -p gpu

#export LD_LIBRARY_PATH=/public/home/sypeng/soft/mpich-install/lib:$LD_LIBRARY_PATH
source /public/home/sypeng/.bashrc 
syp=./nmat2
#srun --mpi=pmi2 /public/software/apps/vasp/5.4.4/intelmpi/vasp_std
#srun --mpi=pmi2 $syp>out 2>err
# PATH
python --version >> out.dat
#mpirun=/home/soft/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/bin/mpirun
#vasp='/home/soft/vasp/vasp.5.3/vasp'
#vasp='/public/home/sypeng/soft/vasp.5.4.4-impi/bin/vasp_std'
ulimit -s unlimited
#ncpu=`cat $PBS_NODEFILE | wc -l`

#if [ -n "$SLURM_CPUS_PER_TASK" ]; then
#    omp_threads=$SLURM_CPUS_PER_TASK
#else
#    omp_threads=1
#fi
#omp_threads=36
export OMP_NUM_THREADS=1

export OMP_STACKSIZE=1g

#cd relax/
#cp CONTCAR ../scf/POSCAR
#cp CONTCAR ../band/POSCAR
echo Job started at `date`
#srun --mpi=pmi2 $syp>>out.dat 2>err
#/public/home/sypeng/soft/pgi-openmpi-cuda-install2/bin/mpirun  -np 2 $syp>out.dat  2>err
/public/home/sypeng/soft/pgi-install2/linux86-64/2019/mpi/openmpi-3.1.3/bin/mpirun  -np 2 $syp>out.dat  2>err
#srun --cpu_bind=cores  $syp>>out.dat 2>err
echo Job ended at `date`
