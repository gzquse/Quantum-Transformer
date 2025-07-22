#!/bin/bash
#SBATCH -N 1                      
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus-per-task=4          
#SBATCH -t 2:00:00                
#SBATCH -q regular                 
#SBATCH -A nintern    
#SBATCH -C gpu                     
#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.out

# set -u  # exit if you try to use an uninitialized variable
# set -e  # bash exits if any statement returns a non-true return value

module load conda
# Activate your Conda environment
conda activate $SCRATCH/cudaq
# Run the Python script
basePath=../paper_AAAI25/data_ionq/
nSamp=30
backN=ionq ; nqaL=" 6 7   "
echo bp=$basePath  nqaL=$nqaL


for nqa in $nqaL; do
    echo 
    nshot=$((2500 * 2 ** nqa))
    echo "nqa=$nqa nshot=$nshot"
    
    expName=qc${nqa}adr_${backN}

    echo expName=$expName shots=$nshot
  
    ./submit_multXY_job.py  --backend $backN --basePath  $basePath  --numQaddr $nqa --numSample $nSamp --numShot ${nshot} --expName $expName  --useRC  -E # ; continue

    ./retrieve_ibmq_job.py  --basePath  $basePath  --expName $expName 

    ./postproc_multXY.py  --basePath  $basePath  --expName  $expName  -p a c
       
    echo "dealt with job for ${nqa} qubits, shots=$nshot"
    echo
        
done  

echo all DONE
