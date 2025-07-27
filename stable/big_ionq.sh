#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value


# stop-me

basePath=/pscratch/sd/g/gzquse/quantum-transformer/paper_AAAI2025/data_ionq
nSamp=30

# backN=ibm_fez
# backN=ibm_kingston ; nqaL=" 2 3 4 5 6   "  

#backN=aer_ideal  ; nqaL=" 2 3 4 5 6 7   "  
backN=ionq; nqaL=" 4 5"
echo bp=$basePath  nqaL=$nqaL


for nqa in $nqaL; do
    echo 
    nshot=$((2500 * 2 ** nqa))
    echo "nqa=$nqa nshot=$nshot"
    
    expName=qc${nqa}adr_${backN}

    echo expName=$expName shots=$nshot
  
    # ./submit_multXY_job.py  --backend $backN --basePath  $basePath  --numQaddr $nqa --numSample $nSamp --numShot ${nshot} --expName $expName  --useRC  -E # ; continue

    ./retrieve_ionq_job.py  --basePath  $basePath  --expName $expName 

    ./postproc_multXY.py  --basePath  $basePath  --expName  $expName  -p a c
       
    echo "dealt with job for ${nqa} qubits, shots=$nshot"
    echo
        
done  

echo all DONE
