#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 Retrieve  results of IBMQ job

Required INPUT:
    --expName: exp_j33ab44

Output:  raw  yields + meta data
'''

import time,os,sys
from pprint import pprint
import numpy as np
sys.path.append(os.path.abspath("../cloud_job/"))

from toolbox.Util_H5io4 import  read4_data_hdf5, write4_data_hdf5
from qiskit_ibm_runtime import QiskitRuntimeService

from time import time, sleep
from toolbox.Util_QiskitV2 import pack_counts_to_numpy
from submit_multXY_job import harvest_sampler_results

from qiskit_ionq import IonQProvider 
import dotenv

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument('-e',"--expName",  default='exp_62a15b79',help='IBMQ experiment name assigned during submission')

    args = parser.parse_args()

    args.inpPath=os.path.join(args.basePath,'jobs')
    args.outPath=os.path.join(args.basePath,'meas')
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
   
    return args

        
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    
    inpF=args.expName+'.ionq.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.inpPath,inpF),verb=args.verb)
      
    pprint(expMD['submit'])

    if args.verb>1: pprint(expMD)

    if 0:    #example decode one Qasm circuit
        rec2=expD['circQasm'][1].decode("utf-8") 
        print('qasm circ:',type(rec2),rec2)
    
    jid=expMD['submit']['job_id']
    #1jid='cns2cfhqygeg00879yv0' # smapler,  cairo

    # ------  construct sampler-job w/o backend ------
    print('M: activate QiskitRuntimeService() ...')

    print('M: retrieve jid:',jid)
    dotenv.load_dotenv()
    api_key = os.getenv('IONQ_API_KEY')
    provider = IonQProvider(api_key)
    backend = provider.get_backend("qpu.aria-1")
    job = backend.retrieve_job(jid)  
    jstat=job.status()
    print('M: got results', jstat)

    harvest_sampler_results(job,expMD,expD)
   
    if args.verb>2: pprint(expMD)
    
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.meas.h5')
    write4_data_hdf5(expD,outF,expMD)


    print('   ./postproc_multXY.py   --basePath  $basePath --expName   %s   -p a    -Y\n'%(expMD['short_name']))
  
    
    
