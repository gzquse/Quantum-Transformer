# ---------------------------------------------------------------------
#  QuantumMultiplier -- reference-accurate
# ---------------------------------------------------------------------
import random, os, sys
from typing import List
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler 
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
here = os.path.dirname(__file__)
# Go up one level to quantum-transformer, and add that:
repo_pkg = os.path.abspath(os.path.join(here, os.pardir))
sys.path.append(repo_pkg)
from datacircuits.ParametricQCrankV2 import  ParametricQCrankV2 as QCrankV2, qcrank_reco_from_yields
from datacircuits.qm import QuantumMultiplier
from toolbox.Util_IOfunc import dateT2Str, iso_to_localtime
from time import time, localtime,mktime
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class QuantumMultiXY:
    """
    compute the batched XY product
    """

    def __init__(self, shots: int = 1024, nq_addr: int = 6, nq_data: int = 2, 
                 features: int = 2, batch_size: int = 1, backend=None, transpiler_seed=42):
        self.shots = shots
        self.nq_addr = nq_addr      # Determines capacity: 2**nq_addr
        self.nq_data = nq_data      # Always 2 for vector pairs
        self.features = features    # Feature dimension of each vector
        self.batch_size = batch_size
        self.backend = backend if backend else AerSimulator()
        self.transpiler_seed = transpiler_seed
        
        # Build template circuit for dot product computation
        self._setup_quantum_circuit()

    @property
    def numQubits(self):
        """Return the number of qubits needed"""
        return (self.nq_addr, self.nq_data)

    def _setup_quantum_circuit(self):
        # Build template circuit (you'll need to define these properly)
        nq_addr, nq_data = self.numQubits  # You need to define this property
        self.qcrankObj = QCrankV2(nq_addr, nq_data, measure=False, barrier=False)    
        qcP = self.qcrankObj.circuit  # parametrized raw qcrank circuit

        # Attached EHands multiplier on last 2 data qubits
        qcP.barrier()
        qa0 = nq_addr
        qcP.rz(np.pi/2, qa0+1)
        qcP.cx(qa0, qa0+1)
        qcP.barrier()
        
        # Assemble final circuit
        qra = QuantumRegister(nq_addr, 'a')
        qrd = QuantumRegister(nq_data, 'd')
        cr = ClassicalRegister(nq_addr+1, 'c')
        qc1 = QuantumCircuit(qra, qrd, cr)
        qcP = qc1.compose(qcP)
        qcP.measure(qrd[1],0)
        for i in range(nq_addr):     qcP.measure(qra[i],nq_addr-i)  # order must be reversed
        # print(qcP)
        # Transpile the circuit
        qcT = transpile(qcP, self.backend, optimization_level=3, 
                       seed_transpiler=self.transpiler_seed)
        self.qcrankObj.circuit = qcT
    
    def evaluate(self, inp_udata):
        """
        Evaluate batched element-wise multiplications
        
        Args:
            inp_udata: numpy array of shape (batch_size, 2, features)
                      where each entry contains [Q_vector, K_vector]
                      
        Returns:
            numpy array of shape (batch_size, features) containing element-wise products
        """

        expected_shape = (2**self.nq_addr, self.nq_data, self.features)
        print(f"Input shape: {inp_udata.shape}")
        print(f"Expected shape: {expected_shape}")
        
        if inp_udata.shape != expected_shape:
            raise ValueError(f"Input shape {inp_udata.shape} doesn't match expected {expected_shape}")
        
        # Bind data to quantum circuit
        self.qcrankObj.bind_data(inp_udata)
        
        # Instantiate circuits for all vector pairs
        qcEL = self.qcrankObj.instantiate_circuits()
        
        # Setup sampler
        options = SamplerOptions()
        options.default_shots = self.shots
        sampler = Sampler(mode=self.backend, options=options)
        
        # Run quantum circuits
        T0 = time()
        job = sampler.run(tuple(qcEL))
        
        # Process results using your ORIGINAL harvest function
        md = {
            'payload': {
                'nq_addr': self.nq_addr,
                'nq_data': self.nq_data
            }
        }
        bigD = {}
        
        # Use your existing harvest function unchanged
        result_data = harvest_sampler_results(job, md, bigD, T0)
        
        # Return the dot product results
        return result_data['rec_udata']  # Shape: (batch_size, features)

def harvest_sampler_results(job, md, bigD, T0=None):  # many circuits
    pmd = md['payload']
    qa = {}
    jobRes = job.result()
   
    jobMetr = job.metrics()    
    
    if T0 != None:  # when run locally
        elaT = time() - T0
        print(' job done, elaT=%.1f min' % (elaT/60.))
        qa['running_duration'] = elaT
        qa['timestamp_running'] = dateT2Str(localtime())

    else:
        jobMetr = job.metrics()
        t1 = iso_to_localtime((jobMetr['timestamps']['running']))
        qa['timestamp_running'] = dateT2Str(t1)
        qa['quantum_seconds'] = jobMetr['usage']['quantum_seconds']
        qa['all_circ_executions'] = jobMetr['executions']
        
        if jobMetr['num_circuits'] > 0:
            qa['one_circ_depth'] = jobMetr['circuit_depths'][0]
        else:
            qa['one_circ_depth'] = None
                
    nCirc = len(jobRes)  # number of circuit in the job
    jstat = str(job.status())
    
    countsL = [jobRes[i].data.c.get_counts() for i in range(nCirc)]

    # collect job performance info
    res0cl = jobRes[0].data.c
    qa['status'] = jstat
    qa['num_circ'] = nCirc
    qa['shots'] = res0cl.num_shots
    qa['num_clbits'] = res0cl.num_bits
    md['job_qa'] = qa
    
    # Use your existing qcrank_reco_from_yields function
    bigD['rec_udata'], bigD['rec_udata_err'] = qcrank_reco_from_yields(
        countsL, pmd['nq_addr'], 1)

    return bigD

def quantum_dot_product_vectorized(Q, K):
    batch, seq_len, features = Q.shape
    
    # Calculate total pairs needed
    total_pairs = batch * seq_len * seq_len
    
    # Calculate required nq_addr such that 2**nq_addr >= total_pairs
    nq_addr = int(np.ceil(np.log2(max(1, total_pairs))))
    max_entries = 2**nq_addr
    
    print(f"Total pairs needed: {total_pairs}")
    print(f"Using nq_addr: {nq_addr}")
    print(f"Max entries (2**nq_addr): {max_entries}")
    
    # Prepare input data: shape (2**nq_addr, 2, features)
    # nq_data = 2 means each entry has 2 vectors
    # features is the third dimension (k in the error message)
    inp_udata = np.zeros((max_entries, 2, features))
    
    pair_idx = 0
    for b in range(batch):
        for i in range(seq_len):
            for j in range(seq_len):
                if pair_idx < max_entries:
                    inp_udata[pair_idx, 0, :] = Q[b, i, :].numpy()  # Q vector
                    inp_udata[pair_idx, 1, :] = K[b, j, :].numpy()  # K vector
                pair_idx += 1
    
    # Initialize quantum multiplier
    qm = QuantumMultiXY(
        shots=16384, 
        nq_addr=nq_addr,  # This gives us 2**nq_addr processing capacity
        nq_data=2,        # Always 2 (for vector pairs)
        features=features, # Feature dimension
        batch_size=1
    )
    
    # Compute element-wise products for all pairs
    # Output shape: (2**nq_addr, features)
    element_products = qm.evaluate(inp_udata)
    
    # Sum over features to get dot products
    # Only take the first total_pairs results (ignore padding)
    dot_products = np.sum(element_products[:total_pairs], axis=1)
    
    # Reshape back to attention matrix format
    attention_scores = torch.tensor(dot_products).view(batch, seq_len, seq_len)
    
    return attention_scores

def test_quantum_vs_classical_dot_batched():
    torch.manual_seed(42)
    batch, seq_len, features = 1, 6, 2
    Q = np.random.uniform(-1, 1., size = (batch, seq_len, features))
    K = np.random.uniform(-1, 1., size = (batch, seq_len, features))
    Q = torch.tensor(Q, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)
    # Classical dot product (per batch)
    S_classical = torch.matmul(Q, K.transpose(-2, -1))  # shape: (batch, seq_len, seq_len)
    print("Classical dot product matrix:\n", S_classical[0])
    print(S_classical.shape)
    # Quantum dot product (per batch)
    # S_quantum = quantum_dot_product(Q, K)
    S_quantum = quantum_dot_product_vectorized(Q, K)  # shape: (batch, seq_len, seq_len)
    print("Classical dot product matrix:\n", S_quantum[0])
    print(S_quantum.shape)
    # Compare
    diff = S_quantum - S_classical
    print("Difference (Quantum - Classical):\n", diff)
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()
    success_rate = (diff.abs() < 0.03).float().mean().item()

    print(f"Mean absolute error: {mae:.4f}")
    print(f"Max absolute error: {max_err:.4f}")
    print(f"Success rate (|diff| < 0.03): {success_rate*100:.1f}%")
    import matplotlib.pyplot as plt

    # Create a figure with two subplots: histogram and heatmap
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram of absolute differences
    axs[0].hist(diff.abs().flatten().numpy(), bins=20, color='skyblue', edgecolor='black')
    axs[0].axvline(0.03, color='red', linestyle='--', label='Threshold = 0.03')
    # axs[0].set_title('Histogram of |Quantum - Classical| Differences')
    axs[0].set_xlabel('Absolute Difference')
    axs[0].set_ylabel('Count')
    axs[0].set_yticks(range(0, 7))
    axs[0].set_ylim(0, 6)
    axs[0].legend()
    axs[0].set_xlim(left=0)

    # Heatmap of the difference matrix for the first batch
    im = axs[1].imshow(diff[0].numpy(), cmap='bwr', vmin=-0.05, vmax=0.05)
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label='Quantum - Classical')
    # axs[1].set_title('Difference Matrix (Quantum - Classical)')
    axs[1].set_xlabel('K index')
    axs[1].set_ylabel('Q index')

    plt.tight_layout()
    plt.savefig('quantum_classical_diff_combined.svg')
    plt.close()

def roys_fontset(plt):
    print('load Roys fontest')
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    #plt.rcParams['text.usetex'] = True  #Needs new Docker image

    tick_major = 6
    tick_minor = 4
    plt.rcParams["xtick.major.size"] = tick_major
    plt.rcParams["xtick.minor.size"] = tick_minor
    plt.rcParams["ytick.major.size"] = tick_major
    plt.rcParams["ytick.minor.size"] = tick_minor

    font_small = 12
    font_medium = 13
    font_large = 14
    plt.rc('font', size=font_small)          # controls default text sizes
    plt.rc('axes', titlesize=font_small)    # fontsize of the axes title
    plt.rc('axes', labelsize=font_small)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
    
    plt.rc('figure', titlesize=font_large)   # fontsize of the figure title

    # legend box
    plt.rc('legend', frameon=False)  # remove it the frame
    plt.rc('legend', fontsize=font_small)    # legend fontsize
    
if __name__ == "__main__":
    test_quantum_vs_classical_dot_batched()
