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
from toolbox.Util_IOfunc import dateT2Str, iso_to_localtime
from datacircuits.ParametricQCrankV2 import  ParametricQCrankV2 as QCrankV2, qcrank_reco_from_yields
from datacircuits.qm import QuantumMultiplier
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

    def __init__(self, shots: int = 1024, nq_addr: int = 6,
                  nq_data: int = 2, backend=None, transpiler_seed=42):
        self.shots = shots
        self.nq_addr = nq_addr      # Determines capacity: 2**nq_addr
        self.nq_data = nq_data      # Always 2 for vector pairs
        self.backend = backend if backend else AerSimulator(from_backend=FakeMarrakesh())
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
        return result_data['rec_udata']  # Shape: (batch_size, 1, features)

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
    
    # Convert to numpy
    Q_np = Q.detach().numpy() if hasattr(Q, 'detach') else Q
    K_np = K.detach().numpy() if hasattr(K, 'detach') else K
    
    total_pairs = batch * seq_len * seq_len
    nq_addr = int(np.ceil(np.log2(max(1, total_pairs))))
    max_entries = 2**nq_addr

    print(f"Computing Q·K^T with shape ({batch}, {seq_len}, {seq_len})")
    print(f"Total pairs needed: {total_pairs}")
    print(f"Using nq_addr: {nq_addr}")

    # Prepare input data: shape (max_entries, 2, features)
    inp_udata = np.zeros((max_entries, 2, features))
    
    pair_idx = 0
    for b in range(batch):
        for i in range(seq_len):  # Q rows
            for j in range(seq_len):  # K rows
                if pair_idx < max_entries:
                    inp_udata[pair_idx, 0, :] = Q_np[b, i, :]  # Q[b,i,:]
                    inp_udata[pair_idx, 1, :] = K_np[b, j, :]  # K[b,j,:]
                pair_idx += 1

    # Quantum evaluation - ADD the missing parameters
    qm = QuantumMultiXY(
        shots=3200000,
        nq_addr=nq_addr,
        nq_data=2,
    )
    
    element_products = qm.evaluate(inp_udata)

    # Sum to get dot products: Q[b,i,:] · K[b,j,:]
    dot_products = np.sum(element_products[:total_pairs, 0, :], axis=-1)

    # Reshape to attention matrix
    attention_scores = dot_products.reshape(batch, seq_len, seq_len)
    
    return torch.tensor(attention_scores, dtype=torch.float32)

def test_quantum_vs_classical_dot_batched():
    torch.manual_seed(42)
    batch, seq_len, features = 10, 10, 10
    Q = np.random.uniform(-1, 1., size = (batch, seq_len, features))
    K = np.random.uniform(-1, 1., size = (batch, seq_len, features))
    Q = torch.tensor(Q, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)
    # Classical dot product (per batch)
    # old
    # S_classical = torch.matmul(Q, K.transpose(-2, -1))  # shape: (batch, seq_len, seq_len)
    S_classical = torch.einsum('bij,bkj->bik', Q, K)
    print("Classical dot product matrix:\n", S_classical[0])
    print(S_classical.shape)
    # Quantum dot product (per batch)
    # S_quantum = quantum_dot_product(Q, K)
    S_quantum = quantum_dot_product_vectorized(Q, K)  # shape: (batch, seq_len, seq_len)
    print("Quantum dot product matrix:\n", S_quantum[0])
    print(S_quantum.shape)
    # Compare
    diff = S_quantum - S_classical
    print("Difference (Quantum - Classical):\n", diff)
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()
    success_rate = (diff.abs() < 0.07).float().mean().item()

    print(f"Mean absolute error: {mae:.4f}")
    print(f"Max absolute error: {max_err:.4f}")
    print(f"Success rate (|diff| < 0.07): {success_rate*100:.1f}%")
    import matplotlib.pyplot as plt
    roys_fontset(plt)
    # Create a figure with two subplots: histogram and heatmap
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    # Histogram of absolute differences
    abs_diff = diff.abs().flatten().numpy()
    axs[0].hist(abs_diff, bins=20, color='skyblue', edgecolor='black')

    # Compute statistics
    mean = abs_diff.mean()
    std = abs_diff.std()
    q25 = np.quantile(abs_diff, 0.25)
    q50 = np.quantile(abs_diff, 0.50)
    q75 = np.quantile(abs_diff, 0.75)

    # Mark mean and std
    axs[0].axvline(mean, color='orange', linestyle='--', label=f'Mean = {mean:.3f}')
    # axs[0].axvline(mean + std, color='red', linestyle=':', label=f'1σ = {mean+std:.3f}')
    axs[0].axvline(mean - std, color='red', linestyle=':', label=f'1σ = {mean-std:.3f}')

    # Mark quantiles
    # axs[0].axvline(q25, color='green', linestyle='-.', label=f'25% = {q25:.3f}')
    # axs[0].axvline(q50, color='purple', linestyle='-.', label=f'50% = {q50:.3f}')
    axs[0].axvline(q75, color='brown', linestyle='-.', label=f'75% = {q75:.3f}')

    axs[0].set_xlabel('Absolute Difference')
    axs[0].set_ylabel('Count')
    axs[0].legend()
    axs[0].set_xlim(left=0)

    # Heatmap of the difference matrix for the first batch
    err   = diff[0].numpy()                 # signed difference
    vmax  = np.abs(err).max()               # largest absolute error
    from matplotlib import colors, cm
    # --- helper: brighten an existing colormap -----------------
    def lighten_cmap(cmap_name="PuBu", alpha=.5):
        """Return a lighter copy of *cmap_name* by mixing every colour with white.

        alpha = 0 → original cmap,  alpha = 1 → pure white.
        """
        base   = cm.get_cmap(cmap_name, 256)
        colors_array = base(np.linspace(0, 1, 256))
        colors_array[:, :3] = (1 - alpha) * colors_array[:, :3] + alpha  # mix with white
        return colors.ListedColormap(colors_array, name=f"light_{cmap_name}")

    # -----------------------------------------------------------

    err_abs = np.abs(diff[0].numpy())

    vmin, vmax = 0, err_abs.max()             # full data range
    light_PuBu = lighten_cmap("PuBu", alpha=.35)

    im = axs[1].imshow(err_abs, cmap=light_PuBu, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    # axs[1].set_title('Difference Matrix (Quantum - Classical)')
    axs[1].set_xlabel('K index')
    axs[1].set_ylabel('Q index')

    plt.tight_layout()
    plt.savefig('exploratory/out/qttention.pdf')
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
    plt.rc('font', size=font_large)          # controls default text sizes
    plt.rc('axes', titlesize=font_large)    # fontsize of the axes title
    plt.rc('axes', labelsize=font_large)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_large)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_large)    # fontsize of the tick labels
    
    plt.rc('figure', titlesize=font_large)   # fontsize of the figure title

    # legend box
    plt.rc('legend', frameon=False)  # remove it the frame
    plt.rc('legend', fontsize=font_large)    # legend fontsize
    
if __name__ == "__main__":
    test_quantum_vs_classical_dot_batched()
