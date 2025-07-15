# ---------------------------------------------------------------------
#  QuantumMultiplier -- reference-accurate (tags 0 & 1)
# ---------------------------------------------------------------------
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
import numpy as np

class QuantumMultiplier:
    """
    tag = 0  →  ⟨Z0⟩ = w·x1 + (1–w)·x2
    tag = 1  →  ⟨Z1⟩ = x1·x2
    """

    def __init__(self, shots: int = 1024, tag: int = 0):
        assert tag in (0, 1), "Only tags 0,1 implemented here"
        self.tag   = tag
        self.shots = shots
        self.backend = AerSimulator()
        qc, self.θ, self.α = self._make_template(tag=tag)
        self.tpl = transpile(qc, self.backend, optimization_level=3)
        self._bind_map = {self.θ[0]: 0.0, self.θ[1]: 0.0, self.α: 0.0}  # cache

    # ---------------------------------------------------------------
    def _make_template(self, tag):
        """Return (QuantumCircuit, θ-vector, α-param) for given tag."""
        qr, cr = QuantumRegister(2, "q"), ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        θ = ParameterVector("θ", 2)     # x1,x2
        α = Parameter("α")              # weight w

        qc.ry(θ[0], 0)
        qc.ry(θ[1], 1)
        qc.barrier()

        qc.rz(np.pi/2, 1)
        qc.cx(0, 1)

        qc.ry( α/2, 0)
        qc.cx(1, 0)
        qc.ry(-α/2, 0)

        if tag == 0:  qm = 0            # Z-basis on q0   (weighted sum)
        elif tag == 1: qm = 1           # Z-basis on q1   (product)

        qc.measure(tag, 0)
        return qc, θ, α
    # ---------------- helpers ----------------

    @staticmethod
    def _θ(x, x_min=-1, x_max=1):
        if not  isinstance(x, np.ndarray):
            x=np.array(x)
        # Normalize x to [-1, 1] for arccos
        return float(np.arccos(x))

    @staticmethod
    def _α(w, w_min=0, w_max=1):
        # Normalize w to [0, 1] for arccos(1-2w)
        if not  isinstance(w, np.ndarray):
            w=np.array(w)
        return float(np.arccos(1. - 2 * w))

    # ---------------- single evaluation ----------------
    def evaluate(self, x1: float, x2: float, w=0.5):
        bd = self._bind_map
        bd[self.θ[0]] = self._θ(x1)
        bd[self.θ[1]] = self._θ(x2)
        bd[self.α]    = self._α(w)
        circ = self.tpl.assign_parameters(bd, inplace=False)
        cnt  = self.backend.run(circ, shots=self.shots).result().get_counts()
        p1 = cnt.get('1', 0) / self.shots
        return 1-2*p1           # = ⟨Z⟩ of measured qubit

    # x1*x2
    def evaluate_product(self, x1: float, x2: float, w=0.5):
        bd = self._bind_map
        bd[self.θ[0]] = self._θ(x1)
        bd[self.θ[1]] = self._θ(x2)
        bd[self.α]    = self._α(w)
        circ = self.tpl.assign_parameters(bd, inplace=False)
        cnt  = self.backend.run(circ, shots=self.shots).result().get_counts()
        p1 = cnt.get('1', 0) / self.shots
        return 1-2*p1           # = ⟨Z⟩ of measured qubit
    
    def evaluate_batch(self,
                    x1: np.ndarray,   # shape (N,)
                    x2: np.ndarray,
                    w : float | np.ndarray = 0.5,
                    ) -> np.ndarray:   # returns (N,)
        x1 = np.asarray(x1);  x2 = np.asarray(x2)
        w  = np.asarray(w) if np.ndim(w) else np.full_like(x1, w)
        circs = []
        for a,b,c in zip(x1,x2,w):
            bd = {self.θ[0]: self._θ(float(a)),
                    self.θ[1]: self._θ(float(b)),
                    self.α   : self._α(float(c))}
            circs.append(self.tpl.assign_parameters(bd, inplace=False))

        job = self.backend.run(circs, shots=self.shots)
        counts = job.result().get_counts()
        return np.array([1 - 2*cnt.get('1',0)/self.shots for cnt in counts],
                        dtype=np.float32)
    