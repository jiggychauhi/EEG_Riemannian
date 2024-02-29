import numpy as np
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, ADAM
from qiskit.quantum_info import Statevector

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.seed = 42


def ansatz(num_qubits):
    return RealAmplitudes(num_qubits, reps=5)


def auto_encoder_circuit(num_latent, num_trash):
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    auxiliary_qubit = num_latent + 2 * num_trash
    # swap test
    circuit.h(auxiliary_qubit)
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)

    circuit.h(auxiliary_qubit)
    circuit.measure(auxiliary_qubit, cr[0])
    return circuit


from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler


class BasicQnnAutoencoder(TransformerMixin):

    def __init__(self, num_latent=3, num_trash=2):
        self.fm = None
        self.ae = None
        self.num_latent = num_latent
        self.num_trash = num_trash

    def fit(self, X, _y=None, **kwargs):
        _, n_features = X.shape

        n_qubits = self.num_latent + self.num_trash
        print(f'raw feature size: {2 ** n_qubits} and feature size: {n_features}')
        assert 2 ** n_qubits == n_features

        circuit = auto_encoder_circuit(self.num_latent, self.num_trash)
        # circuit.draw("mpl",reverse_bits=True)

        self.fm = RawFeatureVector(2 ** n_qubits)

        self.ae = auto_encoder_circuit(self.num_latent, self.num_trash)

        qc = QuantumCircuit(self.num_latent + 2 * self.num_trash + 1, 1)
        qc = qc.compose(self.fm, range(n_qubits))
        qc = qc.compose(self.ae)

        qnn = SamplerQNN(
            circuit=qc,
            input_params=self.fm.parameters,
            weight_params=self.ae.parameters,
            interpret=lambda x: x,
            output_shape=2,
        )

        def cost_func(params_values):
            print(params_values)
            probabilities = qnn.forward(X, params_values)
            cost = np.sum(probabilities[:, 1]) / X.shape[0]
            # keep track of the cost
            if not hasattr(self, "cost_"):
                self.cost_ = []
            self.cost_.append(cost)
            return cost

        opt = ADAM(maxiter=1000)
        initial_point = algorithm_globals.random.random(self.ae.num_parameters)
        opt_result = opt.minimize(
            fun=cost_func,
            x0=initial_point)

        # encoder/decoder circuit

        test_qc = QuantumCircuit(n_qubits)
        test_qc = test_qc.compose(self.fm)
        ansatz_qc = ansatz(n_qubits)
        test_qc = test_qc.compose(ansatz_qc)
        test_qc.barrier()
        for i in range(self.num_trash):
            test_qc.reset(self.num_latent + i)
        test_qc.barrier()
        test_qc = test_qc.compose(ansatz_qc.inverse())
        self.test_qc = test_qc
        self.opt_result = opt_result

        # compute fidelity
        fidelities = []
        for trial in X:
            param_values = np.concatenate((trial, self.opt_result.x))
            output_qc = test_qc.assign_parameters(param_values)
            output_state = Statevector(output_qc).data

            original_qc = self.fm.assign_parameters(trial)
            original_state = Statevector(original_qc).data

            fidelity = np.sqrt(np.dot(original_state.conj(), output_state) ** 2)
            fidelities.append(fidelity.real)

        print(f"fidelity: {np.mean(fidelities)}")

        return self

    def transform(self, X, **kwargs):
        _, n_features = X.shape
        outputs = []
        for trial in X:
            param_values = np.concatenate((trial, self.opt_result.x))
            output_qc = self.test_qc.assign_parameters(param_values)
            output_sv = Statevector(output_qc).data
            output_sv = np.reshape(np.abs(output_sv) ** 2, n_features)
            outputs.append(output_sv)
        return np.array(outputs)
