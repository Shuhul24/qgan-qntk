
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
import tensorflow as tf
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit.quantum_info import state_fidelity
import pickle
from qiskit_finance.circuit.library import NormalDistribution
from qiskit.opflow import (StateFn, PauliSumOp, Gradient, AerPauliExpectation)
from qiskit_machine_learning.neural_networks import OpflowQNN
import matplotlib.pyplot as plt

# Function for generator's loss function
def generator_cost(gen_params):
        curr_params = np.append(disc_params.numpy(),
                                gen_params.numpy())
        state_probs = Statevector(gen_disc_circuit.bind_parameters(curr_params)).probabilities()
        prob_fake_true = np.sum(state_probs[0b100:])
        cost = -prob_fake_true
        return cost


# Function for layerwise-circuit drawing for discriminator
def layer(qubits, num_layers, circ, disc_weights):
        for i in range(qubits):
                circ.h(i)
        for i in range(num_layers):
                for j in range(qubits):
                        circ.rx(disc_weights[(3 * j) + ((3 * qubits) * i)], j)
                        circ.ry(disc_weights[(3 * j) + 1 + ((3 * qubits) * i)], j)
                        circ.rz(disc_weights[(3 * j) + 2 + ((3 * qubits) * i)], j)
                for k in range(qubits):
                        if k != qubits - 1:
                                circ.cx(k, qubits - 1)
        circ.rx(disc_weights[-3], qubits-1)
        circ.ry(disc_weights[-2], qubits-1)
        circ.rz(disc_weights[-1], qubits-1)

        return circ

# Function for discriminator's loss function
def discriminator_cost(disc_params):
        curr_params = np.append(disc_params.numpy(),
                                gen_params.numpy())
        gendisc_probs = Statevector(gen_disc_circuit.bind_parameters(curr_params)).probabilities()
        realdisc_probs = Statevector(real_disc_circuit.bind_parameters(disc_params.numpy())).probabilities()
        prob_fake_true = np.sum(gendisc_probs[0b100:])
        prob_real_true = np.sum(realdisc_probs[0b100:])
        cost = prob_fake_true - prob_real_true
        return cost

def calculate_kl_divergence(model_distribution: dict, target_distribution: dict):
        kl_div = 0
        for bitstring, p_data in target_distribution.items():
                if np.isclose(p_data, 0, atol=1e-8):
                        continue
                if bitstring in model_distribution.keys():
                        kl_div += (p_data * np.log(p_data) - p_data * np.log(model_distribution[bitstring]))

                else:
                        kl_div += p_data * np.log(p_data) - p_data * np.log
        return kl_div

if __name__=="__main__":
        EPOCHS = 300 # Number of iterations / training epochs
        REAL_DIST_NQUBITS = 3 # Number of qubits
        measurement = 'Z' + ('I' * REAL_DIST_NQUBITS)
        real_circuit = NormalDistribution(REAL_DIST_NQUBITS, mu=0, sigma=0.15) # This is the circuit for target state
        layers = 2 # Number of layers in discriminator
# For generator circuit, we have taken an in-built function 'TwoLocal'
# 'TwoLocal' consists of alternating rotation and entanglement gates
        generator = TwoLocal(REAL_DIST_NQUBITS,
                        ['ry', 'rz'],
                        'cz',
                        'full',
                        reps=2,
                        parameter_prefix='θ_g',
                        name='Generator')

# Parameter Vector for discriminator
        disc_weights = ParameterVector('θ_d', (3 * (REAL_DIST_NQUBITS + 1) * layers) + 3)

        circuit = QuantumCircuit(REAL_DIST_NQUBITS+1, name="Discriminator")
# Discriminator circuit using function 'layer' indicating how many layers of block are being used
        discriminator = layer(REAL_DIST_NQUBITS+1, layers, circuit, disc_weights)
        
        N_GPARAMS = generator.num_parameters
        N_DPARAMS = discriminator.num_parameters

        gen_disc_circuit = QuantumCircuit(REAL_DIST_NQUBITS+1)
        gen_disc_circuit.compose(generator, inplace=True)
        gen_disc_circuit.compose(discriminator, inplace=True)

        real_disc_circuit = QuantumCircuit(REAL_DIST_NQUBITS+1)
        real_disc_circuit.compose(real_circuit, inplace=True)
        real_disc_circuit.compose(discriminator, inplace=True)


        expval = AerPauliExpectation() # method to calculate expectation value
        gradient = Gradient()
# Creating a Quantum Instance (statevector)
        qi_sv = QuantumInstance(Aer.get_backend('aer_simulator_statevector'))
# Circuit's wavefunction using 'StateFn' method
        gen_disc_sfn = StateFn(gen_disc_circuit)
        real_disc_sfn = StateFn(real_disc_circuit)

        H1 = StateFn(PauliSumOp.from_list([(measurement, 1.0)]))

# Combine operator and circuit
        gendisc_op = ~H1 @ gen_disc_sfn
        realdisc_op = ~H1 @ real_disc_sfn

# Calling QNN for updating the generator's weights
        gen_opqnn = OpflowQNN(gendisc_op,
            gen_disc_circuit.parameters[:N_DPARAMS], # input parameters (generator weights)
            gen_disc_circuit.parameters[N_DPARAMS:], # differentiable parameters (discriminator weights)
            expval,
            gradient,
            qi_sv)

# Calling QNN for updating the discriminator's weights
        disc_fake_opqnn = OpflowQNN(gendisc_op,  
            gen_disc_circuit.parameters[N_DPARAMS:], # input parameters (generator weights)
            gen_disc_circuit.parameters[:N_DPARAMS], # differentiable parameters (discriminator weights)
            expval,  
            gradient,
            qi_sv)

        disc_real_opqnn = OpflowQNN(realdisc_op,
            [],
            gen_disc_circuit.parameters[:N_DPARAMS], # differentiable parameters (discriminator weights)
            expval,
            gradient,
            qi_sv)

        init_gen_params = tf.Variable(np.random.uniform(low=-np.pi,
                                                    high=np.pi,
                                                    size=(N_GPARAMS)))
        init_disc_params = tf.Variable(np.random.uniform(low=-np.pi,
                                                    high=np.pi,
                                                    size=(N_DPARAMS)))
        gen_params = init_gen_params
        disc_params = init_disc_params
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

        best_gen_params = init_gen_params
        gloss, dloss, kl_div, fidelity = [], [], [], []
        for epoch in range(EPOCHS):
                D_STEPS = 5
                for disc_train_step in range(D_STEPS):
                        # Differential with respect to discriminator weights (considering input parameters of generator circuit)
                        grad_dcost_fake = disc_fake_opqnn.backward(gen_params,
                                disc_params)[1][0, 0]
                        # Differential with respect to discriminator weights (considering input parameters of target circuit)
                        grad_dcost_real = disc_real_opqnn.backward([],
                                disc_params)[1][0, 0]
                        # Evaluate the discriminator's cost
                        grad_dcost = grad_dcost_real - grad_dcost_fake
                        grad_dcost = tf.convert_to_tensor(grad_dcost)
                        discriminator_optimizer.apply_gradients(zip([grad_dcost],
                                [disc_params]))
                        if disc_train_step % D_STEPS == 0:
                                dloss.append(discriminator_cost(disc_params))
                for gen_train_step in range(1):
                        # Differential with respect to generator weights
                        grad_gcost = gen_opqnn.backward(disc_params, 
                                gen_params)[1][0, 0]
                        grad_gcost = tf.convert_to_tensor(grad_gcost)
                        generator_optimizer.apply_gradients(zip([grad_gcost],
                                [gen_params]))
                        gloss.append(generator_cost(gen_params))

                gen_checkpoint_circuit = generator.bind_parameters(gen_params.numpy())
                # Evaluating the state from generator in 'Statevector'
                gen_prob_dict = Statevector(gen_checkpoint_circuit).probabilities_dict()
                real_prob_dict = Statevector(real_circuit).probabilities_dict()
                current_kl = calculate_kl_divergence(gen_prob_dict, real_prob_dict)
                kl_div.append(current_kl)
                # Measuring the fidelities between generated and target state using 'state_fidelity' method
                fidelity.append(state_fidelity(Statevector(gen_checkpoint_circuit), Statevector(real_circuit)))

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        val = []
        val.append(gloss)
        val.append(dloss)
        val.append(fidelity)
        res = np.array(val)

        np.save('./result_data.npy', res)

# Plotting
        fig, (loss, fid) = plt.subplots(2, sharex=True,
                                        gridspec_kw={'height_ratios': [1, 1]},
                                        figsize = (12,12))
        fig.supxlabel('Training Epochs')
        loss.plot(range(len(gloss)), gloss, label="Generator Loss")
        loss.plot(range(len(dloss)), dloss, label="Discriminator Loss")
        loss.legend()
        loss.set(ylabel='Loss')
        loss.grid(ls='-.')
        fid.plot(range(len(fidelity)), fidelity, label="Fidelity", color="C3")
        fid.set(ylabel='Fidelity')
        fid.legend()
        fid.grid(ls='-.')
        fig.tight_layout();
        fig.savefig('./figure.png') 