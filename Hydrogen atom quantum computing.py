from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator 

# 1️⃣ Define the H₂ molecule with PySCF
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
es_problem = driver.run()  # Run the driver to get the electronic structure problem

# 2️⃣ Define an Active Space Transformer (optional, but helps reduce complexity)
transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
es_problem = transformer.transform(es_problem)

# 3️⃣ Extract the Hamiltonian
hamiltonian = es_problem.hamiltonian.second_q_op()

# 4️⃣ Use the Jordan-Wigner Mapper to map to qubits
mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(hamiltonian)

# 5️⃣ Define a Variational Ansatz for VQE
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

# 6️⃣ Initialize the correct Estimator
estimator = Estimator()  # The latest API uses this

# 7️⃣ Solve using VQE (corrected `VQE` initialization)
vqe = VQE(estimator, ansatz, optimizer=COBYLA())  # Correct order of arguments

# 8️⃣ Use GroundStateEigensolver instead of direct VQE computation
solver = GroundStateEigensolver(mapper, vqe)
result = solver.solve(es_problem)

# 9️⃣ Print the computed ground-state energy
print("Estimated Ground State Energy:", result.total_energies[0])
