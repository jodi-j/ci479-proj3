from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

bayesNet = BayesianNetwork()
bayesNet.add_node("M")
bayesNet.add_node("U")
bayesNet.add_node("R")
bayesNet.add_node("B")
bayesNet.add_node("S")

bayesNet.add_edge("M", "R")
bayesNet.add_edge("U", "R")
bayesNet.add_edge("B", "R")
bayesNet.add_edge("B", "S")
bayesNet.add_edge("R", "S")
cpd_A = TabularCPD('M', 2, values=[[.95], [.05]])
cpd_U = TabularCPD('U', 2, values=[[.85], [.15]])
cpd_H = TabularCPD('B', 2, values=[[.90], [.10]])
pd_S = TabularCPD('S', 2, values=[[0.98, .88, .95, .6], [.02, .12, .05, .40]],
                   evidence=['R', 'B'], evidence_card=[2, 2])

cpd_R = TabularCPD('R', 2,
                   values=[[0.96, .86, .94, .82, .24, .15, .10, .05], [.04, .14, .06, .18, .76, .85, .90, .95]],
                   evidence=['M', 'B', 'U'], evidence_card=[2, 2,2])
bayesNet.add_cpds(cpd_A, cpd_U, cpd_H, pd_S, cpd_R)
bayesNet.check_model()
print("Model is correct.")
solver = VariableElimination(bayesNet)
result = solver.query(variables=['R'])
print(result)
result = solver.query(variables=['R'], evidence={'M': 1})
print("R| M", result)

