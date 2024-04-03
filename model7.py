from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

bayesNet = BayesianNetwork()
bayesNet.add_node("A") #Patient Condition
bayesNet.add_node("B") #Test Result
bayesNet.add_node("C") #Treatment Decision
bayesNet.add_node("D") #Outcome

bayesNet.add_edge("C", "D")
bayesNet.add_edge("A", "B")
bayesNet.add_edge("A", "C")
bayesNet.add_edge("B", "C")

cpd_PatientCondition = TabularCPD('A', 3, values = [[0.7], [0.2], [0.1]],
                                  state_names={'A': ['good', 'fair', 'poor']})
cpd_TestResult = TabularCPD('B', 2, values = [[0.8, 0.5, 0.3], 
                                              [0.2, 0.5, 0.7]],
                            evidence=['A'], evidence_card=[3],
                            state_names={'B': ['positive', 'negative'],
                                         'A': ['good', 'fair', 'poor']})
cpd_TreatmentDecision = TabularCPD('C', 2, values = [[0.9, 0.7, 0.5, 0.3, 0.1, 0.1],
                                                     [0.1, 0.3, 0.5, 0.7, 0.9, 0.9]],
                                   evidence=['B', 'A'], evidence_card=[2, 3],
                                   state_names={'C': ['treated', 'not treated'],
                                                'B': ['positive', 'negative'],
                                                'A': ['good', 'fair', 'poor']})
cpd_Outcome = TabularCPD('D', 2, values = [[0.8, 0.2], [0.2, 0.8]],
                         evidence=['C'], evidence_card=[2],
                         state_names={'D': ['positive', 'negative'],
                                      'C': ['treated', 'not treated']})
bayesNet.add_cpds(cpd_PatientCondition, cpd_TestResult, cpd_TreatmentDecision, cpd_Outcome)

bayesNet.check_model()
print("Model is correct.")
solver = VariableElimination(bayesNet)

#Query 1
result = solver.query(variables=['A'])
print("\nQuery 1: Probability of Patient Condition A")
print(result)

#Query 2
result = solver.query(variables=['B'], evidence={'A': 'poor'})
print("\nQuery 2: Probability of Test Result B")
print(result)

#Query 3


#Query 4


#Query 5


#Query 6


#Query 7


#Query 8


#original code from model 7
"""
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
cpd_S = TabularCPD('S', 2, values=[[0.98, .88, .95, .6], [.02, .12, .05, .40]],
                   evidence=['R', 'B'], evidence_card=[2, 2])
cpd_R = TabularCPD('R', 2,
                   values=[[0.96, .86, .94, .82, .24, .15, .10, .05], [.04, .14, .06, .18, .76, .85, .90, .95]],
                   evidence=['M', 'B', 'U'], evidence_card=[2, 2,2])
                   
bayesNet.add_cpds(cpd_A, cpd_U, cpd_H, pd_S, cpd_R)
"""
