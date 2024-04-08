from pgmpy.models import BayesianNetwork #import necessary pgmpy modules
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import pandas as pd

bayesNet = BayesianNetwork() #initialize network
bayesNet.add_node("A") #Patient Condition
bayesNet.add_node("B") #Test Result
bayesNet.add_node("C") #Treatment Decision
bayesNet.add_node("D") #Outcome

bayesNet.add_edge("C", "D") #establish connections between edges based on dependency
bayesNet.add_edge("A", "B")
bayesNet.add_edge("A", "C")
bayesNet.add_edge("B", "C")

cpd_PatientCondition = TabularCPD('A', 3, values = [[0.7], [0.2], [0.1]],
                                  state_names={'A': ['good', 'fair', 'poor']}) #initialize CPT for A
cpd_TestResult = TabularCPD('B', 2, values = [[0.8, 0.5, 0.3], 
                                              [0.2, 0.5, 0.7]],
                            evidence=['A'], evidence_card=[3],
                            state_names={'B': ['positive', 'negative'],
                                         'A': ['good', 'fair', 'poor']}) #initialize CPT for B considering A
cpd_TreatmentDecision = TabularCPD('C', 2, values = [[0.9, 0.7, 0.5, 0.3, 0.1, 0.1],
                                                     [0.1, 0.3, 0.5, 0.7, 0.9, 0.9]],
                                   evidence=['B', 'A'], evidence_card=[2, 3],
                                   state_names={'C': ['treated', 'not treated'],
                                                'B': ['positive', 'negative'],
                                                'A': ['good', 'fair', 'poor']}) #initialize CPT for C considering A, B
cpd_Outcome = TabularCPD('D', 2, values = [[0.8, 0.2], [0.2, 0.8]],
                         evidence=['C'], evidence_card=[2],
                         state_names={'D': ['positive', 'negative'],
                                      'C': ['treated', 'not treated']}) #initialize CPT for D considering C
bayesNet.add_cpds(cpd_PatientCondition, cpd_TestResult, cpd_TreatmentDecision, cpd_Outcome) #add to network

bayesNet.check_model() #check if model is correct
print("Model is correct.")
solver = VariableElimination(bayesNet) #solver runs variable elimination algorithm to predict probabilities based on evidence

#Query 1
result_a = solver.query(variables=['A'])
print("\nQuery 1: Probability of Patient Condition A")
print(result_a)

#Query 2
result_b = solver.query(variables=['B'], evidence={'A': 'poor'})
print("\nQuery 2: Probability of Test Result B")
print(result_b)

#Query 3
result_c = solver.query(variables=['C'], evidence={'A': 'fair', 'B': 'negative'})
print("\nQuery 3: Probability of Treatment Decision C")
print(result_c)

#Query 4
result = solver.query(variables=['D'], evidence={'A': 'fair', 'B': 'positive', 'C': 'not treated'})
print("\nQuery 4: Probability of Outcome D")
print(result)

# Extract the probabilities from the result
probabilities_a = result_a.values
probabilities_b = result_b.values
probabilities_c = result_c.values

# #Query 5
print("\nQuery 5: Most Likely Outcome D")
# Get the most likely outcome for D which is dependent on C given as not treated
most_likely_outcome = solver.map_query(variables=['D'], evidence={'C': 'not treated'}, show_progress=False)
print("The most likely outcome of D given the observed values of variables A, B, and C is:", most_likely_outcome['D'])


# Query 6
print("\nQuery 6: Predicting Treatment Decision")
# Get the most likely states for the patient's condition (A) and the test result (B)
most_likely_state_a = result_a.state_names['A'][np.argmax(probabilities_a)]
most_likely_state_b = result_b.state_names['B'][np.argmax(probabilities_b)]

# Perform a MAP query to find the most likely treatment decision given the patient's condition and test result
most_likely_treatment_decision = solver.map_query(variables=['C'], evidence={'A': most_likely_state_a, 'B': most_likely_state_b}, show_progress=False)
print("The most likely treatment decision given the patient's condition and the test result is:", most_likely_treatment_decision['C'])


# Query 7: Sensitivity Analysis
#helper CPT printed to show how dependent D is on C being treated or not treated
result_treated = solver.query(variables=['D'], evidence={'C': 'treated'})
result_untreated = solver.query(variables=['D'], evidence={'C': 'not treated'})
print("\nQuery 7: Sensitivity Analysis - Probability distribution of D for different values of C")
print(result_treated)
print(result_untreated)

# Query 8
data = [] #create empty list to store all possible combinations
for condition_A in ['good', 'fair', 'poor']: #for each possible A
    for test_result_B in ['positive', 'negative']: #for each possible B
        for treatment_decision_C in ['treated', 'not treated']: #for each possible C
            result = solver.query(variables=['D'], evidence={'A': condition_A, 'B': test_result_B, 'C': treatment_decision_C}) #get all possible D outcomes dependent on the 3 preceding conditions
            probability_of_outcome_D = result.values #extract probabilities for comparison
            data.append({'Condition A': condition_A, 'Test Result B': test_result_B,
                         'Treatment Decision C': treatment_decision_C,
                         'Probability of Outcome D': probability_of_outcome_D[1]}) #store data
df = pd.DataFrame(data) #store data in dataframe to print as seq
print("\nQuery 8: Probability of Patient Outcome based on Varying Conditions")
print(df)