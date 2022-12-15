from valueIterationAgents import ValueIterationAgent
from mdp import MarkovDecisionProcess
import util


class PolicyIterationAgent(ValueIterationAgent):
    def __init__(self, mdp: MarkovDecisionProcess, discount=0.9, iterations=100,value_iteration_tolerance=.01):
        """
        Your policy iteration agent should take an mdp on
        construction, run the indicated number of iterations
        or stop on convergeance and then act according to the
        resulting stored policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, 0)
        self.policy = {
            state: mdp.getPossibleActions(state)[0]
            if not mdp.isTerminal(state)
            else None
            for state in mdp.getStates()
        }
        self.value_iteration_tolerance = value_iteration_tolerance
        self.runPolicyIteration(iterations)

    def runPolicyIteration(self, iterations: int):
        """ do NOT modify this function """
        converged = False
        self.iterations_to_converge = 0
        while not converged and self.iterations_to_converge < iterations:
            self.iterations_to_converge += 1
            self.updateValues()
            converged = self.updatePolicy()
        
    def updateValues(self):
        """Update the values until convergeance. Instead of 
        maximizing over the arms when computing new values, 
        you should select the action defined by the agent's 
        current policy stored in self.policy. 
        You can consider that the values converged when the
        biggest change in value is smaller than 
        self.value_iteration_tolerance."""

        # Selon le Slide 15 de Markov decison process 2.
        i_maxChange = 0
        b_converged = False
        while not b_converged:
            tempCounter = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    tempCounter[state] = 0
                else:
                    tempCounter[state] = self.getQValue(state, self.getPolicy(state))
                    i_maxChange = max(i_maxChange, abs(tempCounter[state] - self.values[state]))
            for state in self.mdp.getStates():
                self.values[state] = tempCounter[state]

            if i_maxChange < self.value_iteration_tolerance:
                b_converged = True
            i_maxChange = 0

    def updatePolicy(self):
        """In this method you should update the agent's policy, 
        stored in self.policy, and return a boolean 
        indicating whether the policy has converged."""

        # Selon le Slide 15 de Markov decison process 2, et avec une discussion entre plusieurs étudiants.

        b_converged = True
        for state in self.mdp.getStates():
            max_action = self.computeActionFromValues(state)
            if self.policy[state] != max_action:
                b_converged = False
                self.policy[state] = max_action
        return b_converged
       

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]
