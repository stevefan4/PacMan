import random

# 1. Q-Learning
class QLearningAgent:
    """Implement Q Reinforcement Learning Agent using Q-table."""

    def __init__(self, game, discount, learning_rate, explore_prob):
        """Store any needed parameters into the agent object.
        Initialize Q-table
        """
        self.game = game
        self.discount = discount
        self.lr = learning_rate
        self.e = explore_prob
        self.q = {}
        
    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        if (state, action) not in self.q:
            self.q[(state, action)] = 0
        return self.q[(state, action)]
            
    def get_value(self, state):
        """Compute state value from Q-values using Bellman Equation.
        V(s) = max_a Q(s,a)
        """
        actions = self.game.get_actions(state)
        if len(actions) == 0:
            return 0.0
        pref_act = self.get_best_policy(state)
        return self.get_q_value(state, pref_act)
    
    def get_best_policy(self, state):
        """Compute the best action to take in the state using Policy Extraction.
        π(s) = argmax_a Q(s,a)

        If there are ties, return a random one for better performance.
        Hint: use random.choice().
        """
        actions = self.game.get_actions(state)
        q_vals = []
        if len(actions) == 0:
            return 0.0
        for act in actions:
            best_act = self.get_q_value(state, act)
            q_vals.append((self.get_q_value(state, act), act))
        best_act = max(q_vals, key = lambda ele : ele[0])[1]
        return best_act
                    
    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.
        """
        cur = self.get_q_value(state, action)
        upd = self.get_value(next_state)
        q_val = (1-self.lr) * cur + self.lr * (reward + (self.discount * upd))
        self.q[(state, action)] = q_val
        
    # 2. Epsilon Greedy
    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.
        """
        actions = self.game.get_actions(state)
        if len(actions) == 0:
            return None
        else: 
            if random.random() < self.e:
                return random.choice(list(actions))
            else:
                return self.get_best_policy(state)

# 3. Bridge Crossing Revisited
def question3():
    epsilon = None
    learning_rate = None
    return "NOT POSSIBLE"
    # If not possible, return 'NOT POSSIBLE'


# 5. Approximate Q-Learning
class ApproximateQAgent(QLearningAgent):
    """Implement Approximate Q Learning Agent using weights."""

    def __init__(self, *args, extractor):
        """Initialize parameters and store the feature extractor.
        Initialize weights table."""

        super().__init__(*args)
        self.weights = {}
        self.feat = extractor
        self.alpha = 0.1
        

    def get_weight(self, feature):
        """Get weight of a feature.
        Never seen feature should have a weight of 0.
        """
        if feature not in self.weights:
            self.weights[feature] = 0
        return self.weights[feature]

    def get_q_value(self, state, action):
        """Compute Q value based on the dot product of feature components and weights.
        Q(s,a) = w_1 * f_1(s,a) + w_2 * f_2(s,a) + ... + w_n * f_n(s,a)
        """
        features = self.feat(state, action).items()
        q_val = 0
        for feat, val in features:
            q_val += self.get_weight(feat) * val
        return q_val

    def update(self, state, action, next_state, reward):
        """Update weights using least-squares approximation.
        Δ = R + γ V(s') - Q(s,a)
        Then update weights: w_i = w_i + α * Δ * f_i(s, a)
        """
        features = self.feat(state, action)
        curr = self.get_q_value(state, action)
        nex = self.get_value(next_state)
        delta = reward + self.discount * nex - curr
        for feat in features:
            self.weights[feat] = self.weights[feat] + self.lr * features[feat] * delta

# 6. Feedback
# Just an approximation is fine.
feedback_question_1 = 8

feedback_question_2 = """
Type your response here.
Allowing the Max Q-Value to be negative (not just 0) was the chink in my armor
that sucked up a good chunk of debugging time. That said, all the talk on piazza
allowed me to dial in on the problem relatively quickly. 
"""

feedback_question_3 = """
Type your response here.
My Favorite part was the machine learning side of things. Also, considering how 
we can use some of the other tools we learned in the class to design evaluation functions
that lead to important, orthogonal features. 
"""

