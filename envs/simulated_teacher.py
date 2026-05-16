class SimulatedTeacher:
    
    def __init__(self, feedback_budget):
       
        self.feedback_budget = feedback_budget
        
    def evaluate_preference(self, segment_A, segment_B, env_reward_function):
       
        if self.feedback_budget <= 0:
            print("Warning: Simulated Teacher feedback budget exhausted.")
            return None
            
        # Decrement budget
        self.feedback_budget -= 1
        
        states_A, actions_A = segment_A
        states_B, actions_B = segment_B
        
        # Calculate the ground truth sum of rewards for segment A
        rewards_A = [env_reward_function(s, a) for s, a in zip(states_A, actions_A)]
        sum_R_A = sum(rewards_A)
        
        # Calculate the ground truth sum of rewards for segment B
        rewards_B = [env_reward_function(s, a) for s, a in zip(states_B, actions_B)]
        sum_R_B = sum(rewards_B)
        
        # Determine the preference label based on strict inequalities
        if sum_R_A > sum_R_B:
            return 1.0
        elif sum_R_B > sum_R_A:
            return 0.0
        else:
            return 0.5  # Soft label for ties
