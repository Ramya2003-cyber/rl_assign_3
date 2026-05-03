class SimulatedTeacher:
    """
    An Oracle that provides preference labels between pairs of trajectory segments
    based on a ground-truth reward function, acting as the human teacher in PEBBLE.
    """
    def __init__(self, feedback_budget):
        """
        Args:
            feedback_budget: Total number of queries the teacher is allowed to answer.
        """
        self.feedback_budget = feedback_budget
        
    def evaluate_preference(self, segment_A, segment_B, env_reward_function):
        """
        Evaluates a preference between two segments using a ground-truth reward function.
        
        Args:
            segment_A: Tuple of (states, actions) representing the first trajectory.
            segment_B: Tuple of (states, actions) representing the second trajectory.
            env_reward_function: A function that takes a state and an action and 
                                 returns the ground-truth reward.
                                 
        Returns:
            label: 1.0 if segment A is strictly preferred over B,
                   0.0 if segment B is strictly preferred over A,
                   0.5 if there is a tie.
            Returns None if the feedback budget is exhausted.
        """
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
