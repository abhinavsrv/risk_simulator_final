# Phase 2: Agent Policy and Reward Functions
# This file will define placeholder policy networks and reward functions for the agents.

def get_borrower_policy_network(agent_id):
    """Placeholder for Borrower Agent's policy network (e.g., an LSTM model)."""
    print(f"Time (Policy): Borrower {agent_id} policy network accessed (placeholder).")
    # In a real scenario, this would return a loaded/initialized neural network.
    return None # Or a mock policy object

def get_lender_policy_network(agent_id):
    """Placeholder for Lender Agent's policy network (e.g., an RL model like PPO)."""
    print(f"Time (Policy): Lender {agent_id} policy network accessed (placeholder).")
    # In a real scenario, this would return a loaded/initialized RL agent/policy.
    return None # Or a mock policy object

def get_regulator_policy_network(agent_id):
    """Placeholder for Regulator Agent's policy network (if not purely rule-based)."""
    print(f"Time (Policy): Regulator {agent_id} policy network accessed (placeholder).")
    return None

def calculate_borrower_reward(action_outcome, market_env_state):
    """Calculates reward for a Borrower Agent based on action outcome."""
    reward = 0
    if action_outcome == "loan_repaid_on_time":
        reward = 10
    elif action_outcome == "loan_defaulted":
        reward = -100
    elif action_outcome == "loan_obtained":
        reward = 5
    # print(f"Time (Reward): Borrower reward calculated: {reward} for outcome {action_outcome}")
    return reward

def calculate_lender_reward(action_outcome, loan_amount, market_env_state):
    """Calculates reward for a Lender Agent. Reward = loan ROI - risk penalty - systemic risk impact."""
    reward = 0
    # Simplified reward structure
    if action_outcome == "loan_approved_and_repaid":
        # Assume some interest earned
        interest_earned = loan_amount * market_env_state.get("interest_rates", 0.05) * 0.1 # Simplified interest
        reward = interest_earned 
    elif action_outcome == "loan_approved_and_defaulted":
        reward = -loan_amount # Loss of principal
    elif action_outcome == "loan_rejected_correctly": # (e.g. avoided a likely default)
        reward = 5 # Small reward for good risk assessment
    elif action_outcome == "loan_rejected_incorrectly": # (e.g. missed a good opportunity)
        reward = -2
    
    # Placeholder for risk penalty and systemic risk impact
    risk_penalty = 0 # Could be based on VaR, CVaR of the loan
    systemic_risk_impact = 0 # Could be based on contribution to overall market instability

    final_reward = reward - risk_penalty - systemic_risk_impact
    # print(f"Time (Reward): Lender reward calculated: {final_reward} for outcome {action_outcome}")
    return final_reward

def calculate_regulator_reward(action_outcome, market_env_state):
    """Calculates reward for a Regulator Agent (if it's a learning agent)."""
    reward = 0
    # Example: Reward for maintaining low systemic risk
    if market_env_state.get("system_health_metrics", {}).get("default_rate", 1) < 0.05:
        reward += 50
    if action_outcome == "stabilized_market":
        reward += 100
    elif action_outcome == "failed_to_prevent_crisis":
        reward -= 200
    # print(f"Time (Reward): Regulator reward calculated: {reward} for outcome {action_outcome}")
    return reward

if __name__ == "__main__":
    print("Testing policy and reward function placeholders...")
    mock_market_state = {
        "interest_rates": 0.07,
        "system_health_metrics": {"default_rate": 0.03}
    }
    get_borrower_policy_network("B001")
    get_lender_policy_network("L001")
    print(f"Borrower reward (repaid): {calculate_borrower_reward('loan_repaid_on_time', mock_market_state)}")
    print(f"Borrower reward (defaulted): {calculate_borrower_reward('loan_defaulted', mock_market_state)}")
    print(f"Lender reward (repaid): {calculate_lender_reward('loan_approved_and_repaid', 10000, mock_market_state)}")
    print(f"Lender reward (defaulted): {calculate_lender_reward('loan_approved_and_defaulted', 10000, mock_market_state)}")
    print(f"Regulator reward (stable market): {calculate_regulator_reward('stabilized_market', mock_market_state)}")

