# Phase 2: Agent Design - Regulator Agent
# This file will contain the implementation of the Regulator Agent.

import random

class RegulatorAgent:
    def __init__(self, agent_id, market_env, initial_rules=None):
        self.agent_id = agent_id
        self.market_env = market_env
        self.rules = initial_rules if initial_rules else {
            "max_interest_rate_ceiling": 0.20, # Example: 20% cap
            "min_capital_adequacy_ratio": 0.08 # Example: 8% for lenders
        }
        self.historical_actions = []

        # Placeholder for a trained model if not rule-based
        # self.policy_model = None

    def get_id(self):
        return self.agent_id

    def monitor_systemic_risk(self):
        """Monitors key systemic risk indicators from the market environment."""
        market_state = self.market_env.get_state_variables()
        system_outputs = self.market_env.get_system_outputs()
        
        # Example: Check overall default rate
        current_default_rate = system_outputs["system_health_metrics"]["default_rate"]
        print(f"Time {self.market_env.env.now}: Regulator {self.agent_id} monitoring. Current system default rate: {current_default_rate:.4f}")

        # Potential risk flags (simplified)
        if current_default_rate > 0.15: # If default rate exceeds 15%
            print(f"Time {self.market_env.env.now}: Regulator {self.agent_id} - ALERT: High systemic default rate detected: {current_default_rate:.4f}")
            return True # Indicates a need for potential intervention
        return False

    def decide_regulatory_action(self):
        """Decides on regulatory actions based on monitored risk and its policy/rules."""
        # This could be rule-based or driven by a trained model
        action_taken = None
        intervention_needed = self.monitor_systemic_risk()

        if intervention_needed:
            # Example rule-based action: Tighten capital requirements if risk is high
            new_capital_requirement = self.rules["min_capital_adequacy_ratio"] * random.uniform(1.05, 1.2) # Increase by 5-20%
            print(f"Time {self.market_env.env.now}: Regulator {self.agent_id} deciding to adjust capital requirements to {new_capital_requirement:.4f}.")
            # The actual call to market_env.regulator_adjust_capital_requirements would be done in the main simulation loop
            action_taken = ("adjust_capital_requirements", new_capital_requirement)
            self.historical_actions.append({
                "time": self.market_env.env.now,
                "action": action_taken,
                "reason": "High systemic default rate"
            })
        else:
            # Example: Periodically adjust interest rate ceilings slightly
            if self.market_env.env.now % 10 == 0: # Every 10 time steps
                new_ceiling = self.rules["max_interest_rate_ceiling"] * random.uniform(0.98, 1.02)
                print(f"Time {self.market_env.env.now}: Regulator {self.agent_id} deciding to adjust max interest rate ceiling to {new_ceiling:.4f}.")
                action_taken = ("set_interest_ceiling", new_ceiling)
                self.historical_actions.append({
                    "time": self.market_env.env.now,
                    "action": action_taken,
                    "reason": "Periodic review"
                })
        return action_taken

# Example usage (will be integrated into the main simulation)
if __name__ == "__main__":
    import simpy
    # Mock market environment for standalone testing
    class MockMarketEnv:
        def __init__(self):
            self.env = simpy.Environment()
            self.now = self.env.now
            self.default_rate = 0.05

        def get_state_variables(self):
            return {"time": self.env.now, "interest_rates": 0.06, "market_volatility": 0.1}

        def get_system_outputs(self):
            # Simulate increasing default rate for testing
            if self.env.now > 5:
                self.default_rate = 0.16
            return {"system_health_metrics": {"default_rate": self.default_rate, "liquidity_ratio": 0.9, "solvency_ratio": 0.9}}
        
        def regulator_adjust_capital_requirements(self, new_requirement):
            print(f"MockMarket: Regulator adjusted capital requirements to {new_requirement}")

        def regulator_set_interest_ceiling(self, new_ceiling):
            print(f"MockMarket: Regulator set interest ceiling to {new_ceiling}")

    mock_market = MockMarketEnv()
    regulator1 = RegulatorAgent("R001", mock_market)

    def test_regulator_proc(env, regulator):
        for i in range(15):
            action = regulator.decide_regulatory_action()
            if action:
                print(f"Time {env.now}: Regulator {regulator.get_id()} decided action: {action}")
                # Simulate applying the action to the market
                if action[0] == "adjust_capital_requirements":
                    mock_market.regulator_adjust_capital_requirements(action[1])
                elif action[0] == "set_interest_ceiling":
                    mock_market.regulator_set_interest_ceiling(action[1])
            yield env.timeout(1)
            mock_market.now = env.now

    mock_market.env.process(test_regulator_proc(mock_market.env, regulator1))
    mock_market.env.run(until=15)

