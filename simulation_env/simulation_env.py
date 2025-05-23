# Phase 1: Simulation Environment Setup
# This file contains the core logic for the simulation environment.

import simpy
import random

class FinancialMarket:
    def __init__(self, env):
        self.env = env
        self.current_time = 0
        # Initial market conditions
        self.interest_rate = 0.05  # Annual interest rate
        self.market_volatility = 0.1 # Market volatility index
        self.liquidity_index = 1.0   # Liquidity index (1.0 = normal)
        self.asset_prices = {"collateral_type_A": 100} # Example asset prices

        # System metrics
        self.total_loans_issued = 0
        self.total_defaults = 0
        self.total_repayments = 0
        self.active_loans = [] # List to store active loan objects or IDs
        self.system_health_metrics = {
            "default_rate": 0.0,
            "liquidity_ratio": 1.0, # Example metric
            "solvency_ratio": 1.0 # Example metric
        }

    def update_market_conditions(self):
        """Dynamically updates market conditions at each time step."""
        # Simple stochastic model for interest rates and volatility
        self.interest_rate = max(0.01, self.interest_rate + random.uniform(-0.005, 0.005))
        self.market_volatility = max(0.05, self.market_volatility + random.uniform(-0.01, 0.01))
        # Asset prices could also fluctuate
        for asset in self.asset_prices:
            self.asset_prices[asset] *= (1 + random.uniform(-self.market_volatility/2, self.market_volatility/2))
        
        print(f"Time {self.env.now}: Market conditions updated - Interest Rate: {self.interest_rate:.4f}, Volatility: {self.market_volatility:.4f}, Asset A Price: {self.asset_prices['collateral_type_A']:.2f}")

    def get_state_variables(self):
        """Returns the current state variables of the market."""
        return {
            "system_time": self.env.now,
            "interest_rates": self.interest_rate,
            "asset_prices": self.asset_prices.copy(),
            "market_volatility": self.market_volatility,
            "liquidity_index": self.liquidity_index,
        }

    def get_system_outputs(self):
        """Returns key system outputs and health metrics."""
        if self.total_loans_issued > 0:
            self.system_health_metrics["default_rate"] = self.total_defaults / self.total_loans_issued
        else:
            self.system_health_metrics["default_rate"] = 0.0
        # Placeholder for other metrics like liquidity and solvency
        return {
            "total_loans_issued": self.total_loans_issued,
            "total_defaults": self.total_defaults,
            "total_repayments": self.total_repayments,
            "active_loans_count": len(self.active_loans),
            "system_health_metrics": self.system_health_metrics.copy()
        }

    # Placeholder for Agent Actions - to be called by agent objects
    def borrower_request_loan(self, borrower_id, amount, collateral_type, collateral_amount):
        print(f"Time {self.env.now}: Borrower {borrower_id} requests loan of {amount} with {collateral_amount} of {collateral_type}.")
        # Logic for lender to approve/reject will be separate
        pass

    def borrower_repay_loan(self, loan_id, amount):
        print(f"Time {self.env.now}: Borrower repays {amount} for loan {loan_id}.")
        self.total_repayments += 1 # Simplified, assumes full repayment for count
        # Actual loan object update logic needed here
        pass

    def borrower_default_loan(self, loan_id):
        print(f"Time {self.env.now}: Borrower defaults on loan {loan_id}.")
        self.total_defaults += 1
        # Actual loan object update logic needed here
        pass

    def lender_approve_loan(self, loan_request_id, lender_id):
        print(f"Time {self.env.now}: Lender {lender_id} approves loan request {loan_request_id}.")
        self.total_loans_issued += 1
        # Logic to create a loan object and add to active_loans
        pass

    def lender_reject_loan(self, loan_request_id, lender_id):
        print(f"Time {self.env.now}: Lender {lender_id} rejects loan request {loan_request_id}.")
        pass

    def regulator_adjust_capital_requirements(self, new_requirement):
        print(f"Time {self.env.now}: Regulator adjusts capital requirements to {new_requirement}.")
        # Logic to update system parameters based on this
        pass

# Main simulation loop (event-based)
def simulation_process(env, market):
    """The core simulation process, managing time steps and agent interactions."""
    print(f"--- Simulation Started at Time {env.now} ---")
    # Example agents (will be replaced by actual agent objects)
    borrower_agents = [{"id": "B1"}, {"id": "B2"}]
    lender_agents = [{"id": "L1"}]
    regulator_agents = [{"id": "R1"}]

    while True:
        print(f"\n--- Time Step: {env.now} ---")
        market.update_market_conditions()

        # --- Agent Actions Phase ---
        # This is a simplified representation. In a full model, agents would make decisions based on market state.
        
        # Example Borrower Actions
        if env.now % 5 == 0 and env.now > 0: # Every 5 steps, a borrower might request a loan
            borrower = random.choice(borrower_agents)
            market.borrower_request_loan(borrower["agent_id"], random.randint(1000, 5000), "collateral_type_A", random.randint(10,50))

        # Example Lender Actions (reacting to requests - this would be more complex)
        # For now, let's assume a lender reviews outstanding requests (not implemented yet)
        if market.total_loans_issued < 5 and env.now % 6 == 0 and env.now > 0: # Simplified approval
             market.lender_approve_loan(f"req_{env.now}", lender_agents[0]["agent_id"]) # Dummy request ID

        # Example Repayment/Default (highly simplified)
        if market.active_loans and env.now % 10 == 0 and env.now > 0:
            loan_to_act_on = random.choice(market.active_loans) # This needs proper loan objects
            if random.random() < 0.1: # 10% chance of default on an active loan
                market.borrower_default_loan(loan_to_act_on) 
            else:
                market.borrower_repay_loan(loan_to_act_on, random.randint(100,500)) # Dummy repayment

        # Example Regulator Action
        if env.now % 20 == 0 and env.now > 0: # Every 20 steps
            market.regulator_adjust_capital_requirements(random.uniform(0.08, 0.12))

        # --- Environment Outputs Phase ---
        current_market_state = market.get_state_variables()
        system_outputs = market.get_system_outputs()
        print(f"Time {env.now}: Current Market State: {current_market_state}")
        print(f"Time {env.now}: System Outputs: {system_outputs}")

        if env.now >= 50: # Stop condition for this test run
            print("--- Simulation Max Time Reached ---")
            break

        yield env.timeout(1) # Advance simulation by one time step (e.g., one day)

if __name__ == "__main__":
    # Setup and start the simulation
    env = simpy.Environment()
    market = FinancialMarket(env)
    env.process(simulation_process(env, market))
    env.run()

    print("\n--- Simulation Finished ---")
    final_state = market.get_state_variables()
    final_outputs = market.get_system_outputs()
    print(f"Final Market State Variables: {final_state}")
    print(f"Final System Outputs: {final_outputs}")

