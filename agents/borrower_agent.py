# Phase 2: Agent Design - Borrower Agent
# This file will contain the implementation of the Borrower Agent.

import numpy as np
import random

class BorrowerAgent:
    def __init__(self, agent_id, initial_profile, market_env):
        self.agent_id = agent_id
        self.profile = initial_profile # e.g., {"trust_score": 0.7, "income": 50000, "collateral": {}}
        self.market_env = market_env
        self.active_loans = []
        self.historical_behavior = [] # To store past actions and outcomes for LSTM training

        # Placeholder for LSTM model - to be developed
        # self.behavior_model = None 

    def get_id(self):
        return self.agent_id

    def update_profile(self, new_data):
        """Updates the borrower's dynamic credit profile."""
        self.profile.update(new_data)
        print(f"Time {self.market_env.env.now}: Borrower {self.agent_id} profile updated: {self.profile}")

    def decide_action(self):
        """Decides whether to request a loan, repay, or default based on its model and market conditions."""
        # This will be driven by the LSTM model in a more advanced version
        # For now, simple probabilistic logic
        market_state = self.market_env.get_state_variables()
        action = None
        
        # Example: Decide to request a loan
        if not self.active_loans and random.random() < 0.1: # 10% chance to request if no active loan
            loan_amount = random.randint(1000, self.profile.get("income", 20000) // 4)
            collateral_value = self.profile.get("collateral", {}).get("collateral_type_A", 0)
            # Requesting a loan (simplified)
            print(f"Time {self.market_env.env.now}: Borrower {self.agent_id} deciding to request a loan of {loan_amount}.")
            action = ("request_loan", loan_amount, "collateral_type_A", collateral_value * 0.1)
            # self.market_env.borrower_request_loan(self.agent_id, loan_amount, "collateral_type_A", collateral_value * 0.1)

        # Example: Decide on repayment or default for an active loan
        elif self.active_loans:
            loan_to_consider = self.active_loans[0] # Simplified: considers the first active loan
            # Simplified decision logic
            if random.random() < 0.05 and market_state["market_volatility"] > 0.15: # Higher chance of default in volatile market
                print(f"Time {self.market_env.env.now}: Borrower {self.agent_id} deciding to default on loan {loan_to_consider['loan_id']}.")
                action = ("default", loan_to_consider["loan_id"])
                # self.market_env.borrower_default_loan(loan_to_consider["loan_id"])
                # self.active_loans.pop(0) # Remove from active loans
            elif random.random() < 0.8: # 80% chance to repay if not defaulting
                repayment_amount = loan_to_consider["amount"] * 0.1 # Simplified: repay 10% of principal
                print(f"Time {self.market_env.env.now}: Borrower {self.agent_id} deciding to repay {repayment_amount} for loan {loan_to_consider['loan_id']}.")
                action = ("repay", loan_to_consider["loan_id"], repayment_amount)
                # self.market_env.borrower_repay_loan(loan_to_consider["loan_id"], repayment_amount)
                # Potentially update loan status or remove if fully paid
        
        if action:
            self.historical_behavior.append({"time": self.market_env.env.now, "action": action, "market_state": market_state})
        return action

    def receive_loan_approval(self, loan_details):
        self.active_loans.append(loan_details)
        print(f"Time {self.market_env.env.now}: Borrower {self.agent_id} received approval for loan: {loan_details['loan_id']}.")

    def receive_loan_rejection(self, request_id):
        print(f"Time {self.market_env.env.now}: Borrower {self.agent_id} loan request {request_id} rejected.")

    def handle_default_consequences(self, loan_id):
        # Logic for what happens to borrower after default (e.g., trust score drops)
        self.profile["trust_score"] = max(0, self.profile.get("trust_score", 0.7) - 0.2)
        self.active_loans = [loan for loan in self.active_loans if loan["loan_id"] != loan_id]
        print(f"Time {self.market_env.env.now}: Borrower {self.agent_id} processed default consequences for loan {loan_id}. New trust score: {self.profile['trust_score']:.2f}")

    def handle_repayment_confirmation(self, loan_id, remaining_balance):
        # Update loan status, potentially remove if fully paid
        for loan in self.active_loans:
            if loan["loan_id"] == loan_id:
                loan["remaining_balance"] = remaining_balance
                if remaining_balance <= 0:
                    self.active_loans.remove(loan)
                    print(f"Time {self.market_env.env.now}: Borrower {self.agent_id} fully repaid loan {loan_id}.")
                break

# Example usage (will be integrated into the main simulation)
if __name__ == "__main__":
    import simpy
    import random
    # A mock market environment for standalone testing
    class MockMarketEnv:
        def __init__(self):
            self.env = simpy.Environment()
            self.now = self.env.now # for direct access in agent
        def get_state_variables(self):
            return {"time": self.env.now, "interest_rate": 0.05, "market_volatility": random.uniform(0.05, 0.2)}
        def borrower_request_loan(self, borrower_id, amount, collateral_type, collateral_amount):
            print(f"MockMarket: Borrower {borrower_id} requests loan {amount} with {collateral_amount} {collateral_type}")
        def borrower_repay_loan(self, loan_id, amount):
            print(f"MockMarket: Borrower repays {amount} for loan {loan_id}")
        def borrower_default_loan(self, loan_id):
            print(f"MockMarket: Borrower defaults on loan {loan_id}")

    mock_market = MockMarketEnv()
    borrower1_profile = {"trust_score": 0.8, "income": 60000, "collateral": {"collateral_type_A": 5000}}
    borrower1 = BorrowerAgent("B001", borrower1_profile, mock_market)

    def test_borrower_proc(env, borrower):
        for _ in range(5):
            action_decision = borrower.decide_action()
            if action_decision:
                print(f"Time {env.now}: Borrower {borrower.get_id()} decided: {action_decision}")
            # Simulate loan approval for testing
            if action_decision and action_decision[0] == "request_loan":
                 borrower.receive_loan_approval({"loan_id": f"L{env.now}", "amount": action_decision[1], "remaining_balance": action_decision[1]})
            yield env.timeout(1)
            mock_market.now = env.now # Update mock market time

    mock_market.env.process(test_borrower_proc(mock_market.env, borrower1))
    mock_market.env.run(until=10)

