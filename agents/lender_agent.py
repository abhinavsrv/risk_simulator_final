# Phase 2: Agent Design - Lender Agent
# This file will contain the implementation of the Lender Agent.

import random
# Placeholder for RL library imports (e.g., Stable-Baselines3)
# from stable_baselines3 import PPO

class LenderAgent:
    def __init__(self, agent_id, initial_capital, risk_tolerance, market_env):
        self.agent_id = agent_id
        self.capital = initial_capital
        self.risk_tolerance = risk_tolerance # e.g., a value between 0 (low) and 1 (high)
        self.market_env = market_env
        self.portfolio = [] # List of active loans managed by this lender
        self.historical_decisions = []

        # Placeholder for RL model - to be developed
        # self.policy_model = None # This would be the trained RL agent

    def get_id(self):
        return self.agent_id

    def assess_loan_request(self, loan_request):
        """Assesses a loan request based on borrower profile, loan terms, and market conditions."""
        # This will be driven by the RL model in a more advanced version
        # For now, simple rule-based logic
        borrower_profile = loan_request.get("borrower_profile", {"trust_score": 0.5})
        loan_amount = loan_request.get("amount", 0)
        # Simple decision: approve if trust score is high and amount is within capital limits
        approve = False
        if borrower_profile.get("trust_score", 0) > (0.5 + (0.5 - self.risk_tolerance) * 0.4) and loan_amount <= self.capital * 0.1:
            approve = True
        
        decision = "approve" if approve else "reject"
        print(f"Time {self.market_env.env.now}: Lender {self.agent_id} assessing loan request {loan_request.get('request_id')}. Decision: {decision}")
        self.historical_decisions.append({
            "time": self.market_env.env.now,
            "request_id": loan_request.get("request_id"),
            "decision": decision,
            "loan_amount": loan_amount,
            "borrower_trust_score": borrower_profile.get("trust_score", 0)
        })
        return decision

    def allocate_capital(self, loan_details):
        """Allocates capital to an approved loan and adds it to the portfolio."""
        loan_amount = loan_details["amount"]
        if self.capital >= loan_amount:
            self.capital -= loan_amount
            new_loan = {
                "loan_id": loan_details["loan_id"],
                "borrower_id": loan_details["borrower_id"],
                "lender_id": self.agent_id,
                "amount": loan_amount,
                "remaining_balance": loan_amount,
                "interest_rate": self.market_env.get_state_variables()["interest_rates"], # Use current market rate
                "status": "active"
            }
            self.portfolio.append(new_loan)
            print(f"Time {self.market_env.env.now}: Lender {self.agent_id} allocated {loan_amount} to loan {new_loan['loan_id']}. Remaining capital: {self.capital}")
            return new_loan
        else:
            print(f"Time {self.market_env.env.now}: Lender {self.agent_id} has insufficient capital to fund loan {loan_details['loan_id']}.")
            return None

    def handle_repayment(self, loan_id, repayment_amount):
        """Processes a loan repayment."""
        for loan in self.portfolio:
            if loan["loan_id"] == loan_id:
                # Simplified: repayment directly adds to capital (interest not explicitly handled here yet)
                self.capital += repayment_amount 
                loan["amount"] -= repayment_amount # Reduce principal
                if loan["amount"] <= 0:
                    loan["status"] = "repaid"
                    # self.portfolio.remove(loan) # Or mark as repaid
                print(f"Time {self.market_env.env.now}: Lender {self.agent_id} received repayment of {repayment_amount} for loan {loan_id}. Remaining capital: {self.capital}")
                break

    def handle_default(self, loan_id):
        """Processes a loan default."""
        for loan in self.portfolio:
            if loan["loan_id"] == loan_id:
                defaulted_amount = loan["amount"]
                loan["status"] = "defaulted"
                # Capital is not recovered in this simple model upon default
                print(f"Time {self.market_env.env.now}: Lender {self.agent_id} processed default of {defaulted_amount} for loan {loan_id}. Remaining capital: {self.capital}")
                # self.portfolio.remove(loan) # Or mark as defaulted
                # RL agent would receive a negative reward here
                break

    def optimize_portfolio(self):
        """Periodically reviews and rebalances the portfolio (placeholder for RL-driven optimization)."""
        # In an RL setup, the agent would take actions (e.g., adjust lending criteria, sell loans) 
        # based on its learned policy to maximize long-term rewards (e.g., profit, Sharpe ratio).
        print(f"Time {self.market_env.env.now}: Lender {self.agent_id} is reviewing its portfolio (currently {len(self.portfolio)} active loans). Capital: {self.capital}")
        # For now, no specific action taken here.

# Example usage (will be integrated into the main simulation)
if __name__ == "__main__":
    import simpy
    # Mock market environment and loan request for standalone testing
    class MockMarketEnv:
        def __init__(self):
            self.env = simpy.Environment()
            self.now = self.env.now
        def get_state_variables(self):
            return {"time": self.env.now, "interest_rates": 0.06, "market_volatility": 0.1}

    mock_market = MockMarketEnv()
    lender1 = LenderAgent("L001", 100000, 0.5, mock_market)

    def test_lender_proc(env, lender):
        # Simulate a loan request
        loan_req1 = {"request_id": "R001", "borrower_id": "B001", "amount": 5000, "borrower_profile": {"trust_score": 0.75}}
        decision = lender.assess_loan_request(loan_req1)
        yield env.timeout(1)
        mock_market.now = env.now
        if decision == "approve":
            loan_details = {"loan_id": "LN001", "borrower_id": "B001", "amount": 5000}
            lender.allocate_capital(loan_details)
        
        yield env.timeout(5)
        mock_market.now = env.now
        lender.optimize_portfolio()
        
        # Simulate a repayment
        if lender.portfolio:
             lender.handle_repayment("LN001", 1000)

    mock_market.env.process(test_lender_proc(mock_market.env, lender1))
    mock_market.env.run(until=10)
