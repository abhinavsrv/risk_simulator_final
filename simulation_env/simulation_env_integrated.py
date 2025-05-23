# Phase 1 & 2: Simulation Environment Setup & Agent Integration
# This file contains the core logic for the simulation environment and integrates agents.

import simpy
import random
import sys
import os

# Adjust Python path to include the project root directory for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import agent classes and policy/reward functions using absolute paths
from agents.borrower_agent import BorrowerAgent
from agents.lender_agent import LenderAgent
from agents.regulator_agent import RegulatorAgent
from agents.agent_policies_rewards import (
    get_borrower_policy_network, calculate_borrower_reward,
    get_lender_policy_network, calculate_lender_reward,
    get_regulator_policy_network, calculate_regulator_reward
)

class FinancialMarket:
    def __init__(self, env):
        self.env = env
        # Initial market conditions
        self.interest_rate = 0.05
        self.market_volatility = 0.1
        self.liquidity_index = 1.0
        self.asset_prices = {"collateral_type_A": 100}
        self.loan_requests = [] # To store pending loan requests
        self.next_loan_id = 1
        self.next_request_id = 1

        # System metrics
        self.total_loans_issued = 0
        self.total_defaults = 0
        self.total_repayments = 0
        self.active_loans = []
        self.system_health_metrics = {
            "default_rate": 0.0,
            "liquidity_ratio": 1.0,
            "solvency_ratio": 1.0,
            "current_capital_adequacy_requirement": 0.08 # Initial regulator setting
        }

    def update_market_conditions(self):
        self.interest_rate = max(0.01, self.interest_rate + random.uniform(-0.005, 0.005))
        self.market_volatility = max(0.05, self.market_volatility + random.uniform(-0.01, 0.01))
        for asset in self.asset_prices:
            self.asset_prices[asset] *= (1 + random.uniform(-self.market_volatility/2, self.market_volatility/2))
        print(f"Time {self.env.now}: Market updated - IR: {self.interest_rate:.4f}, Vol: {self.market_volatility:.4f}, Asset A: {self.asset_prices['collateral_type_A']:.2f}")

    def get_state_variables(self):
        return {
            "system_time": self.env.now,
            "interest_rates": self.interest_rate,
            "asset_prices": self.asset_prices.copy(),
            "market_volatility": self.market_volatility,
            "liquidity_index": self.liquidity_index,
        }

    def get_system_outputs(self):
        if self.total_loans_issued > 0:
            self.system_health_metrics["default_rate"] = self.total_defaults / self.total_loans_issued
        else:
            self.system_health_metrics["default_rate"] = 0.0
        return {
            "total_loans_issued": self.total_loans_issued,
            "total_defaults": self.total_defaults,
            "total_repayments": self.total_repayments,
            "active_loans_count": len(self.active_loans),
            "pending_loan_requests": len(self.loan_requests),
            "system_health_metrics": self.system_health_metrics.copy()
        }

    # --- Agent Interaction Methods ---
    def submit_loan_request(self, borrower_agent, amount, collateral_type, collateral_amount):
        request_id = f"REQ{self.next_request_id}"
        self.next_request_id += 1
        loan_req = {
            "request_id": request_id,
            "borrower_id": borrower_agent.get_id(),
            "borrower_profile": borrower_agent.profile, # Lender needs this
            "amount": amount,
            "collateral_type": collateral_type,
            "collateral_amount": collateral_amount,
            "status": "pending"
        }
        self.loan_requests.append(loan_req)
        print(f"Time {self.env.now}: Borrower {borrower_agent.get_id()} submitted loan request {request_id} for {amount}.")
        return request_id

    def process_loan_decision(self, lender_agent, request_id, decision, borrower_agents):
        # Find the request
        req_to_process = next((req for req in self.loan_requests if req["request_id"] == request_id), None)
        
        if not req_to_process or req_to_process["status"] != "pending":
            print(f"Time {self.env.now}: Lender {lender_agent.get_id()} - Request {request_id} not found or already processed.")
            return

        borrower_agent = next((b for b in borrower_agents if b.get_id() == req_to_process["borrower_id"]), None)

        if decision == "approve":
            loan_id = f"LN{self.next_loan_id}"
            self.next_loan_id += 1
            loan_details = {
                "loan_id": loan_id,
                "request_id": request_id,
                "borrower_id": req_to_process["borrower_id"],
                "lender_id": lender_agent.get_id(),
                "amount": req_to_process["amount"],
                "collateral_type": req_to_process["collateral_type"],
                "collateral_value_at_origination": self.asset_prices.get(req_to_process["collateral_type"], 0) * req_to_process["collateral_amount"],
                "interest_rate": self.interest_rate, # Market rate at time of approval
                "status": "active",
                "remaining_balance": req_to_process["amount"]
            }
            funded_loan = lender_agent.allocate_capital(loan_details) # Lender updates its own books
            if funded_loan:
                self.active_loans.append(funded_loan)
                self.total_loans_issued += 1
                req_to_process["status"] = "approved"
                if borrower_agent: borrower_agent.receive_loan_approval(funded_loan)
                print(f"Time {self.env.now}: Lender {lender_agent.get_id()} APPROVED request {request_id} (Loan {loan_id}).")
            else:
                # Lender had insufficient capital
                req_to_process["status"] = "rejected_by_lender_capital"
                if borrower_agent: borrower_agent.receive_loan_rejection(request_id)
                print(f"Time {self.env.now}: Lender {lender_agent.get_id()} REJECTED request {request_id} due to insufficient capital.")
        else: # decision == "reject"
            req_to_process["status"] = "rejected_by_lender_policy"
            if borrower_agent: borrower_agent.receive_loan_rejection(request_id)
            print(f"Time {self.env.now}: Lender {lender_agent.get_id()} REJECTED request {request_id} by policy.")
        self.loan_requests.remove(req_to_process)

    def process_repayment(self, borrower_id, loan_id, amount, lender_agents):
        """Processes a loan repayment."""
        loan = next((l for l in self.active_loans if l["loan_id"] == loan_id and l["borrower_id"] == borrower_id), None)
        if loan and loan["status"] == "active":
            # Find the lender for this loan
            lender = next((l_agent for l_agent in lender_agents if l_agent.get_id() == loan["lender_id"]), None)
            loan["remaining_balance"] -= amount
            if lender: lender.handle_repayment(loan_id, amount)
            print(f"Time {self.env.now}: Borrower {borrower_id} repaid {amount} for loan {loan_id}. Remaining: {loan['remaining_balance']:.2f}")
            if loan["remaining_balance"] <= 0:
                loan["status"] = "repaid"
                self.total_repayments += 1
                # Reward calculation would happen here
            return loan["remaining_balance"]
        return -1 # Error or loan not found

    def process_default(self, borrower_id, loan_id, lender_agents, borrower_agents):
        """Processes a loan default."""
        loan = next((l for l in self.active_loans if l["loan_id"] == loan_id and l["borrower_id"] == borrower_id), None)
        if loan and loan["status"] == "active":
            # Find the lender and borrower for this loan
            lender = next((l_agent for l_agent in lender_agents if l_agent.get_id() == loan["lender_id"]), None)
            borrower = next((b_agent for b_agent in borrower_agents if b_agent.get_id() == borrower_id), None)
            loan["status"] = "defaulted"
            self.total_defaults += 1
            if lender: lender.handle_default(loan_id)
            if borrower: borrower.handle_default_consequences(loan_id)
            print(f"Time {self.env.now}: Borrower {borrower_id} DEFAULTED on loan {loan_id}.")
            # Reward calculation would happen here
            return True
        return False

    def apply_regulator_action(self, action_details):
        action_type, value = action_details
        if action_type == "adjust_capital_requirements":
            self.system_health_metrics["current_capital_adequacy_requirement"] = value
            print(f"Time {self.env.now}: Market - Regulator adjusted capital requirements to {value:.4f}.")
        elif action_type == "set_interest_ceiling":
            # This would need to be enforced on lenders or new loan rates
            print(f"Time {self.env.now}: Market - Regulator set interest ceiling to {value:.4f} (enforcement TBD).")

# Main simulation process
def simulation_process_with_agents(env, market, borrowers, lenders, regulators):
    print(f"--- Simulation Started with Agents at Time {env.now} ---")

    # Initial agent setup (e.g. loading policies - done in agent init for now)
    for b in borrowers: b.policy_model = get_borrower_policy_network(b.get_id())
    for l_agent in lenders: l_agent.policy_model = get_lender_policy_network(l_agent.get_id())
    for r_agent in regulators: r_agent.policy_model = get_regulator_policy_network(r_agent.get_id())

    simulation_duration = 50 # Number of time steps
    for i in range(simulation_duration):
        print(f"\n--- Time Step: {env.now} ---")
        market.update_market_conditions()

        # --- Agent Action Phase ---
        # 1. Borrowers decide actions
        for b_agent in borrowers:
            action = b_agent.decide_action() # Agent decides based on its internal logic/policy
            if action:
                action_type = action[0]
                if action_type == "request_loan":
                    _, amount, c_type, c_amount = action
                    market.submit_loan_request(b_agent, amount, c_type, c_amount)
                elif action_type == "repay":
                    _, loan_id, repay_amount = action
                    market.process_repayment(b_agent.get_id(), loan_id, repay_amount, lenders)
                elif action_type == "default":
                    _, loan_id = action
                    market.process_default(b_agent.get_id(), loan_id, lenders, borrowers)
        
        # 2. Lenders decide on pending loan requests
        # Create a copy for iteration as market.loan_requests can be modified
        pending_requests_copy = list(market.loan_requests)
        for req in pending_requests_copy:
            # Simple: first lender processes all for now. Could be more complex (e.g. matching)
            if lenders:
                lender_agent = lenders[0] 
                decision = lender_agent.assess_loan_request(req) # Lender uses its policy
                market.process_loan_decision(lender_agent, req["request_id"], decision, borrowers)

        # 3. Lenders optimize portfolio (periodic)
        if env.now % 10 == 0: # Example: every 10 steps
            for l_agent in lenders:
                l_agent.optimize_portfolio()

        # 4. Regulators decide actions
        for r_agent in regulators:
            reg_action = r_agent.decide_regulatory_action()
            if reg_action:
                market.apply_regulator_action(reg_action)

        # --- Environment Outputs Phase ---
        current_market_state = market.get_state_variables()
        system_outputs = market.get_system_outputs()
        print(f"Time {env.now}: Market State: {current_market_state}")
        print(f"Time {env.now}: System Outputs: {system_outputs}")

        # --- Reward Calculation Phase (Example) ---
        # This would be more sophisticated, linking outcomes to specific agent actions for RL
        # For now, just illustrative prints if an action led to a reward-triggering event
        # calculate_borrower_reward(...), calculate_lender_reward(...), calculate_regulator_reward(...)

        yield env.timeout(1)

    print(f"\n--- Simulation Max Time ({simulation_duration} steps) Reached ---")

if __name__ == "__main__":
    env = simpy.Environment()
    market = FinancialMarket(env)

    # Initialize Agents
    borrower_agents = [
        BorrowerAgent("B001", {"trust_score": 0.8, "income": 60000, "collateral": {"collateral_type_A": 500}}, market),
        BorrowerAgent("B002", {"trust_score": 0.6, "income": 40000, "collateral": {"collateral_type_A": 200}}, market)
    ]
    lender_agents = [
        LenderAgent("L001", 1000000, 0.5, market) # 1M capital, medium risk tolerance
    ]
    regulator_agents = [
        RegulatorAgent("R001", market)
    ]

    env.process(simulation_process_with_agents(env, market, borrower_agents, lender_agents, regulator_agents))
    env.run()

    print("\n--- Simulation Finished ---")
    final_state = market.get_state_variables()
    final_outputs = market.get_system_outputs()
    print(f"Final Market State Variables: {final_state}")
    print(f"Final System Outputs: {final_outputs}")
    print(f"Lender L001 final capital: {lender_agents[0].capital}")
    print(f"Borrower B001 active loans: {len(borrower_agents[0].active_loans)}")
    print(f"Borrower B002 active loans: {len(borrower_agents[1].active_loans)}")

