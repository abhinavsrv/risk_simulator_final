# Phase 4: Portfolio Optimization for DeFi Credit Risk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import scipy.optimize as sco
from scipy import stats
import cvxpy as cp

class PortfolioOptimizer:
    """
    Portfolio Optimizer for DeFi lending strategies.
    
    This class implements portfolio optimization techniques including:
    - Markowitz Mean-Variance Optimization
    - Conditional Value at Risk (CVaR) Optimization
    - Risk-adjusted return optimization using Sharpe Ratio
    
    The optimizer helps lenders allocate capital across different borrower segments
    to maximize returns while controlling risk.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the portfolio optimizer.
        
        Args:
            data_dir (str): Directory containing the loan data files
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        else:
            self.data_dir = data_dir
            
        self.results_dir = os.path.join(self.data_dir, 'portfolio_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load credit risk model for PD predictions if available
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      'ml_models', 'saved_models', 'credit_risk_model.pkl')
        self.credit_model = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.credit_model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.feature_names = saved_data['feature_names']
        
    def load_loan_data(self):
        """
        Load and prepare loan data for portfolio optimization.
        
        Returns:
            pandas.DataFrame: Processed loan data with segment information
        """
        # Load the complete loan data
        data_file = os.path.join(self.data_dir, 'complete_loan_data.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}. Run synthetic_data_generator.py first.")
        
        df = pd.read_csv(data_file)
        print(f"Loaded loan data with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Create borrower segments based on trust score
        trust_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        trust_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['trust_segment'] = pd.cut(df['trust_score'], bins=trust_bins, labels=trust_labels)
        
        # Create loan amount segments
        amount_bins = [0, 5000, 10000, 20000, 50000, float('inf')]
        amount_labels = ['Micro', 'Small', 'Medium', 'Large', 'Very Large']
        df['amount_segment'] = pd.cut(df['loan_amount'], bins=amount_bins, labels=amount_labels)
        
        # Create combined segments for portfolio allocation
        df['portfolio_segment'] = df['trust_segment'].astype(str) + '_' + df['amount_segment'].astype(str)
        
        # Calculate historical returns for each loan
        # Return = interest earned - losses from defaults
        # Check if days_to_event exists, otherwise use loan_term
        if 'days_to_event' in df.columns:
            days_column = 'days_to_event'
        else:
            days_column = 'loan_term'
            
        df['interest_earned'] = np.where(
            df['defaulted'] == 1,
            df['loan_amount'] * df['interest_rate'] * (df[days_column] / 365),
            df['loan_amount'] * df['interest_rate']
        )
        
        df['loss_amount'] = df['loss_amount'].fillna(0)
        df['net_return'] = df['interest_earned'] - df['loss_amount']
        df['return_rate'] = df['net_return'] / df['loan_amount']
        
        return df
    
    def calculate_segment_statistics(self, loan_data):
        """
        Calculate return and risk statistics for each portfolio segment.
        
        Args:
            loan_data (pandas.DataFrame): Loan data with segment information
            
        Returns:
            pandas.DataFrame: Segment statistics including expected return and risk
        """
        # Group by portfolio segment and calculate statistics
        segment_stats = loan_data.groupby('portfolio_segment').agg({
            'return_rate': ['mean', 'std', 'count'],
            'defaulted': 'mean',
            'loan_amount': 'sum'
        })
        
        # Flatten multi-level column names
        segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns.values]
        
        # Rename columns for clarity
        segment_stats = segment_stats.rename(columns={
            'return_rate_mean': 'expected_return',
            'return_rate_std': 'risk',
            'return_rate_count': 'num_loans',
            'defaulted_mean': 'default_rate',
            'loan_amount_sum': 'total_amount'
        })
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        segment_stats['sharpe_ratio'] = (segment_stats['expected_return'] - risk_free_rate) / segment_stats['risk']
        
        # Calculate portfolio weights based on historical allocation
        segment_stats['historical_weight'] = segment_stats['total_amount'] / segment_stats['total_amount'].sum()
        
        # Sort by Sharpe ratio
        segment_stats = segment_stats.sort_values('sharpe_ratio', ascending=False)
        
        return segment_stats
    
    def calculate_covariance_matrix(self, loan_data):
        """
        Calculate the covariance matrix of returns between portfolio segments.
        
        Args:
            loan_data (pandas.DataFrame): Loan data with segment information
            
        Returns:
            pandas.DataFrame: Covariance matrix of segment returns
        """
        # Create pivot table of returns by segment
        # For segments with multiple loans, we'll use the average return per time period
        loan_data['year_month'] = pd.to_datetime(loan_data['application_date']).dt.to_period('M')
        
        # Calculate average return per segment per month
        segment_returns = loan_data.groupby(['year_month', 'portfolio_segment'])['return_rate'].mean().unstack()
        
        # Fill missing values with the mean return for that segment
        segment_means = segment_returns.mean()
        segment_returns = segment_returns.fillna(segment_means)
        
        # Calculate covariance matrix
        cov_matrix = segment_returns.cov()
        
        return cov_matrix
    
    def markowitz_optimization(self, expected_returns, cov_matrix, risk_tolerance=None, target_return=None):
        """
        Perform Markowitz portfolio optimization.
        
        Args:
            expected_returns (pandas.Series): Expected returns for each segment
            cov_matrix (pandas.DataFrame): Covariance matrix of segment returns
            risk_tolerance (float): Risk tolerance parameter (None for minimum variance)
            target_return (float): Target return (None for unconstrained optimization)
            
        Returns:
            dict: Optimized portfolio weights and metrics
        """
        num_assets = len(expected_returns)
        
        # Define the optimization problem
        weights = cp.Variable(num_assets)
        returns = expected_returns.values @ weights
        risk = cp.quad_form(weights, cov_matrix.values)
        
        # Set up the objective function based on parameters
        if risk_tolerance is not None:
            # Maximize return - risk_tolerance * risk
            objective = cp.Maximize(returns - risk_tolerance * risk)
        else:
            # Minimize risk (minimum variance portfolio)
            objective = cp.Minimize(risk)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= 0  # No short selling
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append(returns >= target_return)
        
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Check if the problem was solved successfully
        if problem.status != 'optimal':
            raise ValueError(f"Optimization failed with status: {problem.status}")
        
        # Extract the optimal weights
        optimal_weights = weights.value
        
        # Calculate portfolio metrics
        portfolio_return = expected_returns.values @ optimal_weights
        portfolio_risk = np.sqrt(optimal_weights.T @ cov_matrix.values @ optimal_weights)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk
        
        # Create a dictionary of results
        results = {
            'weights': pd.Series(optimal_weights, index=expected_returns.index),
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }
        
        return results
    
    def cvar_optimization(self, loan_data, alpha=0.05, target_return=None):
        """
        Perform Conditional Value at Risk (CVaR) optimization.
        
        Args:
            loan_data (pandas.DataFrame): Loan data with segment information
            alpha (float): Confidence level for CVaR (e.g., 0.05 for 95% confidence)
            target_return (float): Target return (None for unconstrained optimization)
            
        Returns:
            dict: Optimized portfolio weights and metrics
        """
        # Create pivot table of returns by segment
        loan_data['year_month'] = pd.to_datetime(loan_data['application_date']).dt.to_period('M')
        segment_returns = loan_data.groupby(['year_month', 'portfolio_segment'])['return_rate'].mean().unstack()
        
        # Fill missing values with the mean return for that segment
        segment_means = segment_returns.mean()
        segment_returns = segment_returns.fillna(segment_means)
        
        # Get dimensions
        n_scenarios, n_assets = segment_returns.shape
        
        # Define the optimization variables
        weights = cp.Variable(n_assets)
        aux_vars = cp.Variable(n_scenarios)
        var = cp.Variable(1)
        
        # Expected returns
        expected_returns = segment_returns.mean()
        portfolio_return = expected_returns.values @ weights
        
        # CVaR constraints
        scenario_returns = segment_returns.values @ weights
        cvar_constraints = [aux_vars >= 0, aux_vars >= var - scenario_returns]
        
        # Set up the objective function to minimize CVaR
        objective = cp.Minimize(var + (1/(alpha*n_scenarios)) * cp.sum(aux_vars))
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= 0,  # No short selling
        ] + cvar_constraints
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append(portfolio_return >= target_return)
        
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Check if the problem was solved successfully
        if problem.status != 'optimal':
            raise ValueError(f"Optimization failed with status: {problem.status}")
        
        # Extract the optimal weights
        optimal_weights = weights.value
        
        # Calculate portfolio metrics
        portfolio_return = expected_returns.values @ optimal_weights
        cvar_value = problem.value
        
        # Calculate standard deviation for comparison
        cov_matrix = segment_returns.cov()
        portfolio_risk = np.sqrt(optimal_weights.T @ cov_matrix.values @ optimal_weights)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk
        
        # Create a dictionary of results
        results = {
            'weights': pd.Series(optimal_weights, index=segment_returns.columns),
            'return': portfolio_return,
            'risk': portfolio_risk,
            'cvar': cvar_value,
            'sharpe_ratio': sharpe_ratio
        }
        
        return results
    
    def efficient_frontier(self, expected_returns, cov_matrix, num_portfolios=50):
        """
        Calculate the efficient frontier for portfolio visualization.
        
        Args:
            expected_returns (pandas.Series): Expected returns for each segment
            cov_matrix (pandas.DataFrame): Covariance matrix of segment returns
            num_portfolios (int): Number of portfolios to generate
            
        Returns:
            pandas.DataFrame: Efficient frontier portfolios
        """
        # Generate a range of target returns
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        # Calculate optimal portfolios for each target return
        portfolios = []
        for target in target_returns:
            try:
                result = self.markowitz_optimization(
                    expected_returns, cov_matrix, risk_tolerance=None, target_return=target
                )
                
                # Add to portfolios list
                portfolio = {
                    'return': result['return'],
                    'risk': result['risk'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                }
                portfolios.append(portfolio)
            except Exception as e:
                print(f"Optimization failed for target return {target}: {e}")
        
        # Convert to DataFrame
        frontier_df = pd.DataFrame([
            {'return': p['return'], 'risk': p['risk'], 'sharpe_ratio': p['sharpe_ratio']}
            for p in portfolios
        ])
        
        # Add weights as separate columns
        for i, p in enumerate(portfolios):
            for segment, weight in p['weights'].items():
                frontier_df.loc[i, f'weight_{segment}'] = weight
        
        return frontier_df
    
    def visualize_efficient_frontier(self, frontier_df, historical_weights=None, cvar_weights=None):
        """
        Visualize the efficient frontier and optimal portfolios.
        
        Args:
            frontier_df (pandas.DataFrame): Efficient frontier portfolios
            historical_weights (pandas.Series): Historical portfolio weights
            cvar_weights (pandas.Series): CVaR optimized weights
            
        Returns:
            None
        """
        plt.figure(figsize=(12, 8))
        
        # Plot the efficient frontier
        plt.scatter(
            frontier_df['risk'], 
            frontier_df['return'], 
            c=frontier_df['sharpe_ratio'], 
            cmap='viridis', 
            s=30, 
            alpha=0.7
        )
        
        # Add colorbar for Sharpe ratio
        cbar = plt.colorbar()
        cbar.set_label('Sharpe Ratio')
        
        # Plot the maximum Sharpe ratio portfolio
        max_sharpe_idx = frontier_df['sharpe_ratio'].idxmax()
        max_sharpe_portfolio = frontier_df.loc[max_sharpe_idx]
        plt.scatter(
            max_sharpe_portfolio['risk'], 
            max_sharpe_portfolio['return'], 
            marker='*', 
            color='red', 
            s=300, 
            label='Maximum Sharpe Ratio'
        )
        
        # Plot the minimum variance portfolio
        min_var_idx = frontier_df['risk'].idxmin()
        min_var_portfolio = frontier_df.loc[min_var_idx]
        plt.scatter(
            min_var_portfolio['risk'], 
            min_var_portfolio['return'], 
            marker='o', 
            color='green', 
            s=200, 
            label='Minimum Variance'
        )
        
        # Plot historical portfolio if provided
        if historical_weights is not None:
            # Get expected returns from segment statistics
            segment_stats = self.calculate_segment_statistics(self.load_loan_data())
            expected_returns = segment_stats['expected_return']
            cov_matrix = self.calculate_covariance_matrix(self.load_loan_data())
            
            # Calculate historical portfolio metrics
            hist_return = (expected_returns * historical_weights).sum()
            hist_risk = np.sqrt(historical_weights.T @ cov_matrix @ historical_weights)
            
            plt.scatter(
                hist_risk, 
                hist_return, 
                marker='s', 
                color='blue', 
                s=200, 
                label='Historical Portfolio'
            )
        
        # Plot CVaR optimized portfolio if provided
        if cvar_weights is not None:
            # Get expected returns from segment statistics if not already calculated
            if 'expected_returns' not in locals():
                segment_stats = self.calculate_segment_statistics(self.load_loan_data())
                expected_returns = segment_stats['expected_return']
                cov_matrix = self.calculate_covariance_matrix(self.load_loan_data())
            
            # Calculate CVaR portfolio metrics
            cvar_return = (expected_returns * cvar_weights).sum()
            cvar_risk = np.sqrt(cvar_weights.T @ cov_matrix @ cvar_weights)
            
            plt.scatter(
                cvar_risk, 
                cvar_return, 
                marker='D', 
                color='purple', 
                s=200, 
                label='CVaR Optimized'
            )
        
        # Add labels and title
        plt.xlabel('Portfolio Risk (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier for DeFi Lending Portfolio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(self.results_dir, 'efficient_frontier.png'))
        plt.close()
    
    def visualize_portfolio_weights(self, weights_dict):
        """
        Visualize and compare portfolio weights across different optimization methods.
        
        Args:
            weights_dict (dict): Dictionary of portfolio weights from different methods
            
        Returns:
            None
        """
        # Extract weights from dictionary
        weights_df = pd.DataFrame(weights_dict)
        
        # Sort by the maximum Sharpe ratio weights
        if 'max_sharpe' in weights_df.columns:
            weights_df = weights_df.sort_values('max_sharpe', ascending=False)
        
        # Create a stacked bar chart
        plt.figure(figsize=(14, 10))
        weights_df.plot(kind='bar', stacked=False, figsize=(14, 10))
        
        # Add labels and title
        plt.xlabel('Portfolio Segment')
        plt.ylabel('Allocation Weight')
        plt.title('Portfolio Allocation Comparison Across Optimization Methods')
        plt.legend(title='Optimization Method')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.results_dir, 'portfolio_weights_comparison.png'))
        plt.close()
        
        # Create a heatmap for better visualization of segment allocations
        plt.figure(figsize=(12, 10))
        sns.heatmap(weights_df, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
        plt.title('Portfolio Allocation Heatmap')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.results_dir, 'portfolio_weights_heatmap.png'))
        plt.close()
    
    def calculate_reward_signal(self, current_weights, optimal_weights, current_sharpe, optimal_sharpe):
        """
        Calculate reward signal for reinforcement learning based on Sharpe ratio improvement.
        
        Args:
            current_weights (pandas.Series): Current portfolio weights
            optimal_weights (pandas.Series): Optimal portfolio weights
            current_sharpe (float): Current portfolio Sharpe ratio
            optimal_sharpe (float): Optimal portfolio Sharpe ratio
            
        Returns:
            float: Reward signal
        """
        # Calculate the distance between current and optimal weights
        weight_distance = np.sqrt(((current_weights - optimal_weights) ** 2).sum())
        
        # Calculate the Sharpe ratio improvement
        sharpe_improvement = optimal_sharpe - current_sharpe
        
        # Combine into a reward signal
        # Higher reward for getting closer to optimal weights and improving Sharpe ratio
        reward = sharpe_improvement - 0.5 * weight_distance
        
        return reward
    
    def run_portfolio_optimization(self):
        """
        Run the complete portfolio optimization pipeline.
        
        Returns:
            dict: Dictionary of optimization results
        """
        print("Starting Portfolio Optimization Pipeline...")
        
        # Load loan data
        print("\nLoading and preprocessing loan data...")
        loan_data = self.load_loan_data()
        
        # Calculate segment statistics
        print("\nCalculating segment statistics...")
        segment_stats = self.calculate_segment_statistics(loan_data)
        print("\nSegment Statistics:")
        print(segment_stats)
        
        # Save segment statistics to CSV
        segment_stats.to_csv(os.path.join(self.results_dir, 'segment_statistics.csv'))
        
        # Calculate covariance matrix
        print("\nCalculating covariance matrix...")
        cov_matrix = self.calculate_covariance_matrix(loan_data)
        
        # Extract expected returns
        expected_returns = segment_stats['expected_return']
        
        # Run Markowitz optimization for minimum variance portfolio
        print("\nRunning Markowitz optimization for minimum variance portfolio...")
        min_var_results = self.markowitz_optimization(
            expected_returns, cov_matrix, risk_tolerance=None, target_return=None
        )
        print(f"Minimum Variance Portfolio - Return: {min_var_results['return']:.4f}, Risk: {min_var_results['risk']:.4f}, Sharpe: {min_var_results['sharpe_ratio']:.4f}")
        
        # Run Markowitz optimization for maximum Sharpe ratio portfolio
        print("\nRunning Markowitz optimization for maximum Sharpe ratio portfolio...")
        # Use a range of risk tolerances and select the one with highest Sharpe ratio
        risk_tolerances = np.linspace(0.1, 5, 50)
        max_sharpe_results = None
        max_sharpe = -np.inf
        
        for rt in risk_tolerances:
            try:
                results = self.markowitz_optimization(
                    expected_returns, cov_matrix, risk_tolerance=rt, target_return=None
                )
                if results['sharpe_ratio'] > max_sharpe:
                    max_sharpe = results['sharpe_ratio']
                    max_sharpe_results = results
            except Exception as e:
                continue
        
        print(f"Maximum Sharpe Ratio Portfolio - Return: {max_sharpe_results['return']:.4f}, Risk: {max_sharpe_results['risk']:.4f}, Sharpe: {max_sharpe_results['sharpe_ratio']:.4f}")
        
        # Run CVaR optimization
        print("\nRunning CVaR optimization...")
        try:
            cvar_results = self.cvar_optimization(loan_data, alpha=0.05, target_return=None)
            print(f"CVaR Optimized Portfolio - Return: {cvar_results['return']:.4f}, Risk: {cvar_results['risk']:.4f}, CVaR: {cvar_results['cvar']:.4f}, Sharpe: {cvar_results['sharpe_ratio']:.4f}")
        except Exception as e:
            print(f"CVaR optimization failed: {e}")
            cvar_results = None
        
        # Generate efficient frontier
        print("\nGenerating efficient frontier...")
        frontier_df = self.efficient_frontier(expected_returns, cov_matrix, num_portfolios=30)
        
        # Visualize efficient frontier
        print("\nVisualizing efficient frontier...")
        self.visualize_efficient_frontier(
            frontier_df, 
            historical_weights=segment_stats['historical_weight'],
            cvar_weights=cvar_results['weights'] if cvar_results else None
        )
        
        # Visualize portfolio weights
        print("\nVisualizing portfolio weights...")
        weights_dict = {
            'historical': segment_stats['historical_weight'],
            'min_variance': min_var_results['weights'],
            'max_sharpe': max_sharpe_results['weights']
        }
        
        if cvar_results:
            weights_dict['cvar_optimized'] = cvar_results['weights']
        
        self.visualize_portfolio_weights(weights_dict)
        
        # Calculate reward signal for reinforcement learning
        print("\nCalculating reward signal for reinforcement learning...")
        reward = self.calculate_reward_signal(
            segment_stats['historical_weight'],
            max_sharpe_results['weights'],
            (segment_stats['expected_return'] * segment_stats['historical_weight']).sum() / segment_stats['risk'].mean(),
            max_sharpe_results['sharpe_ratio']
        )
        print(f"Reward Signal: {reward:.4f}")
        
        # Save optimization results
        print("\nSaving optimization results...")
        results = {
            'segment_stats': segment_stats,
            'min_variance': min_var_results,
            'max_sharpe': max_sharpe_results,
            'cvar': cvar_results,
            'efficient_frontier': frontier_df,
            'reward_signal': reward
        }
        
        # Save results to pickle file
        with open(os.path.join(self.results_dir, 'portfolio_optimization_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        print("\nPortfolio Optimization Pipeline completed successfully.")
        return results

if __name__ == "__main__":
    # Run the portfolio optimization pipeline
    optimizer = PortfolioOptimizer()
    results = optimizer.run_portfolio_optimization()
