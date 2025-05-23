# Phase 3: Machine Learning and Risk Modeling
# This file implements synthetic data generation for credit risk modeling

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

class SyntheticDataGenerator:
    """
    Generates synthetic data for credit risk modeling in DeFi environments.
    
    This class creates realistic borrower profiles, loan applications, and 
    loan performance data that can be used to train machine learning models
    for credit risk assessment.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the synthetic data generator with optional random seed.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Define parameters for data generation
        self.num_borrowers = 1000
        self.loans_per_borrower_range = (1, 5)
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2025, 1, 1)
        
        # Define feature distributions
        self.feature_distributions = {
            # Borrower features
            'trust_score': {'type': 'beta', 'a': 2, 'b': 2},  # Beta distribution centered around 0.5
            'income': {'type': 'lognormal', 'mean': 10.5, 'sigma': 0.6},  # Log-normal for income (USD)
            'debt_to_income': {'type': 'beta', 'a': 2, 'b': 5},  # Beta distribution skewed towards lower values
            'age': {'type': 'normal', 'mean': 35, 'sigma': 10},  # Normal distribution for age
            'employment_length': {'type': 'lognormal', 'mean': 1.5, 'sigma': 0.8},  # Log-normal for employment length (years)
            'num_delinquencies': {'type': 'poisson', 'lam': 0.5},  # Poisson for count of delinquencies
            'num_credit_lines': {'type': 'poisson', 'lam': 5},  # Poisson for count of credit lines
            
            # Loan features
            'loan_amount': {'type': 'lognormal', 'mean': 8.5, 'sigma': 1.0},  # Log-normal for loan amount (USD)
            'loan_term': {'type': 'choice', 'options': [30, 60, 90, 180, 365]},  # Loan term in days
            'interest_rate': {'type': 'normal', 'mean': 0.08, 'sigma': 0.03},  # Normal distribution for interest rate
            'collateral_ratio': {'type': 'normal', 'mean': 1.5, 'sigma': 0.3},  # Normal distribution for collateral ratio
            
            # Market features
            'market_volatility': {'type': 'normal', 'mean': 0.2, 'sigma': 0.1},  # Normal distribution for market volatility
            'eth_price': {'type': 'lognormal', 'mean': 7.5, 'sigma': 0.3},  # Log-normal for ETH price
            'btc_price': {'type': 'lognormal', 'mean': 10.5, 'sigma': 0.3},  # Log-normal for BTC price
            'defi_tvl': {'type': 'lognormal', 'mean': 23, 'sigma': 0.5},  # Log-normal for DeFi TVL
        }
        
        # Define default probability model parameters
        # These coefficients determine how features affect default probability
        self.default_model_coefficients = {
            'intercept': -2.5,
            'trust_score': -2.0,  # Higher trust score -> lower default probability
            'income': -0.3,  # Higher income -> lower default probability
            'debt_to_income': 1.5,  # Higher DTI -> higher default probability
            'loan_amount': 0.2,  # Higher loan amount -> higher default probability
            'interest_rate': 1.0,  # Higher interest rate -> higher default probability
            'collateral_ratio': -1.0,  # Higher collateral -> lower default probability
            'market_volatility': 0.8,  # Higher volatility -> higher default probability
            'num_delinquencies': 0.5,  # More delinquencies -> higher default probability
        }
        
        # Paths for saving data
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _generate_borrower_features(self, num_borrowers):
        """
        Generate synthetic borrower profiles.
        
        Args:
            num_borrowers (int): Number of borrower profiles to generate
            
        Returns:
            pandas.DataFrame: DataFrame containing borrower features
        """
        borrowers = {}
        
        # Generate borrower features based on defined distributions
        borrowers['borrower_id'] = [f'B{i:04d}' for i in range(1, num_borrowers + 1)]
        
        # Generate trust scores from beta distribution
        borrowers['trust_score'] = np.random.beta(
            self.feature_distributions['trust_score']['a'],
            self.feature_distributions['trust_score']['b'],
            num_borrowers
        )
        
        # Generate income from log-normal distribution
        borrowers['income'] = np.random.lognormal(
            self.feature_distributions['income']['mean'],
            self.feature_distributions['income']['sigma'],
            num_borrowers
        )
        
        # Generate debt-to-income ratio from beta distribution
        borrowers['debt_to_income'] = np.random.beta(
            self.feature_distributions['debt_to_income']['a'],
            self.feature_distributions['debt_to_income']['b'],
            num_borrowers
        )
        
        # Generate age from normal distribution, clipped to reasonable values
        borrowers['age'] = np.clip(
            np.random.normal(
                self.feature_distributions['age']['mean'],
                self.feature_distributions['age']['sigma'],
                num_borrowers
            ),
            18, 80  # Minimum and maximum age
        ).astype(int)
        
        # Generate employment length from log-normal distribution
        borrowers['employment_length'] = np.random.lognormal(
            self.feature_distributions['employment_length']['mean'],
            self.feature_distributions['employment_length']['sigma'],
            num_borrowers
        )
        
        # Generate number of delinquencies from Poisson distribution
        borrowers['num_delinquencies'] = np.random.poisson(
            self.feature_distributions['num_delinquencies']['lam'],
            num_borrowers
        )
        
        # Generate number of credit lines from Poisson distribution
        borrowers['num_credit_lines'] = np.random.poisson(
            self.feature_distributions['num_credit_lines']['lam'],
            num_borrowers
        )
        
        # Create DataFrame from dictionary
        borrowers_df = pd.DataFrame(borrowers)
        
        return borrowers_df
    
    def _generate_loan_applications(self, borrowers_df):
        """
        Generate synthetic loan applications based on borrower profiles.
        
        Args:
            borrowers_df (pandas.DataFrame): DataFrame containing borrower features
            
        Returns:
            pandas.DataFrame: DataFrame containing loan application features
        """
        loan_applications = []
        
        # Generate random number of loan applications for each borrower
        for _, borrower in borrowers_df.iterrows():
            num_loans = random.randint(*self.loans_per_borrower_range)
            
            for _ in range(num_loans):
                # Generate loan application date
                days_offset = random.randint(0, (self.end_date - self.start_date).days)
                application_date = self.start_date + timedelta(days=days_offset)
                
                # Generate loan amount based on income (with some randomness)
                income_factor = borrower['income'] / 50000  # Normalize income
                base_loan_amount = np.random.lognormal(
                    self.feature_distributions['loan_amount']['mean'],
                    self.feature_distributions['loan_amount']['sigma']
                )
                loan_amount = base_loan_amount * (0.5 + 0.5 * income_factor)
                
                # Generate loan term
                loan_term = random.choice(self.feature_distributions['loan_term']['options'])
                
                # Generate interest rate (affected by trust score)
                base_interest_rate = np.random.normal(
                    self.feature_distributions['interest_rate']['mean'],
                    self.feature_distributions['interest_rate']['sigma']
                )
                # Lower trust score -> higher interest rate
                interest_rate = max(0.01, base_interest_rate * (1.5 - 0.5 * borrower['trust_score']))
                
                # Generate collateral ratio (affected by trust score and loan amount)
                base_collateral_ratio = np.random.normal(
                    self.feature_distributions['collateral_ratio']['mean'],
                    self.feature_distributions['collateral_ratio']['sigma']
                )
                # Lower trust score or higher loan amount -> higher collateral requirement
                trust_factor = 1.5 - 0.5 * borrower['trust_score']
                amount_factor = 1.0 + 0.1 * (loan_amount / 10000)
                collateral_ratio = max(1.0, base_collateral_ratio * trust_factor * amount_factor)
                
                # Generate market conditions at time of application
                market_volatility = max(0.05, np.random.normal(
                    self.feature_distributions['market_volatility']['mean'],
                    self.feature_distributions['market_volatility']['sigma']
                ))
                
                eth_price = np.random.lognormal(
                    self.feature_distributions['eth_price']['mean'],
                    self.feature_distributions['eth_price']['sigma']
                )
                
                btc_price = np.random.lognormal(
                    self.feature_distributions['btc_price']['mean'],
                    self.feature_distributions['btc_price']['sigma']
                )
                
                defi_tvl = np.random.lognormal(
                    self.feature_distributions['defi_tvl']['mean'],
                    self.feature_distributions['defi_tvl']['sigma']
                )
                
                # Create loan application record
                loan_application = {
                    'loan_id': f'L{len(loan_applications) + 1:06d}',
                    'borrower_id': borrower['borrower_id'],
                    'application_date': application_date,
                    'loan_amount': loan_amount,
                    'loan_term': loan_term,
                    'interest_rate': interest_rate,
                    'collateral_ratio': collateral_ratio,
                    'market_volatility': market_volatility,
                    'eth_price': eth_price,
                    'btc_price': btc_price,
                    'defi_tvl': defi_tvl,
                    'trust_score': borrower['trust_score'],
                    'income': borrower['income'],
                    'debt_to_income': borrower['debt_to_income'],
                    'age': borrower['age'],
                    'employment_length': borrower['employment_length'],
                    'num_delinquencies': borrower['num_delinquencies'],
                    'num_credit_lines': borrower['num_credit_lines']
                }
                
                loan_applications.append(loan_application)
        
        # Create DataFrame from list of dictionaries
        loan_applications_df = pd.DataFrame(loan_applications)
        
        return loan_applications_df
    
    def _calculate_default_probability(self, loan_application):
        """
        Calculate the probability of default for a loan application.
        
        Args:
            loan_application (pandas.Series): A single loan application record
            
        Returns:
            float: Probability of default (between 0 and 1)
        """
        # Calculate logit score using coefficients
        logit = self.default_model_coefficients['intercept']
        
        for feature, coefficient in self.default_model_coefficients.items():
            if feature != 'intercept' and feature in loan_application:
                logit += coefficient * loan_application[feature]
        
        # Add some random noise to make the model less deterministic
        logit += np.random.normal(0, 0.5)
        
        # Convert logit to probability using sigmoid function
        probability = 1 / (1 + np.exp(-logit))
        
        return probability
    
    def _generate_loan_performance(self, loan_applications_df):
        """
        Generate synthetic loan performance data based on loan applications.
        
        Args:
            loan_applications_df (pandas.DataFrame): DataFrame containing loan applications
            
        Returns:
            pandas.DataFrame: DataFrame containing loan performance data
        """
        loan_performance = []
        
        for _, loan in loan_applications_df.iterrows():
            # Calculate default probability
            default_probability = self._calculate_default_probability(loan)
            
            # Determine if loan defaulted
            defaulted = random.random() < default_probability
            
            # Calculate days to default or maturity
            if defaulted:
                # Defaulted loans typically default before maturity
                days_to_event = random.randint(1, loan['loan_term'])
                event_type = 'default'
            else:
                # Non-defaulted loans reach maturity
                days_to_event = loan['loan_term']
                event_type = 'maturity'
            
            # Calculate event date
            event_date = loan['application_date'] + timedelta(days=days_to_event)
            
            # Calculate loss given default (if applicable)
            if defaulted:
                # Loss given default is affected by collateral ratio
                # Higher collateral ratio means lower loss
                base_lgd = random.uniform(0.1, 0.9)
                lgd = base_lgd / loan['collateral_ratio']
                lgd = max(0, min(1, lgd))  # Ensure LGD is between 0 and 1
                loss_amount = loan['loan_amount'] * lgd
            else:
                lgd = 0
                loss_amount = 0
            
            # Create loan performance record
            performance = {
                'loan_id': loan['loan_id'],
                'borrower_id': loan['borrower_id'],
                'application_date': loan['application_date'],
                'event_date': event_date,
                'days_to_event': days_to_event,
                'event_type': event_type,
                'defaulted': defaulted,
                'loss_given_default': lgd,
                'loss_amount': loss_amount,
                'default_probability': default_probability
            }
            
            loan_performance.append(performance)
        
        # Create DataFrame from list of dictionaries
        loan_performance_df = pd.DataFrame(loan_performance)
        
        return loan_performance_df
    
    def generate_data(self):
        """
        Generate complete synthetic dataset for credit risk modeling.
        
        Returns:
            tuple: (borrowers_df, loan_applications_df, loan_performance_df)
        """
        print("Generating synthetic borrower profiles...")
        borrowers_df = self._generate_borrower_features(self.num_borrowers)
        
        print("Generating synthetic loan applications...")
        loan_applications_df = self._generate_loan_applications(borrowers_df)
        
        print("Generating synthetic loan performance data...")
        loan_performance_df = self._generate_loan_performance(loan_applications_df)
        
        # Merge loan applications and performance for a complete dataset
        complete_df = pd.merge(
            loan_applications_df,
            loan_performance_df[['loan_id', 'event_date', 'event_type', 'defaulted', 'loss_given_default', 'loss_amount', 'default_probability']],
            on='loan_id'
        )
        
        # Ensure defaulted is numeric (convert boolean to int)
        complete_df['defaulted'] = complete_df['defaulted'].astype(int)
        
        # Save datasets to CSV files
        borrowers_df.to_csv(os.path.join(self.data_dir, 'borrowers.csv'), index=False)
        loan_applications_df.to_csv(os.path.join(self.data_dir, 'loan_applications.csv'), index=False)
        loan_performance_df.to_csv(os.path.join(self.data_dir, 'loan_performance.csv'), index=False)
        complete_df.to_csv(os.path.join(self.data_dir, 'complete_loan_data.csv'), index=False)
        
        print(f"Data generation complete. Files saved to {self.data_dir}")
        
        return borrowers_df, loan_applications_df, loan_performance_df, complete_df
    
    def analyze_and_visualize(self, complete_df):
        """
        Analyze and visualize the synthetic data.
        
        Args:
            complete_df (pandas.DataFrame): Complete loan data
            
        Returns:
            None
        """
        # Create directory for visualizations
        viz_dir = os.path.join(self.data_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Calculate default rate
        default_rate = complete_df['defaulted'].mean()
        print(f"Overall default rate: {default_rate:.2%}")
        
        # Analyze default rate by trust score buckets
        trust_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        trust_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        complete_df['trust_bucket'] = pd.cut(complete_df['trust_score'], bins=trust_bins, labels=trust_labels)
        trust_default_rates = complete_df.groupby('trust_bucket')['defaulted'].mean()
        
        plt.figure(figsize=(10, 6))
        trust_default_rates.plot(kind='bar', color='skyblue')
        plt.title('Default Rate by Trust Score')
        plt.xlabel('Trust Score Bucket')
        plt.ylabel('Default Rate')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'default_rate_by_trust.png'))
        
        # Analyze default rate by loan amount buckets
        amount_bins = [0, 5000, 10000, 20000, 50000, 100000, float('inf')]
        amount_labels = ['0-5K', '5K-10K', '10K-20K', '20K-50K', '50K-100K', '100K+']
        complete_df['amount_bucket'] = pd.cut(complete_df['loan_amount'], bins=amount_bins, labels=amount_labels)
        amount_default_rates = complete_df.groupby('amount_bucket')['defaulted'].mean()
        
        plt.figure(figsize=(10, 6))
        amount_default_rates.plot(kind='bar', color='lightgreen')
        plt.title('Default Rate by Loan Amount')
        plt.xlabel('Loan Amount Bucket')
        plt.ylabel('Default Rate')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'default_rate_by_amount.png'))
        
        # Analyze correlation between features and default
        # Select only numeric columns for correlation analysis
        numeric_cols = complete_df.select_dtypes(include=['number']).columns
        # Ensure 'defaulted' is in the numeric columns
        if 'defaulted' in numeric_cols:
            correlation_with_default = complete_df[numeric_cols].corr()['defaulted'].sort_values(ascending=False)
            print("\nCorrelation with default:")
            print(correlation_with_default)
            
            plt.figure(figsize=(12, 8))
            correlation_with_default.drop('defaulted').plot(kind='bar', color='salmon')
            plt.title('Feature Correlation with Default')
            plt.xlabel('Features')
            plt.ylabel('Correlation Coefficient')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'feature_correlation.png'))
        else:
            print("\nWarning: 'defaulted' column not found in numeric columns. Skipping correlation analysis.")
            print(f"Available numeric columns: {list(numeric_cols)}")
        
        # Create scatter plot of trust score vs default probability
        plt.figure(figsize=(10, 6))
        plt.scatter(complete_df['trust_score'], complete_df['default_probability'], 
                   alpha=0.5, c=complete_df['defaulted'], cmap='coolwarm')
        plt.title('Trust Score vs Default Probability')
        plt.xlabel('Trust Score')
        plt.ylabel('Default Probability')
        plt.colorbar(label='Defaulted')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'trust_vs_default_prob.png'))
        
        # Create histogram of default probabilities
        plt.figure(figsize=(10, 6))
        plt.hist(complete_df['default_probability'], bins=30, alpha=0.7, color='purple')
        plt.title('Distribution of Default Probabilities')
        plt.xlabel('Default Probability')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'default_prob_distribution.png'))
        
        print(f"Visualizations saved to {viz_dir}")

if __name__ == "__main__":
    # Generate synthetic data
    data_generator = SyntheticDataGenerator(seed=42)
    borrowers_df, loan_applications_df, loan_performance_df, complete_df = data_generator.generate_data()
    
    # Analyze and visualize the data
    data_generator.analyze_and_visualize(complete_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of borrowers: {len(borrowers_df)}")
    print(f"Number of loan applications: {len(loan_applications_df)}")
    print(f"Number of defaulted loans: {loan_performance_df['defaulted'].sum()}")
    print(f"Default rate: {loan_performance_df['defaulted'].mean():.2%}")
