# Phase 3: Machine Learning and Risk Modeling - Credit Risk Prediction Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import xgboost as xgb
import shap

class CreditRiskModel:
    """
    Credit Risk Model for predicting loan defaults in DeFi environments.
    
    This class implements an XGBoost-based model for predicting the probability
    of default (PD) for loans, with SHAP-based explainability.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the credit risk model.
        
        Args:
            data_dir (str): Directory containing the synthetic data files
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        else:
            self.data_dir = data_dir
            
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml_models', 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.results_dir = os.path.join(self.data_dir, 'model_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None
        
    def load_data(self):
        """
        Load and prepare the synthetic data for modeling.
        
        Returns:
            tuple: X (features), y (target), feature_names
        """
        # Load the complete loan data
        data_file = os.path.join(self.data_dir, 'complete_loan_data.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}. Run synthetic_data_generator.py first.")
        
        df = pd.read_csv(data_file)
        print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Select features for modeling
        feature_columns = [
            'trust_score', 'income', 'debt_to_income', 'age', 'employment_length',
            'num_delinquencies', 'num_credit_lines', 'loan_amount', 'loan_term',
            'interest_rate', 'collateral_ratio', 'market_volatility', 'eth_price',
            'btc_price', 'defi_tvl'
        ]
        
        # Ensure all required columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        # Prepare features and target
        X = df[feature_columns]
        y = df['defaulted'].astype(int)
        
        return X, y, feature_columns
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """
        Preprocess the data for modeling.
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): Target variable
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, param_grid=None, cv=5):
        """
        Train an XGBoost model for credit risk prediction.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            param_grid (dict): Parameter grid for hyperparameter tuning
            cv (int): Number of cross-validation folds
            
        Returns:
            xgboost.XGBClassifier: Trained model
        """
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
                'min_child_weight': [1, 3]
            }
        
        # Initialize base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
        
        # Perform grid search with cross-validation
        print("Starting hyperparameter tuning with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1
        )
        
        # Handle class imbalance with scale_pos_weight
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train the model
        grid_search.fit(
            X_train, 
            y_train,
            sample_weight=np.where(y_train == 1, pos_weight, 1)
        )
        
        # Get the best model
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Save feature names for later use
        self.feature_names = X_train.columns.tolist()
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series): Test target
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        
        # Create ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'))
        
        # Create precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.results_dir, 'precision_recall_curve.png'))
        
        return metrics
    
    def explain_predictions(self, X_test, n_samples=100):
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            X_test (pandas.DataFrame): Test features
            n_samples (int): Number of samples to use for SHAP explanations
            
        Returns:
            shap.Explainer: SHAP explainer object
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
        
        # Create SHAP explainer
        print("Generating SHAP explanations...")
        self.explainer = shap.Explainer(self.model)
        
        # Use a subset of test data for explanations if dataset is large
        if len(X_test) > n_samples:
            X_sample = X_test.sample(n_samples, random_state=42)
        else:
            X_sample = X_test
        
        # Calculate SHAP values
        shap_values = self.explainer(X_sample)
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'shap_summary.png'))
        
        # Create bar plot of mean absolute SHAP values
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
        plt.title('Mean Impact on Model Output Magnitude')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'shap_importance.png'))
        
        # Create dependence plots for top features
        mean_abs_shap = np.abs(shap_values.values).mean(0)
        top_indices = np.argsort(mean_abs_shap)[-3:]  # Top 3 features
        
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                idx, 
                shap_values.values, 
                X_sample, 
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f'SHAP Dependence Plot for {feature_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'shap_dependence_{feature_name}.png'))
        
        return self.explainer
    
    def save_model(self, filename='credit_risk_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filename (str): Filename for the saved model
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
        
        # Create full path
        model_path = os.path.join(self.models_dir, filename)
        
        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename='credit_risk_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filename (str): Filename of the saved model
            
        Returns:
            xgboost.XGBClassifier: Loaded model
        """
        # Create full path
        model_path = os.path.join(self.models_dir, filename)
        
        # Load model and scaler
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.feature_names = saved_data['feature_names']
        
        print(f"Model loaded from {model_path}")
        return self.model
    
    def predict_default_probability(self, loan_features):
        """
        Predict the probability of default for a new loan.
        
        Args:
            loan_features (dict or pandas.DataFrame): Features of the loan
            
        Returns:
            float: Probability of default
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
        
        # Convert dict to DataFrame if necessary
        if isinstance(loan_features, dict):
            loan_features = pd.DataFrame([loan_features])
        
        # Ensure all required features are present
        missing_features = [f for f in self.feature_names if f not in loan_features.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features according to the model's expectations
        loan_features = loan_features[self.feature_names]
        
        # Scale features
        loan_features_scaled = self.scaler.transform(loan_features)
        
        # Predict probability
        default_probability = self.model.predict_proba(loan_features_scaled)[:, 1][0]
        
        return default_probability
    
    def explain_loan_prediction(self, loan_features):
        """
        Explain the prediction for a specific loan using SHAP.
        
        Args:
            loan_features (dict or pandas.DataFrame): Features of the loan
            
        Returns:
            dict: SHAP values for the loan features
        """
        if self.model is None or self.explainer is None:
            raise ValueError("Model or explainer not initialized. Call train_model and explain_predictions first.")
        
        # Convert dict to DataFrame if necessary
        if isinstance(loan_features, dict):
            loan_features = pd.DataFrame([loan_features])
        
        # Ensure all required features are present
        missing_features = [f for f in self.feature_names if f not in loan_features.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features according to the model's expectations
        loan_features = loan_features[self.feature_names]
        
        # Calculate SHAP values
        shap_values = self.explainer(loan_features)
        
        # Create force plot
        plt.figure(figsize=(12, 3))
        shap.force_plot(
            shap_values.base_values[0], 
            shap_values.values[0], 
            loan_features.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.title('SHAP Force Plot for Loan Default Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'loan_shap_force_plot.png'))
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=loan_features.iloc[0].values,
            feature_names=self.feature_names
        ), show=False)
        plt.title('SHAP Waterfall Plot for Loan Default Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'loan_shap_waterfall_plot.png'))
        
        # Return SHAP values as dictionary
        shap_dict = dict(zip(self.feature_names, shap_values.values[0]))
        return shap_dict

def run_credit_risk_modeling():
    """
    Run the complete credit risk modeling pipeline.
    """
    print("Starting Credit Risk Modeling Pipeline...")
    
    # Initialize model
    model = CreditRiskModel()
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X, y, feature_names = model.load_data()
    X_train, X_test, y_train, y_test = model.preprocess_data(X, y)
    
    # Train model with simplified parameter grid for demonstration
    print("\nTraining XGBoost model...")
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }
    model.train_model(X_train, y_train, param_grid=param_grid, cv=3)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = model.evaluate_model(X_test, y_test)
    
    # Generate explanations
    print("\nGenerating SHAP explanations...")
    model.explain_predictions(X_test, n_samples=100)
    
    # Save model
    print("\nSaving model...")
    model_path = model.save_model()
    
    # Example prediction for a new loan
    print("\nExample prediction for a new loan...")
    example_loan = {
        'trust_score': 0.7,
        'income': 75000,
        'debt_to_income': 0.3,
        'age': 35,
        'employment_length': 5,
        'num_delinquencies': 0,
        'num_credit_lines': 3,
        'loan_amount': 10000,
        'loan_term': 90,
        'interest_rate': 0.05,
        'collateral_ratio': 1.5,
        'market_volatility': 0.2,
        'eth_price': 2000,
        'btc_price': 30000,
        'defi_tvl': 10000000000
    }
    
    default_probability = model.predict_default_probability(example_loan)
    print(f"Predicted default probability: {default_probability:.4f}")
    
    # Explain the prediction
    print("\nExplaining the prediction...")
    shap_values = model.explain_loan_prediction(example_loan)
    
    # Print top factors
    print("\nTop factors influencing the prediction:")
    sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, value in sorted_shap[:5]:
        direction = "increasing" if value > 0 else "decreasing"
        print(f"{feature}: {value:.4f} ({direction} default risk)")
    
    print("\nCredit Risk Modeling Pipeline completed successfully.")
    return model, metrics

if __name__ == "__main__":
    # Run the credit risk modeling pipeline
    model, metrics = run_credit_risk_modeling()
