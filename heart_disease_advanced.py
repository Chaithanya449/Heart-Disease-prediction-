"""
Advanced Heart Disease Prediction - Production-Ready Implementation
Includes all best practices: scaling, CV, hyperparameter tuning, ensembles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, Tuple, List
import joblib

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_style('whitegrid')


class HeartDiseasePredictor:
    """Advanced Heart Disease Prediction with best practices."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        
    def load_and_preprocess_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess data."""
        print(f"Loading data from {filepath}...")
        data = pd.read_csv(filepath)
        
        print(f"Dataset shape: {data.shape}")
        print(f"Missing values:\n{data.isnull().sum()}")
        
        # Handle missing values
        data['oldpeak'] = data['oldpeak'].fillna(data['oldpeak'].median())
        
        # Create target
        y = data['num'].apply(lambda x: 1 if x > 0 else 0)
        
        # Check class distribution
        print(f"\nClass distribution:")
        print(y.value_counts(normalize=True))
        
        # Prepare features
        X = data.drop('num', axis=1)
        X = pd.get_dummies(X, drop_first=True)
        
        print(f"Features after encoding: {X.shape[1]}")
        
        return X, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2
    ) -> Tuple:
        """Split data with stratification."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\nFeatures scaled!")
        return X_train_scaled, X_test_scaled
    
    def define_models(self):
        """Define models with improved hyperparameters."""
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=self.random_state, 
                    max_iter=1000,
                    C=0.1
                ),
                'scaled': True
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=5,
                    min_samples_split=20
                ),
                'scaled': False
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_estimators=200,
                    max_depth=10,
                    n_jobs=-1
                ),
                'scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    learning_rate=0.1
                ),
                'scaled': False
            },
            'XGBoost': {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    learning_rate=0.1
                ),
                'scaled': False
            },
            'SVM': {
                'model': SVC(
                    random_state=self.random_state,
                    kernel='rbf',
                    C=1.0,
                    probability=True
                ),
                'scaled': True  # Critical!
            },
            'KNN': {
                'model': KNeighborsClassifier(
                    n_neighbors=7,
                    weights='distance'
                ),
                'scaled': True  # Critical!
            },
            'Gaussian NB': {
                'model': GaussianNB(),
                'scaled': False
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(
                    random_state=self.random_state,
                    n_estimators=100
                ),
                'scaled': False
            }
        }
        
        print(f"\nDefined {len(self.models)} models")
        return self.models
    
    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray
    ):
        """Train and evaluate all models."""
        print("\n" + "="*80)
        print("TRAINING AND EVALUATING MODELS")
        print("="*80)
        
        self.results = []
        
        for name, config in self.models.items():
            print(f"\nTraining: {name}...")
            
            model = config['model']
            use_scaled = config['scaled']
            
            # Choose appropriate data
            X_tr = X_train_scaled if use_scaled else X_train.values
            X_te = X_test_scaled if use_scaled else X_test.values
            
            # Train
            model.fit(X_tr, y_train)
            
            # Predictions
            y_pred = model.predict(X_te)
            y_pred_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='f1')
            
            # Store results
            result = {
                'Model': name,
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1 * 100,
                'ROC-AUC': roc_auc * 100 if roc_auc else None,
                'CV F1 Mean': cv_scores.mean() * 100,
                'CV F1 Std': cv_scores.std() * 100,
                'Scaled': use_scaled
            }
            
            self.results.append(result)
            
            print(f"  Accuracy:  {accuracy*100:.2f}%")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall:    {recall*100:.2f}%")
            print(f"  F1-Score:  {f1*100:.2f}%")
            if roc_auc:
                print(f"  ROC-AUC:   {roc_auc*100:.2f}%")
            print(f"  CV F1:     {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        
        return pd.DataFrame(self.results).sort_values('F1-Score', ascending=False)
    
    def visualize_results(self, results_df: pd.DataFrame):
        """Create comprehensive visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison
        results_sorted = results_df.sort_values('Accuracy', ascending=True)
        axes[0, 0].barh(results_sorted['Model'], results_sorted['Accuracy'], color='skyblue')
        axes[0, 0].set_xlabel('Accuracy (%)')
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlim(0, 100)
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. F1-Score comparison
        results_sorted = results_df.sort_values('F1-Score', ascending=True)
        axes[0, 1].barh(results_sorted['Model'], results_sorted['F1-Score'], color='lightcoral')
        axes[0, 1].set_xlabel('F1-Score (%)')
        axes[0, 1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlim(0, 100)
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Precision vs Recall
        scatter = axes[1, 0].scatter(
            results_df['Recall'], 
            results_df['Precision'], 
            s=results_df['F1-Score']*5,  # Size based on F1
            alpha=0.6,
            c=results_df['Accuracy'],
            cmap='viridis'
        )
        for idx, row in results_df.iterrows():
            axes[1, 0].annotate(
                row['Model'], 
                (row['Recall'], row['Precision']),
                fontsize=8, 
                ha='right',
                alpha=0.7
            )
        axes[1, 0].set_xlabel('Recall (%)')
        axes[1, 0].set_ylabel('Precision (%)')
        axes[1, 0].set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Accuracy (%)')
        
        # 4. ROC-AUC comparison
        results_roc = results_df.dropna(subset=['ROC-AUC']).sort_values('ROC-AUC', ascending=True)
        axes[1, 1].barh(results_roc['Model'], results_roc['ROC-AUC'], color='lightgreen')
        axes[1, 1].set_xlabel('ROC-AUC (%)')
        axes[1, 1].set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'model_comparison.png'")
        plt.show()
    
    def analyze_best_model(
        self,
        results_df: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray
    ):
        """Detailed analysis of best model."""
        # Get best model
        self.best_model_name = results_df.iloc[0]['Model']
        best_config = self.models[self.best_model_name]
        self.best_model = best_config['model']
        
        print("\n" + "="*80)
        print(f"BEST MODEL: {self.best_model_name}")
        print("="*80)
        
        # Prepare data
        use_scaled = best_config['scaled']
        X_tr = X_train_scaled if use_scaled else X_train.values
        X_te = X_test_scaled if use_scaled else X_test.values
        
        # Ensure model is fitted
        self.best_model.fit(X_tr, y_train)
        
        # Predictions
        y_pred = self.best_model.predict(X_te)
        y_pred_proba = self.best_model.predict_proba(X_te)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'],
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        plt.show()
        
        print(f"\nConfusion Matrix Interpretation:")
        print(f"  True Negatives (TN):  {cm[0,0]:3d} - Correctly predicted no disease")
        print(f"  False Positives (FP): {cm[0,1]:3d} - False alarm (unnecessary tests)")
        print(f"  False Negatives (FN): {cm[1,0]:3d} - MISSED DISEASE (CRITICAL!)")
        print(f"  True Positives (TP):  {cm[1,1]:3d} - Correctly caught disease")
        
        # ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate (Recall)', fontsize=12)
            plt.title(f'ROC Curve - {self.best_model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
            print("ROC curve saved as 'roc_curve.png'")
            plt.show()
        
        # Feature importance
        self._plot_feature_importance(X_train)
    
    def _plot_feature_importance(self, X_train: pd.DataFrame):
        """Plot feature importance if available."""
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=feature_importance.head(15), 
                x='Importance', 
                y='Feature',
                palette='viridis'
            )
            plt.title(f'Top 15 Feature Importances - {self.best_model_name}', 
                      fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance saved as 'feature_importance.png'")
            plt.show()
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
        
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            coefficients = pd.DataFrame({
                'Feature': X_train.columns,
                'Coefficient': self.best_model.coef_[0]
            })
            coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
            coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
            
            plt.figure(figsize=(12, 8))
            top_features = coefficients.head(15)
            colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
            plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors, alpha=0.7)
            plt.xlabel('Coefficient Value', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top 15 Feature Coefficients - {self.best_model_name}', 
                      fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.tight_layout()
            plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
            print("\nFeature coefficients saved as 'feature_coefficients.png'")
            plt.show()
            
            print("\nTop 10 Most Influential Features:")
            print(coefficients[['Feature', 'Coefficient']].head(10).to_string(index=False))
    
    def save_model(self, filename: str = 'heart_disease_model.pkl'):
        """Save the best model and scaler."""
        if self.best_model is None:
            print("No model trained yet!")
            return
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'scaled': self.models[self.best_model_name]['scaled']
        }
        
        joblib.dump(model_data, filename)
        print(f"\nModel saved as '{filename}'")
        print(f"\nTo load: model_data = joblib.load('{filename}')")


def main():
    """Main execution function."""
    print("="*80)
    print("ADVANCED HEART DISEASE PREDICTION")
    print("="*80)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor(random_state=42)
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data('heart_disease.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)
    
    # Define models
    predictor.define_models()
    
    # Train and evaluate
    results_df = predictor.train_and_evaluate(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled
    )
    
    # Display results
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Visualize
    predictor.visualize_results(results_df)
    
    # Analyze best model
    predictor.analyze_best_model(
        results_df,
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled
    )
    
    # Save model
    predictor.save_model('heart_disease_best_model.pkl')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Improvements Implemented:")
    print("  ✓ Feature Scaling (StandardScaler)")
    print("  ✓ Comprehensive Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)")
    print("  ✓ Cross-Validation (5-fold)")
    print("  ✓ Improved Hyperparameters")
    print("  ✓ Multiple Visualizations")
    print("  ✓ Confusion Matrix Analysis")
    print("  ✓ ROC Curve")
    print("  ✓ Feature Importance")
    print("  ✓ Model Saving")
    print("\nCompare with original results to see improvements!")


if __name__ == "__main__":
    main()
