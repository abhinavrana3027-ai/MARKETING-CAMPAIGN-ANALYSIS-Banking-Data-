import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader

class ConversionPredictor:
    """Machine learning model for predicting customer conversion."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def preprocess_data(self, test_size=0.2):
        """Prepare data for modeling."""
        df = self.df.copy()
        
        # Target variable
        y = (df['y'] == 'yes').astype(int)
        
        # Drop irrelevant columns
        X = df.drop(['y', 'duration'], axis=1)
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.scalers['main'] = scaler
        
        print(f"\u2713 Data preprocessing complete")
        print(f"  Training set: {self.X_train.shape}")
        print(f"  Test set: {self.X_test.shape}")
        print(f"  Conversion rate in training: {self.y_train.mean()*100:.2f}%")
        print(f"  Conversion rate in test: {self.y_test.mean()*100:.2f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple classification models."""
        print("\n=== Model Training ===")
        
        # Logistic Regression
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        lr_pred = lr.predict(self.X_test)
        lr_prob = lr.predict_proba(self.X_test)[:, 1]
        lr_score = roc_auc_score(self.y_test, lr_prob)
        self.models['Logistic Regression'] = {
            'model': lr, 'predictions': lr_pred, 'proba': lr_prob, 'auc': lr_score
        }
        print(f"  ROC-AUC Score: {lr_score:.4f}")
        
        # Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        rf_pred = rf.predict(self.X_test)
        rf_prob = rf.predict_proba(self.X_test)[:, 1]
        rf_score = roc_auc_score(self.y_test, rf_prob)
        self.models['Random Forest'] = {
            'model': rf, 'predictions': rf_pred, 'proba': rf_prob, 'auc': rf_score
        }
        print(f"  ROC-AUC Score: {rf_score:.4f}")
        
        # Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        gb_pred = gb.predict(self.X_test)
        gb_prob = gb.predict_proba(self.X_test)[:, 1]
        gb_score = roc_auc_score(self.y_test, gb_prob)
        self.models['Gradient Boosting'] = {
            'model': gb, 'predictions': gb_pred, 'proba': gb_prob, 'auc': gb_score
        }
        print(f"  ROC-AUC Score: {gb_score:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all models."""
        print("\n=== Model Evaluation ===")
        
        results = []
        for name, model_dict in self.models.items():
            predictions = model_dict['predictions']
            y_true = self.y_test
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions)
            recall = recall_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)
            auc = model_dict['auc']
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': auc
            })
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {auc:.4f}")
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self):
        """Extract feature importance from tree-based models."""
        print("\n=== Feature Importance ===")
        
        rf_model = self.models['Random Forest']['model']
        feature_names = self.df.drop(['y', 'duration'], axis=1).columns
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return importance_df


if __name__ == '__main__':
    # Load data
    loader = DataLoader('Bank.csv')
    df = loader.load_data()
    
    if df is not None:
        # Initialize predictor
        predictor = ConversionPredictor(df)
        
        # Preprocess data
        predictor.preprocess_data()
        
        # Train models
        predictor.train_models()
        
        # Evaluate models
        results = predictor.evaluate_models()
        
        # Get feature importance
        importance_df = predictor.get_feature_importance()
        
        print("\n✓ Model training and evaluation complete!")
