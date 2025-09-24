import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}

    def load_sample_data(self, dataset_type='heart'):
        """Generate sample medical data for demonstration"""
        np.random.seed(42)

        if dataset_type == 'heart':
            # Heart Disease Dataset Features
            n_samples = 1000
            data = {
                'age': np.random.normal(54, 9, n_samples),
                'sex': np.random.choice([0, 1], n_samples),
                'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),
                'resting_bp': np.random.normal(132, 18, n_samples),
                'cholesterol': np.random.normal(246, 52, n_samples),
                'fasting_bs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
                'resting_ecg': np.random.choice([0, 1, 2], n_samples),
                'max_hr': np.random.normal(149, 23, n_samples),
                'exercise_angina': np.random.choice([0, 1], n_samples, p=[0.68, 0.32]),
                'st_depression': np.random.exponential(1, n_samples),
                'st_slope': np.random.choice([0, 1, 2], n_samples)
            }

            # Create target based on realistic correlations
            risk_score = (
                (data['age'] > 60) * 0.3 +
                data['sex'] * 0.2 +
                (data['chest_pain_type'] > 1) * 0.2 +
                (data['cholesterol'] > 240) * 0.15 +
                data['exercise_angina'] * 0.25 +
                (data['max_hr'] < 130) * 0.2 +
                np.random.normal(0, 0.1, n_samples)
            )
            data['heart_disease'] = (risk_score > 0.4).astype(int)

        elif dataset_type == 'diabetes':
            # Diabetes Dataset Features
            n_samples = 800
            data = {
                'pregnancies': np.random.poisson(3, n_samples),
                'glucose': np.random.normal(120, 32, n_samples),
                'blood_pressure': np.random.normal(69, 19, n_samples),
                'skin_thickness': np.random.normal(21, 16, n_samples),
                'insulin': np.random.exponential(80, n_samples),
                'bmi': np.random.normal(32, 8, n_samples),
                'diabetes_pedigree': np.random.exponential(0.5, n_samples),
                'age': np.random.normal(33, 12, n_samples)
            }

            # Create target based on realistic correlations
            risk_score = (
                (data['glucose'] > 140) * 0.4 +
                (data['bmi'] > 30) * 0.2 +
                (data['age'] > 45) * 0.15 +
                (data['diabetes_pedigree'] > 0.5) * 0.2 +
                np.random.normal(0, 0.15, n_samples)
            )
            data['diabetes'] = (risk_score > 0.3).astype(int)

        return pd.DataFrame(data)

    def explore_data(self, df, target_col):
        """Perform exploratory data analysis"""
        print("Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"\nTarget distribution:")
        print(df[target_col].value_counts(normalize=True))

        # Missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())

        # Basic statistics
        print(f"\nBasic Statistics:")
        print(df.describe())

        # Correlation with target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()[target_col].sort_values(key=abs, ascending=False)
        print(f"\nCorrelations with {target_col}:")
        print(correlations)

        return correlations

    def preprocess_data(self, df, target_col, test_size=0.2):
        """Preprocess the data for ML models"""
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Handle categorical variables if any
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }

        # Train models
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}

        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            results[name] = {
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score'],
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

        self.results = results
        return results

    def cross_validate_models(self, X_train, y_train, cv=5):
        """Perform cross-validation for all models"""
        cv_results = {}

        for name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean_cv_score': scores.mean(),
                'std_cv_score': scores.std()
            }

        return cv_results

    def plot_results(self, y_test):
        """Plot model comparison and ROC curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Model comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.results.keys())

        comparison_data = []
        for metric in metrics:
            metric_values = [self.results[model][metric] for model in model_names]
            comparison_data.append(metric_values)

        x = np.arange(len(model_names))
        width = 0.15

        for i, metric in enumerate(metrics):
            ax1.bar(x + i * width, comparison_data[i], width, label=metric)

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ROC Curves
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            ax2.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")

        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Confusion Matrix for best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        cm = confusion_matrix(y_test, self.results[best_model]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'Confusion Matrix - {best_model}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')

        # Feature importance (for tree-based models)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': range(len(rf_model.feature_importances_)),
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)

            ax4.barh(range(len(feature_importance)), feature_importance['importance'])
            ax4.set_yticks(range(len(feature_importance)))
            ax4.set_yticklabels([f'Feature_{i}' for i in feature_importance['feature']])
            ax4.set_xlabel('Importance')
            ax4.set_title('Feature Importance (Random Forest)')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_results_summary(self):
        """Print a summary of all results"""
        print("\n" + "="*60)
        print("DISEASE PREDICTION MODEL RESULTS SUMMARY")
        print("="*60)

        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  Accuracy:  {result['accuracy']:.3f}")
            print(f"  Precision: {result['precision']:.3f}")
            print(f"  Recall:    {result['recall']:.3f}")
            print(f"  F1-Score:  {result['f1_score']:.3f}")
            print(f"  ROC-AUC:   {result['roc_auc']:.3f}")

        # Best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        print(f"\nBest Model (by ROC-AUC): {best_model}")
        print(f"ROC-AUC Score: {self.results[best_model]['roc_auc']:.3f}")

    def predict_new_case(self, features, model_name='Random Forest'):
        """Predict disease probability for new case"""
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None

        # Ensure features is 2D array
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        model = self.models[model_name]
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return {
            'prediction': prediction,
            'probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }

# Example usage and demonstration
def main():
    # Initialize predictor
    predictor = DiseasePredictor()

    # Load sample heart disease data
    print("Loading Heart Disease Dataset...")
    heart_data = predictor.load_sample_data('heart')

    # Explore data
    correlations = predictor.explore_data(heart_data, 'heart_disease')

    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = predictor.preprocess_data(
        heart_data, 'heart_disease'
    )

    # Train models
    predictor.train_models(X_train, y_train)

    # Cross-validation
    cv_results = predictor.cross_validate_models(X_train, y_train)
    print("\nCross-Validation Results:")
    for model, scores in cv_results.items():
        print(f"{model}: {scores['mean_cv_score']:.3f} (+/- {scores['std_cv_score']:.3f})")

    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)

    # Print results summary
    predictor.print_results_summary()

    # Plot results
    predictor.plot_results(y_test)

    # Example prediction for new patient
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION FOR NEW PATIENT")
    print("="*60)

    # Sample patient data (age, sex, chest_pain_type, resting_bp, cholesterol,
    # fasting_bs, resting_ecg, max_hr, exercise_angina, st_depression, st_slope)
    new_patient = [65, 1, 2, 145, 280, 1, 1, 120, 1, 2.1, 1]

    prediction_result = predictor.predict_new_case(new_patient)
    if prediction_result:
        print(f"Patient Risk Assessment:")
        print(f"  Prediction: {'Positive' if prediction_result['prediction'] else 'Negative'}")
        print(f"  Probability: {prediction_result['probability']:.3f}")
        print(f"  Risk Level: {prediction_result['risk_level']}")

    return predictor

# Run the main function
if __name__ == "__main__":
    predictor = main()