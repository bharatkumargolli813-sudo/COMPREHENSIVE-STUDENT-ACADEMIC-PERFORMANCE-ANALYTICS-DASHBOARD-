"""
Predictive Model Module - Implements ML-based student performance prediction
using regression, classification, and clustering algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class StudentPerformancePredictor:
    """
    ML-based Student Performance Prediction System.
    Uses multiple algorithms to predict student outcomes.
    """
    
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, students):
        """
        Extract and prepare features from student data for ML models.
        Features include attendance, internal marks, and derived metrics.
        """
        features = []
        targets = []
        risk_labels = []
        
        for student in students:
            # Feature extraction
            attendance_vals = list(student['attendance'].values())
            internal_vals = []
            for subj in student['subjects']:
                if subj in student['internal_marks']:
                    internal_vals.extend(student['internal_marks'][subj])
            exam_vals = list(student['exam_marks'].values())
            
            # Aggregate features
            avg_attendance = np.mean(attendance_vals)
            min_attendance = np.min(attendance_vals)
            std_attendance = np.std(attendance_vals)
            avg_internal = np.mean(internal_vals) if internal_vals else 0
            min_internal = np.min(internal_vals) if internal_vals else 0
            std_internal = np.std(internal_vals) if internal_vals else 0
            avg_exam = np.mean(exam_vals) if exam_vals else 0
            cgpa = student['cgpa']
            num_subjects = len(student['subjects'])
            num_low_attendance = sum(1 for a in attendance_vals if a < 70)
            num_failing_internal = sum(1 for m in internal_vals if m < 40)
            attendance_trend = attendance_vals[-1] - attendance_vals[0] if len(attendance_vals) > 1 else 0
            
            feature_vector = [
                avg_attendance, min_attendance, std_attendance,
                avg_internal, min_internal, std_internal,
                avg_exam, cgpa, num_subjects,
                num_low_attendance, num_failing_internal, attendance_trend
            ]
            features.append(feature_vector)
            targets.append(avg_exam)
            
            # Risk classification label
            if avg_exam < 40 or avg_attendance < 60:
                risk_labels.append('High Risk')
            elif avg_exam < 50 or avg_attendance < 70:
                risk_labels.append('Moderate Risk')
            elif avg_exam < 60 or avg_attendance < 75:
                risk_labels.append('Low Risk')
            else:
                risk_labels.append('No Risk')
        
        return np.array(features), np.array(targets), np.array(risk_labels)
    
    def train_regression_model(self, X, y):
        """Train Random Forest Regression model for score prediction."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        self.regression_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.regression_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.regression_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'rmse': round(rmse, 4),
            'r2_score': round(r2, 4),
            'train_score': round(self.regression_model.score(X_train, y_train), 4),
            'test_score': round(self.regression_model.score(X_test, y_test), 4)
        }
        
        # Also compute linear regression for comparison
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        lr_r2 = r2_score(y_test, lr_pred)
        results['linear_rmse'] = round(lr_rmse, 4)
        results['linear_r2'] = round(lr_r2, 4)
        
        return results
    
    def train_classification_model(self, X, labels):
        """Train Random Forest Classifier for risk prediction."""
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        self.classification_model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        )
        self.classification_model.fit(X_train, y_train)
        
        y_pred = self.classification_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': round(accuracy, 4),
            'train_accuracy': round(self.classification_model.score(X_train, y_train), 4),
            'test_accuracy': round(self.classification_model.score(X_test, y_test), 4),
            'class_distribution': dict(zip(*np.unique(y_encoded, return_counts=True))),
            'feature_importance': self.classification_model.feature_importances_.tolist()
        }
        
        return results
    
    def train_clustering_model(self, X, n_clusters=4):
        """Train KMeans clustering for student segmentation."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.clustering_model.fit_predict(X_scaled)
        
        # Compute cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            mask = clusters == i
            cluster_stats[f'Cluster_{i}'] = {
                'count': int(np.sum(mask)),
                'avg_attendance': round(np.mean(X[mask, 0]), 2),
                'avg_internal_marks': round(np.mean(X[mask, 3]), 2),
                'avg_exam_score': round(np.mean(X[mask, 6]), 2),
                'avg_cgpa': round(np.mean(X[mask, 7]), 2)
            }
        
        inertia = self.clustering_model.inertia_
        
        return clusters, cluster_stats, inertia
    
    def predict_student(self, student_data):
        """Predict performance for a single student."""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        # Prepare features for single student
        attendance_vals = list(student_data['attendance'].values())
        internal_vals = []
        for subj in student_data['subjects']:
            if subj in student_data['internal_marks']:
                internal_vals.extend(student_data['internal_marks'][subj])
        exam_vals = list(student_data['exam_marks'].values())
        
        avg_attendance = np.mean(attendance_vals)
        min_attendance = np.min(attendance_vals)
        std_attendance = np.std(attendance_vals)
        avg_internal = np.mean(internal_vals) if internal_vals else 0
        min_internal = np.min(internal_vals) if internal_vals else 0
        std_internal = np.std(internal_vals) if internal_vals else 0
        avg_exam = np.mean(exam_vals) if exam_vals else 0
        cgpa = student_data['cgpa']
        num_subjects = len(student_data['subjects'])
        num_low_attendance = sum(1 for a in attendance_vals if a < 70)
        num_failing_internal = sum(1 for m in internal_vals if m < 40)
        attendance_trend = attendance_vals[-1] - attendance_vals[0] if len(attendance_vals) > 1 else 0
        
        X = np.array([[
            avg_attendance, min_attendance, std_attendance,
            avg_internal, min_internal, std_internal,
            avg_exam, cgpa, num_subjects,
            num_low_attendance, num_failing_internal, attendance_trend
        ]])
        
        # Predictions
        predicted_score = self.regression_model.predict(X)[0]
        risk_pred = self.classification_model.predict(X)[0]
        
        # Cluster assignment
        X_scaled = self.scaler.transform(X)
        cluster = self.clustering_model.predict(X_scaled)[0]
        
        return {
            'predicted_score': round(predicted_score, 2),
            'risk_level': ['No Risk', 'Low Risk', 'Moderate Risk', 'High Risk'][risk_pred],
            'cluster': int(cluster),
            'confidence': round(self.classification_model.predict_proba(X)[0].max(), 4)
        }
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance rankings."""
        if self.regression_model is None:
            return {}
        
        importance = self.regression_model.feature_importances_
        
        if feature_names is None:
            feature_names = [
                'Avg Attendance', 'Min Attendance', 'Std Attendance',
                'Avg Internal Marks', 'Min Internal Marks', 'Std Internal Marks',
                'Avg Exam Score', 'CGPA', 'Num Subjects',
                'Low Attendance Subjects', 'Failing Internal Tests', 'Attendance Trend'
            ]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df.to_dict('records')


def generate_correlation_matrix(students):
    """Generate correlation matrix between different academic metrics."""
    data = []
    for s in students:
        avg_att = np.mean(list(s['attendance'].values()))
        avg_int = np.mean([m for marks in s['internal_marks'].values() for m in marks])
        avg_exam = np.mean(list(s['exam_marks'].values()))
        cgpa = s['cgpa']
        num_low_att = sum(1 for a in s['attendance'].values() if a < 70)
        
        data.append({
            'attendance': avg_att,
            'internal_marks': avg_int,
            'exam_marks': avg_exam,
            'cgpa': cgpa,
            'low_attendance_subjects': num_low_att
        })
    
    df = pd.DataFrame(data)
    correlation = df.corr()
    return correlation


def generate_regression_data(students):
    """Generate data for regression visualization."""
    x_attendance = []
    y_scores = []
    x_internal = []
    
    for s in students:
        avg_att = np.mean(list(s['attendance'].values()))
        avg_exam = np.mean(list(s['exam_marks'].values()))
        avg_int = np.mean([m for marks in s['internal_marks'].values() for m in marks])
        
        x_attendance.append(avg_att)
        y_scores.append(avg_exam)
        x_internal.append(avg_int)
    
    return {
        'attendance_vs_score': {
            'x': x_attendance,
            'y': y_scores
        },
        'internal_vs_score': {
            'x': x_internal,
            'y': y_scores
        }
    }


if __name__ == '__main__':
    import json
    
    # Load data
    with open('/home/ubuntu/project/academic_analytics/students_data.json', 'r') as f:
        students = json.load(f)
    
    print("=" * 60)
    print("Student Performance Predictive Model Training")
    print("=" * 60)
    
    predictor = StudentPerformancePredictor()
    
    # Prepare features
    print("\n[1/4] Preparing features...")
    X, y, labels = predictor.prepare_features(students)
    print(f"  Features shape: {X.shape}")
    print(f"  Labels: {len(np.unique(labels))} classes")
    
    # Train regression
    print("\n[2/4] Training Regression Model...")
    reg_results = predictor.train_regression_model(X, y)
    print(f"  Random Forest RMSE: {reg_results['rmse']}")
    print(f"  Random Forest R2: {reg_results['r2_score']}")
    print(f"  Linear Regression RMSE: {reg_results['linear_rmse']}")
    print(f"  Linear Regression R2: {reg_results['linear_r2']}")
    
    # Train classification
    print("\n[3/4] Training Classification Model...")
    cls_results = predictor.train_classification_model(X, labels)
    print(f"  Accuracy: {cls_results['accuracy']}")
    print(f"  Class Distribution: {cls_results['class_distribution']}")
    
    # Train clustering
    print("\n[4/4] Training Clustering Model...")
    clusters, cluster_stats, inertia = predictor.train_clustering_model(X)
    predictor.is_trained = True
    print(f"  Clusters: {len(cluster_stats)}")
    print(f"  Inertia: {round(inertia, 2)}")
    for name, stats in cluster_stats.items():
        print(f"  {name}: {stats['count']} students, Avg CGPA: {stats['avg_cgpa']}")
    
    # Feature importance
    print("\nFeature Importance:")
    importance = predictor.get_feature_importance()
    for item in importance:
        print(f"  {item['Feature']}: {round(item['Importance'], 4)}")
    
    # Correlation matrix
    print("\nCorrelation Matrix:")
    corr = generate_correlation_matrix(students)
    print(corr.round(3))
    
    # Sample predictions
    print("\nSample Student Predictions:")
    for s in students[:5]:
        pred = predictor.predict_student(s)
        print(f"  {s['name']}: Predicted Score={pred['predicted_score']}, "
              f"Risk={pred['risk_level']}, Cluster={pred['cluster']}")
    
    print("\nModel training complete!")
