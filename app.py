"""
Main Application Module - Academic Performance Analytics Dashboard
Integrates data generation, predictive modeling, and visualization.
"""

import json
import numpy as np
import pandas as pd
from data_generator import generate_students, generate_subject_analytics, generate_branch_comparison, predict_student_performance
from predictive_model import StudentPerformancePredictor, generate_correlation_matrix, generate_regression_data
from visualization import (
    fig1_overall_dashboard, fig2_attendance_analysis, fig3_internal_marks_analysis,
    fig4_predictive_insights, fig5_subject_wise_performance, fig6_attendance_trends,
    fig7_ml_results, fig8_branch_comparison, fig9_system_architecture,
    fig10_correlation_analysis, fig11_risk_prediction_dashboard, fig12_alert_notification_system
)


def main():
    print("=" * 70)
    print("COMPREHENSIVE STUDENT ACADEMIC PERFORMANCE ANALYTICS DASHBOARD")
    print("=" * 70)
    
    # Step 1: Generate Data
    print("\n[1/6] Generating sample student data...")
    students = generate_students()
    subject_analytics = generate_subject_analytics(students)
    branch_comparison = generate_branch_comparison(students)
    
    with open('students_data.json', 'w') as f:
        json.dump(students, f, indent=2)
    with open('analytics_data.json', 'w') as f:
        json.dump({'subject_analytics': subject_analytics, 'branch_comparison': branch_comparison}, f, indent=2)
    
    print(f"  Generated {len(students)} student profiles")
    print(f"  Analyzed {len(subject_analytics)} subjects")
    print(f"  Compared {len(branch_comparison)} branches")
    
    # Step 2: Predictive Analysis
    print("\n[2/6] Running predictive analysis...")
    predictor = StudentPerformancePredictor()
    X, y, labels = predictor.prepare_features(students)
    print(f"  Features prepared: {X.shape[1]} features for {X.shape[0]} students")
    
    # Step 3: Train ML Models
    print("\n[3/6] Training Machine Learning models...")
    reg_results = predictor.train_regression_model(X, y)
    print(f"  Regression Model: RMSE={reg_results['rmse']}, R²={reg_results['r2_score']}")
    
    cls_results = predictor.train_classification_model(X, labels)
    print(f"  Classification Model: Accuracy={cls_results['accuracy']}")
    
    clusters, cluster_stats, inertia = predictor.train_clustering_model(X)
    predictor.is_trained = True
    print(f"  Clustering: {len(cluster_stats)} clusters, Inertia={inertia:.2f}")
    
    # Step 4: Generate Predictions
    print("\n[4/6] Generating student predictions...")
    predictions = []
    for s in students:
        pred = predictor.predict_student(s)
        pred['name'] = s['name']
        pred['branch'] = s['branch']
        pred['actual_score'] = np.mean(list(s['exam_marks'].values()))
        predictions.append(pred)
    
    print(f"  Predictions generated for {len(predictions)} students")
    high_risk = sum(1 for p in predictions if p['risk_level'] == 'High Risk')
    mod_risk = sum(1 for p in predictions if p['risk_level'] == 'Moderate Risk')
    print(f"  High Risk: {high_risk}, Moderate Risk: {mod_risk}")
    
    # Step 5: Generate Visualizations
    print("\n[5/6] Generating visualizations...")
    fig1_overall_dashboard(students, branch_comparison)
    fig2_attendance_analysis(students)
    fig3_internal_marks_analysis(students)
    fig4_predictive_insights(students)
    fig5_subject_wise_performance(students)
    fig6_attendance_trends(students, branch_comparison)
    fig7_ml_results(predictor, X, y, labels)
    fig8_branch_comparison(students, branch_comparison)
    fig9_system_architecture()
    fig10_correlation_analysis(students)
    fig11_risk_prediction_dashboard(students)
    fig12_alert_notification_system(students)
    
    # Step 6: Summary Report
    print("\n[6/6] Generating summary report...")
    print(f"\n  {'Metric':<30} {'Value':<15}")
    print(f"  {'-'*45}")
    print(f"  {'Total Students':<30} {len(students):<15}")
    print(f"  {'Total Subjects':<30} {len(subject_analytics):<15}")
    print(f"  {'Total Branches':<30} {len(branch_comparison):<15}")
    print(f"  {'Average CGPA':<30} {np.mean([s['cgpa'] for s in students]):.2f}")
    print(f"  {'Average Attendance':<30} {np.mean([s['overall_attendance'] for s in students]):.1f}%")
    print(f"  {'Average Exam Score':<30} {np.mean([np.mean(list(s['exam_marks'].values())) for s in students]):.1f}")
    print(f"  {'High Risk Students':<30} {high_risk:<15}")
    print(f"  {'Moderate Risk Students':<30} {mod_risk:<15}")
    print(f"  {'Regression R² Score':<30} {reg_results['r2_score']:.4f}")
    print(f"  {'Classification Accuracy':<30} {cls_results['accuracy']:.4f}")
    
    print("\n" + "=" * 70)
    print("All visualizations and analysis complete!")
    print("Images saved in: /home/ubuntu/report_images/")
    print("=" * 70)


if __name__ == '__main__':
    main()
