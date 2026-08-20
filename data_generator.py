"""
Data Generator Module - Generates realistic sample student academic data
for the Comprehensive Student Academic Performance Analytics Dashboard.
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# Configuration
NUM_STUDENTS = 100
SEMESTER = 5
BRANCHES = ['Computer Science Engineering', 'Electronics & Communication Engineering',
            'Information Technology', 'Mechanical Engineering', 'Civil Engineering',
            'Electrical Engineering']
SUBJECTS = {
    'Computer Science Engineering': [
        'Data Structures & Algorithms', 'Operating Systems', 'Database Management Systems',
        'Computer Networks', 'Software Engineering', 'Artificial Intelligence',
        'Web Technologies', 'Machine Learning'
    ],
    'Electronics & Communication Engineering': [
        'Digital Signal Processing', 'Microprocessors & Microcontrollers', 'Communication Systems',
        'VLSI Design', 'Control Systems', 'Embedded Systems', 'IoT Fundamentals', 'Signal Processing'
    ],
    'Information Technology': [
        'Data Structures & Algorithms', 'Operating Systems', 'Cloud Computing',
        'Computer Networks', 'Software Engineering', 'Data Analytics',
        'Web Technologies', 'Cyber Security'
    ],
    'Mechanical Engineering': [
        'Thermodynamics', 'Fluid Mechanics', 'Machine Design',
        'Heat Transfer', 'Manufacturing Processes', 'CAD/CAM',
        'Industrial Engineering', 'Robotics'
    ],
    'Civil Engineering': [
        'Structural Analysis', 'Concrete Technology', 'Surveying',
        'Geotechnical Engineering', 'Transportation Engineering', 'Water Resources',
        'Environmental Engineering', 'Construction Management'
    ],
    'Electrical Engineering': [
        'Power Systems', 'Electrical Machines', 'Control Systems',
        'Power Electronics', 'Measurements & Instrumentation', 'High Voltage Engineering',
        'Renewable Energy Systems', 'Smart Grid Technology'
    ]
}


def generate_students(num_students=NUM_STUDENTS):
    """Generate comprehensive student profiles with academic data."""
    students = []
    for i in range(1, num_students + 1):
        branch = random.choice(BRANCHES)
        subjects = SUBJECTS[branch]
        
        # Generate attendance data for each subject (60% to 95%)
        attendance = {}
        for subj in subjects:
            # Some students have low attendance (at-risk indicator)
            if random.random() < 0.15:  # 15% chance of low attendance
                att = round(random.uniform(50, 65), 1)
            else:
                att = round(random.uniform(70, 98), 1)
            attendance[subj] = att
        
        # Generate internal marks (3 out of 3 tests per subject)
        internal_marks = {}
        for subj in subjects:
            marks = []
            for test in range(1, 4):
                # Correlate with attendance somewhat
                base = attendance[subj] * 0.8
                if random.random() < 0.1:  # Some poor performers
                    marks.append(round(random.uniform(30, 55), 1))
                else:
                    marks.append(round(random.uniform(55, 95), 1))
            internal_marks[subj] = marks
        
        # Generate external/exam marks
        exam_marks = {}
        for subj in subjects:
            avg_internal = np.mean(internal_marks[subj])
            # Exam marks correlated with internal marks
            exam = round(avg_internal * 0.7 + random.uniform(5, 20), 1)
            exam = min(100, max(0, exam))
            exam_marks[subj] = exam
        
        # Calculate CGPA
        total_credits = len(subjects) * 3
        total_grade_points = 0
        for subj, exam in exam_marks.items():
            if exam >= 90: gp = 10
            elif exam >= 80: gp = 9
            elif exam >= 70: gp = 8
            elif exam >= 60: gp = 7
            elif exam >= 50: gp = 6
            elif exam >= 40: gp = 5
            else: gp = 0
            total_grade_points += gp * 3
        
        cgpa = round(total_grade_points / total_credits, 2)
        
        # Overall attendance percentage
        overall_attendance = round(np.mean(list(attendance.values())), 1)
        
        # Student details
        first_names = ['Aarav', 'Vihaan', 'Aditya', 'Ishaan', 'Arjun', 'Siddharth',
                       'Krishna', 'Rohan', 'Aryan', 'Kabir', 'Ananya', 'Priya',
                       'Sanya', 'Kavya', 'Meera', 'Nisha', 'Riya', 'Aditi', 'Sneha', 'Pooja']
        last_names = ['Sharma', 'Gupta', 'Singh', 'Kumar', 'Patel', 'Jain',
                      'Reddy', 'Nair', 'Iyer', 'Rao', 'Mehta', 'Verma',
                      'Chopra', 'Malhotra', 'Kapoor', 'Saxena', 'Agarwal', 'Srivastava', 'Tiwari', 'Mishra']
        
        student = {
            'student_id': f'STU{i:04d}',
            'name': f'{random.choice(first_names)} {random.choice(last_names)}',
            'branch': branch,
            'semester': SEMESTER,
            'attendance': attendance,
            'overall_attendance': overall_attendance,
            'internal_marks': internal_marks,
            'exam_marks': exam_marks,
            'cgpa': cgpa,
            'subjects': subjects,
            'status': 'Active',
            'date_registered': (datetime(2025, 1, 1) + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
            'risk_level': determine_risk_level(overall_attendance, cgpa)
        }
        students.append(student)
    
    return students


def determine_risk_level(attendance, cgpa):
    """Determine student risk level based on attendance and CGPA."""
    if attendance < 60 or cgpa < 5.0:
        return 'High Risk'
    elif attendance < 70 or cgpa < 6.5:
        return 'Moderate Risk'
    elif attendance < 75 or cgpa < 7.5:
        return 'Low Risk'
    else:
        return 'No Risk'


def predict_student_performance(student, historical_data=None):
    """
    Predict student performance using a weighted scoring model.
    Combines attendance, internal marks, and historical patterns.
    """
    predictions = {}
    
    for subject in student['subjects']:
        # Weighted prediction formula
        att_weight = 0.25
        internal_weight = 0.35
        historical_weight = 0.15
        trend_weight = 0.25
        
        # Attendance score (0-100)
        att_score = student['attendance'][subject]
        
        # Internal marks score (average of 3 tests)
        internal_avg = np.mean(student['internal_marks'][subject])
        
        # Trend score (improvement or decline)
        marks = student['internal_marks'][subject]
        if len(marks) >= 2:
            trend = marks[-1] - marks[0]
            trend_score = 50 + trend  # Center at 50
            trend_score = max(0, min(100, trend_score))
        else:
            trend_score = internal_avg
        
        # Historical average (if available)
        if historical_data and subject in historical_data:
            hist_score = historical_data[subject]
        else:
            hist_score = internal_avg * 0.8 + 15  # Estimated
        
        # Weighted prediction
        predicted = (att_score * att_weight + 
                    internal_avg * internal_weight +
                    hist_score * historical_weight +
                    trend_score * trend_weight)
        
        # Convert to grade prediction
        if predicted >= 90:
            predicted_grade = 'O'
        elif predicted >= 80:
            predicted_grade = 'A+'
        elif predicted >= 70:
            predicted_grade = 'A'
        elif predicted >= 60:
            predicted_grade = 'B+'
        elif predicted >= 50:
            predicted_grade = 'B'
        elif predicted >= 40:
            predicted_grade = 'C'
        else:
            predicted_grade = 'F'
        
        predictions[subject] = {
            'predicted_score': round(predicted, 2),
            'predicted_grade': predicted_grade,
            'risk_factor': 'At Risk' if predicted < 50 else 'Low Risk' if predicted < 65 else 'Safe'
        }
    
    return predictions


def generate_subject_analytics(students):
    """Generate subject-wise analytics across all students."""
    all_subjects = set()
    for s in students:
        all_subjects.update(s['subjects'])
    
    subject_stats = {}
    for subj in all_subjects:
        scores = []
        attendances = []
        for s in students:
            if subj in s['subjects']:
                scores.append(s['exam_marks'].get(subj, 0))
                attendances.append(s['attendance'].get(subj, 0))
        
        if scores:
            subject_stats[subj] = {
                'avg_score': round(np.mean(scores), 2),
                'max_score': round(np.max(scores), 2),
                'min_score': round(np.min(scores), 2),
                'std_dev': round(np.std(scores), 2),
                'pass_rate': round((np.sum(np.array(scores) >= 40) / len(scores)) * 100, 1),
                'avg_attendance': round(np.mean(attendances), 2),
                'students_enrolled': len(scores),
                'high_performers': int(np.sum(np.array(scores) >= 80)),
                'at_risk_students': int(np.sum(np.array(scores) < 50))
            }
    
    return subject_stats


def generate_branch_comparison(students):
    """Generate branch-wise comparison statistics."""
    branches = set(s['branch'] for s in students)
    branch_stats = {}
    
    for branch in branches:
        branch_students = [s for s in students if s['branch'] == branch]
        cgpas = [s['cgpa'] for s in branch_students]
        attendances = [s['overall_attendance'] for s in branch_students]
        
        branch_stats[branch] = {
            'num_students': int(len(branch_students)),
            'avg_cgpa': float(round(np.mean(cgpas), 2)),
            'max_cgpa': float(round(np.max(cgpas), 2)),
            'min_cgpa': float(round(np.min(cgpas), 2)),
            'std_cgpa': float(round(np.std(cgpas), 2)),
            'avg_attendance': float(round(np.mean(attendances), 2)),
            'high_risk_count': int(sum(1 for s in branch_students if s['risk_level'] == 'High Risk')),
            'moderate_risk_count': int(sum(1 for s in branch_students if s['risk_level'] == 'Moderate Risk')),
            'low_risk_count': int(sum(1 for s in branch_students if s['risk_level'] == 'Low Risk')),
            'no_risk_count': int(sum(1 for s in branch_students if s['risk_level'] == 'No Risk'))
        }
    
    return branch_stats


def generate_attendance_trends(students, num_weeks=20):
    """Generate weekly attendance trends for visualization."""
    trends = {}
    for branch in BRANCHES:
        branch_students = [s for s in students if s['branch'] == branch]
        weekly_attendance = []
        for week in range(1, num_weeks + 1):
            # Simulate attendance trend (slight decline over semester)
            base_att = np.mean([s['overall_attendance'] for s in branch_students])
            noise = random.uniform(-5, 5)
            decline = week * 0.15  # Slight decline factor
            weekly_att = max(40, base_att - decline + noise)
            weekly_attendance.append(round(weekly_att, 1))
        trends[branch] = weekly_attendance
    return trends


def generate_marks_distribution(students):
    """Generate marks distribution data for histogram/box plots."""
    all_marks = []
    for s in students:
        all_marks.extend(s['exam_marks'].values())
    
    distribution = {
        'ranges': ['0-20', '21-40', '41-60', '61-80', '81-100'],
        'counts': [0, 0, 0, 0, 0]
    }
    
    for mark in all_marks:
        if mark <= 20:
            distribution['counts'][0] += 1
        elif mark <= 40:
            distribution['counts'][1] += 1
        elif mark <= 60:
            distribution['counts'][2] += 1
        elif mark <= 80:
            distribution['counts'][3] += 1
        else:
            distribution['counts'][4] += 1
    
    return distribution


if __name__ == '__main__':
    print("Generating student data...")
    students = generate_students()
    
    print(f"Generated {len(students)} student profiles")
    
    # Generate analytics
    subject_analytics = generate_subject_analytics(students)
    branch_comparison = generate_branch_comparison(students)
    attendance_trends = generate_attendance_trends(students)
    marks_distribution = generate_marks_distribution(students)
    
    # Save data
    with open('/home/ubuntu/project/academic_analytics/students_data.json', 'w') as f:
        json.dump(students, f, indent=2)
    
    with open('/home/ubuntu/project/academic_analytics/analytics_data.json', 'w') as f:
        json.dump({
            'subject_analytics': subject_analytics,
            'branch_comparison': branch_comparison,
            'marks_distribution': marks_distribution
        }, f, indent=2)
    
    # Print summary
    print(f"\nSubject Analytics: {len(subject_analytics)} subjects analyzed")
    print(f"Branch Comparison: {len(branch_comparison)} branches compared")
    
    # Print sample predictions
    sample_student = students[0]
    predictions = predict_student_performance(sample_student)
    print(f"\nSample Predictions for {sample_student['name']}:")
    for subj, pred in list(predictions.items())[:5]:
        print(f"  {subj}: {pred['predicted_score']} ({pred['predicted_grade']}) - {pred['risk_factor']}")
    
    print("\nData generation complete!")
