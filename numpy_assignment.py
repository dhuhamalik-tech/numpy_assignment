# ==========================================
# NumPy Assignment - Student Scores Analysis
# ==========================================

# Import NumPy
import numpy as np

# -------------------------------
# Create the datasets
# -------------------------------

# Student test scores (Math, Science, English)
scores = np.array([
    [85, 92, 78],
    [90, 88, 95],
    [75, 70, 85],
    [88, 95, 92],
    [65, 72, 68],
    [95, 88, 85],
    [78, 85, 82],
    [92, 89, 90]
])

# Student names
names = np.array([
    'Alice',
    'Bob',
    'Charlie',
    'David',
    'Eve',
    'Frank',
    'Grace',
    'Henry'
])

# Random matrices
np.random.seed(42)  # Makes results reproducible
matrix_A = np.random.randint(1, 10, size=(4, 4))
matrix_B = np.random.randint(1, 10, size=(4, 4))

# =====================================================
# Task 1.1 - Average score for each student
# =====================================================

# Mean across each row (student)
average_scores = scores.mean(axis=1)

print("\nTask 1.1: Average Scores Per Student")
for name, avg in zip(names, average_scores):
    print(f"{name}: {avg:.2f}")

# =====================================================
# Task 1.2 - Highest score in each subject
# =====================================================

highest_scores = scores.max(axis=0)

print("\nTask 1.2: Highest Score in Each Subject")
print("Math:", highest_scores[0])
print("Science:", highest_scores[1])
print("English:", highest_scores[2])

# =====================================================
# Task 1.3 - Students scoring above 90 in any subject
# =====================================================

above_90 = np.any(scores > 90, axis=1)

print("\nTask 1.3: Students Who Scored Above 90")
print(names[above_90])

# =====================================================
# Task 1.4 - Students passing all subjects
# Passing score = 70
# =====================================================

passed_all = np.all(scores >= 70, axis=1)

print("\nTask 1.4: Students Passing All Subjects")
for name, passed in zip(names, passed_all):
    print(f"{name}: {passed}")

# =====================================================
# Task 2.1 - Reshape scores to 12 x 2
# =====================================================

reshaped_scores = scores.reshape(12, 2)

print("\nTask 2.1: Reshaped Scores (12x2)")
print(reshaped_scores)

# =====================================================
# Task 2.2 - Standardized Scores
# Formula: (x - mean) / std
# =====================================================

mean = scores.mean()
std = scores.std()

standardized_scores = (scores - mean) / std

print("\nTask 2.2: Standardized Scores")
print(standardized_scores)

# =====================================================
# Task 2.3 - Sort students by average score
# =====================================================

sorted_indices = np.argsort(average_scores)[::-1]

print("\nTask 2.3: Students Sorted by Average Score")

for i in sorted_indices:
    print(f"{names[i]}: {average_scores[i]:.2f}")

# =====================================================
# Task 2.4 - Min, Max, Mean for each subject
# =====================================================

print("\nTask 2.4: Subject Statistics")

subjects = ["Math", "Science", "English"]

for i, subject in enumerate(subjects):
    print(f"\n{subject}")
    print("Minimum:", scores[:, i].min())
    print("Maximum:", scores[:, i].max())
    print("Mean:", scores[:, i].mean())

# =====================================================
# Task 3.1 - Matrix Multiplication
# =====================================================

matrix_product = matrix_A @ matrix_B

print("\nTask 3.1: Matrix Multiplication")
print(matrix_product)

# =====================================================
# Task 3.2 - Determinant
# =====================================================

determinant = np.linalg.det(matrix_A)

print("\nTask 3.2: Determinant of Matrix A")
print(determinant)

# =====================================================
# Task 3.3 - Inverse of Matrix A
# =====================================================

print("\nTask 3.3: Inverse of Matrix A")

if determinant != 0:
    inverse = np.linalg.inv(matrix_A)
    print(inverse)
else:
    print("Matrix A has no inverse.")

# =====================================================
# Task 3.4 - Eigenvalues
# =====================================================

eigenvalues = np.linalg.eigvals(matrix_A)

print("\nTask 3.4: Eigenvalues")
print(eigenvalues)

# =====================================================
# Task 4.1 - Add 5 points to Math scores
# =====================================================

updated_scores = scores.copy()

updated_scores[:, 0] += 5

print("\nTask 4.1: Scores After Adding 5 Points to Math")
print(updated_scores)

# =====================================================
# Task 4.2 - Unique scores
# =====================================================

unique_scores = np.unique(scores)

print("\nTask 4.2: Unique Scores")
print(unique_scores)

# =====================================================
# Task 4.3 - Students Above Average in Every Subject
# =====================================================

subject_means = scores.mean(axis=0)

above_average = np.all(scores > subject_means, axis=1)

print("\nTask 4.3: Students Above Average in All Subjects")
print(names[above_average])

# =====================================================
# Bonus Challenge
# =====================================================

def student_report(student_name):
    """
    Returns:
    - Individual scores
    - Ranking in each subject
    - Whether student is in top 3 overall
    """

    if student_name not in names:
        print("Student not found.")
        return

    index = np.where(names == student_name)[0][0]

    student_scores = scores[index]

    overall_rank = np.argsort(average_scores)[::-1]
    top3 = index in overall_rank[:3]

    print("\n===============================")
    print("Student Report")
    print("===============================")

    print("Name:", student_name)
    print("Scores:")
    print("Math:", student_scores[0])
    print("Science:", student_scores[1])
    print("English:", student_scores[2])

    print("\nRankings")

    for i, subject in enumerate(subjects):
        ranking = np.argsort(scores[:, i])[::-1]
        rank = np.where(ranking == index)[0][0] + 1
        print(f"{subject}: Rank {rank}")

    print("\nTop 3 Overall:", top3)


# Example
student_report("Bob")