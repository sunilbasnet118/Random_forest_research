import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load trained model
model = joblib.load('trained_model.pkl')

# Load actual data
actual_df = pd.read_json('actual_dataset.json')

# Preprocess actual data (similar to data preprocessing)
actual_df['Rating'] = 3 / ((1/actual_df['Self_Rating']) + (1/actual_df['Supervisor_Rating']) + (1/actual_df['Peer_Rating']))

# Select features
X_actual = actual_df[['Objective_Score', 'Competencies_Score', 'Self_Rating', 'Supervisor_Rating', 'Peer_Rating']]

# Predict performance score
predicted_performance_score = model.predict(X_actual)

# Load actual scores from actual_score.json
actual_score_df = pd.read_json('actual_score.json')

# Compare actual and predicted scores
comparison_df = pd.DataFrame({
    'Employee_ID': actual_score_df['Employee_ID'].astype(int),
    'Actual_Score': actual_score_df['Actual_Score'],
    'Predicted_Score': predicted_performance_score
})

# Plot scatter plot based on actual vs predicted score with regression line
plt.figure(figsize=(12, 6))

# Scatter plot with regression line
plt.subplot(1, 2, 1)
sns.regplot(x='Actual_Score', y='Predicted_Score', data=comparison_df, color='blue', scatter_kws={'s': 50}, label='Data')
regression_model = LinearRegression()
regression_model.fit(comparison_df[['Actual_Score']], comparison_df['Predicted_Score'])
plt.plot(comparison_df['Actual_Score'], regression_model.predict(comparison_df[['Actual_Score']]), color='red', label='Regression Line')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Actual vs Predicted Performance Score')
plt.legend()

# Scatter plot only
# Scatter plot only
plt.subplot(1, 2, 2)
plt.scatter(comparison_df['Actual_Score'], comparison_df['Predicted_Score'], color='blue', label='Data')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Actual vs Predicted Performance Score')

plt.tight_layout()

# Display table of actual and predicted scores
# Round Actual_Score and Predicted_Score to 2 decimal places
comparison_df['Actual_Score'] = comparison_df['Actual_Score'].round(2)
comparison_df['Predicted_Score'] = comparison_df['Predicted_Score'].round(2)

# Convert Employee_ID to integer to remove decimal places
comparison_df['Employee_ID'] = comparison_df['Employee_ID'].astype(int)

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(10, 7))

# Turn off the axis
ax.axis('tight')
ax.axis('off')

# Set the title above the table
plt.title('Comparison of Actual and Predicted Performance Scores', pad=20)

# Prepare table data with formatting
table_data = comparison_df.values.tolist()
formatted_table_data = [[f"{int(row[0])}", f"{row[-2]:.2f}", f"{row[-1]:.2f}"] for row in table_data]

# New column labels
col_labels = ['Employee ID', 'Actual Score', 'Predicted Score']

# Create the table
table = ax.table(cellText=formatted_table_data, colLabels=col_labels, loc='center', cellLoc='center', colColours=['lightblue']*len(col_labels))

# Auto-adjust the table font size
table.auto_set_font_size(False)
table.set_fontsize(10)

# Adjust column widths
cell_dict = table.get_celld()
for i in range(len(formatted_table_data) + 1):  # +1 for header
    for j in range(len(col_labels)):
        cell = cell_dict[(i, j)]
        cell.set_width(0.20)
        cell.set_height(0.05)

# Display the plot
plt.show()

# Display the comparison table in the terminal
print("\nComparison of Actual and Predicted Performance Scores")
print(comparison_df.to_string(index=False))