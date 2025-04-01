import os

# Check the current working directory
print("Current Working Directory:", os.getcwd())

# Verify if actual_df contains data
print("Shape of actual_df:", actual_df.shape)
print("Columns in actual_df:", actual_df.columns)

# Save results
actual_df[['Objective_Score', 'Competencies_Score', 'Self_Rating', 'Supervisor_Rating', 'Peer_Rating', 'Actual_Score']].to_json('actual_score.json')

# Check if the file is created
print("File 'actual_score.json' exists:", os.path.exists('actual_score.json'))