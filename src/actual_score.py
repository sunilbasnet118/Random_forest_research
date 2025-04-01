import pandas as pd

def calculate_actual_score(actual_data_file):
    # Load actual data
    actual_df = pd.read_json(actual_data_file)

    # Preprocess actual data (similar to data preprocessing)
    actual_df['Rating'] = 3 / ((1/actual_df['Self_Rating']) + (1/actual_df['Supervisor_Rating']) + (1/actual_df['Peer_Rating']))

    # Calculate actual score using weights
    weight_objective = 0.45
    weight_competencies = 0.35
    weight_rating = 0.20
    actual_df['Actual_Score'] = (actual_df['Objective_Score'] * weight_objective) + \
                                 (actual_df['Competencies_Score'] * weight_competencies) + \
                                 ((actual_df['Self_Rating'] + actual_df['Supervisor_Rating'] + actual_df['Peer_Rating']) / 3 * weight_rating)

    # Save results with employees' IDs
    actual_df[['Employee_ID', 'Objective_Score', 'Competencies_Score', 'Self_Rating', 'Supervisor_Rating', 'Peer_Rating', 'Actual_Score']].to_json('actual_score.json', orient='records')

# Example usage
calculate_actual_score('actual_dataset.json')