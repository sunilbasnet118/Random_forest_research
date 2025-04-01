import pandas as pd

def preprocess_data(data_file):
    # Read data from JSON file into a pandas DataFrame
    df = pd.read_json(data_file)

    # Feature engineering: Calculate rating using harmonic mean
    df['Rating'] = 3 / ((1/df['Self_Rating']) + (1/df['Supervisor_Rating']) + (1/df['Peer_Rating']))

    # Calculate the weighted score
    weight_objective = 0.45
    weight_competencies = 0.35
    weight_rating = 0.20
    df['Weighted_Score'] = (df['Objective_Score'] * weight_objective) + \
                            (df['Competencies_Score'] * weight_competencies) + \
                            (df['Rating'] * weight_rating)

    # Return preprocessed data
    return df[['Objective_Score', 'Competencies_Score', 'Self_Rating', 'Supervisor_Rating', 'Peer_Rating']], df['Weighted_Score']