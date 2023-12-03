import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.model_selection import train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

EVALUATION_DATA_DIR = "./benchmark/data/"
MODEL_CHECKPOINT_PATH = "./models/CheckpointLightFM.pickle"

merged_data = pd.read_csv(f"{EVALUATION_DATA_DIR}evaluation_data.csv")

# Encode categorical features like 'gender' and 'occupation' using label encoding
gender_encoder = LabelEncoder()
occupation_encoder = LabelEncoder()

merged_data['gender'] = gender_encoder.fit_transform(merged_data['gender'])
merged_data['occupation'] = occupation_encoder.fit_transform(
    merged_data['occupation'])

# Create a dataset for LightFM
# For users and items, only their ids are needed
user_ids = merged_data['user_id'].unique()
item_ids = merged_data['movie_id'].unique()

# Create the feature lists for users and items
user_features = merged_data[['user_id', 'age',
                             'gender', 'occupation', 'zip_code']].drop_duplicates()
item_features = merged_data[['movie_id', 'movie_name', 'action', 'adventure', 'animation', "children's", 'comedy', 'crime', 'documentary', 'drama',
                             'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western', 'release_year']].drop_duplicates()

# Convert features to a list of strings, as required by LightFM
user_features['features'] = user_features.apply(lambda x: [str(
    x['age']), str(x['gender']), str(x['occupation']), x['zip_code']], axis=1)
item_features['features'] = item_features.apply(
    lambda x: [x['movie_name']] + list(x['action':'release_year'].astype(str)), axis=1)

item_features = item_features[[
    'movie_id', 'movie_name', 'release_year', 'features']]
# Extract the interaction data
interaction_data = merged_data[['user_id', 'movie_id', 'rating']]


# Initialize the LightFM dataset
dataset = Dataset()

# Fit the dataset to the users and items
dataset.fit(
    users=user_ids,
    items=item_ids,
    user_features=user_features['features'].explode(),
    item_features=item_features['features'].explode())

# Split the data
train_data, test_data = train_test_split(
    interaction_data, test_size=0.2, random_state=42)

# Build the interaction matrices for training and testing
(interactions_train, _) = dataset.build_interactions(
    [(row['user_id'], row['movie_id'], row['rating']) for idx, row in train_data.iterrows()])
(interactions_test, _) = dataset.build_interactions(
    [(row['user_id'], row['movie_id'], row['rating']) for idx, row in test_data.iterrows()])

# Build user and item feature matrices
user_features_matrix = dataset.build_user_features(
    [(row['user_id'], row['features']) for idx, row in user_features.iterrows()], normalize=False)
item_features_matrix = dataset.build_item_features(
    [(row['movie_id'], row['features']) for idx, row in item_features.iterrows()], normalize=False)


model = pickle.load(open(MODEL_CHECKPOINT_PATH, 'rb'))


auc = auc_score(model, interactions_test, user_features=user_features_matrix,
                item_features=item_features_matrix).mean()

precision = precision_at_k(
    model, interactions_test, user_features=user_features_matrix, item_features=item_features_matrix, k=10).mean()

recall = recall_at_k(model, interactions_test, user_features=user_features_matrix,
                     item_features=item_features_matrix, k=10).mean()

f1score = 2*precision*recall/(precision+recall)

print(f"F1-Score: {f1score:0.2}")
print(f"Area Under The Curve Score: {auc:0.2}")
print(f"Precision: {precision:0.2}")
print(f"Recall: {recall:0.2}")
