import warnings
import numpy as np
import pandas as pd
import random
import os
import json
import torch
import python_avatars as pa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    adjusted_rand_score,
)
from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

from langchain import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import google.generativeai as genai
from dotenv import load_dotenv

def initialize():
    load_dotenv()

    API_KEY = os.getenv("API_KEY")

    if not API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env file")

    genai.configure(api_key=os.environ["API_KEY"])
    global model
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

def clean_data(dframe, is_labeled=True):
    """
    Cleans the DataFrame by handling missing values, dropping columns with constant values,
    and label encoding categorical columns.

    Parameters:
    - dframe: Original DataFrame
    - is_labeled: Boolean indicating whether the DataFrame is labeled or unlabeled

    Returns:
    - Cleaned DataFrame
    """
    df = dframe.copy()

    # Fill missing values with mean for numeric columns
    for column in df.columns:
        if df[column].dtype != object:
            df[column] = df[column].fillna(df[column].mean())

    # Drop columns with constant values
    columns_to_drop = [col for col in df.columns if len(df[col].unique()) == 1]
    df.drop(columns=columns_to_drop, inplace=True)

    # Label encode text label columns
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = LabelEncoder().fit_transform(df[column_name])

    # Check if the last column is label encoded or not
    if is_labeled and (df.iloc[:, -1].dtype == object or df.iloc[:, -1].dtype == bool):
        print("Last column is not label encoded. Label encoding it now...")
        df.iloc[:, -1] = LabelEncoder().fit_transform(df.iloc[:, -1])

    return df

def normalize_columns(df, list_of_columns, is_labeled=True):
    """
    Normalizes specified columns using StandardScaler.

    Parameters:
    - df: Original DataFrame
    - list_of_columns: List of columns to normalize

    Returns:
    - DataFrame with normalized columns
    """
    dframe = df.copy()
    if len(list_of_columns) > 0:
        # If the dataset is labeled and the last column is not to be normalized
        if is_labeled:
            last_column = dframe.columns[-1]
            list_of_columns = [col for col in list_of_columns if col != last_column]

            # Normalize the specified columns
        for column_name in list_of_columns:
            dframe[column_name] = StandardScaler().fit_transform(dframe[column_name].values.reshape(-1, 1))

    return dframe

def feature_selection_variance_threshold(df, t_value, is_labeled=True):
    """
    Perform feature selection using VarianceThreshold.

    Parameters:
    - df: Original DataFrame
    - t_value: Threshold value for variance

    Returns:
    - DataFrame with selected features
    """
    dframe = df.copy()
    selector = VarianceThreshold(threshold=t_value)
    # Separate the target variable (last column)
    if is_labeled:
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]
        # Create a VarianceThreshold selector
        # Fit the selector to the features
        X_selected = selector.fit_transform(X)
        # Get the indices of the selected features
        features_to_keep = selector.get_support(indices=True)
        # Select the features using the indices
        selected_features = X.iloc[:, features_to_keep]
        # Concatenate the selected features with the target column
        result_df = pd.concat([selected_features, y], axis=1)
        return result_df
    else:
        selector.fit(dframe)
        features_to_keep = selector.get_support(indices=True)
        return dframe.iloc[:,features_to_keep]

def feature_selection_k_best(df, k_value, is_labeled=True):
    """
    Perform feature selection using SelectKBest with mutual_info_classif.

    Parameters:
    - df: Original DataFrame
    - k_value: Number of top features to select

    Returns:
    - DataFrame with selected features
    """
    if is_labeled:
    # Separate the target variable (last column)
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = np.zeros(df.shape[0])

    # Create a SelectKBest selector
    selector = SelectKBest(score_func=mutual_info_classif, k=k_value)

    # Fit the selector to the features
    X_selected = selector.fit_transform(X, y)

    # Get the indices of the selected features
    features_to_keep = selector.get_support(indices=True)

    # Select the features using the indices
    selected_features = X.iloc[:, features_to_keep]

    if is_labeled:
        # Concatenate the selected features with the target column
        result_df = pd.concat([selected_features, y], axis=1)
        return result_df
    else:
        selected_feature_names = X.columns[features_to_keep]
    return X[selected_feature_names]

def plot_pairplot(df, cols_plot_info, hue_column):
    sns.pairplot(data=df[cols_plot_info], hue=hue_column, palette=["#3EC1D3", "#FFD36E"], corner=True)
    plt.show()

def cluster_kmeans(original_data, df):
    """
    Perform K-Means clustering on the given DataFrame for a range of K values.
    Output optimal K values based on GridSearchCV.

    Parameters:
    - df: DataFrame for clustering
    - k_values: Maximum number of clusters to consider

    Returns:
    - None
    """

    # Hyperparameter tuning using GridSearchCV
    kmeans = KMeans()
    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'init': ['k-means++', 'random'],
        'max_iter': [100, 300, 500],
        'random_state': [0, 50]
    }
    grid_search = GridSearchCV(estimator=kmeans, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X=df.iloc[:, :-1], y=df.iloc[:, -1])
    best_params = grid_search.best_params_

    # Fit K-Means clustering with the optimal hyperparameters
    kmeans = KMeans(n_clusters=best_params['n_clusters'], init=best_params['init'], 
                    max_iter=best_params['max_iter'], random_state=best_params['random_state'])
    kmeans.fit(df)

    clustered_df = df.copy()
    cluster_prediction = kmeans.predict(clustered_df)
    original_data["label"] = cluster_prediction

    label_mapping = {i: chr(65 + i) for i in range(kmeans.n_clusters)}
    original_data["label"] = clustered_df['label'].replace(label_mapping)
    
    original_data.to_csv('./data/original_data_kmeans.csv', index=False)

def cluster_kmeans_gpu(original_data, df, gpu_indx=0):
    """
    Perform K-Means clustering on the given DataFrame for a range of K values.
    Output optimal K values based on GridSearchCV.

    Parameters:
    - original_data: Original DataFrame to add clustering labels to.
    - df: DataFrame for clustering.
    - gpu_indx: Index of the GPU to use (default is 0).

    Returns:
    - None
    """
    # Select device
    device = torch.device(f'cuda:{gpu_indx}' if torch.cuda.is_available() else 'cpu')
    
    # Convert DataFrame to PyTorch tensor and move to selected device
    data_tensor = torch.tensor(df.values, dtype=torch.float32).to(device)

    # Hyperparameter tuning using GridSearchCV
    kmeans = KMeans()
    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'init': ['k-means++', 'random'],
        'max_iter': [100, 300, 500],
        'random_state': [0, 50]
    }
    grid_search = GridSearchCV(estimator=kmeans, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit GridSearchCV using CPU
    grid_search.fit(X=df.values[:, :-1], y=df.values[:, -1])
    best_params = grid_search.best_params_

    # Fit K-Means clustering with the optimal hyperparameters
    kmeans = KMeans(n_clusters=best_params['n_clusters'], init=best_params['init'], 
                    max_iter=best_params['max_iter'], random_state=best_params['random_state'])
    kmeans.fit(df.values)  # Use CPU for fitting

    clustered_df = df.copy()
    cluster_prediction = kmeans.predict(df.values)
    clustered_df["label"] = cluster_prediction
    original_data["label"] = cluster_prediction

    label_mapping = {i: chr(65 + i) for i in range(kmeans.n_clusters)}
    clustered_df['label'] = clustered_df['label'].replace(label_mapping)
    original_data['label'] = original_data['label'].replace(label_mapping)
    clustered_df.to_csv('./data/labeled_result_kmeans.csv', index=False)
    original_data.to_csv('./data/original_data_kmeans.csv', index=False)

def process_and_visualize_labeled_data(file_path, colors):
    """
    Process and visualize labeled data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file containing labeled data.
    - colors: List of colors for the plot.

    Returns:
    - df_res: DataFrame with group sizes.
    - df_plot: DataFrame sorted by label.
    - Plot saved as 'group_count_plot.png' in the './assets/' folder.
    """
    df = pd.read_csv(file_path)

    # Calculate group sizes
    df_res = round(df['label'].value_counts(normalize=True), 2).rename_axis('group').to_frame('group_size').sort_index()

    # Sort the data by labels
    df_plot = df.sort_values(by='label').reset_index(drop=True)

    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df_plot["label"], palette=colors)
    plt.xlabel('Group', fontsize=12)
    plt.ylabel('Number of Customers in the Group', fontsize=12)
    plt.title('Customer Group Distribution')
    
    save_dir = '../frontend/public/assets/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_dir + 'group_count_plot.png')
    plt.close()

    return df_res, df_plot

def analyze_summary_dynamic(df, label_column='label', output_file='./data/summary_result.csv'):
    """
    Performs statistical analysis on all columns (except the label column) grouped by clusters.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_column (str): The column name for cluster labels. Default is 'label'.
        output_file (str): Path to save the summary CSV file. Default is './data/summary_result.csv'.

    Returns:
        pd.DataFrame: A DataFrame containing the summary results.
    """
    
    # Exclude the label column from analysis
    cols_to_analyze = df.columns[df.columns != label_column]

    # Initialize the result DataFrame with group sizes
    df_res = round(df[label_column].value_counts(normalize=True), 2).rename_axis('group').to_frame('group_size').sort_index()

    # Calculate quartiles and ranks for each column based on its type
    for col in cols_to_analyze:
        if pd.api.types.is_numeric_dtype(df[col]):
            df_res[f"{col}_q1"] = df.groupby(label_column)[col].quantile(0.25).to_list()
            df_res[f"{col}_q3"] = df.groupby(label_column)[col].quantile(0.75).to_list()
            df_res[f"{col}_rank_q1"] = df_res[f"{col}_q1"].rank(method='max', ascending=False).astype(int).to_list()
            df_res[f"{col}_rank_q3"] = df_res[f"{col}_q3"].rank(method='max', ascending=False).astype(int).to_list()
        elif pd.api.types.is_bool_dtype(df[col]):
            df_res[f"{col}_mean"] = df.groupby(label_column)[col].mean().round(2).to_list()
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # For categorical or object types, calculate mode (most frequent value)
            df_res[f"{col}_mode"] = df.groupby(label_column)[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_list()

    df_res.to_csv(output_file)

    return df_res

def generate_avatar(gender):
    female_hair_type = [pa.HairType.STRAIGHT_STRAND, pa.HairType.BOB, pa.HairType.LONG_NOT_TOO_LONG,
                        pa.HairType.BRIDE, pa.HairType.CURLY_2, pa.HairType.MIA_WALLACE,
                        pa.HairType.STRAIGHT_1, pa.HairType.STRAIGHT_2]
    male_hair_type = [pa.HairType.SHAGGY, pa.HairType.SHORT_FLAT, pa.HairType.CAESAR,
                      pa.HairType.CAESAR_SIDE_PART, pa.HairType.SHORT_WAVED, pa.HairType.POMPADOUR,
                      pa.HairType.ELVIS, pa.HairType.BUZZCUT]
    clothing_color = ["#E5CB93", "#A9DBEA", "#DBD2F4", "#FF8B71", "#878ECD", "#046582",
                      "#DAD8D7", "#DAD8D7", "#C0D8C0", "#DD4A48", "#FEA82F", "#FF6701"]
    clothing_style = [pa.ClothingType.SHIRT_SCOOP_NECK, pa.ClothingType.BLAZER_SWEATER,
                      pa.ClothingType.COLLAR_SWEATER, pa.ClothingType.HOODIE,
                      pa.ClothingType.SHIRT_CREW_NECK]
    hair_color = ["#6B3307", "#000000", "#C4942D", "#B05A08", "#3F4E4F", "#A27B5C",
                  "#A19882", "#555555", "#7F7C82", "#FEA82F"]

    if gender == 'Female':
        my_avatar = pa.Avatar(
            style=pa.AvatarStyle.CIRCLE,
            background_color="#F4F9F9",
            top=random.choice(female_hair_type),
            eyebrows=pa.EyebrowType.DEFAULT_NATURAL,
            eyes=pa.EyeType.DEFAULT,
            nose=pa.NoseType.DEFAULT,
            mouth=pa.MouthType.SMILE,
            facial_hair=pa.FacialHairType.NONE,
            skin_color="#FBD9BF",
            hair_color=random.choice(hair_color),
            accessory=pa.AccessoryType.NONE,
            clothing=random.choice(clothing_style),
            clothing_color=random.choice(clothing_color)
        )
    else:
        my_avatar = pa.Avatar(
            style=pa.AvatarStyle.CIRCLE,
            background_color="#F4F9F9",
            top=random.choice(male_hair_type),
            eyebrows=pa.EyebrowType.DEFAULT_NATURAL,
            eyes=pa.EyeType.DEFAULT,
            nose=pa.NoseType.DEFAULT,
            mouth=pa.MouthType.SMILE,
            facial_hair=pa.FacialHairType.NONE,
            skin_color="#FBD9BF",
            hair_color=random.choice(hair_color),
            accessory=pa.AccessoryType.NONE,
            clothing=random.choice(clothing_style),
            clothing_color=random.choice(clothing_color)
        )

    image_svg = my_avatar.render()
    image_svg = image_svg.replace("264px", "100%")
    image_svg = image_svg.replace("280px", "100%")

    return image_svg

def generate_persona(summary: dict) -> dict:
    """Generates a customer persona based on input summary.

    Args:
        summary (dict): Summary dictionary containing demographic, interest, and behavior information.

    Returns:
        dict: A dictionary containing the generated customer persona.
    """

    persona_schema = {
        "name": "response",
        "description": """{
            "cluster_summaries": [],
            "cluster_personas": []
        }"""
    }

    response_schema = [persona_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = """
    I want you to generate two things in valid JSON format:
    1. Generate a detailed textual cluster summary for each group (defined in the summary first column) in a comma-separated string without keys.
    2. Generate a detailed customer persona for each group (defined in the summary first column) in the format defined in the example given below (please give one value for demographic data and not ranges):

    IMPORTANT: The output should be a JSON object of two arrays cluster_summaries and cluster_personas. Make sure the JSON is valid.

    Example:
    {{
        "cluster_summaries": [
        {{'textual summary of cluster 1', ...}},
        ],
        "cluster_personas": [
        {{
        "demographics": {{
            "name": "Anna",
            "age": "age_value",
            "gender": "gender_value",
            "marital_status": "marital_status_value",
            "family_structure": "family_structure_value",
            "income_level": "income_level_value",
            "location": "location_value",
            "occupation": "occupation_value"
        }},
        "psychographics": {{
            "values_and_beliefs": "values_and_beliefs_value",
            "interests_and_hobbies": "interests_and_hobbies_value",
            "lifestyle_choices": "lifestyle_choices_value",
            "technology_usage": "technology_usage_value",
            "brand_preferences": "brand_preferences_value",
            "community_engagement": "community_engagement_value",
            "health_and_wellness": "health_and_wellness_value",
            "family_dynamics": "family_dynamics_value",
            "financial_goals": "financial_goals_value",
            "media_consumption": "media_consumption_value",
            "environmental_consciousness": "environmental_consciousness_value",
            "cultural_influences": "cultural_influences_value"
        }},
        "needs_and_pain_points": {{
            "needs": "needs_value",
            "pain_points": "pain_points_value"
        }},
        "behavioral_data": {{
            "behavioral_drivers": "behavioral_drivers_value",
            "obstacles_to_purchasing": "obstacles_to_purchasing_value",
            "expectations": "expectations_value",
            "marketing_suggestions": "marketing_suggestions_value"
        }}
    }}
    ]
    }}

    Summary:
    {summary}

    {format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["summary"],
        template=prompt_template,
        partial_variables={"format_instructions": format_instructions},
    )

    response = model.generate_content(prompt.format(summary=json.dumps(summary)))
    output = ""
    for chunk in response:
        output += chunk.text


    try:
        personas = json.loads(output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Generated content is not valid JSON: {e}")
    
    for persona in personas.get("response", {}).get("cluster_personas", personas.get("cluster_personas", [])):
        gender = persona.get("demographics", {}).get("gender", "string")
        avatar_url = generate_avatar(gender)
        persona["avatar"] = avatar_url
            
    return personas