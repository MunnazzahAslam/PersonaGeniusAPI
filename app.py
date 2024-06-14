from flask import Flask,send_file, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import os
from utils.segmentation import clean_data, normalize_columns, feature_selection_variance_threshold, feature_selection_k_best, cluster_kmeans_gpu, process_and_visualize_labeled_data, analyze_summary_dynamic, generate_persona, initialize

initialize()
app = Flask(__name__)
CORS(app)

data_store = {}

@app.route('/', methods=['GET'])
def root():
    return 'Persona Genius API'

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files.get('file')
    if not uploaded_file or not uploaded_file.filename.endswith(".csv"):
        return jsonify({"error": "Unsupported file format. Please upload a CSV file."}), 400

    try:
        data = pd.read_csv(uploaded_file, sep=None, engine="python")
        data_store['raw_data'] = data
        total_records = len(data)

        return jsonify({"message": "Data uploaded successfully", "total_records": total_records, "steps_output": {"raw_data": data.head(10).to_dict(orient='list')}}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing data: {e}"}), 400

@app.route('/eda', methods=['GET'])
def eda_file():
    try:
        cleaned_data = clean_data(data_store['raw_data'], is_labeled=False)
        normalized_data = normalize_columns(cleaned_data, cleaned_data.columns.tolist(), is_labeled=False)
        var_selected_data = feature_selection_variance_threshold(normalized_data, 0.2, is_labeled=False)
        k_best_selected_data = feature_selection_k_best(var_selected_data, var_selected_data.shape[1] - 1, is_labeled=False)
        
        k_best_selected_data.to_csv('./data/cleaned_data.csv', index=False)
        data_store['cleaned_data'] = k_best_selected_data

        return jsonify({"message": "Data cleaned and EDA done successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing data: {e}"}), 400


@app.route('/cluster', methods=['GET'])
def cluster_data():
    try:
        cleaned_data = data_store['cleaned_data']
        raw_data = data_store['raw_data']
        
        cluster_kmeans_gpu(raw_data, cleaned_data)
        
        file_path = './data/labeled_result_kmeans.csv'
        colors = ['#FF165D', '#ADE792', '#3EC1D3', '#FFD36E']
        process_and_visualize_labeled_data(file_path, colors)

        return jsonify({"message": "Data clustered successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Error clustering data: {e}"}), 400

@app.route('/summary', methods=['GET'])
def generate_summary():
    try:
        dt = pd.read_csv('./data/original_data_kmeans.csv', index_col=0)
        analyze_summary_dynamic(dt, 'label')
        
        summary = pd.read_csv('./data/summary_result.csv')
        summary_dict = summary.to_dict(orient='list')
        personas = generate_persona(summary_dict)
        
        with open('./data/personas_output.json', 'w') as file:
            json.dump(personas, file, indent=4)

        return jsonify(personas), 200

    except Exception as e:
        return jsonify({"error": f"Error generating summary and personas: {e}"}), 400

if __name__ == "__main__":
    app.run(port=8000)