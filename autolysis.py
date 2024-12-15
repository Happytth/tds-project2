# /// script
# requires-python = ">=3.7"
# dependencies = [
#   "requests",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn",
#   "numpy",
#   "ipykernel"
# ]
# ///

import os
import sys
import json
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')

# Set your AIPROXY_TOKEN here
os.environ["AIPROXY_TOKEN"] = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDUwMTRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.4M6INtFWixQ7nWjWl2nkN3Wcopybkd_ZWa-tVf4d8FM"

class DataAnalysisTool:
    def __init__(self, data_path: str):
        self.data_path = data_path
        # Do not create a directory; use the current directory for saving outputs
        self.output_dir = os.getcwd()  # Current working directory
    
        # Try multiple encodings in sequence
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'utf-16', 'utf-32', 'cp1250', 'mac_roman']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(data_path, encoding=encoding)
                print(f"File successfully loaded with {encoding} encoding.")
                break
            except UnicodeDecodeError:
                print(f"Failed to load the file with {encoding} encoding.")
                continue
        else:
            raise ValueError("Failed to load the file with supported encodings.")
    
        self.ai_proxy_token = os.environ.get("AIPROXY_TOKEN")
        if not self.ai_proxy_token:
            raise ValueError("AIPROXY_TOKEN environment variable not set")
    
        self.api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"


    def compute_entropy(self, column: pd.Series) -> float:
        """Calculates Shannon entropy of a given column."""
        value_counts = column.value_counts(normalize=True)
        return -sum(value_counts * np.log2(value_counts))

    def perform_analysis(self) -> Dict[str, Any]:
        analysis_results = {
            "dataset_summary": {
                "num_rows": len(self.df),
                "columns": list(self.df.columns),
                "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "missing_data": self.df.isnull().sum().to_dict(),
                "sample_data": self.df.head().to_dict()
            }
        }

        # Entropy Analysis
        entropy_values = {}
        for column in self.df.select_dtypes(include=['object', 'category']).columns:
            entropy_values[column] = self.compute_entropy(self.df[column])

        analysis_results['entropy_analysis'] = entropy_values

        return analysis_results
        
    #LDA Analysis
    def perform_lda_analysis(self, num_topics: int = 5) -> Dict[str, Any]:
        """Perform Latent Dirichlet Allocation (LDA) on text columns."""
        text_columns = self.df.select_dtypes(include=['object', 'category']).columns
        lda_results = {}

        for column in text_columns:
            print(f"Performing LDA on column: {column}")
            vectorizer = CountVectorizer(stop_words='english')
            data_vectorized = vectorizer.fit_transform(self.df[column].dropna())

            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(data_vectorized)

            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                topic_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]]
                topics.append(f"Topic {topic_idx + 1}: {' '.join(topic_terms)}")
            
            lda_results[column] = topics

        return lda_results

    def create_visualizations(self, analysis_results: Dict[str, Any]):
        plt.figure(figsize=(20, 20))
        
        # Correlation Heatmap
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            plt.subplot(1, 2, 1)
            correlation_matrix = self.df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            heatmap_path = os.path.join(self.output_dir, 'correlation_matrix.png')  # Direct path
            plt.savefig(heatmap_path)
            plt.close()

        # Distribution of Numeric Columns
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(numeric_cols[:3], 1):
            plt.subplot(1, 3, i)
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
        dist_path = os.path.join(self.output_dir, 'numeric_distributions.png')  # Direct path
        plt.tight_layout()
        plt.savefig(dist_path)
        plt.close()

        return heatmap_path, dist_path  # Return paths to the visualizations

    def generate_narrative(self, analysis_results: Dict[str, Any], visualization_paths: list) -> str:
        prompt = f"""
        Write a detailed analysis of this dataset with the following components:
        1. Description of the dataset, including size and column details.
        2. Insights on entropy for categorical columns.
        3. Key findings from the Latent Dirichlet Allocation (LDA) analysis for topics.
        4. Include relevant visualizations in Markdown formatting and write down what you found interesting in the images (e.g., correlation heatmap, distribution plots).
        5. Any other potential observations and implications for further analysis.
        6. Write the interesting findings.

        Dataset Details:
        {json.dumps(analysis_results['dataset_summary'], indent=2)}
        Entropy Analysis:
        {json.dumps(analysis_results['entropy_analysis'], indent=2)}

        Visualizations:
        ![Correlation Heatmap]({visualization_paths[0]})
        ![Numeric Distributions]({visualization_paths[1]})
        """

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ai_proxy_token}"
        }

        try:
            print("Requesting analysis...")
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print("Analysis received!")
                return data.get('choices', [{}])[0].get('message', {}).get('content', 'No content available')
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return f"Error generating response: {response.status_code}"
        except Exception as e:
            print("Error encountered!")
            print(f"Error details: {e}")
            return f"Error generating response: {str(e)}"

    def execute(self):
        print("Starting the analysis...")
        analysis_results = self.perform_analysis()
        visualization_paths = self.create_visualizations(analysis_results)

        lda_results = self.perform_lda_analysis(num_topics=5)
        analysis_results['lda_analysis'] = lda_results

        print("Generating narrative...")
        narrative = self.generate_narrative(analysis_results, visualization_paths)

        print(f"Saving README.md to {self.output_dir}")
        with open(os.path.join(self.output_dir, 'README.md'), 'w', encoding='utf-8') as file:
            file.write(narrative)
        print("README.md saved!")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_data.py <dataset.csv>")
        sys.exit(1)

    data_path = sys.argv[1]
    analyzer = DataAnalysisTool(data_path)
    analyzer.execute()

if __name__ == "__main__":
    main()
