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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for matplotlib

# Ensure the AI Proxy Token is set correctly
AI_PROXY_ENV = "AIPROXY_TOKEN"
if not os.getenv(AI_PROXY_ENV):
    raise ValueError("Error: AIPROXY_TOKEN environment variable not set.")

# Define the main class
class DataAnalysisTool:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.output_dir = os.getcwd()  # Current directory for saving outputs
        
        # Load data with fallback encodings
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(data_path, encoding=encoding)
                print(f"File successfully loaded with {encoding} encoding.")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Failed to load the file with supported encodings.")
        
        self.ai_proxy_token = os.getenv(AI_PROXY_ENV)
        self.api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    def compute_entropy(self, column: pd.Series) -> float:
        """Calculate Shannon entropy for a given column."""
        value_counts = column.value_counts(normalize=True)
        return -sum(value_counts * np.log2(value_counts))

    def perform_analysis(self) -> Dict[str, Any]:
        """Generate a basic summary and entropy analysis of the dataset."""
        summary = {
            "num_rows": len(self.df),
            "columns": list(self.df.columns),
            "column_types": self.df.dtypes.astype(str).to_dict(),
            "missing_data": self.df.isnull().sum().to_dict(),
        }

        # Entropy analysis for categorical columns
        entropy_analysis = {
            col: self.compute_entropy(self.df[col])
            for col in self.df.select_dtypes(include=['object', 'category'])
        }

        return {"summary": summary, "entropy_analysis": entropy_analysis}

    def perform_lda_analysis(self, num_topics: int = 5) -> Dict[str, Any]:
        """Perform LDA topic modeling on text columns."""
        text_columns = self.df.select_dtypes(include=['object', 'category']).columns
        lda_results = {}

        for column in text_columns:
            print(f"Performing LDA on column: {column}")
            vectorizer = CountVectorizer(stop_words='english')
            data_vectorized = vectorizer.fit_transform(self.df[column].dropna())

            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(data_vectorized)

            topics = [
                " ".join(
                    [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
                )
                for topic in lda.components_
            ]
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

    # Kept generate_narrative completely intact
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
        """Run all steps of the analysis."""
        print("Running analysis...")
        analysis = self.perform_analysis()
        lda_results = self.perform_lda_analysis()
        analysis['lda_analysis'] = lda_results

        vis_paths = self.create_visualizations(analysis)
        narrative = self.generate_narrative(analysis, vis_paths)

        with open(os.path.join(self.output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(narrative)
        print("README.md saved successfully!")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_data.py <dataset.csv>")
        sys.exit(1)

    data_path = sys.argv[1]
    analyzer = DataAnalysisTool(data_path)
    analyzer.execute()


if __name__ == "__main__":
    main()
