Enhancing Disease Detection Through Integrating Patient Segmentation and ChatGPT
Overview
This project integrates disease detection and patient segmentation using machine learning and ChatGPT. It is divided into two parts:
1. Disease Detection with Python and Machine Learning: Implements clustering and segmentation for patient data using Python.
2. ChatGPT Integration: Uses the segmented data, along with additional disease information, to create an AI chatbot that provides diagnostic insights.

Project Structure
1. Files and Folders
* Patient Segmentation.ipynb: Handles data preprocessing, clustering using K-means, and saving the segmented data.
* langchain.ipynb: Handles data indexing and storage in a Pinecone database for ChatGPT access.
* utils.py: Contains functions to retrieve and refine responses from the AI model.
* main.py: A Streamlit-based UI for user interaction with the AI chatbot.
* requirements.txt: Lists all dependencies required for the project.
2. Dataset
* data/: Contains datasets such as patient_segmented_disease_dataset.csv and related PDF files for additional disease-related information.

Implementation Guide
Part 1: Disease Detection
1. Upload Patient Segmentation.ipynb to Jupyter Notebook, Google Colab, or VS Code.
2. Run all cells to:
o Perform clustering using the elbow method and K-means.
o Generate and save patient clusters in patient_segmented_disease_dataset.csv.
Part 2: ChatGPT Integration
Pre-requisites:
* Install Anaconda for environment management.
* Ensure Pinecone and OpenAI API keys are available.
Steps:
1. Create a Conda Environment:
bash
CopyEdit
conda create -n mchatgpt python=3.8 -y
conda activate mchatgpt
2. Install Requirements:
bash
CopyEdit
pip install -r requirements.txt
pip install --upgrade jupyter ipywidgets
pip install chardet charset_normalizer streamlit_chat
3. Run Langchain Indexing:
o Navigate to the folder containing the files.
o Run the langchain.ipynb notebook in Jupyter to set up the Pinecone database.
4. Run the Backend:
bash
CopyEdit
python utils.py
5. Start the Streamlit UI:
bash
CopyEdit
streamlit run main.py
6. Open the UI and interact with the chatbot.

Features
1. Patient Clustering:
o Identifies clusters based on patient data.
o Stores segmented data for enhanced disease classification.
2. ChatGPT Integration:
o Enables querying using natural language.
o Provides detailed explanations based on lab reports and conditions.

Sample Queries
Try the following queries in the UI:
1. "What diseases could be indicated by the following lab results: HAEMATOCRIT 30.9, HAEMOGLOBINS 9.9, ERYTHROCYTE 4.23, etc.?"
2. "What are normal values for blood count, and what conditions can abnormal values indicate?"
3. "What tests should I take to confirm anemia or clotting disorders?"

Troubleshooting
* Missing Libraries: Use pip install <library_name> to resolve missing dependencies.
* API Errors: Verify Pinecone and OpenAI API keys are correctly configured.
* Streamlit Issues: Ensure the correct Conda environment is activated before running main.py.

