------ Information Retrieval System (IR System) ------

Overview
This project implements an Information Retrieval (IR) system using various search methods: Boolean Search, TF-IDF, BM25, and Hybrid Search. It processes a collection of documents, builds indices (Inverted Index, TF-IDF, and BM25), and allows users to query the documents through a search interface.

Key Features
    - Preprocessing:Tokenization, stemming, stopword removal.
    - Search Methods:
        Boolean Search: Retrieves documents containing all query terms.
        TF-IDF Search: Retrieves documents based on term relevance using the TF-IDF algorithm.
        BM25 Search: Uses BM25 ranking function to score documents based on query terms.
        Hybrid Search: Combines TF-IDF and BM25 results for improved accuracy.
    - Indexing: Builds Inverted Index, TF-IDF Index, and BM25 Index for efficient document retrieval.
    - Evaluation: Supports precision, recall, F1-score, MAP, MRR, and NDCG evaluations for query results.
Setup
    - Install Dependencies:
        Run the following command to install required libraries:
            pip install -r requirements.txt
    - Download NLTK Data:
        The system requires specific NLTK corpora (stopwords, punkt). These are downloaded automatically when running the system.

Running the System
    - Building the Indexes
        This script processes the documents and builds the necessary indexes (Inverted, TF-IDF, BM25):
            python build_indexes.py
        It will save index pkl files in ir_cache folder
    - Evaluating the System
        After building the indexes, run the evaluation script to assess the performance of the IR system:
            python evaluate_system.py
        It saves detailed evaluation results (e.g., MAP, MRR, Precision@K) in evaluation_results_headings.json.
    - Search System (Query)
        After the indexes are built and evaluated, you can interactively search using:
            python search_system.py
        Usage: Enter a search query, choose a search method (Boolean, TF-IDF, BM25, Hybrid), and get the top K results.

File Descriptions
    - build_indexes.py: Builds the inverted index, TF-IDF, and BM25 indices.
    - evaluate_system.py: Evaluates the IR system using generated queries and computes precision, recall, and other metrics.
    - search_system.py: Provides an interactive command-line interface for searching the indexed documents.
    - ir_system_new.py: Contains the main LocalIRSystem class with methods for building indexes, searching, and evaluating.

Required Libraries:
    - pandas>=2.2.2
    - numpy>=2.2.6
    - nltk>=3.9.1
    - scikit-learn>=1.5.1

Evaluation Metrics
    - The system evaluates its search methods using:
        Precision: Proportion of relevant documents in the top K results.
        Recall: Proportion of relevant documents retrieved from the total relevant documents.
        F1 Score: Harmonic mean of Precision and Recall.
        MAP (Mean Average Precision): Measures precision at different levels of recall.
        MRR (Mean Reciprocal Rank): Measures the rank of the first relevant document.
        NDCG (Normalized Discounted Cumulative Gain): Measures ranking quality of the results.

Important Notes
    - Ensure that the Articles.csv file is properly formatted before running the scripts.
    - The NLTK corpora (punkt, stopwords) will be downloaded automatically if not found on the first run.
    - The system is designed to handle large document collections efficiently through indexing.