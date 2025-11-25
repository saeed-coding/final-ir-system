import pandas as pd
import numpy as np
import re
import string
from collections import defaultdict
import math
from typing import List, Dict, Tuple
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')


class LocalIRSystem:
    def __init__(self, cache_dir="ir_cache"):
        self.documents = []
        self.doc_ids = []
        self.headings = []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize other attributes
        self.vocabulary = set()
        self.inverted_index = defaultdict(list)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25_index = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for indexing"""
        if not isinstance(text, str):
            return []

        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'\d+', '', text)

        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]

        return tokens

    def build_inverted_index(self, documents: List[str]):
        """Build inverted index"""
        self.inverted_index = defaultdict(list)

        for doc_id, doc_text in enumerate(documents):
            tokens = self.preprocess_text(doc_text)
            self.vocabulary.update(tokens)

            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for token, freq in term_freq.items():
                self.inverted_index[token].append((doc_id, freq))

    def build_tfidf_index(self, documents: List[str]):
        """Build TF-IDF index"""
        preprocessed_docs = [' '.join(self.preprocess_text(doc)) for doc in documents]

        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=False,
            max_features=10000,
            min_df=2,
            max_df=0.8,
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_docs)

    def build_bm25_index(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        """Build BM25 index"""
        preprocessed_docs = [self.preprocess_text(doc) for doc in documents]

        doc_term_freq = []
        doc_lengths = []

        for doc_tokens in preprocessed_docs:
            term_freq = defaultdict(int)
            for token in doc_tokens:
                term_freq[token] += 1
            doc_term_freq.append(term_freq)
            doc_lengths.append(len(doc_tokens))

        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

        self.bm25_index = {
            'doc_term_freq': doc_term_freq,
            'doc_lengths': doc_lengths,
            'avg_doc_length': avg_doc_length,
            'k1': k1,
            'b': b,
            'doc_freq': defaultdict(int),
            'total_docs': len(documents)
        }

        for term_freq in doc_term_freq:
            for term in term_freq.keys():
                self.bm25_index['doc_freq'][term] += 1

    def save_indexes(self):
        """Save indexes to disk"""
        print("💾 Saving indexes to disk...")

        with open(self.cache_dir / "inverted_index.pkl", 'wb') as f:
            pickle.dump(dict(self.inverted_index), f)

        with open(self.cache_dir / "tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        from scipy.sparse import save_npz
        if self.tfidf_matrix is not None:
            save_npz(self.cache_dir / "tfidf_matrix.npz", self.tfidf_matrix)

        with open(self.cache_dir / "bm25_index.pkl", 'wb') as f:
            pickle.dump(self.bm25_index, f)

        with open(self.cache_dir / "vocabulary.pkl", 'wb') as f:
            pickle.dump(list(self.vocabulary), f)

        with open(self.cache_dir / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)

        with open(self.cache_dir / "headings.pkl", 'wb') as f:
            pickle.dump(self.headings, f)

        print("✅ Indexes saved successfully!")

    def load_indexes(self):
        """Load indexes from disk"""
        required_files = [
            "inverted_index.pkl", "tfidf_vectorizer.pkl", "tfidf_matrix.npz",
            "bm25_index.pkl", "vocabulary.pkl", "documents.pkl",  "headings.pkl"
        ]

        if all((self.cache_dir / file).exists() for file in required_files):
            try:
                print("📂 Loading cached indexes...", end=" ")

                with open(self.cache_dir / "inverted_index.pkl", 'rb') as f:
                    inverted_data = pickle.load(f)
                    self.inverted_index = defaultdict(list, inverted_data)

                with open(self.cache_dir / "tfidf_vectorizer.pkl", 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)

                from scipy.sparse import load_npz
                self.tfidf_matrix = load_npz(self.cache_dir / "tfidf_matrix.npz")

                with open(self.cache_dir / "bm25_index.pkl", 'rb') as f:
                    self.bm25_index = pickle.load(f)

                with open(self.cache_dir / "vocabulary.pkl", 'rb') as f:
                    self.vocabulary = set(pickle.load(f))

                with open(self.cache_dir / "documents.pkl", 'rb') as f:
                    self.documents = pickle.load(f)

                with open(self.cache_dir / "headings.pkl", 'rb') as f:
                    self.headings = pickle.load(f)

                self.doc_ids = list(range(len(self.documents)))

                print("Done!")
                return True

            except Exception as e:
                print(f"Error: {e}")
                return False

        return False

    def load_data(self, csv_file_path: str, text_column: str = 'Article', headings_column: str = 'Headings'):
        """Load data from CSV file including headings"""
        data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
        self.documents = data[text_column].fillna('').tolist()
        self.headings = data[headings_column].fillna('').tolist()  # Load headings
        self.doc_ids = list(range(len(self.documents)))
        print(f"Loaded {len(self.documents)} documents with {len(self.headings)} headings")

    def build_indexes(self, force_rebuild=False):
        """Build all indexes"""
        if not force_rebuild and self.load_indexes():
            return

        print("🔨 Building indexes from scratch...")
        start_time = time.time()

        print("📊 Building inverted index...")
        self.build_inverted_index(self.documents)

        print("📈 Building TF-IDF index...")
        self.build_tfidf_index(self.documents)

        print("🎯 Building BM25 index...")
        self.build_bm25_index(self.documents)

        self.save_indexes()

        total_time = time.time() - start_time
        print(f"✅ Index building completed in {total_time:.2f} seconds")

    # Search methods (boolean_search, tfidf_search, bm25_search, hybrid_search, evaluate_query)
    # ... [Include all the search methods from previous implementation]

    def boolean_search(self, query: str) -> List[int]:
        """Boolean search (AND operation)"""
        query_terms = self.preprocess_text(query)

        if not query_terms:
            return []

        doc_sets = []
        for term in query_terms:
            if term in self.inverted_index:
                doc_ids = [doc_id for doc_id, _ in self.inverted_index[term]]
                doc_sets.append(set(doc_ids))
            else:
                return []

        result_docs = set.intersection(*doc_sets) if doc_sets else set()
        return list(result_docs)

    def tfidf_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """TF-IDF based search"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built.")

        preprocessed_query = ' '.join(self.preprocess_text(query))
        query_vector = self.tfidf_vectorizer.transform([preprocessed_query])

        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]

        return results

    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 search"""
        if self.bm25_index is None:
            raise ValueError("BM25 index not built.")

        query_terms = self.preprocess_text(query)
        scores = [0.0] * len(self.documents)

        k1 = self.bm25_index['k1']
        b = self.bm25_index['b']
        avg_dl = self.bm25_index['avg_doc_length']
        total_docs = self.bm25_index['total_docs']

        for doc_id in range(len(self.documents)):
            doc_length = self.bm25_index['doc_lengths'][doc_id]
            doc_term_freq = self.bm25_index['doc_term_freq'][doc_id]

            for term in query_terms:
                if term in doc_term_freq:
                    tf = doc_term_freq[term]
                    df = self.bm25_index['doc_freq'].get(term, 1)

                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avg_dl))

                    scores[doc_id] += idf * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

        return results

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Hybrid search combining TF-IDF and BM25"""
        tfidf_results = self.tfidf_search(query, top_k * 2)
        bm25_results = self.bm25_search(query, top_k * 2)

        combined_scores = defaultdict(float)

        for doc_id, score in tfidf_results:
            combined_scores[doc_id] += score * 0.4

        for doc_id, score in bm25_results:
            combined_scores[doc_id] += score * 0.6

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def evaluate_query(self, query: str, method: str = 'hybrid', top_k: int = 10):
        """Evaluate a query and return results"""
        start_time = time.time()

        if method == 'boolean':
            results = self.boolean_search(query)
            formatted_results = [(doc_id, 1.0) for doc_id in results][:top_k]
        elif method == 'tfidf':
            formatted_results = self.tfidf_search(query, top_k)
        elif method == 'bm25':
            formatted_results = self.bm25_search(query, top_k)
        else:  # hybrid
            formatted_results = self.hybrid_search(query, top_k)

        search_time = time.time() - start_time

        print(f"⏱️ Search completed in {search_time:.4f} seconds")
        print(f"📄 Found {len(formatted_results)} results")

        return formatted_results

    # Evaluation
    def evaluate_system(self, test_queries: Dict[str, List[int]], top_k_values: List[int] = [5, 10, 20]):
        """
        Comprehensive evaluation of the IR system

        Args:
            test_queries: Dictionary with queries as keys and list of relevant document IDs as values
            top_k_values: List of k values for precision@k, recall@k, etc.
        """
        print("\n" + "=" * 60)
        print("EVALUATING IR SYSTEM PERFORMANCE")
        print("=" * 60)

        results = {
            'map': self.mean_average_precision(test_queries),
            'mrr': self.mean_reciprocal_rank(test_queries),
            'precision_recall': {},
            'ndcg': {}
        }

        # Calculate precision@k, recall@k, f1@k for different k values
        for k in top_k_values:
            precision_k = self.precision_at_k(test_queries, k)
            recall_k = self.recall_at_k(test_queries, k)
            f1_k = self.f1_score_at_k(test_queries, k)
            ndcg_k = self.ndcg_at_k(test_queries, k)

            results['precision_recall'][k] = {
                'precision': precision_k,
                'recall': recall_k,
                'f1': f1_k
            }
            results['ndcg'][k] = ndcg_k

        # Print results
        self._print_evaluation_results(results, top_k_values)

        return results

    def precision_at_k(self, test_queries: Dict[str, List[int]], k: int = 10) -> float:
        """
        Calculate average precision at K

        Precision@K = (Number of relevant documents in top K) / K
        """
        precisions = []

        for query, relevant_docs in test_queries.items():
            # Get top K results using hybrid search (you can change the method)
            results = self.hybrid_search(query, top_k=k)
            retrieved_docs = [doc_id for doc_id, _ in results]

            # Count relevant documents in top K
            relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
            precision = relevant_retrieved / k if k > 0 else 0
            precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    def recall_at_k(self, test_queries: Dict[str, List[int]], k: int = 10) -> float:
        """
        Calculate average recall at K

        Recall@K = (Number of relevant documents in top K) / (Total relevant documents)
        """
        recalls = []

        for query, relevant_docs in test_queries.items():
            results = self.hybrid_search(query, top_k=k)
            retrieved_docs = [doc_id for doc_id, _ in results]

            relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
            total_relevant = len(relevant_docs)
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
            recalls.append(recall)

        return np.mean(recalls) if recalls else 0.0

    def f1_score_at_k(self, test_queries: Dict[str, List[int]], k: int = 10) -> float:
        """
        Calculate F1 score at K
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        f1_scores = []

        for query, relevant_docs in test_queries.items():
            results = self.hybrid_search(query, top_k=k)
            retrieved_docs = [doc_id for doc_id, _ in results]

            relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
            total_relevant = len(relevant_docs)

            precision = relevant_retrieved / k if k > 0 else 0
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)

        return np.mean(f1_scores) if f1_scores else 0.0

    def average_precision(self, query: str, relevant_docs: List[int]) -> float:
        """
        Calculate Average Precision for a single query
        """
        results = self.hybrid_search(query, top_k=len(self.documents))
        retrieved_docs = [doc_id for doc_id, _ in results]

        precision_values = []
        num_relevant = 0

        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precision_values.append(precision_at_i)

        if not precision_values:
            return 0.0

        return np.mean(precision_values)

    def mean_average_precision(self, test_queries: Dict[str, List[int]]) -> float:
        """
        Calculate Mean Average Precision (MAP)
        """
        average_precisions = []

        for query, relevant_docs in test_queries.items():
            ap = self.average_precision(query, relevant_docs)
            average_precisions.append(ap)

        return np.mean(average_precisions) if average_precisions else 0.0

    def reciprocal_rank(self, query: str, relevant_docs: List[int]) -> float:
        """
        Calculate Reciprocal Rank for a single query
        """
        results = self.hybrid_search(query, top_k=len(self.documents))
        retrieved_docs = [doc_id for doc_id, _ in results]

        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)

        return 0.0

    def mean_reciprocal_rank(self, test_queries: Dict[str, List[int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        """
        reciprocal_ranks = []

        for query, relevant_docs in test_queries.items():
            rr = self.reciprocal_rank(query, relevant_docs)
            reciprocal_ranks.append(rr)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def ndcg_at_k(self, test_queries: Dict[str, List[int]], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K
        """
        ndcg_scores = []

        for query, relevant_docs in test_queries.items():
            results = self.hybrid_search(query, top_k=k)

            # Binary relevance (1 for relevant, 0 for non-relevant)
            relevance_scores = [1 if doc_id in relevant_docs else 0 for doc_id, _ in results]

            # Calculate DCG
            dcg = 0.0
            for i, rel in enumerate(relevance_scores):
                dcg += rel / np.log2(i + 2)  # i+2 because i starts from 0

            # Calculate IDCG (ideal DCG)
            ideal_relevance = [1] * min(len(relevant_docs), k)
            # Pad with zeros if needed
            ideal_relevance.extend([0] * (k - len(ideal_relevance)))

            idcg = 0.0
            for i, rel in enumerate(ideal_relevance):
                idcg += rel / np.log2(i + 2)

            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def compare_search_methods(self, test_queries: Dict[str, List[int]], top_k: int = 10):
        """
        Compare performance of different search methods
        """
        print("\n" + "=" * 60)
        print("COMPARING SEARCH METHODS")
        print("=" * 60)

        methods = ['boolean', 'tfidf', 'bm25', 'hybrid']
        results = {}

        for method in methods:
            print(f"\nEvaluating {method.upper()} method...")

            # Temporarily change evaluate_query to use specific method
            precisions = []
            recalls = []

            for query, relevant_docs in test_queries.items():
                if method == 'boolean':
                    retrieved_docs = self.boolean_search(query)
                    retrieved_docs = retrieved_docs[:top_k]
                elif method == 'tfidf':
                    results_list = self.tfidf_search(query, top_k)
                    retrieved_docs = [doc_id for doc_id, _ in results_list]
                elif method == 'bm25':
                    results_list = self.bm25_search(query, top_k)
                    retrieved_docs = [doc_id for doc_id, _ in results_list]
                else:  # hybrid
                    results_list = self.hybrid_search(query, top_k)
                    retrieved_docs = [doc_id for doc_id, _ in results_list]

                relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
                precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
                recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0

                precisions.append(precision)
                recalls.append(recall)

            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (
                                                                                                avg_precision + avg_recall) > 0 else 0

            results[method] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': f1
            }

            print(f"  Precision: {avg_precision:.4f}")
            print(f"  Recall:    {avg_recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

        return results

    def _print_evaluation_results(self, results: dict, top_k_values: List[int]):
        """Print evaluation results in a formatted way"""
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"   Mean Average Precision (MAP):  {results['map']:.4f}")
        print(f"   Mean Reciprocal Rank (MRR):    {results['mrr']:.4f}")

        print(f"\n📈 Precision@K / Recall@K / F1@K:")
        for k in top_k_values:
            metrics = results['precision_recall'][k]
            print(f"   K={k:2d}: "
                  f"Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, "
                  f"F1: {metrics['f1']:.4f}")

        print(f"\n🎯 NDCG@K:")
        for k in top_k_values:
            ndcg = results['ndcg'][k]
            print(f"   K={k:2d}: NDCG: {ndcg:.4f}")

    def generate_queries_from_headings(self, num_queries: int = 10, min_relevant: int = 2, max_relevant: int = 20):
        """
        Generate high-quality test queries from document headings

        Args:
            num_queries: Number of queries to generate
            min_relevant: Minimum number of relevant documents per query
            max_relevant: Maximum number of relevant documents per query
        """
        print(f"🎯 Generating test queries from {len(self.headings)} headings...")

        # Preprocess headings to find good candidate queries
        candidate_queries = {}

        for doc_id, heading in enumerate(self.headings):
            if not isinstance(heading, str) or len(heading.strip()) < 10:
                continue

            # Clean the heading and use it as a potential query
            clean_heading = self._clean_heading(heading)
            if clean_heading and len(clean_heading) > 5:
                candidate_queries[clean_heading] = doc_id

        # Remove duplicates and find queries that match multiple documents
        final_queries = {}

        for query, source_doc_id in candidate_queries.items():
            # Find documents relevant to this query
            relevant_docs = self._find_relevant_docs_for_heading(query, source_doc_id)

            # Filter by relevance count
            if min_relevant <= len(relevant_docs) <= max_relevant:
                final_queries[query] = relevant_docs

            # Stop when we have enough queries
            if len(final_queries) >= num_queries:
                break

        print(f"✅ Generated {len(final_queries)} quality test queries from headings")
        return final_queries

    def _clean_heading(self, heading: str) -> str:
        """Clean and normalize heading for use as query"""
        # Remove extra whitespace, special characters, etc.
        cleaned = re.sub(r'[^\w\s]', ' ', heading)  # Remove punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()

        # Remove common prefixes/suffixes
        prefixes = ['article about', 'discussion on', 'study of', 'research on', 'analysis of']
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # Ensure reasonable length
        words = cleaned.split()
        if len(words) < 2 or len(words) > 8:
            return ""  # Too short or too long

        return cleaned

    def _find_relevant_docs_for_heading(self, query: str, source_doc_id: int) -> List[int]:
        """
        Find documents relevant to a heading-based query
        The source document is always relevant, plus others that contain similar content
        """
        relevant_docs = [source_doc_id]  # The source document is always relevant

        # Use TF-IDF similarity to find other relevant documents
        try:
            # Preprocess query
            preprocessed_query = ' '.join(self.preprocess_text(query))
            query_vector = self.tfidf_vectorizer.transform([preprocessed_query])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Find documents with similarity above threshold
            threshold = 0.1  # Adjust based on your data
            for doc_id, similarity in enumerate(similarities):
                if doc_id != source_doc_id and similarity > threshold:
                    relevant_docs.append(doc_id)

        except Exception as e:
            print(f"Warning: Could not compute similarities for query '{query}': {e}")
            # Fallback: just use the source document

        return relevant_docs

    def generate_headings_based_queries_advanced(self, num_queries: int = 15):
        """
        More advanced query generation using headings with better quality control
        """
        print(f"🎯 Generating advanced test queries from headings...")

        # Step 1: Collect and clean all headings
        clean_headings = []
        for heading in self.headings:
            if isinstance(heading, str) and heading.strip():
                clean_heading = self._clean_heading(heading)
                if clean_heading:
                    clean_headings.append(clean_heading)

        # Step 2: Find frequent headings (these make good test queries)
        from collections import Counter
        heading_freq = Counter(clean_headings)

        # Step 3: Select the best candidate queries
        final_queries = {}
        selected_queries = set()

        for heading, freq in heading_freq.most_common(100):  # Top 100 most frequent headings
            if len(final_queries) >= num_queries:
                break

            # Skip if we've already used a similar query
            if self._is_similar_to_existing(heading, selected_queries):
                continue

            # Find relevant documents for this heading
            relevant_docs = []
            for doc_id, doc_heading in enumerate(self.headings):
                if isinstance(doc_heading, str) and heading.lower() in doc_heading.lower():
                    relevant_docs.append(doc_id)

            # Quality control: ensure reasonable number of relevant docs
            if 3 <= len(relevant_docs) <= 25:
                final_queries[heading] = relevant_docs
                selected_queries.add(heading)

        print(f"✅ Generated {len(final_queries)} high-quality test queries")

        # Show some examples
        if final_queries:
            print("📝 Sample queries generated:")
            for i, query in enumerate(list(final_queries.keys())[:5]):
                print(f"   {i + 1}. '{query}' -> {len(final_queries[query])} relevant docs")

        return final_queries

    def _is_similar_to_existing(self, new_query: str, existing_queries: set, similarity_threshold: float = 0.7) -> bool:
        """
        Check if a new query is too similar to existing queries
        """
        new_terms = set(self.preprocess_text(new_query))

        for existing_query in existing_queries:
            existing_terms = set(self.preprocess_text(existing_query))

            # Simple Jaccard similarity
            intersection = len(new_terms & existing_terms)
            union = len(new_terms | existing_terms)

            if union > 0 and intersection / union > similarity_threshold:
                return True

        return False

