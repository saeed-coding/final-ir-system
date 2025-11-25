from ir_system_new import LocalIRSystem
import json
import time


def main():
    print("=== IR System Evaluation Using Headings ===")
    print("This uses document headings to create realistic test queries.\n")

    # Initialize and load system
    ir_system = LocalIRSystem()

    if not ir_system.load_indexes():
        print("❌ No indexes found. Please run build_indexes.py first.")
        return

    print(f"✅ System loaded with {len(ir_system.documents)} documents")
    print(f"📚 Available headings: {len([h for h in ir_system.headings if h])}")

    # Generate queries from headings
    print("\n" + "=" * 50)
    print("GENERATING TEST QUERIES FROM HEADINGS")
    print("=" * 50)

    # Method 1: Simple heading-based queries
    # test_queries_simple = ir_system.generate_queries_from_headings(num_queries=10)

    # Method 2: Advanced heading-based queries (recommended)
    test_queries_advanced = ir_system.generate_headings_based_queries_advanced(num_queries=15)

    # Use the advanced queries
    test_queries = test_queries_advanced

    if not test_queries:
        print("❌ No suitable test queries generated from headings.")
        print("Trying fallback method...")
        test_queries = ir_system.generate_queries_from_headings(num_queries=10)

    if not test_queries:
        print("❌ No test queries available for evaluation.")
        return

    print(f"\n📝 Using {len(test_queries)} heading-based test queries")

    # Run comprehensive evaluation
    print("\n" + "=" * 50)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("=" * 50)

    start_time = time.time()
    evaluation_results = ir_system.evaluate_system(test_queries, top_k_values=[5, 10, 15, 20])
    evaluation_time = time.time() - start_time

    # Compare search methods
    print("\n" + "=" * 50)
    print("COMPARING SEARCH METHODS")
    print("=" * 50)

    method_comparison = ir_system.compare_search_methods(test_queries, top_k=10)

    # Save detailed results
    results_to_save = {
        'evaluation_results': evaluation_results,
        'method_comparison': method_comparison,
        'test_queries_used': test_queries,
        'evaluation_settings': {
            'total_documents': len(ir_system.documents),
            'total_queries': len(test_queries),
            'evaluation_time_seconds': evaluation_time,
            'average_relevant_docs_per_query': sum(len(docs) for docs in test_queries.values()) / len(test_queries)
        }
    }

    with open("evaluation_results_headings.json", "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n💾 Detailed results saved to 'evaluation_results_headings.json'")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    best_method = max(method_comparison.items(), key=lambda x: x[1]['f1'])
    print(f"🏆 Best performing method: {best_method[0].upper()} "
          f"(F1: {best_method[1]['f1']:.4f})")
    print(f"📊 Overall MAP: {evaluation_results['map']:.4f}")
    print(f"🎯 Overall MRR: {evaluation_results['mrr']:.4f}")
    print(f"⏱️ Evaluation time: {evaluation_time:.2f} seconds")

    # Show query performance breakdown
    print(f"\n📈 Query Performance Breakdown:")
    for query, relevant_docs in list(test_queries.items())[:5]:  # Show first 5
        precision = ir_system.precision_at_k({query: relevant_docs}, k=10)
        recall = ir_system.recall_at_k({query: relevant_docs}, k=10)
        print(f"   '{query[:40]}...': P={precision:.3f}, R={recall:.3f}")


if __name__ == "__main__":
    main()