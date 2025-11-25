from ir_system_new import LocalIRSystem
import time


def main():
    print("=== IR Search System ===")
    print("Loading pre-built indexes...")

    start_load = time.time()
    ir_system = LocalIRSystem()

    # Try to load cached indexes
    if ir_system.load_indexes():
        load_time = time.time() - start_load
        print(f"✅ System ready in {load_time:.2f} seconds!")
        print(f"📚 Loaded {len(ir_system.documents)} documents")
        print("\nAvailable search methods: boolean, tfidf, bm25, hybrid")
        print("Type 'quit' to exit\n")
    else:
        print("❌ No pre-built indexes found!")
        print("Please run 'build_indexes.py' first to prepare the dataset.")
        return

    # Interactive search loop
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()

            if query.lower() == 'quit':
                print("Goodbye!")
                break

            if not query:
                continue

            # Method selection with default
            method = input("Choose method (boolean/tfidf/bm25/hybrid) [hybrid]: ").strip().lower()
            if not method or method not in ['boolean', 'tfidf', 'bm25', 'hybrid']:
                method = 'hybrid'

            # Get top-k results
            try:
                top_k = int(input("Number of results [10]: ").strip() or "10")
            except ValueError:
                top_k = 10

            # Execute search
            print(f"\nSearching with {method.upper()}...")
            results = ir_system.evaluate_query(query, method, top_k)

            if not results:
                print("No results found. Try a different query or method.")
            else:
                print(f"\n📄 Top {len(results)} results:")
                for rank, (doc_id, score) in enumerate(results, 1):
                    # Show a clean snippet
                    doc_heading = ir_system.headings[doc_id]
                    doc_text = ir_system.documents[doc_id]
                    snippet = doc_text.replace('\n', ' ').strip()
                    if len(snippet) > 150:
                        snippet = snippet[:147] + "..."

                    print(f"{rank}. [Doc {doc_id}] Score: {score:.4f}")
                    print(f"Heading: {doc_heading}")
                    print(f"   {snippet}")
                    print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()