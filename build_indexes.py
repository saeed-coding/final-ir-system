from ir_system_new import LocalIRSystem
import time


if __name__ == "__main__":
    print("=== IR System Index Builder ===")
    print("This script will build indexes including headings for better evaluation.\n")

    ir_system = LocalIRSystem()

    csv_file_path = "Articles.csv"  # Replace with your actual file path

    print(f"Loading data from {csv_file_path}...")
    # Now load both articles and headings
    ir_system.load_data(csv_file_path, text_column='Article', headings_column='Heading')

    print(f"Loaded {len(ir_system.documents)} documents with {len(ir_system.headings)} headings")

    # Build indexes
    print("\nBuilding indexes...")
    start_time = time.time()
    ir_system.build_indexes(force_rebuild=True)
    end_time = time.time()

    print(f"\n✅ Index building completed in {end_time - start_time:.2f} seconds!")
    print("You can now run 'evaluate_with_headings.py' for meaningful evaluation.")