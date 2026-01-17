"""
This is the database management utility for tracking our database.
Provides commands to view, verify, and manage our document tracking database.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from document_tracker import DocumentTracker
from document_loaders import get_supported_extensions
from config import TRACKING_DB_PATH, RAW_DOCS_DIR


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


def show_stats():
    """Display database statistics."""
    tracker = DocumentTracker(TRACKING_DB_PATH)
    stats = tracker.get_ingestion_stats()
    
    print_header("DATABASE STATISTICS")
    print(f"Total documents:    {stats['total_documents']}")
    print(f"Total chunks:       {stats['total_chunks']}")
    print(f"First ingestion:    {stats['first_ingestion'] or 'N/A'}")
    print(f"Last ingestion:     {stats['last_ingestion'] or 'N/A'}")
    print(f"Database location:  {TRACKING_DB_PATH}")


def list_documents():
    """List all ingested documents."""
    tracker = DocumentTracker(TRACKING_DB_PATH)
    docs = tracker.list_ingested_documents()
    
    print_header(f"INGESTED DOCUMENTS ({len(docs)} total)")
    
    if not docs:
        print("No documents in database.")
        return
    
    for doc in docs:
        file_name = Path(doc['file_path']).name
        ingested = datetime.fromisoformat(doc['ingested_at']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{file_name}")
        print(f"   Path:     {doc['file_path']}")
        print(f"   Ingested: {ingested}")
        print(f"   Chunks:   {doc['num_chunks']}")
        print(f"   Status:   {doc['status']}")


def check_new_documents():
    """Check for documents in raw directory that haven't been ingested."""
    tracker = DocumentTracker(TRACKING_DB_PATH)
    raw_path = Path(RAW_DOCS_DIR)
    
    # Get all document files using supported extensions
    supported_extensions = get_supported_extensions()
    file_patterns = [f"*{ext}" for ext in supported_extensions]
    files = []
    for pattern in file_patterns:
        files.extend(raw_path.glob(pattern))
    
    # Find new files
    new_files = []
    for file_path in files:
        if not tracker.is_document_ingested(file_path):
            new_files.append(file_path)
    
    print_header(f"NEW DOCUMENTS CHECK")
    print(f"Raw directory:      {RAW_DOCS_DIR}")
    print(f"Total files found:  {len(files)}")
    print(f"New (not ingested): {len(new_files)}")
    
    if new_files:
        print("\nFiles ready for ingestion:")
        for file_path in new_files:
            size_kb = file_path.stat().st_size / 1024
            print(f"   - {file_path.name} ({size_kb:.1f} KB)")
        print("\nRun 'python src/ingest.py' to ingest these files.")
    else:
        print("\nAll files in raw directory have been ingested.")


def verify_integrity():
    """Verify database integrity."""
    tracker = DocumentTracker(TRACKING_DB_PATH)
    
    print_header("DATABASE INTEGRITY CHECK")
    
    is_ok = tracker.verify_database_integrity()
    
    if is_ok:
        print("Database integrity: OK")
    else:
        print("Database integrity: FAILED")
        print("   Consider backing up and recreating the database.")
    
    # Check if database file exists
    db_path = Path(TRACKING_DB_PATH)
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        print(f"\nDatabase file:")
        print(f"   Path: {db_path}")
        print(f"   Size: {size_kb:.1f} KB")
    else:
        print("\nDatabase file not found (will be created on first use)")


def remove_document_by_name(filename):
    """Remove a document from tracking by filename."""
    tracker = DocumentTracker(TRACKING_DB_PATH)
    docs = tracker.list_ingested_documents()
    
    # Find matching document
    matching_docs = [d for d in docs if Path(d['file_path']).name == filename]
    
    if not matching_docs:
        print(f"Document not found: {filename}")
        return
    
    if len(matching_docs) > 1:
        print(f"Multiple documents match '{filename}':")
        for doc in matching_docs:
            print(f"   - {doc['file_path']}")
        print("Please specify the full path.")
        return
    
    doc = matching_docs[0]
    file_path = Path(doc['file_path'])
    
    print(f"Removing: {file_path.name}")
    tracker.remove_document(file_path)
    print("Document removed from tracking database.")
    print("Note: The actual file and vector embeddings are NOT deleted.")


def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Database management utility for document tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/db_manager.py --stats          Show database statistics
  python src/db_manager.py --list           List all ingested documents
  python src/db_manager.py --check          Check for new documents
  python src/db_manager.py --verify         Verify database integrity
  python src/db_manager.py --remove doc.pdf Remove document from tracking
        """
    )
    
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--list', action='store_true',
                       help='List all ingested documents')
    parser.add_argument('--check', action='store_true',
                       help='Check for new documents in raw directory')
    parser.add_argument('--verify', action='store_true',
                       help='Verify database integrity')
    parser.add_argument('--remove', type=str, metavar='FILENAME',
                       help='Remove a document from tracking by filename')
    parser.add_argument('--all', action='store_true',
                       help='Show all information')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute requested commands
    if args.all or args.stats:
        show_stats()
    
    if args.all or args.list:
        list_documents()
    
    if args.all or args.check:
        check_new_documents()
    
    if args.all or args.verify:
        verify_integrity()
    
    if args.remove:
        remove_document_by_name(args.remove)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
