from pathlib import Path
import pymupdf4llm
import argparse
from typing import Optional, List
from markdown_translator import MarkdownTranslator


def process_pdf(pdf_path: Path, translator: MarkdownTranslator, output_dir: Path) -> None:
    """Process a single PDF file: extract markdown and translate to Spanish markdown."""
    print(f"\n=== Processing PDF: {pdf_path.name} ===")
    
    # Extract markdown from PDF
    print("Extracting markdown from PDF...")
    
    # For larger files, use more careful extraction settings
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 0.3:  # Files larger than 300KB
        print(f"Large PDF detected ({file_size_mb:.2f} MB) - using careful extraction")
        # Use page-by-page extraction for better order preservation
        md_text = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, write_images=False)
    else:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
    
    print(f"Extracted {len(md_text)} characters from PDF")

    # Save original extracted markdown for debugging
    debug_path = output_dir / f"{pdf_path.stem}_original_extracted.md"
    debug_path.write_text(md_text, encoding="utf-8")
    print(f"Original extraction saved for debugging: {debug_path}")

    # Translate to Spanish
    print("Translating to Spanish...")
    translated_text, _ = translator.translate_full_markdown(md_text, provider="openai")
    print("Translation completed")

    # Save translated markdown
    output_path = output_dir / f"{pdf_path.stem}_spanish.md"
    output_path.write_text(translated_text, encoding="utf-8")
    print(f"Spanish markdown saved: {output_path}")


def process_markdown(md_path: Path, translator: MarkdownTranslator, output_dir: Path) -> None:
    """Process a single markdown file: translate to Spanish markdown."""
    print(f"\n=== Processing Markdown: {md_path.name} ===")
    
    # Read markdown content
    print("Reading markdown file...")
    md_text = md_path.read_text(encoding="utf-8")
    print(f"Read {len(md_text)} characters from markdown file")

    # Translate to Spanish
    print("Translating to Spanish...")
    translated_text, _ = translator.translate_full_markdown(md_text, provider="openai")
    print("Translation completed")

    # Save translated markdown
    output_path = output_dir / f"{md_path.stem}_spanish.md"
    output_path.write_text(translated_text, encoding="utf-8")
    print(f"Spanish markdown saved: {output_path}")


def find_files_to_process(input_dir: Path) -> tuple[List[Path], List[Path]]:
    """Find all PDF and markdown files in the input directory, sorted by file size (smallest first)."""
    pdf_files = list(input_dir.glob("*.pdf"))
    md_files = list(input_dir.glob("*.md"))
    
    # Sort files by size (smallest first)
    pdf_files.sort(key=lambda f: f.stat().st_size)
    md_files.sort(key=lambda f: f.stat().st_size)
    
    return pdf_files, md_files


def main(input_dir: str = "pdfs", output_dir: Optional[str] = None):
    """
    Process all PDF and markdown files in the input directory, translate to Spanish using OpenAI GPT-5, and save as markdown files.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Set default output directory if not provided
    if output_dir is None:
        output_path = Path("translated_markdown")
    else:
        output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Find all files to process
    pdf_files, md_files = find_files_to_process(input_path)
    
    total_files = len(pdf_files) + len(md_files)
    if total_files == 0:
        print(f"No PDF or markdown files found in '{input_dir}'")
        return
    
    print(f"Found {len(pdf_files)} PDF files and {len(md_files)} markdown files to process")

    
    # Show files in processing order (smallest to largest)
    if pdf_files:
        print("\nPDF files (smallest to largest):")
        for pdf in pdf_files:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"  {pdf.name} ({size_mb:.2f} MB)")
    
    if md_files:
        print("\nMarkdown files (smallest to largest):")
        for md in md_files:
            size_kb = md.stat().st_size / 1024
            print(f"  {md.name} ({size_kb:.1f} KB)")
    
    print()
    
    # Initialize translator
    translator = MarkdownTranslator()
    
    # Process all PDF files
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{total_files}]", end=" ")
        try:
            process_pdf(pdf_file, translator, output_path)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    
    # Process all markdown files
    for i, md_file in enumerate(md_files, len(pdf_files) + 1):
        print(f"\n[{i}/{total_files}]", end=" ")
        try:
            process_markdown(md_file, translator, output_path)
        except Exception as e:
            print(f"Error processing {md_file.name}: {e}")
    
    print("\n=== Processing Complete ===")
    print(f"All translated markdown files saved to: {output_path}")
    print("\nNext steps:")
    print("1. Review the translated markdown files for accuracy")
    print("2. Make any necessary edits")
    print("3. Use a separate tool to convert to PDF when ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all PDFs and markdown files to Spanish markdown using OpenAI GPT-5")
    parser.add_argument("input_dir", nargs="?", default="pdfs", 
                       help="Input directory containing PDF and markdown files (default: pdfs)")
    parser.add_argument("-o", "--output", help="Output directory for Spanish markdown files (default: translated_markdown)")
    args = parser.parse_args()
    main(args.input_dir, args.output)