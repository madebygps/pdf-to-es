#!/usr/bin/env python3
"""
Helper script to convert reviewed Spanish markdown files to PDFs.
Use this after reviewing and editing the translated markdown files.
"""

import argparse
from pathlib import Path
from typing import List
from markdown_translator import MarkdownTranslator


def find_markdown_files(input_dir: Path) -> List[Path]:
    """Find all markdown files in the input directory, sorted by size."""
    md_files = list(input_dir.glob("*.md"))
    md_files.sort(key=lambda f: f.stat().st_size)
    return md_files


def convert_markdown_to_pdf(md_path: Path, output_dir: Path) -> None:
    """Convert a single markdown file to PDF."""
    print(f"\n=== Converting: {md_path.name} ===")
    
    # Read markdown content
    print("Reading markdown file...")
    md_text = md_path.read_text(encoding="utf-8")
    print(f"Read {len(md_text)} characters from markdown file")

    # Create PDF
    output_path = output_dir / f"{md_path.stem}.pdf"
    print(f"Creating PDF: {output_path}")
    
    translator = MarkdownTranslator()
    translator.markdown_to_pdf(md_text, str(output_path))
    print(f"PDF created successfully: {output_path}")


def main(input_dir: str = "translated_markdown", output_dir: str = "final_pdfs"):
    """
    Convert all reviewed Spanish markdown files to PDFs.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        print("Make sure you've run the translation script first!")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Find all markdown files
    md_files = find_markdown_files(input_path)
    
    if not md_files:
        print(f"No markdown files found in '{input_dir}'")
        return
    
    print(f"\nFound {len(md_files)} markdown files to convert:")
    for md in md_files:
        size_kb = md.stat().st_size / 1024
        print(f"  {md.name} ({size_kb:.1f} KB)")
    
    print()
    
    # Convert all files
    for i, md_file in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}]", end=" ")
        try:
            convert_markdown_to_pdf(md_file, output_path)
        except Exception as e:
            print(f"Error converting {md_file.name}: {e}")
    
    print("\n=== Conversion Complete ===")
    print(f"All PDFs saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert reviewed Spanish markdown files to PDFs")
    parser.add_argument("input_dir", nargs="?", default="translated_markdown", 
                       help="Input directory containing Spanish markdown files (default: translated_markdown)")
    parser.add_argument("-o", "--output", default="final_pdfs",
                       help="Output directory for PDF files (default: final_pdfs)")
    args = parser.parse_args()
    main(args.input_dir, args.output)