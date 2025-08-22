import pathlib
import pymupdf4llm
from markdown_translator import MarkdownTranslator

def main():
    """
    Main PDF translation workflow using the markdown-first approach.
    
    This approach:
    1. Extracts markdown from PDF
    2. Translates markdown with semantic chunking
    3. Generates professional PDFs from Spanish markdown
    """
    # Create output directories
    pathlib.Path("translated").mkdir(exist_ok=True)
    
    # Extract markdown from PDF
    print("Extracting markdown from PDF...")
    md_text = pymupdf4llm.to_markdown("pdfs/perks.pdf")
    
    # Save original English markdown
    pathlib.Path("output.md").write_bytes(md_text.encode())
    print(f"English markdown saved: {len(md_text)} characters")
    
    # Initialize markdown translator
    translator = MarkdownTranslator()
    
    # Translate with OpenAI
    print("\n=== OpenAI Translation ===")
    spanish_md_openai = translator.translate_full_markdown(md_text, provider="openai")
    
    # Save Spanish markdown
    spanish_md_path_openai = "translated/perks_openai_es.md"
    pathlib.Path(spanish_md_path_openai).write_text(spanish_md_openai, encoding='utf-8')
    print(f"Spanish markdown saved: {spanish_md_path_openai}")
    
    # Generate PDF from Spanish markdown
    print("Generating PDF from Spanish markdown...")
    translator.markdown_to_pdf(spanish_md_openai, "translated/perks_openai_es.pdf")
    
    # Translate with Anthropic
    print("\n=== Anthropic Translation ===")
    spanish_md_anthropic = translator.translate_full_markdown(md_text, provider="anthropic")
    
    # Save Spanish markdown
    spanish_md_path_anthropic = "translated/perks_anthropic_es.md"
    pathlib.Path(spanish_md_path_anthropic).write_text(spanish_md_anthropic, encoding='utf-8')
    print(f"Spanish markdown saved: {spanish_md_path_anthropic}")
    
    # Generate PDF from Spanish markdown
    print("Generating PDF from Spanish markdown...")
    translator.markdown_to_pdf(spanish_md_anthropic, "translated/perks_anthropic_es.pdf")
    
    print("\n=== Translation Complete! ===")
    print("Generated files:")
    print("- output.md (original English markdown)")
    print("- translated/perks_openai_es.md (OpenAI Spanish markdown)")
    print("- translated/perks_openai_es.pdf (OpenAI Spanish PDF)")
    print("- translated/perks_anthropic_es.md (Anthropic Spanish markdown)")
    print("- translated/perks_anthropic_es.pdf (Anthropic Spanish PDF)")
    
    # Show comparison stats
    print("\nCharacter counts:")
    print(f"  English: {len(md_text):,} chars")
    print(f"  Spanish (OpenAI): {len(spanish_md_openai):,} chars ({len(spanish_md_openai)/len(md_text)*100:.1f}%)")
    print(f"  Spanish (Anthropic): {len(spanish_md_anthropic):,} chars ({len(spanish_md_anthropic)/len(md_text)*100:.1f}%)")

if __name__ == "__main__":
    main()