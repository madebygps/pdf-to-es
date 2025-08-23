from pathlib import Path
import os
import pymupdf4llm
import argparse
import time
import csv
import json
from typing import Tuple
from markdown_translator import MarkdownTranslator


def run_translation(translator: MarkdownTranslator, md_text: str, provider: str, out_dir: Path) -> Tuple[str, float, int, float]:
    start = time.time()
    try:
        out, tokens = translator.translate_full_markdown(md_text, provider=provider)
    except RuntimeError as exc:
        # Write a small sentinel file so the compare runner can read a file and
        # report file length without crashing.
        skip_path = out_dir / f"perks_{provider}_es.SKIPPED.txt"
        skip_path.write_text(f"SKIPPED: {exc}\n", encoding="utf-8")
        return (str(skip_path), 0.0, 0, 0.0)
    elapsed = time.time() - start
    filename = out_dir / f"perks_{provider}_es.md"
    filename.write_text(out, encoding="utf-8")
    cost = translator.compute_cost(provider, tokens)
    return str(filename), elapsed, tokens, cost


def main(pdf_path: str = "pdfs/perks.pdf", compare: bool = False):
    out_dir = Path("translated")
    out_dir.mkdir(exist_ok=True)

    print("Extracting markdown from PDF...")
    md_text = pymupdf4llm.to_markdown(pdf_path)

    Path("output.md").write_text(md_text, encoding="utf-8")
    print(f"English markdown saved: {len(md_text)} characters")

    translator = MarkdownTranslator()

    if compare:
        providers = ["openai", "anthropic", "mistral"]
        # Require Mistral key if Mistral is included in compare to avoid silent skips
        if "mistral" in providers and not os.getenv("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY is not set. Set it to include Mistral in --compare or run with --providers to exclude it.")
            return
        stats = []
        for p in providers:
            print(f"\n=== {p.upper()} Translation ===")
            fname, elapsed, tokens, cost = run_translation(translator, md_text, p, out_dir)
            # fname may be a SKIPPED sentinel file path
            fname_path = Path(fname)
            try:
                content = fname_path.read_text(encoding='utf-8')
                chars = len(content)
            except Exception:
                chars = 0

            skipped = fname_path.name.endswith('.SKIPPED.txt')
            print(f"Saved: {fname} (took {elapsed:.1f}s, {chars} chars, {tokens} tokens, ${cost:.4f})")
            stats.append({
                "provider": p,
                "filename": str(fname_path),
                "elapsed_seconds": float(elapsed),
                "chars": int(chars),
                "tokens": int(tokens),
                "cost_usd": float(cost),
                "skipped": bool(skipped),
            })

        # Write a CSV and JSON summary
        csv_path = out_dir / "compare_stats.csv"
        json_path = out_dir / "compare_stats.json"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["provider", "filename", "elapsed_seconds", "chars", "tokens", "cost_usd", "skipped"])
            writer.writeheader()
            for row in stats:
                writer.writerow(row)

        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

        print(f"\n=== Compare Complete ===\nSummary saved: {csv_path} and {json_path}")
        return

    # Default behavior: run the two providers we used before
    print("\n=== OpenAI Translation ===")
    fname, elapsed, tokens, cost = run_translation(translator, md_text, "openai", out_dir)
    print(f"Saved: {fname} (took {elapsed:.1f}s, {tokens} tokens, ${cost:.4f})")

    print("\n=== Anthropic Translation ===")
    fname, elapsed, tokens, cost = run_translation(translator, md_text, "anthropic", out_dir)
    print(f"Saved: {fname} (took {elapsed:.1f}s, {tokens} tokens, ${cost:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", default="pdfs/perks.pdf", help="PDF path to translate")
    parser.add_argument("--compare", action="store_true", help="Run translations across openai, anthropic and mistral and save outputs")
    args = parser.parse_args()
    main(args.pdf, compare=args.compare)