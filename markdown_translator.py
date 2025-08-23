"""markdown_translator
Clean, modular translator for markdown-based PDF translation.

This module provides a single `MarkdownTranslator` class and small helpers
to keep prompt loading, chunking, translation and PDF-generation responsibilities
separated and easy to test.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Any

import openai
import anthropic
from dotenv import load_dotenv

load_dotenv()

TYPE_GUIDANCE = {
    "title": "This is a main title - keep it concise and impactful in Spanish",
    "heading": "This is a section heading - maintain clarity and professional tone",
    "subheading": "This is a subsection heading - keep it descriptive but brief",
    "list_item": "These are list items - maintain parallel structure and conciseness",
    "bold": "This is emphasized text - preserve the emphasis and meaning",
    "body": "This is body text - use natural, professional Spanish",
}


def load_prompt_template(path: Optional[Path]) -> Optional[str]:
    """Load an external prompt template from disk, if available.

    Returns the template string or None if it cannot be read.
    """
    if not path:
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return None


class MarkdownTranslator:
    """High-level translator that keeps prompt-loading, chunking and provider
    integration well separated for easier testing and extension.
    """

    def __init__(self, *, prompt_path: Optional[Path] = None):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Determine external prompt template path and cache contents
        default_prompt_path = Path(__file__).parent / "prompts" / "translation_prompt.txt"
        self._prompt_path = Path(prompt_path) if prompt_path is not None else default_prompt_path
        self._template = load_prompt_template(self._prompt_path)
        # Model selection via env vars (allow overriding defaults)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        # Shared decoding settings
        try:
            self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        except Exception:
            self.temperature = 0.2
        # Shared system prompt used across providers for parity
        self.system_prompt = (
            "You are a professional translator specializing in corporate documents.\n"
            "Translate English markdown to Spanish while preserving formatting and style."
        )

    def _build_translation_prompt(self, content: str, chunk_type: str) -> str:
        """Construct the prompt to send to the provider. Uses an external template
        when available, and falls back to a concise inline prompt otherwise.
        """
        guidance = TYPE_GUIDANCE.get(chunk_type, "Translate this text naturally to Spanish")

        if self._template:
            try:
                return self._template.format(guidance=guidance, content=content)
            except Exception:
                # If the external template is malformed, fall back to inline.
                pass

        # Minimal inline fallback to keep functionality if template is missing
        return f"{guidance}\n\nEnglish markdown:\n{content}\n\nSpanish translation:"

    # -- Token estimation & cost helpers ---------------------------------
    def _estimate_tokens(self, *texts: str) -> int:
        """Rudimentary token estimator used when provider doesn't return usage.

        Uses a simple chars->tokens heuristic (~4 chars/token). This is
        intentionally conservative for quick cost estimates.
        """
        total_chars = sum(len(t or "") for t in texts)
        return max(1, int(total_chars / 4))

    def compute_cost(self, provider: str, tokens: int) -> float:
        """Compute cost in USD for given provider and token count.

                Reads per-1k-token rates from environment:
                    OPENAI_COST_PER_1K, ANTHROPIC_COST_PER_1K
        Defaults to 0.0 if not set.
        """
        def _get_rate(env_name: str) -> float:
            try:
                return float(os.getenv(env_name, "0.0"))
            except Exception:
                return 0.0

        if provider == "openai":
            rate = _get_rate("OPENAI_COST_PER_1K")
        elif provider == "anthropic":
            rate = _get_rate("ANTHROPIC_COST_PER_1K")
        else:
            rate = 0.0

        return (tokens / 1000.0) * rate

    # -- Chunking ---------------------------------------------------------
    def split_markdown_for_translation(self, markdown_text: str, max_chunk_chars: int = 2000) -> List[Tuple[str, str]]:
        """Split markdown into logical chunks while preserving structure.

        Returns a list of (chunk_type, content) tuples.
        """
        chunks: List[Tuple[str, str]] = []
        current_chunk = []  # list[str] for efficient concat
        current_type = "body"

        def flush_current():
            nonlocal current_chunk, current_type
            if current_chunk:
                chunks.append((current_type, "".join(current_chunk).strip()))
                current_chunk = []

        for line in markdown_text.splitlines():
            line_with_newline = line + "\n"

            if line.startswith("# "):
                chunk_type = "title"
            elif line.startswith("## "):
                chunk_type = "heading"
            elif line.startswith("### "):
                chunk_type = "subheading"
            elif line.startswith("- ") or line.startswith("* ") or re.match(r"^\d+\. ", line):
                chunk_type = "list_item"
            elif line.startswith("**") and line.endswith("**"):
                chunk_type = "bold"
            elif line.strip() == "":
                chunk_type = "whitespace"
            else:
                chunk_type = "body"

            # if new structural element and current has content -> flush
            if (chunk_type in {"title", "heading", "subheading"} and current_chunk) or \
               (sum(len(s) for s in current_chunk) + len(line_with_newline) > max_chunk_chars and current_chunk):
                flush_current()
                current_chunk = [line_with_newline]
                current_type = chunk_type
            else:
                current_chunk.append(line_with_newline)
                if chunk_type in {"title", "heading", "subheading"} and current_type == "body":
                    current_type = chunk_type

        flush_current()
        return chunks

    # -- Provider dispatch ------------------------------------------------
    def translate_markdown_chunk(self, chunk_content: str, chunk_type: str, provider: str = "openai") -> Tuple[str, int]:
        """Translate a single chunk using the selected provider.

        Returns a tuple of (translated_text, tokens_used).
        """
        providers = {
            "openai": self._translate_chunk_openai,
            "anthropic": self._translate_chunk_anthropic,
        }
        translate_fn = providers.get(provider)
        if translate_fn is None:
            raise ValueError(f"Unsupported provider: {provider}")
        return translate_fn(chunk_content, chunk_type)

    def _translate_chunk_openai(self, content: str, chunk_type: str) -> Tuple[str, int]:
        """Translate a chunk using OpenAI."""
        user_prompt = self._build_translation_prompt(content, chunk_type)
        # Pass temperature to make decoding settings consistent across providers
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}],
            
        )

        # Try common response shapes safely
        # Try to return text and usage if available
        try:
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            tok_count = None
            if usage is not None:
                if isinstance(usage, dict):
                    tok_count = usage.get("total_tokens")
                else:
                    tok_count = getattr(usage, "total_tokens", None)
            if tok_count:
                return text, int(tok_count)
        except Exception:
            pass

        # Fallback to estimate tokens
        return (getattr(getattr(response.choices[0], "message", ""), "content", str(response)), self._estimate_tokens(content))

    def _translate_chunk_anthropic(self, content: str, chunk_type: str) -> Tuple[str, int]:
        """Translate a chunk using Anthropic."""
        prompt = self._build_translation_prompt(content, chunk_type)
        # Use the same system+user prompt pattern as OpenAI for parity
        # Many Anthropics SDKs expect a single prompt string instead of a system/user
        # role separation. Combine system and user prompts to ensure parity with
        # other providers while remaining compatible with different SDK versions.
        combined = f"{self.system_prompt}\n\n{prompt}"
        # Try the completions API shape first (common in newer SDKs), then
        # fall back to the messages API shape for older SDKs.
        response = None
        try:
            response = self.anthropic_client.completions.create(
                model=self.anthropic_model,
                prompt=combined,
                max_tokens_to_sample=2000,
                temperature=self.temperature,
            )
        except Exception:
            try:
                response = self.anthropic_client.messages.create(
                    model=self.anthropic_model,
                    messages=[{"role": "user", "content": combined}],
                    max_tokens=2000,
                    temperature=self.temperature,
                )
            except Exception:
                # If both attempts fail, raise so caller can handle (and skip provider)
                raise

        # Robustly extract text from a variety of response shapes. The Anthropic
        # SDK may return objects with different attributes depending on which
        # resource was used (completions vs messages), so attempt multiple
        # common accesses.
        try:
            # If response is a mapping-like object
            resp_dict: dict[str, Any] = {}
            if hasattr(response, "to_dict"):
                try:
                    resp_dict = response.to_dict()
                except Exception:
                    resp_dict = {}
            elif isinstance(response, dict):
                resp_dict = response
            else:
                # Try to coerce to dict via vars()
                try:
                    candidate = vars(response)
                    if isinstance(candidate, dict):
                        resp_dict = candidate
                except Exception:
                    resp_dict = {}

            # Check a few possible keys and common SDK shapes
            text: Any = None

            # 1) If the SDK returned a 'content' field (Messages API), it's usually
            #    a list of TextBlock-like objects. Extract .text from each block.
            if hasattr(response, "content"):
                try:
                    content_blocks = getattr(response, "content")
                    if isinstance(content_blocks, list) and content_blocks:
                        pieces: list[str] = []
                        for blk in content_blocks:
                            if isinstance(blk, dict):
                                btext = blk.get("text") or blk.get("content")
                                if btext:
                                    pieces.append(str(btext))
                            else:
                                # object from SDK; try attribute access
                                btext = getattr(blk, "text", None)
                                if btext:
                                    pieces.append(str(btext))
                                else:
                                    pieces.append(str(blk))
                        text = "\n\n".join(pieces)
                except Exception:
                    text = None

            # 2) If not messages-style, inspect resp_dict for completions/messages shapes
            if not text and isinstance(resp_dict, dict):
                if "completion" in resp_dict and isinstance(resp_dict["completion"], str):
                    text = resp_dict["completion"]
                elif "text" in resp_dict and isinstance(resp_dict["text"], str):
                    text = resp_dict["text"]
                elif "choices" in resp_dict and isinstance(resp_dict["choices"], list) and len(resp_dict["choices"]) > 0:
                    first = resp_dict["choices"][0]
                    if isinstance(first, dict):
                        text = first.get("text") or first.get("completion") or first.get("content")

            # 3) Fallback to attribute access
            if not text:
                if hasattr(response, "completion"):
                    text = getattr(response, "completion")
                elif hasattr(response, "text"):
                    text = getattr(response, "text")

            # Final fallback
            if text is None:
                text = str(response)

            # Try token usage extraction
            usage = None
            if isinstance(resp_dict, dict):
                usage = resp_dict.get("usage")
            if usage is None and hasattr(response, "usage"):
                usage = getattr(response, "usage")

            tok_count = None
            if usage is not None:
                if isinstance(usage, dict):
                    tok_count = usage.get("total_tokens") or usage.get("total-token")
                else:
                    tok_count = getattr(usage, "total_tokens", None)

            # Ensure we return a str for text
            text_str = str(text)
            if tok_count:
                return text_str, int(tok_count)
            return text_str, self._estimate_tokens(content)
        except Exception:
            return str(response), self._estimate_tokens(content)

    # Support removed
    # -- High-level workflow ------------------------------------------------
    def translate_full_markdown(self, markdown_text: str, provider: str = "openai") -> Tuple[str, int]:
        """Translate an entire markdown document while preserving structure.

        Returns a tuple (translated_text, total_tokens) for the whole document.
        """
        print(f"Splitting markdown into chunks for {provider} translation...")
        chunks = self.split_markdown_for_translation(markdown_text)
        print(f"Created {len(chunks)} chunks")

        translated_chunks: List[str] = []
        total_tokens = 0
        for i, (chunk_type, content) in enumerate(chunks):
            print(f"Translating chunk {i+1}/{len(chunks)} (type: {chunk_type})")
            text, tokens = self.translate_markdown_chunk(content, chunk_type, provider)
            translated_chunks.append(text)
            total_tokens += tokens

        return "\n\n".join(translated_chunks), total_tokens

    # -- PDF generation -----------------------------------------------------
    def markdown_to_pdf(self, markdown_text: str, output_path: str):
        """Convert markdown to PDF using available system tooling or Python fallbacks."""
        try:
            # Try markdown-pdf (npm) then pandoc, then Python library fallback
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_md:
                temp_md.write(markdown_text)
                temp_md_path = temp_md.name

            try:
                subprocess.run(["markdown-pdf", temp_md_path, "-o", output_path], capture_output=True, text=True, check=True)
                print(f"PDF generated successfully: {output_path}")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            try:
                subprocess.run(["pandoc", temp_md_path, "-o", output_path, "--pdf-engine=xelatex"], capture_output=True, text=True, check=True)
                print(f"PDF generated successfully with pandoc: {output_path}")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # final fallback
            self._markdown_to_pdf_python(markdown_text, output_path)

        finally:
            try:
                os.unlink(temp_md_path)
            except Exception:
                pass

    def _markdown_to_pdf_python(self, markdown_text: str, output_path: str):
        """Generate a PDF using markdown -> HTML -> PDF via weasyprint or save HTML as final fallback."""
        try:
            import markdown
            from weasyprint import HTML, CSS

            html_content = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])

            css_style = """
            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px; color: #333; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
            h3 { color: #7f8c8d; }
            ul, ol { padding-left: 30px; }
            li { margin-bottom: 5px; }
            p { margin-bottom: 15px; }
            """

            html_doc = f"<html><head><meta charset='utf-8'></head><body>{html_content}</body></html>"
            HTML(string=html_doc).write_pdf(output_path, stylesheets=[CSS(string=css_style)])
            print(f"PDF generated with weasyprint: {output_path}")

        except ImportError:
            print("Required packages not available. Installing...")
            subprocess.run(["uv", "add", "weasyprint", "markdown"], check=True)
            # Retry once
            self._markdown_to_pdf_python(markdown_text, output_path)

        except Exception as exc:
            print(f"Error in Python PDF generation: {exc}")
            html_path = output_path.replace(".pdf", ".html")
            with open(html_path, "w", encoding="utf-8") as f:
                import markdown as _md

                f.write(f"<html><head><meta charset='utf-8'></head><body>{_md.markdown(markdown_text, extensions=['tables','fenced_code'])}</body></html>")
            print(f"Saved as HTML instead: {html_path}")