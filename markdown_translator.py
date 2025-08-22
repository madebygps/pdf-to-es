"""
Markdown-based PDF translation system.
Translate markdown first, then generate PDF from Spanish markdown.
"""

import os
import re
from typing import List, Tuple
import openai
import anthropic
from dotenv import load_dotenv

load_dotenv()

class MarkdownTranslator:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    def split_markdown_for_translation(self, markdown_text: str, max_chunk_chars: int = 2000) -> List[Tuple[str, str]]:
        """
        Split markdown into logical chunks for translation while preserving structure.
        Returns list of (chunk_type, content) tuples.
        """
        chunks = []
        current_chunk = ""
        current_type = "body"
        
        lines = markdown_text.split('\n')
        
        for line in lines:
            line_with_newline = line + '\n'
            
            # Detect different markdown elements
            if line.startswith('# '):
                chunk_type = "title"
            elif line.startswith('## '):
                chunk_type = "heading"
            elif line.startswith('### '):
                chunk_type = "subheading"
            elif line.startswith('- ') or line.startswith('* ') or re.match(r'^\d+\. ', line):
                chunk_type = "list_item"
            elif line.startswith('**') and line.endswith('**'):
                chunk_type = "bold"
            elif line.strip() == "":
                chunk_type = "whitespace"
            else:
                chunk_type = "body"
            
            # If we're starting a new section or chunk is getting too large
            if (chunk_type in ["title", "heading", "subheading"] and current_chunk.strip()) or \
               (len(current_chunk) + len(line_with_newline) > max_chunk_chars and current_chunk.strip()):
                
                if current_chunk.strip():
                    chunks.append((current_type, current_chunk.strip()))
                current_chunk = line_with_newline
                current_type = chunk_type
            else:
                current_chunk += line_with_newline
                # Update type for the chunk (prefer more specific types)
                if chunk_type in ["title", "heading", "subheading"] and current_type == "body":
                    current_type = chunk_type
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append((current_type, current_chunk.strip()))
        
        return chunks
    
    def translate_markdown_chunk(self, chunk_content: str, chunk_type: str, provider: str = "openai") -> str:
        """Translate a single markdown chunk with type-aware prompts"""
        
        if provider == "openai":
            return self._translate_chunk_openai(chunk_content, chunk_type)
        else:
            return self._translate_chunk_anthropic(chunk_content, chunk_type)
    
    def _translate_chunk_openai(self, content: str, chunk_type: str) -> str:
        """Translate using OpenAI with markdown awareness"""
        
        type_guidance = {
            "title": "This is a main title - keep it concise and impactful in Spanish",
            "heading": "This is a section heading - maintain clarity and professional tone",
            "subheading": "This is a subsection heading - keep it descriptive but brief",
            "list_item": "These are list items - maintain parallel structure and conciseness",
            "bold": "This is emphasized text - preserve the emphasis and meaning",
            "body": "This is body text - use natural, professional Spanish"
        }
        
        guidance = type_guidance.get(chunk_type, "Translate this text naturally to Spanish")
        
        system_prompt = """You are a professional translator specializing in corporate documents. 
        Translate English markdown to Spanish while:
        1. PRESERVING ALL markdown formatting (# ## ### - * ** etc.)
        2. Keeping translations concise and professional
        3. Using natural Spanish business terminology
        4. Maintaining document structure exactly"""
        
        user_prompt = f"""{guidance}

CRITICAL: Preserve ALL markdown formatting exactly. Return ONLY the Spanish translation.

English markdown:
{content}

Spanish translation:"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content or ""
    
    def _translate_chunk_anthropic(self, content: str, chunk_type: str) -> str:
        """Translate using Anthropic with markdown awareness"""
        
        type_guidance = {
            "title": "This is a main title - keep it concise and impactful",
            "heading": "This is a section heading - maintain clarity and professional tone", 
            "subheading": "This is a subsection heading - keep it descriptive but brief",
            "list_item": "These are list items - maintain parallel structure and conciseness",
            "bold": "This is emphasized text - preserve the emphasis and meaning",
            "body": "This is body text - use natural, professional Spanish"
        }
        
        guidance = type_guidance.get(chunk_type, "Translate this text naturally to Spanish")
        
        prompt = f"""Translate this English markdown to Spanish. {guidance}

CRITICAL REQUIREMENTS:
1. Preserve ALL markdown formatting exactly (# ## ### - * ** etc.)
2. Return ONLY the Spanish translation - no analysis or explanations
3. Keep translations concise and professional
4. Use natural Spanish business terminology

English markdown:
{content}

Spanish translation:"""
        
        response = self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Type checking workaround - access text dynamically
        first_content = response.content[0]
        return getattr(first_content, 'text', str(first_content))
    
    def translate_full_markdown(self, markdown_text: str, provider: str = "openai") -> str:
        """Translate entire markdown document maintaining structure"""
        
        print(f"Splitting markdown into chunks for {provider} translation...")
        chunks = self.split_markdown_for_translation(markdown_text)
        print(f"Created {len(chunks)} chunks")
        
        translated_chunks = []
        
        for i, (chunk_type, content) in enumerate(chunks):
            print(f"Translating chunk {i+1}/{len(chunks)} (type: {chunk_type})")
            
            translated = self.translate_markdown_chunk(content, chunk_type, provider)
            translated_chunks.append(translated)
        
        # Join with double newlines to maintain document structure
        return '\n\n'.join(translated_chunks)
    
    def markdown_to_pdf(self, markdown_text: str, output_path: str):
        """Convert markdown to PDF using a markdown-to-PDF library"""
        try:
            # Try using markdown-pdf (requires npm install -g markdown-pdf)
            import subprocess
            import tempfile
            import os
            
            # Create temporary markdown file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_md:
                temp_md.write(markdown_text)
                temp_md_path = temp_md.name
            
            try:
                # Try markdown-pdf first
                result = subprocess.run([
                    'markdown-pdf', 
                    temp_md_path, 
                    '-o', output_path
                ], capture_output=True, text=True, check=True)
                print(f"PDF generated successfully: {output_path}")
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: Try pandoc
                try:
                    result = subprocess.run([
                        'pandoc', 
                        temp_md_path, 
                        '-o', output_path,
                        '--pdf-engine=xelatex'
                    ], capture_output=True, text=True, check=True)
                    print(f"PDF generated successfully with pandoc: {output_path}")
                    
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Final fallback: Use Python libraries
                    self._markdown_to_pdf_python(markdown_text, output_path)
                    
            finally:
                # Clean up temp file
                os.unlink(temp_md_path)
                
        except Exception as e:
            print(f"Error generating PDF: {e}")
            print("Falling back to Python-based PDF generation...")
            self._markdown_to_pdf_python(markdown_text, output_path)
    
    def _markdown_to_pdf_python(self, markdown_text: str, output_path: str):
        """Fallback PDF generation using Python libraries"""
        try:
            # Try using markdown + weasyprint
            import markdown
            from weasyprint import HTML, CSS
            
            # Convert markdown to HTML
            html_content = markdown.markdown(markdown_text, extensions=['tables', 'fenced_code'])
            
            # Basic CSS for professional appearance
            css_style = """
            body { 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                max-width: 800px; 
                margin: 40px auto; 
                padding: 20px;
                color: #333;
            }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
            h3 { color: #7f8c8d; }
            ul, ol { padding-left: 30px; }
            li { margin-bottom: 5px; }
            p { margin-bottom: 15px; }
            """
            
            # Generate PDF
            html_doc = f"<html><head><meta charset='utf-8'></head><body>{html_content}</body></html>"
            HTML(string=html_doc).write_pdf(output_path, stylesheets=[CSS(string=css_style)])
            print(f"PDF generated with weasyprint: {output_path}")
            
        except ImportError:
            print("Required packages not available. Installing...")
            import subprocess
            subprocess.run(['uv', 'add', 'weasyprint', 'markdown'], check=True)
            # Retry
            self._markdown_to_pdf_python(markdown_text, output_path)
            
        except Exception as e:
            print(f"Error in Python PDF generation: {e}")
            # Save as HTML as final fallback
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                import markdown
                html_content = markdown.markdown(markdown_text, extensions=['tables', 'fenced_code'])
                f.write(f"<html><head><meta charset='utf-8'></head><body>{html_content}</body></html>")
            print(f"Saved as HTML instead: {html_path}")