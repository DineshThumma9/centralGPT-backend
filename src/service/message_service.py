import os

from fastapi import APIRouter
from llama_index.llms.groq import Groq
from loguru import logger

system_prompt = """You are a professional AI assistant that MUST follow these strict formatting rules:

    ## TEXT FORMATTING:
    - Always use proper spacing between words and sentences
    - End sentences with periods followed by ONE space
    - Use double line breaks (\\n\\n) between paragraphs
    - Never compress words together or create run-on sentences

    ## CODE FORMATTING:
    - ALWAYS wrap code in proper fenced code blocks with language specification:
      ```python
      # Your code here
      ```
    - For inline code, use single backticks: `code`
    - Never mix inline code with code blocks
    - Always include the language name after the opening ```

    ## LIST FORMATTING:
    - Use proper markdown lists with consistent spacing:
      - Bullet point 1
      - Bullet point 2

      Or numbered:
      1. First item
      2. Second item

    - Always put ONE space after the bullet/number
    - Each list item on its own line
    - Add empty line before and after lists

    ## STRUCTURE REQUIREMENTS:
    - Use proper headers: # ## ### 
    - like for Main Topic #
    - for subtopics ## 
    - Add empty lines before and after headers
    - Use **bold** and *italic* correctly use bold for keywords etc
    - For tables, use proper markdown table format

    ## RESPONSE QUALITY:
    - Be conversational but well-structured
    - Break long responses into clear paragraphs
    - Use examples when explaining concepts
    - Always proofread for proper spacing and formatting

    Remember: Consistent, readable formatting is as important as the content itself."""

rag_prompt = """ 

YOU are smart RAG Model which read content and answer user query you know all filename and dir structure of code and helpful to user



"""

EXT_TO_LANG = {
    # Python
    "py": "python",
    # JavaScript / TypeScript
    "js": "javascript",
    "ts": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
    # HTML/CSS
    "html": "html",
    "css": "css",
    # Java
    "java": "java",
    # C/C++
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    # C#
    "cs": "csharp",
    # PHP
    "php": "php",
    # Ruby
    "rb": "ruby",
    # Go
    "go": "go",
    # Rust
    "rs": "rust",
    # Kotlin
    "kt": "kotlin",
    "kts": "kotlin",
    # Swift
    "swift": "swift",
    # Shell / Bash
    "sh": "bash",
    "bash": "bash",
    # PowerShell
    "ps1": "powershell",
    # Scala
    "scala": "scala",
    # Dart
    "dart": "dart",
    # R
    "r": "r",
    # Julia
    "jl": "julia",
    # SQL
    "sql": "sql",
    # YAML/JSON config
    "yaml": "yaml",
    "yml": "yaml",
    "json": "json",
    # Markdown / Docs
    "md": "markdown",
    "rst": "markdown",
    # Text
    "txt": "text",
    # Docker / CI
    "dockerfile": "docker",
    "env": "text",
    # Make
    "makefile": "make",
}

router = APIRouter()


async def session_title_gen(query):
    try:
        title_gen = Groq(model="compound-beta", api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""Generate a concise, descriptive title (maximum 6 words) for a chat session based on this first message: "{query}"

Rules:
- Maximum 6 words
- No quotes or special characters
- Describe the main topic or question
- Be specific but concise

Title:"""

        session_title = await title_gen.acomplete(prompt)

        result = session_title.content if hasattr(session_title, 'content') else str(session_title)

        # Clean and validate the result
        if result:
            # Remove quotes and clean up
            cleaned_title = result.strip().strip('"').strip("'")
            if cleaned_title and len(cleaned_title) > 0:
                return cleaned_title

        return "New Chat"

    except Exception as e:
        logger.error(f"Error in session_title_gen: {e}")
        return "New Chat"
