from pygments import highlight
from pygments.lexers import (
    PythonLexer, BashLexer, CppLexer,
    HtmlLexer, CssLexer, JavascriptLexer,
    get_lexer_by_name
)
from pygments.formatters import HtmlFormatter
import re


def format_llm_response(response_text):
    # Process code blocks first
    def process_code_block(match):
        language = match.group(1).lower() or 'text'
        code = match.group(2)

        try:
            if language == 'python':
                lexer = PythonLexer()
            elif language == 'bash':
                lexer = BashLexer()
            elif language == 'cpp':
                lexer = CppLexer()
            elif language == 'html':
                lexer = HtmlLexer()
            elif language == 'css':
                lexer = CssLexer()
            elif language == 'javascript':
                lexer = JavascriptLexer()
            else:
                lexer = get_lexer_by_name(language, stripall=True)

            return highlight(code, lexer, HtmlFormatter(
                noclasses=True,
                style='sas',
                linenos=False
            ))
        except:
            return f'<pre><code>{code}</code></pre>'

    # Handle ```language\ncode\n``` blocks
    response_text = re.sub(
        r'```(\w*)\n(.*?)```',
        process_code_block,
        response_text,
        flags=re.DOTALL
    )

    # Basic formatting for the rest
    formatted = response_text.replace('**', '<strong>').replace('**', '</strong>')
    formatted = formatted.replace('*', '<em>').replace('*', '</em>')
    formatted = formatted.replace('\n\n', '</p><p>')

    return f'<div class="llama-response">{formatted}</div>'