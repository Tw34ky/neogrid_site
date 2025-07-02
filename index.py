from flask import Flask, render_template, send_from_directory, abort
import os
from filters import register_filters
from werkzeug.utils import redirect
from flask import request
import docx
from PyPDF2 import PdfReader
import io


app = Flask(__name__)

# Register custom filters
register_filters(app)

# Configuration - set your base directory here
BASE_DIR = os.path.abspath(os.path.expanduser('~'))  # Ensure absolute path


@app.route('/')
def index():
    return list_files(BASE_DIR)


@app.route('/browse/')
def plug():
    return redirect('/')


@app.route('/browse/<path:subpath>')
def browse(subpath):
    # Replace any forward slashes with backslashes (for Windows paths)
    subpath = subpath.replace('/', '\\')

    # Safely join paths and ensure we stay within BASE_DIR
    try:
        full_path = os.path.abspath(os.path.join(BASE_DIR, subpath))
    except:
        abort(404)

    # Security check - prevent directory traversal
    if not full_path.startswith(BASE_DIR):
        abort(403, description="Access denied")

    if not os.path.exists(full_path):
        abort(404)

    return list_files(full_path)


def list_files(directory):
    # Get all files and directories
    items = []
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                items.append({
                    'name': item,
                    'is_dir': os.path.isdir(item_path),
                    'size': os.path.getsize(item_path) if not os.path.isdir(item_path) else 0,
                    'modified': os.path.getmtime(item_path)
                })
            except (OSError, PermissionError):
                continue  # Skip files we can't access

        # Sort directories first, then files
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
    except (OSError, PermissionError):
        abort(403, description="Cannot access directory")

    # Calculate relative path for navigation
    rel_path = os.path.relpath(directory, BASE_DIR)
    print(rel_path)
    # Get parent directory if not at root
    parent_dir = None
    if directory != BASE_DIR:
        parent_dir = os.path.dirname(directory)
        # Ensure parent_dir is still within BASE_DIR
        if not parent_dir.startswith(BASE_DIR):
            parent_dir = BASE_DIR

    # Pre-split the path parts for the template
    path_parts = []
    current_path = BASE_DIR
    if rel_path != '.':
        for part in rel_path.split(os.sep):
            current_path = os.path.join(current_path, part)
            path_parts.append({
                'name': part,
                'path': os.path.relpath(current_path, BASE_DIR)
            })

    search_term = request.args.get('search_term', '')
    return render_template('index.html',
                           items=items,
                           current_dir=directory,
                           parent_dir=parent_dir,
                           path_parts=path_parts,
                           base_dir=BASE_DIR,
                           search_term=search_term)


@app.route('/download/<path:filename>')
def download(filename):
    directory = os.path.dirname(filename)
    file = os.path.basename(filename)
    return send_from_directory(directory, file, as_attachment=False)


# Add os module to template globals
@app.context_processor
def inject_os():
    return {'os': os}

@app.route('search_files', methods=['POST'])
def search_files(filepath):
    item_list = []
    for i in os.listdir(filepath):
        if i.endswith('.txt') or i.endswith('.docx') or i.endswith('.pdf'):
            item_list.append(i)



def search_in_file(filepath, search_term):
    """Search for text in different file types"""
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return search_term.lower() in content.lower()

        elif filepath.endswith('.docx'):
            doc = docx.Document(filepath)
            text = '\n'.join([para.text for para in doc.paragraphs])
            return search_term.lower() in text.lower()

        elif filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                reader = PdfReader(f)
                text = '\n'.join([page.extract_text() for page in reader.pages])
            return search_term.lower() in text.lower()

        return False
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


if __name__ == '__main__':
    app.run(debug=True, port=5000)
