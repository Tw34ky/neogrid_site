from flask import Flask, render_template, send_from_directory, abort, request
import os
import time
from filters import register_filters
from werkzeug.utils import redirect
import docx
from pytesseract_func import pdf_to_text
import pprint
from globals import *
import data_base_lib

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
    def format_file_size(size_in_bytes):
        import math

        if size_in_bytes == 0:
            return ''

        units = ['Bytes', 'KB', 'MB', 'GB', 'TB']
        unit_index = min(int(math.log(size_in_bytes, 1024)), len(units) - 1)
        size = size_in_bytes / (1024 ** unit_index)

        if size.is_integer() or size >= 10 or unit_index == 0:
            return f"{int(round(size))} {units[unit_index]}"
        else:
            return f"{size:.2f}".rstrip('0').rstrip('.') + f" {units[unit_index]}"

    # Get all files and directories
    items = []
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                mod_time = time.localtime(os.path.getmtime(item_path))
                name, is_dir, modified = item, os.path.isdir(item_path), f"{mod_time[2]}.{mod_time[1]}.{mod_time[0]} {mod_time[3]}:{mod_time[4]}"
                if not os.path.isdir(item_path):
                    items.append({
                        'name': name,
                        'is_dir': is_dir,
                        'size': format_file_size(os.path.getsize(item_path)),
                        'modified': modified
                    })
                else:
                    items.append({
                        'name': name,
                        'is_dir': is_dir,
                        'size': '',
                        'modified': modified
                    })
            except (OSError, PermissionError):
                continue
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
    except (OSError, PermissionError):
        abort(403, description="Cannot access directory")

    # Calculate relative path for navigation
    rel_path = os.path.relpath(directory, BASE_DIR)
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
def download(filename: str):
    directory = os.path.dirname(filename)
    file = os.path.basename(filename)
    return send_from_directory(directory, file, as_attachment=False)


@app.context_processor
def inject_os():
    return {'os': os}


@app.route('/search_files')
def search_files():
    start_time = time.time()
    args = request.args
    filepath = args.getlist('current_dir')[0]
    search_prompt = args.getlist('search_term')[0]
    filepath = filepath.replace('/', '\\')

    # Safely join paths and ensure we stay within BASE_DIR
    try:
        full_path = os.path.abspath(os.path.join(BASE_DIR, filepath))
    except:
        abort(404)

    # Security check - prevent directory traversal
    if not full_path.startswith(BASE_DIR):
        abort(403, description="Access denied")

    if not os.path.exists(full_path):
        abort(404)

    item_list = []

    # Рекурсивно обходим все директории и файлы
    for root, dirs, files in os.walk(full_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.docx') or file.endswith('.pdf'):
                item = f"{root}/{file}".replace('\\', '/')
                file_text = db_expansion(filename=item)
                if not file_text:
                    continue
                # item_list.append(
                #     {"type": item.rsplit('.')[-1], "path": item, "name": item.rsplit('/')[-1], "content": file_text})
    pprint.pprint(item_list)
    print()
    print("--- %s seconds ---" % (time.time() - start_time))
    search_prompt = data_base_lib.query_rag(search_prompt)
    return render_template('files.html', lang_model_ans=search_prompt, files=item_list)


def db_expansion(filename):
    retrieved_data = parse_file(filename)
    if retrieved_data:
        params = []
        data_base_lib.populate_database(params, filename)
        """
        ig some db code idk    
        """
        return retrieved_data
    return None


def parse_file(filepath: str):
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                return content


        elif filepath.endswith('.docx') or filepath.endswith('.doc'):
            doc = docx.Document(filepath)
            content = ' '.join([para.text for para in doc.paragraphs])
            return content

        elif filepath.endswith('.pdf'):
            import fitz  # PyMuPDF
            with fitz.open(filepath) as doc:
                content = ""
                for page in doc:
                    content += page.get_text()
                if len(content) == 0:
                    content = pdf_to_text(filepath)
                return content

        elif filepath.endswith('.rtf'):
            from striprtf.striprtf import rtf_to_text

            with open(filepath) as infile:
                content = infile.read()
                content = rtf_to_text(content)
                return content
        return False

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


if __name__ == '__main__':
    app.run(debug=True, port=5000)
