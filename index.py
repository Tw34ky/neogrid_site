from flask import Flask, render_template, abort, request, redirect, url_for, jsonify
import os, json, time, docx, pprint, global_vars, data_base_lib, indexation_check
from filters import register_filters
from werkzeug.utils import redirect
from pytesseract_func import pdf_to_text
from open_path import path_opener
from replace_vars import update_global_vars


app = Flask(__name__)
global_current_dir = global_vars.BASE_DIR
register_filters(app)


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
META_FILE = 'files.json'


def load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


@app.route('/')
def index():
    import global_vars
    rebasing_check_needed = indexation_check.check()
    if rebasing_check_needed or global_vars.reset_data_boolean:
        restart_database()
    return list_files(global_vars.BASE_DIR)


@app.route('/browse/')
def plug():
    return redirect('/')


@app.route('/apply_settings', methods=['POST'])
def apply_settings():
    import inspect

    variables = {}
    print(request.form)
    for name in dir(global_vars):
        obj = getattr(global_vars, name)
        if (not (inspect.isfunction(obj) or inspect.isclass(obj) or name.startswith('__')) and name
                in global_vars.SETTINGS):
            variables[name] = obj
    changed_vars = []
    for field in request.form:
        if request.form.get('reset_data_boolean') == 'on':
            variables['reset_data_boolean'] = True
        elif request.form.get('reset_data_boolean') == 'off':
            variables['reset_data_boolean'] = False
        if variables.get(field) != request.form.get(field):
            variables[field] = request.form.get(field)
            changed_vars.append(field)

    update_global_vars('global_vars.py', variables, changed_vars)

    return redirect('/')


@app.route('/browse/<path:subpath>')
def browse(subpath):
    # Replace any forward slashes with backslashes (for Windows paths)
    subpath = subpath.replace('/', '\\')

    # Safely join paths and ensure we stay within BASE_DIR
    try:
        full_path = os.path.abspath(os.path.join(global_vars.BASE_DIR, subpath))
    except:
        abort(404)

    # Security check - prevent directory traversal
    if not full_path.startswith(global_vars.BASE_DIR):
        abort(403, description="Access denied")

    if not os.path.exists(full_path):
        abort(404)

    return list_files(full_path)


def list_files(directory):
    global global_current_dir

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
                name, is_dir, modified = item, os.path.isdir(
                    item_path), f"{mod_time[2]}.{mod_time[1]}.{mod_time[0]} {mod_time[3]}:{mod_time[4]}"
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

    rel_path = os.path.relpath(directory, global_vars.BASE_DIR)
    parent_dir = None
    if directory != global_vars.BASE_DIR:
        parent_dir = os.path.dirname(directory)
        if not parent_dir.startswith(global_vars.BASE_DIR):
            parent_dir = global_vars.BASE_DIR
    path_parts = []
    current_path = global_vars.BASE_DIR
    if rel_path != '.':
        for part in rel_path.split(os.sep):
            current_path = os.path.join(current_path, part)
            path_parts.append({
                'name': part,
                'path': os.path.relpath(current_path, global_vars.BASE_DIR)})
    global_current_dir = directory
    search_term = request.args.get('search_term', '')
    return render_template('index.html',
                           items=items,
                           current_dir=directory,
                           parent_dir=parent_dir,
                           path_parts=path_parts,
                           base_dir=global_vars.BASE_DIR,
                           search_term=search_term)


@app.route('/settings_page')
def settings_page():
    import inspect

    variables = {}
    for name in dir(global_vars):

        obj = getattr(global_vars, name)
        if not (inspect.isfunction(obj) or inspect.isclass(obj) or name.startswith(
                '__')) and name in global_vars.SETTINGS:
            variables[name] = {'value': obj,
                               'type': str(type(obj))[str(type(obj)).find("'") + 1:str(type(obj)).rfind("'")]}

    del variables['BASE_DIR']
    return render_template('settings.html', settings_data=variables)


@app.context_processor
def inject_os():
    return {'os': os}


@app.route('/invoke_prompt')
def invoke_prompt():
    start_time = time.time()
    args = request.args
    search_prompt, sources = data_base_lib.query_rag(args.getlist('search_term')[0])
    print("\n--- LLaMa answered in %s seconds ---" % (time.time() - start_time))

    time.sleep(5)
    def format_llm_response(response_text):
        # Convert markdown-like formatting to HTML
        formatted = response_text.replace('**', '<strong>').replace('**', '</strong>')
        formatted = formatted.replace('*', '<em>').replace('*', '</em>')

        # Add paragraph breaks
        formatted = formatted.replace('\n\n', '</p><p>')

        # Add code blocks (if you want to handle them)
        formatted = formatted.replace('```python', '<pre><code class="language-python">')
        formatted = formatted.replace('```', '</code></pre>')

        return f'<div class="llm-response">{formatted}</div>'


    return render_template('answer.html', answer_text=search_prompt,
                           edited_text=format_llm_response(search_prompt), sources=sources)  # answer_text=search_prompt

    # return render_template('answer.html', answer_text='Процедура реализации имущества гражданина была введена с 19.02.2025 и продолжалась до 22.07.2025, когда было назначено судебное заседание по вопросу о завершении процедуры.',
    #                             edited_text=format_llm_response('Процедура реализации имущества гражданина была введена с 19.02.2025 и продолжалась до 22.07.2025, когда было назначено судебное заседание по вопросу о завершении процедуры.'), sources=r'C:/Users/Timofey/Desktop/Sample Data/Судебные акты/A45-409-2025_20250722_Opredelenie.pdf:9\\C:/Users/Timofey/Desktop/Sample Data/Судебные акты/A45-409-2025_20250722_Opredelenie.pdf:8\\C:/Users/Timofey/Desktop/Sample Data/Судебные акты/A45-409-2025_20250722_Opredelenie.pdf:1\\C:/Users/Timofey/Desktop/Sample Data/Судебные акты/A45-409-2025_20250722_Opredelenie.pdf:2'.split(r'\\'))  # answer_text=search_prompt


@app.route('/open', methods=['POST'])
def open_file():
    file_path = request.json.get('path')
    print(request.json)
    file_path = file_path[0:file_path.rfind(':')]
    if file_path:
        success = path_opener(file_path)
        return jsonify({"status": "success" if success else "error"})
    return jsonify({"status": "error", "message": "No path provided"}), 400


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    return redirect(url_for('index'))


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    files = load_metadata()
    files = [f for f in files if f['name'] != filename]
    # save_metadata(files)

    return redirect(url_for('index'))


@app.route('/restart_database')
def restart_database():
    global global_current_dir
    data_base_lib.clear_database()
    start_time = time.time()
    args = request.args
    print('Initiated database restart')

    try:
        filepath = args.getlist('current_dir')[0].replace('/', '\\')
    except IndexError:
        try:
            filepath = global_current_dir
        except ValueError:
            filepath = global_vars.BASE_DIR

    print(filepath)
    # Safely join paths and ensure we stay within BASE_DIR
    try:
        full_path = os.path.abspath(os.path.join(global_vars.BASE_DIR, filepath))
    except:
        abort(404)

    # Security check - prevent directory traversal
    if not full_path.startswith(global_vars.BASE_DIR):
        abort(403, description="Access denied")

    if not os.path.exists(full_path):
        abort(404)

    # Рекурсивно обходим все директории и файлы
    for root, dirs, files in os.walk(full_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.docx') or file.endswith('.pdf'):
                item = f"{root}/{file}".replace('\\', '/')
                file_text = db_expansion(filename=item)
                if not file_text:
                    continue
    print("--- Database restart took %s seconds ---" % (time.time() - start_time))
    update_global_vars('global_vars.py', {'reset_data_boolean': False}, ['reset_data_boolean'])
    return redirect('/')


def db_expansion(filename):
    retrieved_data = parse_file(filename)
    if retrieved_data:
        print(filename)
        retrieved_data = retrieved_data.replace('\n', ' ')
        retrieved_data = retrieved_data.replace('\xad', '')
        retrieved_data = retrieved_data.replace('\u00ad', '')
        retrieved_data = retrieved_data.replace('\N{SOFT HYPHEN}', '')
        params = []
        metadata = {'source': filename}
        data_base_lib.populate_database(params, retrieved_data, metadata)
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

            with open(filepath, encoding='utf-8') as infile:
                content = infile.read()
                content = rtf_to_text(content)
                return content
        return False

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


if __name__ == '__main__':
    app.run(port=5000)
