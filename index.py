from flask import Flask, render_template, abort, request, redirect, url_for, jsonify
import os, json, docx, data_base_lib, importlib

from werkzeug.utils import redirect

# from rag_engine import RAGEngine

from funcs import global_vars
from funcs.pytesseract_func import pdf_to_text
from funcs.open_path import path_opener
from funcs.replace_vars import update_global_vars
from funcs.filters import register_filters
from funcs.answer_formatting import format_llm_response
# from funcs.file_utils import db_expansion, parse_file 
from funcs.listing_utils import get_file_items


app = Flask(__name__)
global_current_dir = global_vars.BASE_DIR
register_filters(app)
# rag_engine = RAGEngine()


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
    importlib.reload(global_vars)
    # rebasing_check_needed = indexation_check.check()
    # if rebasing_check_needed or
    #     restart_database()
    if global_vars.reset_data_boolean:
        restart_database()
    
    global global_current_dir
    global_current_dir = global_vars.BASE_DIR

    items, parent_dir, path_parts = get_file_items(global_vars.BASE_DIR)  

    search_term = request.args.get('search_term', '')

    return render_template('index.html',
                        items=items,
                        current_dir='',
                        parent_dir=parent_dir,
                        path_parts=path_parts,
                        base_dir=global_vars.BASE_DIR,
                        search_term=search_term)

@app.route('/browse/')
def plug():
    return redirect('/')


@app.route('/apply_settings', methods=['POST'])
def apply_settings():
    import inspect
    importlib.reload(global_vars)
    variables = {}
    for name in dir(global_vars):
        obj = getattr(global_vars, name)
        if (not (inspect.isfunction(obj) or inspect.isclass(obj) or name.startswith('__')) and name
                in global_vars.SETTINGS):
            variables[name] = obj
    changed_vars = set()
    print(request.form)
    for field in request.form:
        if variables.get(field) != request.form.get(field):
            variables[field] = request.form.get(field)
            changed_vars.add(field)

    if request.form.get('reset_data_boolean') == 'on':
        variables['reset_data_boolean'] = True
        changed_vars.add('reset_data_boolean')
    elif request.form.get('reset_data_boolean') == 'off':
        variables['reset_data_boolean'] = False
        changed_vars.add('reset_data_boolean')

    if request.form.get('use_llm') == 'on':
        variables['use_llm'] = True
        changed_vars.add('use_llm')
    elif request.form.get('use_llm') == 'off':
        variables['use_llm'] = False
        changed_vars.add('use_llm')

    print(variables)
    update_global_vars('funcs/global_vars.py', variables, changed_vars)

    return redirect('/')


@app.route('/browse/<path:subpath>')
def browse(subpath):
    # print(subpath)
    try:
        full_path = os.path.abspath(os.path.join(global_vars.BASE_DIR, subpath))
    except:
        abort(404)

    if not full_path.startswith(global_vars.BASE_DIR):
        abort(403, description="Access denied")

    if not os.path.exists(full_path):
        abort(404)

    global global_current_dir

    try:
        items, parent_dir, path_parts = get_file_items(full_path)
    except PermissionError:
        abort(403, description="Cannot access directory")

    # print(subpath, parent_dir, global_current_dir)
    # if parent_dir and parent_dir + subpath != global_current_dir:

    global_current_dir = full_path

    render_path = os.path.relpath(full_path, global_vars.BASE_DIR)

    search_term = request.args.get('search_term', '')
    print('current_dir:', global_current_dir)
    return render_template('index.html',
                           items=items,
                           current_dir=render_path,
                           parent_dir=parent_dir,
                           path_parts=path_parts,
                           base_dir=global_vars.BASE_DIR,
                           search_term=search_term)


@app.route('/settings_page')
def settings_page():
    import inspect
    importlib.reload(global_vars)
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

    # def format_llm_response(response_text):
    #     # Convert markdown-like formatting to HTML
    #     formatted = response_text.replace('**', '<strong>').replace('**', '</strong>')
    #     formatted = formatted.replace('*', '<em>').replace('*', '</em>')

    #     # Add paragraph breaks
    #     formatted = formatted.replace('\n\n', '</p><p>')

    #     # Add code blocks (if you want to handle them)
    #     formatted = formatted.replace('```python', '<pre><code class="language-python">')
    #     formatted = formatted.replace('```', '</code></pre>')

    #     return f'<div class="llm-response">{formatted}</div>'

    args = request.args
    question = args.getlist('search_term')[0]
    answer, gen_time, sources = rag_engine.query(question)
    print('gen_time:', gen_time)
    # print(format_llm_response(answer))
    return render_template('answer.html', answer_text=answer,
                           edited_text=format_llm_response(answer), sources=sources)  # answer_text=search_prompt

    # return render_template('answer.html', answer_text='Процедура реализации имущества гражданина была введена с 19.02.2025 и продолжалась до 22.07.2025, когда было назначено судебное заседание по вопросу о завершении процедуры.',
    #                             edited_text=format_llm_response('Процедура реализации имущества гражданина была введена с 19.02.2025 и продолжалась до 22.07.2025, когда было назначено судебное заседание по вопросу о завершении процедуры.'), sources=r'C:/Users/Timofey/Desktop/Sample Data/Судебные акты/A45-409-2025_20250722_Opredelenie.pdf:9\\C:/Users/Timofey/Desktop/Sample Data/Судебные акты/A45-409-2025_20250722_Opredelenie.pdf:8\\C:/Users/Timofey/Desktop/Sample Data/Судебные акты/A45-409-2025_20250722_Opredelenie.pdf:1\\C:/Users/Timofey/Desktop/Sample Data/Судебные акты/A45-409-2025_20250722_Opredelenie.pdf:2'.split(r'\\'))  # answer_text=search_prompt


@app.route('/open', methods=['POST'])
def open_file():
    file_path = request.json.get('path')
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
    # Determine directory to index
    filepath = request.args.get('current_dir', global_current_dir)
    full_path = os.path.abspath(os.path.join(global_vars.BASE_DIR, filepath))
    # security check...
    rag_engine.rebuild_index(full_path)
    # set reset flag back to False
    update_global_vars('funcs/global_vars.py', {'reset_data_boolean': False}, ['reset_data_boolean'])
    return redirect('/')


# def db_expansion(filename):
#     retrieved_data = parse_file(filename)
#     if retrieved_data:
#         print(filename)
#         retrieved_data = retrieved_data.replace('\n', ' ')
#         retrieved_data = retrieved_data.replace('\xad', '')
#         retrieved_data = retrieved_data.replace('\u00ad', '')
#         retrieved_data = retrieved_data.replace('\N{SOFT HYPHEN}', '')
#         params = []
#         metadata = {'source': filename}
#         data_base_lib.populate_database(params, retrieved_data, metadata)
#         return retrieved_data
#     return None


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
