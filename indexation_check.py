import time
import difflib
from neogrid_site.globals import BASE_DIR, SUPPORTED_FORMATS
import os
import hashlib


def compute_dir_hash(directory_path):
    dir_hash = hashlib.md5()
    for root, _, files in os.walk(directory_path):
        for file in sorted(files):  # Сортируем для стабильности хеша
            file_path = os.path.join(root, file)
            if file.split('.')[-1] in SUPPORTED_FORMATS:
                print(file_path)
                # Хешируем имя файла и его содержимое
                try:
                    with open(file_path, "rb") as f:
                        dir_hash.update(f.read())
                    dir_hash.update(file.encode())
                except (PermissionError, OSError):
                    continue
    return dir_hash.hexdigest()


def string_similarity(str1, str2):
    matcher = difflib.SequenceMatcher(None, str1, str2)
    return matcher.ratio() * 100


def check():
    is_check_necessary = False
    file_name = 'appdata/timedata_checks.txt'
    with open(file_name, 'r+') as file:
        data = file.read()
        start_time = time.time()
        print(len(data))
        if len(data) == 0:  # If the file is effectively empty
            is_check_necessary = True
            file.write(f'last_check_time:{str(round(time.time()))}\n')
            file.write(f'last_hash_save:{compute_dir_hash(BASE_DIR)}\n')
        else:
            for i in data.split('\n'):
                print(i)
                line_data = i.split(':')
                print(line_data)
                try:
                    if line_data[0] == 'last_check_time':
                        if time.time() - int(line_data[1]) > 86400:
                            is_check_necessary = True
                    elif line_data[1] == 'last_hash_save':
                        if is_check_necessary:
                            current_hash = compute_dir_hash(BASE_DIR)
                            if string_similarity(current_hash, line_data[1]) < 0.95:
                                return True
                except:
                    pass

    print("--- Hashing took %s seconds ---" % (time.time() - start_time))
    return is_check_necessary
