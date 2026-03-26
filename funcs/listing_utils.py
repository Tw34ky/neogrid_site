import os
import time
from funcs.file_utils import format_file_size
from funcs.global_vars import BASE_DIR


def get_file_items(directory):
    """
    Возвращает кортеж (items, parent_dir, path_parts) для переданной директории.
    items – список словарей с информацией о файлах/папках.
    parent_dir – путь к родительской директории (или None, если текущая – корневая).
    path_parts – список частей пути для навигации (хлебные крошки).
    """
    items = []
    g = os.scandir(directory)
    try:
        for item in os.scandir(directory):
            # print('scanning', item.name)
            item_path = os.path.join(directory, item)
            try:
                mod_time = time.localtime(os.path.getmtime(item_path))
                name = item.name
                is_dir = item.is_dir()
                modified = f"{mod_time[2]}.{mod_time[1]}.{mod_time[0]} {mod_time[3]}:{mod_time[4]}"
                if not is_dir:
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
        print('listdir dead')
        raise PermissionError("Cannot access directory")

    # Определяем родительскую директорию
    parent_dir = None
    if directory != BASE_DIR:
        parent_dir = os.path.dirname(directory)
        if not parent_dir.startswith(BASE_DIR):
            parent_dir = BASE_DIR

    # Хлебные крошки
    path_parts = []
    rel_path = os.path.relpath(directory, BASE_DIR)
    if rel_path != '.':
        current_path = BASE_DIR
        for part in rel_path.split(os.sep):
            current_path = os.path.join(current_path, part)
            path_parts.append({
                'name': part,
                'path': os.path.relpath(current_path, BASE_DIR)
            })

    return items, parent_dir, path_parts