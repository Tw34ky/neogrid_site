import hashlib
import json
from pathlib import Path


def calculate_directory_hash(directory, hash_algorithm='sha256', ignore_hidden=True):
    """
    Calculate a combined hash for all files in a directory.

    Args:
        directory: Path to the directory
        hash_algorithm: Hash algorithm to use (default: sha256)
        ignore_hidden: Whether to ignore hidden files (starting with .)

    Returns:
        A dictionary containing file paths and their hashes,
        and a combined hash of all file hashes
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory")

    hasher = hashlib.new(hash_algorithm)
    file_hashes = {}

    for file_path in directory.rglob('*'):
        if file_path.is_file():
            if ignore_hidden and file_path.name.startswith('.'):
                continue

            # Calculate file hash
            file_hash = hashlib.new(hash_algorithm)
            try:
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        file_hash.update(chunk)
            except (IOError, PermissionError) as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue

            file_hashes[str(file_path.relative_to(directory))] = file_hash.hexdigest()
            hasher.update(file_hash.digest())

    return {
        'files': file_hashes,
        'directory_hash': hasher.hexdigest(),
        'algorithm': hash_algorithm
    }


def save_hash_info(directory, hash_info, save_file='.directory_hash.json'):
    """Save hash information to a JSON file."""
    save_path = Path(directory) / save_file
    with open(save_path, 'w') as f:
        json.dump(hash_info, f, indent=2)


def load_hash_info(directory, save_file='.directory_hash.json'):
    """Load previously saved hash information."""
    load_path = Path(directory) / save_file
    try:
        with open(load_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def compare_directory_hashes(directory, hash_algorithm='sha256'):
    """
    Compare current directory state with previously saved state.

    Returns:
        Tuple of (changed_status, changes_dict)
        changed_status: True if directory changed, False otherwise
        changes_dict: Dictionary containing changes (added, removed, modified files)
    """
    current = calculate_directory_hash(directory, hash_algorithm)
    previous = load_hash_info(directory)

    if not previous:
        return True, {'status': 'no_previous_record', 'current': current}

    changes = {
        'added': [],
        'removed': [],
        'modified': [],
        'unchanged': []
    }

    # Compare file lists
    current_files = set(current['files'].keys())
    previous_files = set(previous['files'].keys())

    changes['added'] = list(current_files - previous_files)
    changes['removed'] = list(previous_files - current_files)

    # Check modified files
    common_files = current_files & previous_files
    for file in common_files:
        if current['files'][file] != previous['files'][file]:
            changes['modified'].append(file)
        else:
            changes['unchanged'].append(file)

    # Determine if directory changed
    directory_changed = (len(changes['added']) > 0 or
                         len(changes['removed']) > 0 or
                         len(changes['modified']) > 0)

    return directory_changed, {
        'changes': changes,
        'current': current,
        'previous': previous
    }


# Example usage:
if __name__ == "__main__":
    directory_to_watch = './my_directory'  # Change this to your directory

    # First run - save initial state
    if not load_hash_info(directory_to_watch):
        print("No previous hash found. Creating initial hash...")
        hash_info = calculate_directory_hash(directory_to_watch)
        save_hash_info(directory_to_watch, hash_info)
        print("Initial directory hash saved.")
    else:
        # Subsequent runs - compare with previous state
        changed, changes = compare_directory_hashes(directory_to_watch)

        if changed:
            print("Directory has changed!")
            print(f"Added files: {changes['changes']['added']}")
            print(f"Removed files: {changes['changes']['removed']}")
            print(f"Modified files: {changes['changes']['modified']}")

            # Save new state
            hash_info = calculate_directory_hash(directory_to_watch)
            save_hash_info(directory_to_watch, hash_info)
            print("Updated directory hash saved.")
        else:
            print("Directory has not changed since last check.")