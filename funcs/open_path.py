def path_opener(path):
    """Open file browser at the specified path"""
    import subprocess
    import platform
    import os
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux and other Unix-like
            subprocess.run(["xdg-open", path])
        return True
    except Exception as e:
        print(f"Error opening path: {e}")
        return False