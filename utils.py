import os

def get_files_with_extensions(folder_path: str, target_exts: list[str]):
    """
    Recursively collect all files under folder_path
    that match any extension in target_exts.
    
    target_exts example: [".epub", ".txt", ".pdf"]
    """

    # normalize extensions
    normalized_exts = []
    for ext in target_exts:
        if not ext.startswith("."):
            ext = "." + ext
        normalized_exts.append(ext.lower())

    matches = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in normalized_exts):
                full_path = os.path.join(root, filename)
                matches.append(full_path)

    return matches


def write_to_txt(path : str, text : str):
    """Write text to a .txt file at the specified path"""
    try:
        with open(path, 'w', encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Could not write to .txt : {e}")