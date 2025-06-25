import os

def generate_api_rst(src_dir, rst_file):
    """
    Generates an .rst file with automodule directives for all Python modules
    found in specific subdirectories of the source directory.
    """
    allowed_folders = ['models', 'modules', 'pipeline_blocks']
    with open(rst_file, 'w') as f:
        f.write("API Reference\n")
        f.write("=============\n\n")

        for folder in allowed_folders:
            folder_path = os.path.join(src_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for root, dirs, files in os.walk(folder_path):
                if '__pycache__' in dirs:
                    dirs.remove('__pycache__')

                for file in files:
                    if file.endswith(".py") and not file.startswith("__init__"):
                        module_path = os.path.join(root, file)
                        # Get path relative to src_dir to create module path
                        rel_path = os.path.relpath(module_path, src_dir)
                        module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                        f.write(f".. automodule:: {module_name}\n")
                        f.write("   :members:\n\n")

if __name__ == "__main__":
    # The script is in docs/config, so we go up two levels to the project root, then to src
    src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    output_rst_file = os.path.join(os.path.dirname(__file__), 'api.rst')
    generate_api_rst(src_directory, output_rst_file)
    print(f"Generated {output_rst_file} successfully.")
