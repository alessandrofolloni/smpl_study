import os


def generate_folder_structure(root_folder, indent_level=0, previous_files=None):
    """
    Recursively generates a string representation of a folder's structure,
    displaying file names only if they are not similar to previous ones.

    Args:
        root_folder (str): Path to the root folder to analyze.
        indent_level (int): Current indentation level for printing.
        previous_files (set): Set of previously encountered file names.

    Returns:
        str: A string representation of the folder structure.
    """
    # Initialize previous_files as a set if it's None
    if previous_files is None:
        previous_files = set()

    # Indentation for the current level
    indent = '    ' * indent_level

    # Initialize the structure representation
    structure = f"{indent}{os.path.basename(root_folder)}/\n"

    try:
        # Get list of directories and files
        items = os.listdir(root_folder)
    except PermissionError:
        # Handle permission errors for inaccessible directories
        structure += f"{indent}    [Permission Denied]\n"
        return structure

    # Separate directories and files
    dirs = sorted([d for d in items if os.path.isdir(os.path.join(root_folder, d))])
    files = sorted([f for f in items if os.path.isfile(os.path.join(root_folder, f))])

    # Recursively generate structure for directories
    for dir_name in dirs:
        structure += generate_folder_structure(
            os.path.join(root_folder, dir_name),
            indent_level + 1,
            previous_files
        )

    # Add files to the structure if they are unique
    for file_name in files:
        # Check if the file name (without extension) is similar to any previous file
        file_base = os.path.splitext(file_name)[0]  # File name without extension

        if file_base not in previous_files:
            # If it's unique, add to the structure and mark it as seen
            structure += f"{indent}    {file_name}\n"
            previous_files.add(file_base)

    return structure


def write_folder_structure_to_file(root_folder, output_file):
    """
    Writes the hierarchical structure of a folder to a text file.

    Args:
        root_folder (str): Path to the root folder to analyze.
        output_file (str): Path to the output text file.
    """
    structure = generate_folder_structure(root_folder)

    # Write the structure to the output file
    with open(output_file, 'w') as f:
        f.write(f"Folder Structure for: {root_folder}\n\n")
        f.write(structure)

    print(f"Folder structure has been written to {output_file}")


if __name__ == "__main__":
    # Specify the folder path you want to analyze
    folder_path = '/Users/alessandrofolloni/PycharmProjects/smpl_study/datasets/FIT3D'

    # Specify the output file path
    output_file_path = 'fit3d_structure.txt'  # Change as needed

    write_folder_structure_to_file(folder_path, output_file_path)
