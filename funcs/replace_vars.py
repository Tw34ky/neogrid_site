def update_global_vars(file_path, variables, changed_vars):
    with open(file_path, "r") as file:
        content = file.readlines()

    new_content = []
    changes_made = False

    for line in content:
        line_modified = False
        for element in changed_vars:
            if element in line:
                if not line_modified and line[0:8] != 'SETTINGS':  # Only modify once per line and make an exception for settings list
                    if element == 'reset_data_boolean':
                        new_content.append(f"reset_data_boolean = {str(variables[element]).capitalize()}\n")
                    elif element == 'use_llm':
                        new_content.append(f"use_llm = {str(variables[element]).capitalize()}\n")
                    else:
                        new_content.append(f"{element} = r'{variables[element]}'\n")
                    print(f"Replaced line: {line.strip()} with {new_content[-1].strip()}")
                    line_modified = True
                    changes_made = True
        if not line_modified:
            new_content.append(line)

    if changes_made:
        with open(file_path, "w") as file:
            file.writelines(new_content)
    else:
        print("No changes were made to the file.")