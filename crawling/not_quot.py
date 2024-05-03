import re

# Open the file for reading
with open("1/강점.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()


# Function to replace quotes at the start or end of a sentence
def replace_quotes(line):
    # Replace double quotes at the start of a sentence
    line = re.sub(r'^"', "", line)
    # Replace double quotes at the end of a sentence (including possible newline characters)
    line = re.sub(r'"(\s*)$', r"\1", line)
    return line


# Apply the function to each line
modified_lines = [replace_quotes(line) for line in lines]

# Open the file again for writing and save the modified content
with open("강점_no_qout.txt", "w", encoding="utf-8") as file:
    file.writelines(modified_lines)
