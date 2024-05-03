# Define the headings and corresponding file names
sections = {
    "강점:": "강점.txt",
    "단점:": "단점.txt",
    "지원동기 및 입사 후 포부:": "지원동기 및 입사 후 포부.txt",
    "성공경험:": "성공경험.txt",
}


def process_file(input_filename):
    current_section = None
    section_content = {key: [] for key in sections.keys()}

    with open(input_filename, "r", encoding="utf-8") as file:
        for line in file:
            # Check if the line contains a section heading
            if line.strip() in sections:
                current_section = line.strip()
            elif current_section:
                # Add the line to the correct section
                section_content[current_section].append(line)

    # Write each section to its corresponding file
    for section, content in section_content.items():
        if content:  # Only write if there is content for this section
            with open(sections[section], "w", encoding="utf-8") as file:
                file.writelines(content)


# Example usage:
process_file("3/라벨링_nospace.txt")
