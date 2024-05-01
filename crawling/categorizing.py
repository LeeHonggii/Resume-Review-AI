def classify_and_save_text(file_path):
    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 섹션별로 내용을 저장할 딕셔너리
    sections = {
        "성공경험": [],
        "지원동기 및 입사 후 포부": [],
        "강점": [],
        "단점": []
    }

    # 섹션 이름을 키워드로 사용하여 텍스트 분할
    current_section = None
    lines = content.split("\n")
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.endswith(':'):
            potential_section = stripped_line[:-1]
            if potential_section in sections:
                current_section = potential_section
                continue
        if current_section and stripped_line:
            sections[current_section].append(stripped_line)

    # 분류된 각 섹션별 텍스트를 파일로 저장
    for section, lines in sections.items():
        filename = f"{section}.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("\n".join(lines))


# 텍스트 파일 경로를 설정하세요
file_path = "C:/Users/birth/Desktop/라벨링.txt"
classify_and_save_text(file_path)
