def divide_from_file_filtered(input_filename):
    # 파일 읽기
    with open(input_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    headers = ['성공경험', '지원동기 및 입사 후 포부', '강점', '단점']
    sections = {key: "" for key in headers}
    current_header = None

    # 텍스트를 선별하여 각 섹션 분류, HTTPS 포함 줄 제외
    for line in lines:
        line = line.strip()
        if 'https://' in line:  # HTTPS 포함 줄은 건너뛰기
            continue
        if line == '':
            continue
        if line in headers:
            current_header = line
        elif current_header:
            sections[current_header] += '"' + line + '",' + str(current_header) + '\n'
    
    # 각 섹션을 별도의 파일로 저장
    for header, content in sections.items():
        filename = f"{header}.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Saved {header} to {filename}")

# 파일명 지정
input_filename = "./4th_wo_-.txt"

# 파일 파싱 및 각 섹션 저장, HTTPS 포함 줄 제외
divide_from_file_filtered(input_filename)
