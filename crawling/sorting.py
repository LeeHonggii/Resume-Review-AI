input_file_path = "crawling\Link.txt"
output_file_path = "sorted_link.txt"

# Reading links from the text file
with open(input_file_path, 'r') as file:
    links_from_file = file.readlines()

# Strip newline characters and sort by page number
links_sorted_by_page = sorted(
    links_from_file, key=lambda x: int(x.split("page=")[1].split("&")[0]))

# Rewrite the sorted links to a new text file
with open(output_file_path, 'w') as file:
    for link in links_sorted_by_page:
        file.write(link)
