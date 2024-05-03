inFp = None
inList = []
inStri = ""

with open("data/output.txt", "r", encoding="utf-8") as inFp:
    outFp1 = open("data/21.txt", "w")
    outFp2 = open("data/22.txt", "w")
    outFp3 = open("data/23.txt", "w")
    outFp4 = open("data/24.txt", "w")
    inList = inFp.readlines()
    for inStr in inList:
        if inStr == "\n":
            # print("newline")
            continue
        label = inStr[:3]
        sent = inStr[3:]
        if sent[:1] == '"' and sent[-2:-1] == '"':
            # print(str)
            sent = sent[1:-2]
            sent = sent + '\n'
            sent.strip()
            # print(str)
            # if sent.startswith('"') or sent.startswith('"'):
            #     print(sent)
            #     quit()

        if label == "1. ":
            # print(inStr, end="")
            outFp1.writelines(sent)
        elif label == "2. ":
            # print(inStr, end="")
            outFp2.writelines(sent)
        elif label == "3. ":
            # print(inStr, end="")
            outFp3.writelines(sent)
        elif label == "4. ":
            # print(inStr, end="")
            outFp4.writelines(sent)
        else:
            print(inStr, end="")
        # if inStr[:2] ==
        # print(inStr, end="")
        # print(len(inStr))
    outFp1.close()
    outFp2.close()
    outFp3.close()
    outFp4.close()

# inFp.close()
