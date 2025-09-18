import os

def remove_prefix(text):
    start_index = 0
    for i, char in enumerate(text):
        if '\u4e00' <= char <= '\u9fff':
            start_index = i
            break
    return text[start_index:]

def find_answer(question, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    question_found = False
    answer = ""

    for line in lines:
        line = line.strip()

        if line.endswith('？'):
            current_question = remove_prefix(line[:])
            #current_question = line[3:].strip()

            if current_question == question:
                question_found = True
            else:
                question_found = False

        elif question_found:
            answer += line + '\n'

    return answer.strip()

def search_answers_in_directory(question, directory_path):
    answers = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            result = find_answer(question, file_path)

            if result:
                answers.append(f"{result}")
                #answers.append(f"Answer to '{question}' in {filename}:\n{result}")

    return answers

if __name__ == "__main__":
    # Example usage
    question_to_search = "分配遗产时必须按照均等分割的原则吗？"
    directory_path = "/home/pc/PycharmProjects/chat0.2.4副本/knowledge_base/law_qa/content"  # Replace with the actual file path

    results = search_answers_in_directory(question_to_search, directory_path)

    for result in results:
        print(result)