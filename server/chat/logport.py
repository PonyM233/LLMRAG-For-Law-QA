#此文件已无用
async def logport():
    file_path = '/home/pc/PycharmProjects/chat0.2.4/log/10_25.txt'  # 本地txt文件的路径
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content
