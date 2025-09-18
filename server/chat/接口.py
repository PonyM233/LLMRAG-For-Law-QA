from flask import Flask, request, send_file

app = Flask(__name__)

# 定义接口路由
@app.route('/read_txt_file', methods=['GET'])
def read_txt_file():
    file_path = '/home/pc/PycharmProjects/chat0.2.4/log/10_25.txt'  # 本地txt文件的路径

    # 检查文件是否存在
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found", 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='192.168.1.104',port=7861,debug=True)