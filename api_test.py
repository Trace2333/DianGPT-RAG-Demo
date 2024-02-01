import requests

# 设置接口的URL
url = "http://dian.org.cn:33357/chat"

# 准备请求的数据
while(1):
    data = {
        "query": input("query: ")
    }

    # 发送POST请求
    response = requests.post(url, json=data)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析响应的JSON数据
        json_data = response.json()
        # 提取响应中的回答
        answer = json_data["response"]
        # 打印回答
        print("回答:", answer)
    else:
        # 打印错误信息
        print("请求失败:", response.text)