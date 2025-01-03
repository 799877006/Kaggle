import requests
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import time
import random
import hashlib

def hex_sha256(text):
    """计算字符串的SHA-256哈希值"""
    return hashlib.sha256(text.encode()).hexdigest()

# 在文件开头的导入部分下面添加salt变量
salt = "xV8v4Qu54lUKrEYFZkJhB8cuOh9Asafs"  # 这是示例值，实际值可能需要更新

def get_post_content(post_id):
    """获取帖子详细内容"""
    post_url = "https://bbs-api.miyoushe.com/post/wapi/getPostFull"
    params = {"post_id": post_id}
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,ja;q=0.6",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": "_MHYUUID=c10e1e0f-a52d-4559-af72-b0fdb295e284; _ga=GA1.1.1451395397.1734324577; DEVICEFP_SEED_ID=8e3d5c6599a29d3f; DEVICEFP_SEED_TIME=1734324576981; DEVICEFP=38d8016f34968; MIHOYO_LOGIN_PLATFORM_LIFECYCLE_ID=217cf69a7d; ltoken_v2=v2_7ov09rqZcreYWix32DsEzWes5i0Kg5dA12ji5WitzX-eLA2_XMf1S8X-HjJUsdbkINdx8nIKY3rJP6aTSVfiblTnmZ752cyukmLc44HRGziWhZzYG4zj6sABrnfq62KLo9fyfTg9RWjmAybU.CAE=; ltmid_v2=0ktwjomg6t_mhy; ltuid_v2=180590137; cookie_token_v2=v2_q44iNAA62mdDczXyw0xQLozJKUY1h7zu5ayNLyKoaZmuooOHI4i4g-muXd2kycW_1E3BL075cQCtC9Xqz7KDdOcTogV671aXMaMaOmfX-4bhA3f1abkmqqgyuAvABw6sr_0uuxLJU1B4_M2lUg==.CAE=; account_mid_v2=0ktwjomg6t_mhy; account_id_v2=180590137; cookie_token=mKH7438TFSw1kPaup44dSAYPPKdcVmBJjfaED2Ve; account_id=180590137; ltoken=Bs9i6KdOn8aDSwqQtVSeSnffsfxrB9KdhHypJsxj; ltuid=180590137; acw_tc=ac11000117343541698914121ef985db5e56364f52ee0ace14987ec3b52039; _ga_KS4J8TXSHQ=GS1.1.1734345966.4.1.1734355139.0.0.0",  # 建议替换为你的Cookie
        "Origin": "https://www.miyoushe.com",
        "Referer": f"https://www.miyoushe.com/ys/article/{post_id}",
        "Sec-Ch-Ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A_Brand\";v=\"24\"",
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors", 
        "Sec-Fetch-Site": "same-site",
        "x-rpc-client_type": "4",
        "x-rpc-app_version": "2.44.1",
        "DS": generate_ds()  # 需要实现正确的DS签名生成
    }
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 增加延迟时间
            # time.sleep(random.uniform(3, 5))
            
            response = requests.get(post_url, params=params, headers=headers, timeout=10)
            
            # 如果返回码是1034,需要等待更长时间
            if response.json().get('retcode') == 1034:
                print("触发频率限制,等待30秒...")
                time.sleep(30)
                retry_count += 1
                continue
                
            print(f"获取帖子内容状态码: {response.status_code}")
            
            # 检查响应状态码
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return None
            
            data = response.json()
            print(f"获取帖子内容返回: {data.get('retcode')}, {data.get('message')}")
            
            # 检查返回码
            if data['retcode'] != 0:
                print(f"API返回错误: {data.get('message')}")
                return None
            
            # 检查数据完整性
            if 'data' not in data or 'post' not in data['data']:
                print("返回数据结构不完整")
                return None
            
            post = data['data']['post'].get('post')
            if not post:
                print("帖子数据为空")
                return None
            
            # 检查视频内容
            if data['data']['post'].get("vod_list"):
                print("帖子包含视频，跳过")
                return None
            
            # 检查必要字段
            required_fields = ['subject', 'content', 'post_id']
            if not all(field in post for field in required_fields):
                print("帖子缺少必要字段")
                return None
            
            return post
            
        except requests.Timeout:
            print("请求超时")
            return None
        except requests.RequestException as e:
            print(f"网络请求错误: {str(e)}")
            return None
        except json.JSONDecodeError:
            print("JSON解析错误")
            return None
        except Exception as e:
            print(f"获取帖子内容失败: {str(e)}")
            return None
        finally:
            retry_count += 1
            # time.sleep(5)

def generate_ds():
    t = int(time.time())
    r = ''.join(random.sample('0123456789abcdefghijklmnopqrstuvwxyz', 6))
    c = hex_sha256(f"salt={salt}&t={t}&r={r}")
    return f"{t},{r},{c}"

def get_comments(post_id):
    url = "https://bbs-api.miyoushe.com/post/wapi/getPostReplies"
    
    params = {
        "post_id": post_id,
        "size": 20,
        "last_id": 0,
        "is_hot": "true"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.miyoushe.com",
        "Referer": "https://www.miyoushe.com/"
    }
    
    all_comments = []
    page = 1
    pbar = tqdm(desc="正在获取评论", unit="页")
    
    while True:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if data["retcode"] != 0:
            break
            
        comments = data["data"]["list"]
        if not comments:
            print("\n没有更多评论了")
            break
            
        print(f"\n----- 第{page}页评论 -----")
        for i, comment in enumerate(comments, 1):
            comment_info = {
                "用户名": comment["user"]["nickname"],
                "UID": comment["user"]["uid"],
                "评论内容": comment["reply"]["content"],
                # "发布时间": datetime.fromtimestamp(comment["reply"]["created_at"])
            }
            # print(f"\n评论 {i}:")
            # print(f"用户: {comment_info['用户名']} (UID: {comment_info['UID']})")
            # print(f"内容: {comment_info['评论内容']}")
            # print(f"时间: {comment_info['发布时间']}")
            all_comments.append(comment_info)
        
        pbar.update(1)
        page += 1
        
        # 检查是否是最后一页
        if data["data"].get("is_last", False):
            print("\n已到达最后一页")
            break
            
        params["last_id"] = comments[-1]["reply"]["floor_id"]
    
    pbar.close()
    return all_comments

def get_post_list(forum_id=43, size=20, last_id=None):
    """获取帖子列表
    
    Args:
        forum_id: 论坛id,43为原神攻略区
        size: 每页数量
        last_id: 上一页最后一个帖子的时间戳
    """
    url = "https://bbs-api.miyoushe.com/painter/wapi/getRecentForumPostList"
    
    params = {
        "forum_id": forum_id,
        "gids": 2,
        "is_good": False,
        "page_size": size,
        "sort_type": 1
    }
    
    if last_id:
        params["last_id"] = last_id
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    resp = requests.get(url, params=params, headers=headers)
    data = resp.json()
    
    if data["retcode"] == 0:
        posts = []
        for item in data["data"]["list"]:
            post = {
                "post_id": item["post"]["post_id"],
                "subject": item["post"]["subject"],
            }
            posts.append(post)
            
        # 获取最后一个帖子的时间戳作为下一页的last_id
        next_id = f"{data['data']['last_id']}" if posts else None
        return posts, next_id
    
    return [], None

def process_and_save_data(post_content, comments):
    """处理数据"""
    # 构建基础数据
    data = {
        "post_id": post_content["post_id"],
        "title": post_content["subject"],
        "content": post_content["content"],
        "author": post_content.get("user", {}).get("nickname", "")
    }
    
    # 预先创建评论列
    max_comments = 20
    for i in range(1, max_comments + 1):
        data[f"reply{i}_content"] = ""
        data[f"reply{i}_author"] = ""
    
    # 添加评论数据
    for i in range(min(len(comments), max_comments)):
        data[f"reply{i+1}_content"] = comments[i]["评论内容"]
        data[f"reply{i+1}_author"] = comments[i]["用户名"]
    
    return data

def save_to_csv(data, filepath, mode='a'):
    """保存数据到CSV文件
    
    Args:
        data: 要保存的数据（单个帖子的字典或列表）
        filepath: CSV文件的完整路径
        mode: 'a' 为追加模式，'w' 为覆盖模式
    """
    df = pd.DataFrame([data] if isinstance(data, dict) else data)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if mode == 'a' and os.path.exists(filepath):
        # 追加模式且文件已存在，不写入表头
        df.to_csv(filepath, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        # 新文件或覆盖模式，写入表头
        df.to_csv(filepath, mode='w', index=False, encoding='utf-8-sig')
    
    print(f"\n数据已保存到: {filepath}")

def main():
    forum_id = 43  # 原神攻略区
    page_size = 20
    last_id = 1649145788.552378#3.6版本开始，gemma2开始训练的时候
    
    # 指定保存文件的完整路径
    output_file = "dataset/miyoushe_dataset3.csv"
    
    print("正在获取帖子列表...")
    
    with tqdm(desc="获取帖子列表", unit="页") as pbar:
        while True:
            # 使用last_id分页获取帖子
            posts, next_id = get_post_list(forum_id=forum_id, size=page_size, last_id=last_id)
            
            if not posts:
                print("\n没有更多帖子了")
                break
                
            pbar.update(1)
            
            print(f"\n----- 第{pbar.n}页 -----")
            for post in posts:
                print(f"标题: {post['subject']}")
                print(f"ID: {post['post_id']}")
                print(f"last_id: {last_id}")
            
            # 遍历处理每个帖子
            for post in tqdm(posts, desc="正在爬取帖子内容"):
                try:
                    # 获取帖子内容
                    post_content = get_post_content(post['post_id'])
                    if not post_content:
                        print(f"\n获取帖子 {post['post_id']} 内容失败，跳过")
                        continue
                    else:
                        print(f"\n获取帖子 {post['post_id']} 内容成功")
                    # 获取评论
                    comments = get_comments(post['post_id'])
                    
                    # 处理数据
                    processed_data = process_and_save_data(post_content, comments)
                    
                    # 立即保存到CSV，使用追加模式
                    save_to_csv(processed_data, output_file, 
                              mode='a' if os.path.exists(output_file) else 'w')
                    
                except Exception as e:
                    print(f"\n处理帖子 {post['post_id']} 时出错: {str(e)}")
                    continue
            # 更新last_id用于下一页
            last_id = next_id

if __name__ == "__main__":
    main()