import csv
import json
import asyncio
import aiohttp
from collections import defaultdict
from tqdm import tqdm
import os
from typing import List, Dict, Any, Optional
import time

# API配置
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.v3.cm/v1/chat/completions") 
API_KEY = os.getenv("LLM_API_KEY", "your_key")
CONCURRENCY = 5          # 并发请求数
BATCH_SIZE = 20          # 每批数据量
TIMEOUT = 30             # 请求超时(秒)
MAX_CONTENT_LENGTH = 500 # 单条内容最大长度
MAX_RETRIES = 3          # 最大重试次数

# 分类体系
CATEGORIES = ['军事', '医药健康', '政治', '教育考试', '文体娱乐', 
              '其他', '灾难事故', '社会生活', '科技', '财经商业']

PROMPT_TEMPLATE = """
请严格按以下要求对每条文本进行分类：

一、候选分类（仅限选择下列项）：
{categories}

二、分类规则：
1. 根据文本整体语义判断
2. 优先选择更具体的分类
3. 不确定时选择"其他"

三、返回格式要求以json形式进行：
{{"results": [{{"id": 序号, "category": "分类名称"}},...] }}

四、待分类文本（共{count}条）：
{items}
"""

class BatchClassifier:
    def __init__(self):
        self.stats = defaultdict(int)
        self.semaphore = asyncio.Semaphore(CONCURRENCY)
        self.fallback_semaphore = asyncio.Semaphore(CONCURRENCY // 2 or 1)  # 降级处理的并发限制

    async def classify_batch(self, session: aiohttp.ClientSession, batch: List[Dict], 
                            batch_id: int, is_fallback: bool = False) -> List[Dict]:
        """处理单批次数据"""
        # 避免空批次
        if not batch:
            return []
            
        items_text = "\n\n".join(
            f"ID {idx}: 标题《{item.get('title', '')[:50]}》\n"
            f"内容：{item.get('content', '')[:MAX_CONTENT_LENGTH]}{'...' if len(item.get('content', '')) > MAX_CONTENT_LENGTH else ''}"
            for idx, item in enumerate(batch))
        
        prompt = PROMPT_TEMPLATE.format(
            categories="、".join(CATEGORIES),
            count=len(batch),
            items=items_text
        )
        
        payload = {
            "model": "gemini-1.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }

        # 选择适当的信号量
        semaphore = self.fallback_semaphore if is_fallback else self.semaphore
        
        async with semaphore:
            for retry in range(MAX_RETRIES):
                try:
                    async with session.post(
                        LLM_API_URL,
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json=payload,
                        timeout=TIMEOUT
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise ValueError(f"API返回非200状态码: {response.status}, 响应: {error_text[:200]}")
                            
                        data = await response.json()
                        result = self._process_batch_result(data, batch, batch_id)
                        if result:  # 如果成功处理
                            return result
                        raise ValueError("结果处理失败，将重试") 
                        
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    # 网络错误特殊处理
                    if retry < MAX_RETRIES - 1:
                        wait_time = (retry + 1) * 2  # 指数退避
                        print(f"批次{batch_id}请求失败 (重试 {retry+1}/{MAX_RETRIES}): {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"批次{batch_id}在{MAX_RETRIES}次尝试后仍然失败: {str(e)}")
                        # 单批次所有重试都失败时，如果不是降级模式，尝试降级处理
                        if not is_fallback and len(batch) > 1:
                            print(f"批次{batch_id}降级为单条处理")
                            return await self._fallback_classify(session, batch, batch_id)
                        else:
                            # 返回默认分类结果
                            return self._apply_default_category(batch)
                except Exception as e:
                    if retry < MAX_RETRIES - 1:
                        wait_time = (retry + 1) * 2
                        print(f"批次{batch_id}处理异常 (重试 {retry+1}/{MAX_RETRIES}): {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"批次{batch_id}在{MAX_RETRIES}次尝试后仍然失败: {str(e)}")
                        # 同上，处理降级逻辑
                        if not is_fallback and len(batch) > 1:
                            print(f"批次{batch_id}降级为单条处理")
                            return await self._fallback_classify(session, batch, batch_id)
                        else:
                            return self._apply_default_category(batch)

    def _process_batch_result(self, api_response: Dict, original_batch: List[Dict], batch_id: int) -> List[Dict]:
        """处理API返回结果"""
        try:
            # 适应不同API格式的响应处理
            message_content = ""
            
            # 尝试处理OpenAI格式响应
            if "choices" in api_response and api_response["choices"]:
                if "message" in api_response["choices"][0]:
                    message_content = api_response["choices"][0]["message"].get("content", "")
                elif "text" in api_response["choices"][0]:  # 兼容某些API格式
                    message_content = api_response["choices"][0].get("text", "")
            
            # 尝试处理Google Gemini格式响应
            elif "candidates" in api_response and api_response["candidates"]:
                if "content" in api_response["candidates"][0]:
                    content_parts = api_response["candidates"][0]["content"].get("parts", [])
                    message_content = "".join(part.get("text", "") for part in content_parts if "text" in part)
            
            # 处理直接返回内容的情况
            elif "content" in api_response:
                message_content = api_response["content"]
                
            if not message_content:
                raise ValueError(f"无法从API响应中提取内容: {json.dumps(api_response)[:200]}...")
            
            # 解析JSON内容
            result_data = json.loads(message_content)
            
            # 支持多种可能的响应字段名
            results_key = None
            for key in ["results", "结果", "result", "分类结果"]:
                if key in result_data:
                    results_key = key
                    break
                    
            if not results_key:
                raise ValueError(f"API响应中找不到结果字段: {json.dumps(result_data)[:200]}...")
            
            # 创建ID到分类的映射
            classified_items = {}
            for item in result_data[results_key]:
                item_id = None
                
                # 支持不同的ID字段名
                for id_field in ["id", "ID", "序号", "index"]:
                    if id_field in item:
                        item_id = item[id_field]
                        break
                
                if item_id is None:
                    continue
                    
                # 支持不同的分类字段名
                category = None
                for cat_field in ["category", "分类", "类别", "class"]:
                    if cat_field in item:
                        category = item[cat_field]
                        break
                        
                if category:
                    try:
                        # 确保ID转为整数
                        item_id = int(str(item_id).strip())
                        classified_items[item_id] = category
                    except (ValueError, TypeError):
                        print(f"无效的ID格式: {item_id}")
            
            # 关联原始数据
            processed_batch = []
            for idx, item in enumerate(original_batch):
                if item:  # 确保项目有效
                    category = classified_items.get(idx, "其他")
                    category = self._validate_category(category)
                    item_copy = item.copy()  # 避免修改原始数据
                    item_copy["predicted_category"] = category
                    self.stats[category] += 1
                    processed_batch.append(item_copy)
            
            return processed_batch
            
        except json.JSONDecodeError as e:
            print(f"批次{batch_id}响应不是有效的JSON: {str(e)}")
            return []
        except Exception as e:
            print(f"批次{batch_id}结果解析失败: {str(e)}")
            return []

    def _validate_category(self, category: str) -> str:
        """验证分类结果有效性"""
        if not isinstance(category, str):
            return "其他"
            
        # 去除标点和空白
        category = category.strip("。，、；： \t\n")
        
        # 检查完全匹配
        if category in CATEGORIES:
            return category
            
        # 检查部分匹配（处理可能的错别字或变体）
        for valid_cat in CATEGORIES:
            if valid_cat in category or category in valid_cat:
                return valid_cat
                
        return "其他"

    async def _fallback_classify(self, session: aiohttp.ClientSession, batch: List[Dict], batch_id: int) -> List[Dict]:
        """降级处理：单条分类"""
        results = []
        tasks = []
        
        # 创建单条处理任务
        for idx, item in enumerate(batch):
            if not item:  # 跳过空项
                continue
                
            single_batch = [item]
            # 使用is_fallback=True标记这是降级请求
            task = asyncio.create_task(
                self.classify_batch(session, single_batch, f"{batch_id}-{idx}", is_fallback=True)
            )
            tasks.append(task)
        
        # 限制并发数的处理所有单条请求
        for chunk_tasks in [tasks[i:i+3] for i in range(0, len(tasks), 3)]:
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for result in chunk_results:
                if isinstance(result, Exception):
                    print(f"降级处理异常: {str(result)}")
                    continue
                    
                if result:  # 确保有返回结果
                    results.extend(result)
                    
            # 短暂延迟避免速率限制
            await asyncio.sleep(1)
        
        # 处理所有失败的项目
        failed_items = []
        for item in batch:
            found = False
            for result_item in results:
                if (item.get('title') == result_item.get('title') and 
                    item.get('content') == result_item.get('content')):
                    found = True
                    break
            
            if not found and item:
                item_copy = item.copy()
                item_copy["predicted_category"] = "其他"
                self.stats["其他"] += 1
                failed_items.append(item_copy)
                
        results.extend(failed_items)
        return results
        
    def _apply_default_category(self, batch: List[Dict]) -> List[Dict]:
        """为批次应用默认分类"""
        result = []
        for item in batch:
            if item:  # 确保项目有效
                item_copy = item.copy()
                item_copy["predicted_category"] = "其他"
                self.stats["其他"] += 1
                result.append(item_copy)
        return result

async def process_csv(input_path: str, output_path: str) -> Dict[str, Any]:
    """主处理流程"""
    try:
        # 读取数据
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            items = [row for row in reader if row and any(row.values())]  # 过滤空行
            
        if not items:
            raise ValueError(f"CSV文件为空或格式无效: {input_path}")
            
        # 确保所有必要的字段存在
        required_fields = ['title', 'content']
        for field in required_fields:
            if any(field not in item for item in items):
                print(f"警告: 部分记录缺少'{field}'字段")
        
        # 分批处理
        classifier = BatchClassifier()
        batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
        
        # 并发执行
        tcp_connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
        timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        
        async with aiohttp.ClientSession(connector=tcp_connector, timeout=timeout) as session:
            tasks = [
                classifier.classify_batch(session, batch, idx)
                for idx, batch in enumerate(batches)
            ]
            
            results = []
            successful_tasks = 0
            with tqdm(total=len(tasks), desc="批次处理进度") as pbar:
                for future in asyncio.as_completed(tasks):
                    try:
                        batch_result = await future
                        if batch_result:
                            results.extend(batch_result)
                            successful_tasks += 1
                    except Exception as e:
                        print(f"任务处理异常: {str(e)}")
                    finally:
                        pbar.update(1)
        
        # 保存结果
        total_items = len(items)
        success_items = len(results)
        
        output = {
            "metadata": {
                "total_items": total_items,
                "processed_items": success_items,
                "success_rate": f"{success_items/total_items*100:.1f}%" if total_items > 0 else "0%",
                "batch_size": BATCH_SIZE,
                "categories": CATEGORIES
            },
            "statistics": dict(sorted(
                classifier.stats.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            "data": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            
        return output
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="批量文本分类处理工具")
    parser.add_argument("--input", help="输入CSV文件路径", required=True)
    parser.add_argument("--output", default="output.json", help="输出JSON文件路径")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"批处理大小 (默认: {BATCH_SIZE})")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help=f"并发请求数 (默认: {CONCURRENCY})")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help=f"请求超时秒数 (默认: {TIMEOUT})")
    args = parser.parse_args()
    
    # 更新全局配置
    BATCH_SIZE = args.batch_size
    CONCURRENCY = args.concurrency
    TIMEOUT = args.timeout
    
    print("启动批量分类处理器")
    print(f"• 并发数: {CONCURRENCY}")
    print(f"• 批量大小: {BATCH_SIZE}")
    print(f"• 请求超时: {TIMEOUT}秒")
    print(f"• 可用分类: {', '.join(CATEGORIES)}")
    print(f"• 输入文件: {args.input}")
    print(f"• 输出文件: {args.output}")
    
    try:
        start_time = time.time()
        output = asyncio.run(process_csv(args.input, args.output))
        elapsed_time = time.time() - start_time
        
        print(f"\n处理完成，总耗时: {elapsed_time:.2f}秒")
        print(f"• 总条目数: {output['metadata']['total_items']}")
        print(f"• 成功处理: {output['metadata']['processed_items']}")
        print(f"• 成功率: {output['metadata']['success_rate']}")
        print(f"• 分类统计: ", end="")
        for category, count in list(output['statistics'].items())[:3]:
            print(f"{category}({count})", end=" ")
        print("...")
        print(f"结果已保存至: {args.output}")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        exit(1)
