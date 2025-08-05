import json
import os

def split_trace_file(input_file_path, events_per_chunk=100000, output_dir="./split_traces"):
    """
    将一个大的 Chrome Tracing JSON 文件按 traceEvents 数量拆分成多个小文件。

    Args:
        input_file_path (str): 输入的大 JSON 文件路径。
        events_per_chunk (int): 每个拆分文件包含的最大事件数。默认为 100000。
        output_dir (str): 存放拆分后文件的目录。默认为 "./split_traces"。
    """
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 2. 打开输入文件准备流式读取
    with open(input_file_path, 'r', encoding='utf-8') as f:
        # 读取并解析文件开头，直到 "traceEvents": [ 部分
        # 假设文件结构是标准的 { "otherKey": ..., "traceEvents": [ ... ], "otherKey2": ... }
        initial_part = ""
        char = f.read(1)
        bracket_count = 0
        in_trace_events = False

        # 读取到 traceEvents 开始的 '['
        while char:
            if char == '{':
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
            
            initial_part += char

            # 检查是否进入 traceEvents 数组
            if not in_trace_events and '"traceEvents"' in initial_part and char == '[':
                in_trace_events = True
                break # 停在 '[' 处
            
            char = f.read(1)
        
        if not in_trace_events:
            raise ValueError("未能在文件中找到 'traceEvents' 数组。")

        # 3. 确定文件的结尾部分（traceEvents 之后的内容）
        # 由于是流式处理，我们先不考虑结尾，最后手动补上 '}'
        # 通常结尾是 ], "otherKey": ... } 
        # 我们需要捕获 traceEvents 结束后的所有内容
        
        # 为了简化，我们假设除了 traceEvents，其他元数据都很小，可以先读入内存
        # 重新设计策略：先读取整个文件，找到 traceEvents 的位置，然后流式处理 events
        # 但因为文件很大，重新读取不现实。
        # 最佳策略是：边读边写，只缓存必要的头部和尾部。
        
        # --- 更健壮的流式处理方法 ---
        # 重置文件指针
        f.seek(0)
        decoder = json.JSONDecoder()
        
        # 读取整个文件内容（如果文件过大，这步会失败，需要更复杂的流处理）
        # 但对于单行 JSON，可以尝试读取头部和尾部
        
        file_content = f.read()
        try:
            # 尝试定位 traceEvents
            start_idx = file_content.find('"traceEvents"')
            if start_idx == -1:
                 raise ValueError("未能在文件中找到 'traceEvents' 数组。")
            
            # 找到 "traceEvents": [ 的 '[' 的位置
            colon_idx = file_content.find(':', start_idx)
            array_start_idx = file_content.find('[', colon_idx)
            if array_start_idx == -1:
                 raise ValueError("未能在 'traceEvents' 后找到数组开始 '['。")

            # 找到数组结束 ']'
            # 这很复杂，因为数组内可能有嵌套的 [] 或字符串包含 ]
            # 使用栈来匹配
            stack = []
            i = array_start_idx
            while i < len(file_content):
                if file_content[i] == '[':
                    stack.append('[')
                elif file_content[i] == ']':
                    if stack:
                        stack.pop()
                        if not stack: # 匹配了最外层的 ]
                            array_end_idx = i
                            break
                i += 1
            
            if i == len(file_content):
                raise ValueError("未能找到 'traceEvents' 数组的结束 ']'。")
            
            # 提取头部 (到 traceEvents 开始之前)
            header = file_content[:array_start_idx + 1] # 包含 "traceEvents": [
            
            # 提取尾部 (从 traceEvents 结束之后)
            # 确保尾部格式正确，可能需要添加逗号
            # 例如，如果原文件是 {..."traceEvents": [...], "other":...}
            # 我们需要确保生成的文件也是合法的
            # 检查 array_end_idx 后是否有逗号
            remainder_start = array_end_idx + 1
            # 跳过空格
            while remainder_start < len(file_content) and file_content[remainder_start].isspace():
                remainder_start += 1
            
            # 如果下一个字符是 '}'，说明 traceEvents 是最后一个元素，直接取到末尾
            # 否则，我们需要保留后面的元素
            remainder = file_content[remainder_start:]
            
            # 提取 traceEvents 内容
            trace_events_str = file_content[array_start_idx + 1 : array_end_idx] # 不包含首尾 []
            
        except Exception as e:
            print(f"解析文件结构时出错: {e}")
            return


        # 4. 拆分 traceEvents 字符串
        # 由于是单行，且事件通常由 {} 表示，我们可以按 '}, {' 来近似分割
        # 注意：第一个没有前导 '}'，最后一个没有后导 '{'
        # 更安全的方法是使用 json.loads 加载整个 trace_events_str，但这又回到内存问题
        # 如果确定是单行且格式规范，可以按 '}, {' 分割，但需要处理首尾
        
        # 尝试加载 trace events 到内存（如果失败，则需要更复杂的流式JSON解析）
        try:
            # 这种方法对于超大文件仍然会内存溢出
            # all_events = json.loads(f"[{trace_events_str}]") 
            # 使用流式解析器
            events_str_list = []
            idx = 0
            while idx < len(trace_events_str):
                # 跳过空白字符
                while idx < len(trace_events_str) and trace_events_str[idx].isspace():
                    idx += 1
                if idx >= len(trace_events_str):
                    break
                    
                # 开始解析一个 JSON 对象
                try:
                    event_obj, end_idx = decoder.raw_decode(trace_events_str, idx)
                    events_str_list.append(json.dumps(event_obj, separators=(',', ':'))) # 标准化为无空格的字符串
                    idx = end_idx
                    # 跳过可能的逗号和空白
                    while idx < len(trace_events_str) and (trace_events_str[idx].isspace() or trace_events_str[idx] == ','):
                        idx += 1
                except json.JSONDecodeError as e:
                    print(f"在索引 {idx} 处解析事件时出错: {e}")
                    break
            
            total_events = len(events_str_list)
            print(f"总共解析到 {total_events} 个事件。")

            if total_events == 0:
                print("未找到任何可拆分的事件。")
                return

            # 5. 写入拆分后的文件
            chunk_count = 0
            for i in range(0, total_events, events_per_chunk):
                chunk_events = events_str_list[i:i + events_per_chunk]
                chunk_count += 1
                
                # 构建新文件内容
                # 头部 + 事件列表 + 尾部
                # 注意：事件列表需要用 ',' 连接，并包裹在 [] 中
                chunk_content = header + ','.join(chunk_events) + "]" + remainder

                output_file_name = f"split_trace_part_{chunk_count:03d}.json"
                output_file_path = os.path.join(output_dir, output_file_name)

                with open(output_file_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(chunk_content)
                
                print(f"已写入拆分文件: {output_file_path}")

            print(f"拆分完成，共生成 {chunk_count} 个文件。")

        except json.JSONDecodeError as e:
            print(f"无法将 traceEvents 内容解析为 JSON 数组: {e}")
            # 如果解析失败，可以尝试字符串分割，但这很脆弱
            # print("尝试使用字符串分割方法...")
            # ... (字符串分割逻辑) ...

if __name__ == "__main__":
    # --- 请根据你的实际情况修改以下变量 ---
    INPUT_FILE_PATH = "model_convert/build-output-dit-0729/compiler/debug/subgraph_npu_0/b1/trace_dit_0729.json"  # 替换为你的大 tracing 文件路径
    EVENTS_PER_CHUNK = 50000  # 每个拆分文件最多包含 50000 个事件
    OUTPUT_DIRECTORY = "./trace_dit"  # 拆分后的文件存放目录

    split_trace_file(INPUT_FILE_PATH, EVENTS_PER_CHUNK, OUTPUT_DIRECTORY)
