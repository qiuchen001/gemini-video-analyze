import json

def parse_json_string(json_str):
    # 去除字符串中的转义字符和多余的换行符
    cleaned_str = json_str.replace('\\n', '').replace('\\"', '"')
    
    # 解析 JSON 字符串
    parsed_data = json.loads(cleaned_str)
    
    return parsed_data

# 示例字符串
json_str = '```json\\n{\\n  \\"videoTime\\": 16,\\n  \\"list\\": [\\n    {\\n      \\"analysis\\": \\"黄色出租车在前方突然加速并变道至主车道。\\",\\n      \\"behaviour\\": {\\n        \\"behaviourId\\": \\"B2\\",\\n        \\"behaviourName\\": \\"突然加速/减速\\",\\n        \\"timeRange\\": \\"00:00:00-00:00:02\\"\\n      }\\n    },\\n    {\\n      \\"analysis\\": \\"黄色出租车在前方突然变道至主车道，没有发出信号。\\",\\n      \\"behaviour\\": {\\n        \\"behaviourId\\": \\"B3\\",\\n        \\"behaviourName\\": \\"无警告变道\\",\\n        \\"timeRange\\": \\"00:00:02-00:00:04\\"\\n      }\\n    }\\n  ]\\n}\\n```'

# 解析字符串
parsed_data = parse_json_string(json_str)

# 打印解析结果
print(parsed_data)