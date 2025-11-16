from typing import Any


def _extract_text_from_vllm(response: Any) -> str:
    """
    从vLLM响应对象中提取文本
    
    支持多种格式：
    1. {"outputs": [{"text": "..."}]}  # 标准格式
    2. {"text": "..."}                  # 简化格式
    3. 直接字符串                        # 纯文本
    
    Args:
        response: vLLM响应对象或字符串
    
    Returns:
        提取的文本，如果无法提取则返回空字符串
    """
    if response is None:
        return ""
    
    # 如果已经是字符串，直接返回
    if isinstance(response, str):
        return response.strip()
    
    # 如果是字典，尝试提取
    if isinstance(response, dict):
        # 格式1: {"outputs": [{"text": "..."}]}
        if "outputs" in response:
            outputs = response["outputs"]
            if isinstance(outputs, list) and len(outputs) > 0:
                first_output = outputs[0]
                if isinstance(first_output, dict) and "text" in first_output:
                    return str(first_output["text"]).strip()
        
        # 格式2: {"text": "..."}
        if "text" in response:
            return str(response["text"]).strip()
    
    # 其他情况，尝试转换为字符串
    try:
        return str(response).strip()
    except:
        return ""