import torch


def get_tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def convertFileSize(size):
    # 定义单位列表
    units = 'Bytes', 'KB', 'MB', 'GB', 'TB'
    # 初始化单位为Bytes
    unit = units[0]
    # 循环判断文件大小是否大于1024，如果大于则转换为更大的单位
    for i in range(1, len(units)):
        if size >= 1024:
            size /= 1024
            unit = units[i]
        else:
            break
    # 格式化输出文件大小，保留两位小数
    return '{:.2f} {}'.format(size, unit)


# 示例用法
# g = torch.randn(19618, 4096, requires_grad=False)
x = torch.randn(19618,1024, requires_grad=False)
total_bytes = get_tensor_bytes(x)

print(total_bytes)  # 输出 240
print(convertFileSize(total_bytes))
