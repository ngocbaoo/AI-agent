# Trong file test.py của bạn

import json
from dotenv import load_dotenv
# Import tool đã được nâng cấp
from tools import euipo_trademark_search_tool 

# Tải các biến môi trường
load_dotenv()

print("--- [BẮT ĐẦU] Lấy 10 mẫu đầu tiên từ Sandbox ---")

# Tạo đầu vào cho tool:
# - name: "*" dùng ký tự wildcard để khớp với bất kỳ tên nào.
# - nice_class: Bỏ trống để không lọc theo nhóm.
tool_input = {
    "name": "*" 
}

# Gọi tool
results = euipo_trademark_search_tool.invoke(tool_input)

# In kết quả
print("--- KẾT QUẢ ---")
print(json.dumps(results, indent=2, ensure_ascii=False))
print("--- [KẾT THÚC] ---")