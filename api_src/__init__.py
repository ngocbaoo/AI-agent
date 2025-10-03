from .euipo import EUIPOTradeMarkSource
from .base import BaseSource
import os

# "Nhà máy" này sẽ tạo ra các source dựa trên biến môi trường
def build_sources() -> list[BaseSource]:
    sources: list[BaseSource] = []
    
    # Kích hoạt EUIPO nếu có biến môi trường EUIPO_ENABLE=1
    if os.getenv("EUIPO_ENABLE", "1").strip().lower() in {"1", "true", "yes"}:
        try:
            sources.append(EUIPOTradeMarkSource())
            print("--- INFO: EUIPO source được kích hoạt. ---")
        except Exception as e:
            print(f"--- ERROR: Lỗi khởi tạo EUIPO source: {e}")
            
    return sources

# Khởi tạo các nguồn tra cứu một lần duy nhất khi ứng dụng khởi động
available_sources = build_sources()