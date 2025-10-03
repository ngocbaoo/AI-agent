import os
import time
import requests
from typing import List, Dict, Any, Optional
from requests.exceptions import RequestException
from .base import BaseSource, NormalizedHit

class EUIPOTradeMarkSource(BaseSource):
    """
    Đóng gói toàn bộ logic giao tiếp với EUIPO API Sandbox.
    """
    def __init__(self):
        self.name = "EUIPO Sandbox TM Search"
        self.base_url = os.environ.get("EUIPO_API_BASE", "")
        self.auth_url = os.environ.get("EUIPO_AUTH_URL", "")
        self.client_id = os.environ.get("EU_SANDBOX_ID", "")
        self.client_secret = os.environ.get("EU_SANDBOX_SECRET", "")
        self.scope = "uid"
        self._cached_token: str = ""
        self._token_exp_ts: float = 0.0

    def _get_token(self) -> str:
        now = time.time()
        if self._cached_token and self._token_exp_ts > now + 60:
            return self._cached_token

        print("--- [SOURCE LOG] Yêu cầu token mới từ EUIPO Sandbox...")
        if not all([self.client_id, self.client_secret]):
             return ""

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'grant_type': 'client_credentials', 'client_id': self.client_id, 'client_secret': self.client_secret, 'scope': self.scope}
        
        try:
            response = requests.post(self.auth_url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            response_data = response.json()
            self._cached_token = response_data.get("access_token", "")
            self._token_exp_ts = now + response_data.get("expires_in", 3600)
            print("--- [SOURCE LOG] Đã lấy token thành công. ---")
            return self._cached_token
        except RequestException as e:
            print(f"--- [SOURCE ERROR] Không thể lấy token: {e}")
            self._cached_token = ""
            self._token_exp_ts = 0
            return ""

    def _do_search(self, query_text: str, nice_class: Optional[int] = None) -> List[Dict[str, Any]]:
        rsql_query = f"wordMarkSpecification.verbalElement==*{query_text}*"

        params = {"query": rsql_query, "size": 10}
        
        for attempt in range(2): # Thử lại tối đa 2 lần
            try:
                token = self._get_token()
                if not token: raise ValueError("Không thể lấy access token.")

                headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}', 'X-IBM-Client-Id': self.client_id}
                response = requests.get(self.base_url, headers=headers, params=params, timeout=20)

                if response.status_code == 401 and attempt == 0:
                    print("--- [SOURCE LOG] Lỗi 401, tự động làm mới token và thử lại...")
                    self._cached_token = ""
                    continue

                response.raise_for_status()
                return response.json().get("trademarks", [])
            except RequestException as e:
                print(f"--- [SOURCE ERROR] Lỗi khi tra cứu: {e}")
                if attempt == 1: raise e
        return []

    def _normalize_item(self, item: Dict[str, Any]) -> NormalizedHit:
        title = item.get("wordMarkSpecification", {}).get("verbalElement", "")
        owner_list = item.get("applicants", [])
        owner_names = [o.get("name", "") for o in owner_list if o]
        
        return NormalizedHit(
            source=self.name, kind="trademark", id=item.get("applicationNumber", "-"),
            title=title, url=None, jurisdiction="EU", status=item.get("status", ""),
            filing_date=item.get("applicationDate"), owner=", ".join(filter(None, owner_names)),
            classes=[str(c) for c in item.get("niceClasses", [])], abstract=None
        )

    def search(self, brands: List[str], nice_class: Optional[int] = None) -> List[NormalizedHit]:
        hits: List[NormalizedHit] = []
        for brand_name in brands[:3]:
            try:
                raw_items = self._do_search(brand_name, nice_class)
                for item in raw_items:
                    hits.append(self._normalize_item(item))
            except Exception as e:
                hits.append(NormalizedHit(
                    source=self.name, kind="info", id="-", title=f"Lỗi khi tra cứu '{brand_name}': {e}",
                    url=None, jurisdiction="EU", status=None, filing_date=None,
                    owner=None, classes=None, abstract=None
                ))
        return hits