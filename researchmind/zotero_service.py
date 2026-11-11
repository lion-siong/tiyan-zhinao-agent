"""可选的 Zotero Web API 文献同步。

没有配置 Zotero 凭据时，本地 SQLite 论文库仍可完整使用。配置后既可以只同步
书目元数据，也可以把本地真实 PDF 作为附件上传到 Zotero。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings, settings as default_settings
from researchmind.models import CandidatePaper, PaperRecord


class ZoteroError(RuntimeError):
    """Zotero 未配置、拒绝请求或返回异常时抛出的错误。"""


@dataclass(frozen=True)
class ZoteroStatus:
    configured: bool
    library_type: str
    library_id: str
    message: str


class ZoteroService:
    """通过官方 Zotero Web API 写入书目元数据与 PDF 附件。"""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings

    def status(self) -> ZoteroStatus:
        return ZoteroStatus(
            configured=self.config.zotero_configured,
            library_type=self.config.zotero_library_type,
            library_id=self.config.zotero_library_id,
            message=(
                "Zotero 已配置，可手动同步书目元数据与本地 PDF。"
                if self.config.zotero_configured
                else "未配置 Zotero，当前使用本地论文库。"
            ),
        )

    def add_paper(self, paper: PaperRecord) -> str:
        """把一个论文元数据条目写入 Zotero，并返回 item key。"""

        return self._post_item(self._paper_item(paper))

    def add_candidate(self, candidate: CandidatePaper) -> str:
        """把采集候选的已核验元数据写入 Zotero。"""

        return self._post_item(self._candidate_item(candidate))

    def add_paper_with_pdf(self, paper: PaperRecord) -> tuple[str, str]:
        """把论文元数据与本地 PDF 附件一起写入 Zotero。

        返回 ``(父条目 key, 附件条目 key)``。只有 PDF 文件真实存在且可读取时
        才会创建附件；上传过程使用 Zotero Web API 的授权上传协议。
        """

        pdf_path = Path(paper.pdf_path)
        if not pdf_path.is_file():
            raise ZoteroError(
                f"论文 PDF 不存在，无法上传到 Zotero：{pdf_path}"
            )
        parent_key = self.add_paper(paper)
        attachment_key = self._create_attachment_item(
            parent_key,
            pdf_path,
        )
        self._upload_attachment(
            parent_key=parent_key,
            attachment_key=attachment_key,
            pdf_path=pdf_path,
        )
        return parent_key, attachment_key

    def _post_item(self, item: dict[str, Any]) -> str:
        body = self._request_json(
            self._library_url("items"),
            method="POST",
            payload=[item],
        )
        try:
            successful = body.get("successful", {})
            first = next(iter(successful.values()))
            key = str(first.get("key") or first.get("data", {}).get("key") or "")
        except (ValueError, TypeError, StopIteration, AttributeError) as exc:
            raise ZoteroError("Zotero 返回格式无法识别。") from exc
        if not key:
            raise ZoteroError("Zotero 未返回条目 key。")
        return key

    @staticmethod
    def _paper_item(paper: PaperRecord) -> dict[str, Any]:
        return {
            "itemType": "journalArticle",
            "title": paper.title,
            "creators": [
                {"creatorType": "author", "lastName": author}
                for author in paper.authors
            ],
            "date": paper.year,
            "publicationTitle": paper.journal,
            "DOI": paper.doi,
            "url": paper.source_url,
            "abstractNote": paper.abstract,
            "tags": [{"tag": keyword} for keyword in paper.keywords],
        }

    @staticmethod
    def _candidate_item(candidate: CandidatePaper) -> dict[str, Any]:
        return {
            "itemType": "journalArticle",
            "title": candidate.title,
            "creators": [
                {"creatorType": "author", "lastName": author}
                for author in candidate.authors
            ],
            "date": candidate.year,
            "publicationTitle": candidate.journal,
            "DOI": candidate.doi,
            "url": candidate.source_url,
            "abstractNote": candidate.abstract,
            "tags": [{"tag": keyword} for keyword in candidate.keywords],
        }

    def _create_attachment_item(
        self,
        parent_key: str,
        pdf_path: Path,
    ) -> str:
        item = {
            "itemType": "attachment",
            "parentItem": parent_key,
            "linkMode": "imported_file",
            "title": "Full Text PDF",
            "contentType": "application/pdf",
            "filename": pdf_path.name,
        }
        body = self._request_json(
            self._library_url("items"),
            method="POST",
            payload=[item],
        )
        try:
            successful = body.get("successful", {})
            first = next(iter(successful.values()))
            key = str(first.get("key") or first.get("data", {}).get("key") or "")
        except (TypeError, StopIteration, AttributeError) as exc:
            raise ZoteroError("Zotero 未返回 PDF 附件条目 key。") from exc
        if not key:
            raise ZoteroError("Zotero 未返回 PDF 附件条目 key。")
        return key

    def _upload_attachment(
        self,
        *,
        parent_key: str,
        attachment_key: str,
        pdf_path: Path,
    ) -> None:
        del parent_key
        pdf_bytes = pdf_path.read_bytes()
        authorization = self._request_json(
            self._library_url(f"items/{attachment_key}/file"),
            method="POST",
            raw_body=urllib.parse.urlencode(
                {
                    "md5": hashlib.md5(pdf_bytes).hexdigest(),
                    "filename": pdf_path.name,
                    "filesize": len(pdf_bytes),
                    "mtime": int(pdf_path.stat().st_mtime * 1000),
                }
            ).encode("utf-8"),
            content_type="application/x-www-form-urlencoded",
            extra_headers={"If-None-Match": "*"},
        )
        upload_url = str(authorization.get("url", "")).strip()
        upload_key = str(authorization.get("uploadKey", "")).strip()
        prefix = authorization.get("prefix", "")
        suffix = authorization.get("suffix", "")
        upload_content_type = str(
            authorization.get("contentType", "")
        ).strip()
        if not upload_url or not upload_key or not isinstance(prefix, str):
            raise ZoteroError("Zotero 文件上传授权信息不完整。")
        if not isinstance(suffix, str):
            raise ZoteroError("Zotero 文件上传授权信息不完整。")

        upload_body = prefix.encode("utf-8") + pdf_bytes + suffix.encode("utf-8")
        request = urllib.request.Request(
            upload_url,
            data=upload_body,
            headers={
                "Content-Type": upload_content_type,
                "User-Agent": "ResearchMind/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.llm_timeout_seconds,
            ) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise ZoteroError(
                f"Zotero 文件存储返回 HTTP {exc.code}：{detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ZoteroError(f"无法上传 PDF 到 Zotero：{exc}") from exc

        self._request_json(
            self._library_url(
                f"items/{attachment_key}/file?"
                + urllib.parse.urlencode({"upload": upload_key})
            ),
            method="POST",
            payload={},
            extra_headers={"If-None-Match": "*"},
        )

    def _library_url(self, suffix: str) -> str:
        self._ensure_configured()
        library_type = self.config.zotero_library_type
        if library_type not in {"user", "group"}:
            raise ZoteroError("ZOTERO_LIBRARY_TYPE 必须是 user 或 group。")
        library_segment = "users" if library_type == "user" else "groups"
        encoded_id = urllib.parse.quote(
            self.config.zotero_library_id,
            safe="",
        )
        return (
            f"{self.config.zotero_base_url.rstrip('/')}/"
            f"{library_segment}/{encoded_id}/{suffix.lstrip('/')}"
        )

    def _ensure_configured(self) -> None:
        if not self.config.zotero_configured:
            raise ZoteroError(
                "未配置 ZOTERO_API_KEY 和 ZOTERO_LIBRARY_ID，"
                "无法同步到 Zotero；本地论文库不受影响。"
            )

    def _request_json(
        self,
        url: str,
        *,
        method: str,
        payload: Any | None = None,
        raw_body: bytes | None = None,
        content_type: str = "application/json",
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """向 Zotero 发送请求，并把错误转换为可展示的领域异常。"""

        data = raw_body
        if data is None and payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.zotero_api_key}",
            "Content-Type": content_type,
            "Zotero-API-Version": "3",
            "User-Agent": "ResearchMind/1.0",
            **(extra_headers or {}),
        }
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.llm_timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise ZoteroError(
                f"Zotero 返回 HTTP {exc.code}：{detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ZoteroError(f"无法连接 Zotero：{exc}") from exc
        try:
            parsed = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError as exc:
            raise ZoteroError("Zotero 返回的不是有效 JSON。") from exc
        if not isinstance(parsed, dict):
            raise ZoteroError("Zotero 返回格式无法识别。")
        return parsed
