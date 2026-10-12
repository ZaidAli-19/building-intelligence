"""Download of a PDF that a `/pdf` request produced (held in memory; unguessable id)."""

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from building_with_rag import export_pdf

router = APIRouter()
_ID = re.compile(r"^[0-9a-f]{32}$")


@router.get("/v1/exports/{export_id}.pdf")
def download(export_id: str) -> Response:
    data = export_pdf.get(export_id) if _ID.match(export_id) else None
    if data is None:
        raise HTTPException(status_code=404, detail={
            "code": "export_not_found", "message": "No such export (it may have expired)."})
    return Response(content=data, media_type="application/pdf", headers={
        "Content-Disposition": 'attachment; filename="legal-sections.pdf"'})
