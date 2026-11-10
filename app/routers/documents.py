from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
import html
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from app.services.search_service import documents
from app.core.database import SessionLocal
from app.models.document import Document, DocumentVersion
from app.services.document_titles import display_title
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# The viewer can display a page at 150% of the available viewport width.  The
# previous 108 dpi raster was smaller than that CSS display size, so vector PDF
# text was blurred by browser upscaling. 288 dpi keeps a large 150% view at
# source resolution without producing an unnecessarily large page image.
PAGE_IMAGE_DPI = 288


def _source_path(drawing_number: str, version: str | None = None) -> tuple[Path, str | None, str]:
    doc = next((item for item in documents() if item.drawing_number == drawing_number), None)
    if doc is None:
        raise HTTPException(404, "Drawing not found")
    path = Path(doc.source_path) if doc.source_path else None
    selected_version = None
    title = doc.title
    try:
        with SessionLocal() as session:
            record = session.scalar(select(Document).where(Document.drawing_number == drawing_number))
            if record:
                if version:
                    source = session.scalar(select(DocumentVersion).where(
                        DocumentVersion.document_id == record.id,
                        DocumentVersion.version == version,
                    ))
                else:
                    source = session.get(DocumentVersion, record.current_version_id) if record.current_version_id else None
                if source:
                    path = Path(source.file_path)
                    selected_version = source.version
    except SQLAlchemyError:
        pass
    if version and selected_version is None:
        raise HTTPException(404, "Version not found")
    if path is None or not path.is_file():
        raise HTTPException(404, "PDF not found")
    return path, selected_version, display_title(title, path, drawing_number)


def _bbox_style(bbox: list[float] | None) -> str:
    """Convert normalized 4-corner bbox to responsive CSS percentages."""
    if not bbox or len(bbox) != 8:
        return ""
    try:
        xs = [float(bbox[index]) for index in (0, 2, 4, 6)]
        ys = [float(bbox[index]) for index in (1, 3, 5, 7)]
    except (TypeError, ValueError):
        return ""
    left, top = max(0, min(xs)) * 100, max(0, min(ys)) * 100
    right, bottom = min(1, max(xs)) * 100, min(1, max(ys)) * 100
    if right <= left or bottom <= top:
        return ""
    return f"left:{left:.4f}%;top:{top:.4f}%;width:{right-left:.4f}%;height:{bottom-top:.4f}%;"
@router.get("/")
def list_documents(limit: int = 50, offset: int = 0):
    rows = documents()[offset:offset+limit]
    return {"items": [{"drawing_number": d.drawing_number, "title": display_title(d.title, d.source_path, d.drawing_number), "floor": d.floor, "discipline": d.discipline} for d in rows], "total": len(documents())}

@router.get("/{drawing_number}/file")
def drawing_file(drawing_number: str, version: str | None = None):
    path, _selected_version, _title = _source_path(drawing_number, version)
    media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "text/markdown"
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/{drawing_number}/page-image")
def drawing_page_image(
    drawing_number: str,
    page: int = Query(default=1, ge=1),
    version: str | None = None,
):
    path, _selected_version, _title = _source_path(drawing_number, version)
    if path.suffix.lower() != ".pdf":
        raise HTTPException(415, "Page images are available for PDF sources only")
    import pymupdf

    try:
        with pymupdf.open(path) as pdf:
            if page > len(pdf):
                raise HTTPException(404, "Page not found")
            pixmap = pdf[page - 1].get_pixmap(
                dpi=PAGE_IMAGE_DPI,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            return Response(pixmap.tobytes("png"), media_type="image/png")
    except HTTPException:
        raise
    except (OSError, RuntimeError) as exc:
        raise HTTPException(422, "Unable to render PDF page") from exc


@router.get("/{drawing_number}/viewer", response_class=HTMLResponse, include_in_schema=False)
def citation_viewer(
    drawing_number: str,
    page: int = Query(default=1, ge=1),
    version: str | None = None,
    bbox: str | None = None,
    embedded: bool = False,
):
    path, selected_version, title = _source_path(drawing_number, version)
    if path.suffix.lower() != ".pdf":
        raise HTTPException(415, "Citation viewer supports PDF sources only")
    bbox_values = None
    if bbox:
        values = bbox.split(",")
        if len(values) == 8:
            try:
                bbox_values = [float(value) for value in values]
            except ValueError:
                bbox_values = None
    style = _bbox_style(bbox_values)
    highlight = f'<div class="highlight" style="{html.escape(style)}"></div>' if style else ""
    number = html.escape(drawing_number)
    escaped_title = html.escape(title)
    revision = html.escape(selected_version or "catalog")
    image_url = f"/api/v1/documents/{number}/page-image?page={page}"
    if selected_version:
        image_url += f"&version={html.escape(selected_version)}"
    scope = "bbox領域をハイライト" if style else "ページ全体を根拠として表示（bboxなし）"
    body_class = ' class="embedded"' if embedded else ""
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{number} p.{page} - 根拠ビューア</title>
<style>.embedded .identity,.embedded .meta{{display:none}}.embedded .toolbar{{min-height:52px!important;justify-content:flex-end;padding:4px 8px!important}}.embedded .tools{{width:100%;justify-content:flex-end}}.embedded .zoom{{border-left:0;padding-left:0!important}}.embedded .zoom button{{min-width:58px;height:44px!important;padding:0 10px!important;font-size:12px!important}}</style>
<style>:root{{font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic UI",sans-serif;background:#f3f5f4;color:#202a2d;--teal:#146b68;--line:#d2d9d7}}*{{box-sizing:border-box}}body{{height:100vh;margin:0;display:flex;flex-direction:column;font-size:14px;overflow:hidden}}.toolbar{{flex:none;min-height:62px;display:flex;gap:16px;justify-content:space-between;align-items:center;padding:9px 14px;background:#fff;border-bottom:1px solid #cbd3d1}}.identity{{display:flex;align-items:center;gap:13px;min-width:0}}.back{{color:#0d5e5b;text-decoration:none;font-size:14px;font-weight:700;white-space:nowrap}}.back:hover{{text-decoration:underline}}h1{{margin:0;font-size:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.number{{color:#0d5e5b;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:18px;font-weight:850}}.tools{{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex-wrap:wrap}}.meta{{display:flex;gap:5px;align-items:center;flex-wrap:wrap}}.meta span{{display:inline-flex;border:1px solid #ced7d5;border-radius:3px;background:#f7f9f8;color:#58666a;padding:3px 7px;font-size:12px;font-weight:700}}.meta .revision{{border-color:#aabfbb;background:#eef5f3;color:#345f5b}}.meta .scope{{color:#64716f;font-weight:600}}.zoom{{display:flex;align-items:center;gap:3px;padding-left:8px;border-left:1px solid var(--line)}}.zoom button{{height:30px;border:1px solid #bdc9c6;border-radius:3px;padding:0 8px;background:#fff;color:#425155;font-size:12px;font-weight:700}}.zoom button:hover,.zoom button[aria-pressed="true"]{{border-color:#6f9893;background:#eef5f3;color:#0d5e5b}}main{{flex:1;min-height:0;width:100%;overflow:auto;padding:8px 10px 16px}}.page{{position:relative;display:block;width:calc(100vw - 22px);margin:0 auto;background:#fff;border:1px solid var(--line)}}img{{display:block;width:100%;height:auto}}.highlight{{position:absolute;border:3px solid #a85f1d;background:#d9862838;box-shadow:0 0 0 2px #fff,0 0 0 4px #a85f1dcc;pointer-events:none}}.highlight:before{{content:"根拠";position:absolute;left:-5px;bottom:calc(100% + 7px);border-radius:3px;padding:2px 6px;background:#a85f1d;color:#fff;font-size:12px;line-height:1.3;font-weight:800;white-space:nowrap;box-shadow:0 1px 3px #0004}}.highlight:after{{content:"";position:absolute;left:3px;bottom:calc(100% + 1px);border:4px solid transparent;border-top-color:#a85f1d}}.note{{color:#657276;font-size:13px;margin:8px 2px 0}}.embedded .toolbar{{min-height:46px;padding:6px 9px;gap:8px}}.embedded .identity>.back,.embedded .meta .scope{{display:none}}.embedded .identity{{gap:7px}}.embedded h1{{font-size:13px}}.embedded .number{{font-size:16px}}.embedded .meta span{{padding:2px 5px;font-size:11px}}.embedded .back{{font-size:11px}}.embedded .zoom{{padding-left:5px}}.embedded .zoom button{{height:27px;padding:0 6px;font-size:11px}}.embedded main{{padding:6px 7px 12px}}.embedded .note{{font-size:11px}}button:focus-visible,a:focus-visible{{outline:3px solid #5b9e98;outline-offset:2px}}@media(max-width:800px){{body:not(.embedded){{overflow:auto}}body:not(.embedded) .toolbar{{align-items:flex-start;flex-direction:column;gap:7px}}body:not(.embedded) .identity,body:not(.embedded) .tools{{width:100%}}body:not(.embedded) .tools{{justify-content:space-between}}body:not(.embedded) main{{min-height:72vh;overflow:auto}}body:not(.embedded) h1{{white-space:normal}}}}</style></head>
<body{body_class}><header class="toolbar"><div class="identity"><a class="back" href="javascript:history.back()">← 戻る</a><h1><span class="number">{number}</span> | {escaped_title}</h1></div><div class="tools"><div class="meta" aria-label="図面情報"><span class="revision">Rev {revision}</span><span>p.{page}</span><span class="scope">{scope}</span><a class="back" href="{html.escape(image_url)}" target="_blank" rel="noopener">画像を開く ↗</a></div><div class="zoom" aria-label="表示倍率"><button type="button" data-zoom="fit" aria-pressed="false">Fit</button><button type="button" data-zoom="100" aria-pressed="false">100%</button><button type="button" data-zoom="150" aria-pressed="false">150%</button></div></div></header><main id="viewport">
<div class="page" id="page"><img id="drawing" src="{image_url}" alt="{number} page {page}">{highlight}</div><div class="note">ハイライト位置と周辺の図面情報を確認してください。</div></main>
<script>
const viewport=document.getElementById('viewport'),page=document.getElementById('page'),drawing=document.getElementById('drawing'),highlight=document.querySelector('.highlight'),buttons=[...document.querySelectorAll('[data-zoom]')];
let zoomMode=highlight?'150':'fit';
function focusHighlight(){{if(!highlight)return;const x=page.offsetLeft+highlight.offsetLeft+highlight.offsetWidth/2-viewport.clientWidth/2;const y=page.offsetTop+highlight.offsetTop+highlight.offsetHeight/2-viewport.clientHeight/2;viewport.scrollTo({{left:Math.max(0,x),top:Math.max(0,y),behavior:'auto'}});}}
function setZoom(mode,focus=false){{const available=Math.max(320,viewport.clientWidth-20);page.style.width=(mode==='150'?available*1.5:available)+'px';zoomMode=mode;buttons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.zoom===mode)));if(focus)requestAnimationFrame(focusHighlight);}}
buttons.forEach(button=>button.addEventListener('click',()=>setZoom(button.dataset.zoom,Boolean(highlight))));
if(drawing.complete)setZoom(highlight?'150':'fit',Boolean(highlight));else drawing.addEventListener('load',()=>setZoom(highlight?'150':'fit',Boolean(highlight)),{{once:true}});
window.addEventListener('resize',()=>{{if(zoomMode==='fit'||zoomMode==='150')setZoom(zoomMode,Boolean(highlight));}});
</script></body></html>"""
