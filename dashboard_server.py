"""
Local FastAPI dashboard for the wellfound-apply-helper.

Run:
    python dashboard_server.py
    # open http://localhost:9876

Endpoints:
    GET  /                        — dashboard HTML (live, server-rendered)
    GET  /api/blurb/<id>          — return prepared blurb (used by bookmarklet)
    POST /api/answer_questions    — generate per-question answers for any modal
                                    (used by the smart bookmarklet)
    POST /api/confirm_submitted/<id>   — flip pending → submitted
    POST /api/mark_external/<id>       — flip apply_mode to external
    POST /api/save_blurb/<id>     — save edits to a bundle's blurb
    POST /api/regen/<id>          — regenerate blurb via LLM pipeline
    POST /api/withdraw/<id>       — mark bundle as withdrawn (local label)
    POST /api/delete_pending/<id> — remove a pending bundle directory
    POST /api/ingest_candidates   — bulk-import jobs from external scraper
    POST /api/ingest_pool_chunk   — append to a discovery pool file

All data flows through ./profile.json + ./projects/*.md. No personal info
is hardcoded in this file.
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("FastAPI not installed. Run: pip install -r requirements.txt")
    raise SystemExit(1)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from blurb import compose_blurb, scrub_ai_tells, quick_checks, answer_question_prompt
from llm_utils import openai_client
from profile import load_profile
from project_pool import load_full_pool, format_pool_for_prompt


# ---------- Config + paths --------------------------------------------------

DEFAULT_BUNDLE_ROOT = Path.home() / "auto_apply_wellfound"
BUNDLE_ROOT = Path(os.environ.get("WELLFOUND_HELPER_BUNDLE_DIR", DEFAULT_BUNDLE_ROOT)).expanduser()
PENDING_ROOT = BUNDLE_ROOT / "_pending"
PROFILE_DIR = Path(os.environ.get("WELLFOUND_HELPER_PROFILE_DIR",
                                   Path.home() / ".wellfound_helper_browser")).expanduser()


def _chrome_app_path() -> str | None:
    """Return path to the Chrome binary on this OS, or None if unknown.
    Override with WELLFOUND_HELPER_CHROME_PATH env var."""
    override = os.environ.get("WELLFOUND_HELPER_CHROME_PATH")
    if override:
        return override
    sys_name = platform.system()
    candidates = {
        "Darwin": [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ],
        "Windows": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "Linux": [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ],
    }
    for p in candidates.get(sys_name, []):
        if Path(p).exists():
            return p
    return None


CHROME_APP = _chrome_app_path()


# ---------- bundle IO -------------------------------------------------------

def _bundle_for(listing_id: str) -> Path:
    """Find the bundle directory holding result.json with this listing_id.
    Searches BOTH submitted (BUNDLE_ROOT) and pending (BUNDLE_ROOT/_pending)."""
    for root in (BUNDLE_ROOT, PENDING_ROOT):
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir() or d.name == "_pending":
                continue
            rj = d / "result.json"
            if not rj.exists():
                continue
            try:
                data = json.loads(rj.read_text())
            except Exception:
                continue
            if str(data.get("listing_id")) == str(listing_id):
                return d
    raise HTTPException(404, f"no bundle for listing_id={listing_id}")


def _load_submitted() -> list[dict]:
    out: list[dict] = []
    if not BUNDLE_ROOT.exists():
        return out
    for d in sorted(BUNDLE_ROOT.iterdir()):
        if not d.is_dir() or d.name == "_pending":
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        data["_bundle_name"] = d.name
        data["_is_pending"] = False
        out.append(data)
    out.sort(key=lambda r: r.get("submitted_at") or r.get("_bundle_name", ""), reverse=True)
    return out


def _load_pending() -> list[dict]:
    out: list[dict] = []
    if not PENDING_ROOT.exists():
        return out
    for d in sorted(PENDING_ROOT.iterdir()):
        if not d.is_dir():
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        data["_bundle_name"] = d.name
        data["_is_pending"] = True
        out.append(data)
    out.sort(key=lambda r: r.get("prepared_at") or r.get("_bundle_name", ""), reverse=True)
    return out


def _load_all() -> list[dict]:
    return _load_pending() + _load_submitted()


def _save_bundle(listing_id: str, patch: dict) -> dict:
    d = _bundle_for(listing_id)
    rj = d / "result.json"
    data = json.loads(rj.read_text())
    data.update(patch)
    rj.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def _promote_pending_to_submitted(listing_id: str, *, banner_text: str = "") -> bool:
    """Move a pending bundle to BUNDLE_ROOT with status=submitted."""
    for d in PENDING_ROOT.iterdir() if PENDING_ROOT.exists() else []:
        if not d.is_dir():
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        if str(data.get("listing_id")) != str(listing_id):
            continue
        data["status"] = "submitted"
        data["submitted_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        if banner_text:
            data["success_banner_text"] = banner_text
        rj.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
        dst = BUNDLE_ROOT / d.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(d), str(dst))
        return True
    return False


# ---------- HTML rendering --------------------------------------------------

STATUS_COLOR = {
    "submitted": "#16a34a",
    "withdrawn": "#a3a3a3",
    "dry_run": "#0891b2",
    "already_applied": "#6b7280",
    "resubmitted": "#16a34a",
    "pending": "#f59e0b",
}


def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


# Bookmarklets — both rendered as drag-to-install links in the banner

_FILL_BLURB_JS = (
    "javascript:(async()=>{"
    "const m=location.pathname.match(/jobs\\/(\\d+)-/)||location.search.match(/job_listing_slug=(\\d+)-/);"
    "if(!m){alert('Open a Wellfound job page first');return;}"
    "const id=m[1];"
    "let ta=document.querySelector('textarea[name*=\"customQuestionAnswers\"],textarea[name=\"userNote\"]');"
    "if(!ta){const btn=[...document.querySelectorAll('button')].find(b=>/^apply$/i.test(b.textContent.trim())&&b.offsetParent);if(btn){btn.click();await new Promise(r=>setTimeout(r,1800));ta=document.querySelector('textarea[name*=\"customQuestionAnswers\"],textarea[name=\"userNote\"]');}}"
    "if(!ta){alert('No Apply textarea found. Click Apply first.');return;}"
    "let blurb;"
    "try{const r=await fetch('http://localhost:9876/api/blurb/'+id);if(!r.ok){alert('Dashboard returned '+r.status);return;}blurb=(await r.json()).blurb;}catch(e){alert('Cannot reach dashboard at localhost:9876');return;}"
    "const s=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ta),'value').set;"
    "s.call(ta,blurb);"
    "ta.dispatchEvent(new Event('input',{bubbles:true}));"
    "ta.dispatchEvent(new Event('change',{bubbles:true}));"
    "})();"
)

_SMART_BOOKMARKLET_JS = (
    "javascript:(async()=>{"
    "let lid=null;"
    "const m1=location.pathname.match(/jobs\\/(\\d+)-/);"
    "const m2=location.search.match(/job_listing_slug=(\\d+)-/);"
    "const m3=location.pathname.match(/jobs\\/applications\\/\\d+-(\\d+)/);"
    "if(m1)lid=m1[1]; else if(m2)lid=m2[1]; else if(m3)lid=m3[1];"
    "if(!lid){for(const a of document.querySelectorAll('a[href^=\"/jobs/\"]')){const h=a.getAttribute('href')||'';const mh=h.match(/^\\/jobs\\/(\\d+)-/);if(mh){lid=mh[1];break;}}}"
    "const findTextareas=()=>{let arr=[...document.querySelectorAll('.ReactModal__Content textarea,[role=\"dialog\"] textarea,textarea[name*=\"customQuestionAnswers\"],textarea[name=\"userNote\"]')];if(!arr.length){arr=[...document.querySelectorAll('textarea')].filter(t=>t.offsetParent&&t.offsetHeight>20);}return [...new Set(arr)];};"
    "let tas=findTextareas();"
    "if(!tas.length){const applyBtns=[...document.querySelectorAll('button,a')].filter(b=>{const t=(b.textContent||'').trim();return /^(apply|apply now)$/i.test(t)&&b.offsetParent;});for(const btn of applyBtns){btn.click();await new Promise(r=>setTimeout(r,2500));tas=findTextareas();if(tas.length)break;}}"
    "if(!tas.length){const ext=[...document.querySelectorAll('button,a')].find(el=>/apply on (website|company website)/i.test(el.textContent));if(ext){try{const r=await fetch('http://localhost:9876/api/blurb/'+lid);if(r.ok){const j=await r.json();await navigator.clipboard.writeText(j.blurb||'');alert('External apply detected. Blurb copied to clipboard. Click Apply on website + paste into the company form.');return;}}catch(e){}alert('External apply (no Wellfound textarea). Click Apply on website.');return;}}"
    "const modalRoot=document.querySelector('.ReactModal__Content, [role=\"dialog\"]');"
    "const radioGroups={};"
    "if(modalRoot){for(const radio of modalRoot.querySelectorAll('input[type=\"radio\"]')){if(!radio.name)continue;(radioGroups[radio.name]=radioGroups[radio.name]||[]).push(radio);}}"
    "let radiosFilled=0;"
    "if(Object.keys(radioGroups).length){for(const name of Object.keys(radioGroups)){const radios=radioGroups[name];if(radios.length<2)continue;let qtext='';let p=radios[0].closest('label')?.parentElement||radios[0].parentElement;for(let d=0;d<6&&p;d++){const t=(p.textContent||'').trim();if(t&&t.length>10&&t.length<400){qtext=t.split(/\\n/)[0].trim();break;}p=p.parentElement;}const ql=qtext.toLowerCase();let pick=null;if(/visa|sponsor|h[\\s-]?1b|opt|cpt/.test(ql))pick='yes';else if(/citizen|green card|permanent resident/.test(ql))pick='no';else if(/willing to travel/.test(ql))pick='yes';else if(/relocate|live in.*new york|nyc/.test(ql))pick='yes';else if(/authorized to work|legally allowed/.test(ql))pick='yes';else if(/start.*immediate|available.*start/.test(ql))pick='yes';else if(/over 18|at least 18/.test(ql))pick='yes';else if(/felony|criminal|convicted/.test(ql))pick='no';else if(/(remote|hybrid|onsite|in[- ]?office)/.test(ql))pick='__skip__';if(pick&&pick!=='__skip__'){const target=radios.find(r=>{const lbl=r.closest('label')?.textContent||r.value||'';return new RegExp('^\\\\s*'+pick+'\\\\b','i').test(lbl);});if(target&&!target.checked){target.click();radiosFilled++;}}}}"
    "if(!tas.length){if(radiosFilled){alert('Filled '+radiosFilled+' yes/no question(s).');return;}alert('No textareas or radios found. Click Apply first.');return;}"
    "const getPrompt=(ta)=>{let p=ta.parentElement;for(let d=0;d<6&&p;d++){const txt=(p.textContent||'').replace(ta.value||'','').trim();if(txt&&txt.length>5&&txt.length<500){const first=txt.split(/\\n+/)[0].trim();if(first)return first.slice(0,400);}p=p.parentElement;}return '';};"
    "const questions=tas.map((ta,i)=>({name:ta.name||'textarea_'+i,prompt:getPrompt(ta),idx:i}));"
    "const jdEl=document.querySelector('[class*=\"jdBody\"],[class*=\"description\"],main article,main section');"
    "const jd=((jdEl&&jdEl.innerText)||document.body.innerText||'').slice(0,4500);"
    "const company=document.querySelector('a[href^=\"/company/\"]')?.textContent?.trim()||document.title.split(' at ')[1]?.split(' \\u2022 ')[0]||'';"
    "const title=document.querySelector('h1,h2')?.textContent?.trim()||document.title.split(' at ')[0]||'';"
    "const pd=document.createElement('div');"
    "pd.style.cssText='position:fixed;top:20px;right:20px;background:#fef3c7;border:2px solid #f59e0b;padding:12px 16px;border-radius:8px;font-family:system-ui;font-size:14px;z-index:99999;box-shadow:0 4px 12px rgba(0,0,0,0.15);max-width:320px;';"
    "pd.innerHTML='Generating '+questions.length+' answer(s)...';"
    "document.body.appendChild(pd);"
    "let resp;"
    "try{resp=await(await fetch('http://localhost:9876/api/answer_questions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({listing_id:lid,company,title,jd,questions})})).json();}catch(e){pd.innerHTML='Backend unreachable (localhost:9876).';setTimeout(()=>pd.remove(),5000);return;}"
    "if(!resp.ok){pd.innerHTML='Error: '+(resp.error||'failed');setTimeout(()=>pd.remove(),5000);return;}"
    "let filled=0;"
    "for(const ans of resp.answers||[]){const ta=tas.find(t=>(t.name||'')===ans.name)||tas[(ans.idx||0)];if(!ta||!ans.answer)continue;const setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ta),'value').set;setter.call(ta,ans.answer);ta.dispatchEvent(new Event('input',{bubbles:true}));ta.dispatchEvent(new Event('change',{bubbles:true}));filled++;}"
    "const src=resp.source==='bundle'?'(reused dashboard blurb, no LLM cost)':resp.source==='mixed'?'(bundle blurb + LLM for other questions)':'(gpt-4o-mini)';"
    "pd.innerHTML='Filled '+filled+'/'+tas.length+' textarea(s)'+(radiosFilled?' + '+radiosFilled+' radio(s)':'')+'<br><span style=\"font-size:11px;color:#854d0e;\">'+src+'</span><br>Review + Send. <span style=\"font-size:11px;color:#475569;\">(watching for Send banner...)</span>';"
    "const SUCCESS_RE=/Congrats!?\\s*Your application has been submitted|SUCCESS!?\\s*YOUR APPLICATION HAS BEEN SENT/i;"
    "const watchStart=Date.now();"
    "const watcher=setInterval(async()=>{"
    "if(Date.now()-watchStart>5*60*1000){clearInterval(watcher);pd.innerHTML='Watcher timed out. Click I-submitted-it on dashboard if you sent.';setTimeout(()=>pd.remove(),5000);return;}"
    "if(SUCCESS_RE.test(document.body.innerText)){clearInterval(watcher);if(lid){try{await fetch('http://localhost:9876/api/confirm_submitted/'+lid,{method:'POST'});}catch(e){}}pd.style.background='#dcfce7';pd.style.borderColor='#16a34a';pd.innerHTML='Submitted detected! '+(lid?'Moved to history on dashboard.':'(no listing_id detected, mark manually).');setTimeout(()=>pd.remove(),5000);}"
    "},1500);"
    "})();"
)


def _render_bookmarklet_banner() -> str:
    return f"""
    <details class="banner" open>
      <summary>📌 Bookmarklet install — drag the orange button to your bookmark bar →</summary>
      <div class="banner-body">
        <p style="margin:6px 0 14px;color:#854d0e;font-weight:600;font-size:14px;">
          1️⃣ Press <kbd>⌘+Shift+B</kbd> (macOS) or <kbd>Ctrl+Shift+B</kbd> (Windows/Linux) to show the bookmark bar<br>
          2️⃣ Drag the orange button below onto the bookmark bar — do NOT click<br>
          3️⃣ Use it on any Wellfound apply modal: click <i>Apply</i>, then click the bookmarklet
        </p>

        <div style="margin:18px 0;">
          <a href="{_esc(_SMART_BOOKMARKLET_JS)}" class="bookmarklet-drag"
             onclick="alert('Drag this button to your bookmark bar — do not click it here.'); return false;">
            ⚡ Fill smart  ← drag to bookmark bar
          </a>
        </div>

        <details style="margin-top:14px;">
          <summary style="font-size:12px;color:#854d0e;cursor:pointer;">Legacy: Fill blurb (single textarea, no LLM)</summary>
          <div style="margin:8px 0;">
            <a href="{_esc(_FILL_BLURB_JS)}" class="bookmarklet-drag bookmarklet-drag-secondary"
               onclick="alert('Drag to bookmark bar — do not click here.'); return false;">
              📝 Fill blurb (legacy)
            </a>
          </div>
        </details>

        <div class="banner-howto" style="margin-top:18px;">
          <b>What Fill smart does:</b>
          <ol style="margin:6px 0 0 18px;padding:0;">
            <li>Scans every textarea in the modal + reads each question prompt</li>
            <li>Sends to your local backend at <code>localhost:9876</code></li>
            <li>Backend reuses pre-prepared blurb if the listing is already in your dashboard (zero LLM cost)</li>
            <li>Otherwise generates per-question answers via gpt-4o-mini grounded in your <code>projects/</code> KBs</li>
            <li>Fills all textareas + watches for Wellfound's success banner</li>
            <li>On success → auto-flags the bundle as submitted in your dashboard</li>
          </ol>
        </div>
      </div>
    </details>
    """


def _render_card(b: dict) -> str:
    listing_id = _esc(b.get("listing_id"))
    status = b.get("status", "?")
    color = STATUS_COLOR.get(status, "#dc2626")
    is_pending = bool(b.get("_is_pending"))
    new_blurb = scrub_ai_tells(b.get("resubmit_blurb") or "")
    original_answer = ""
    for ans in b.get("answers") or []:
        original_answer = ans.get("answer") or ans.get("value") or ""
        if original_answer:
            break
    original_answer = scrub_ai_tells(original_answer)
    ts = b.get("withdrawn_at") or b.get("submitted_at") or b.get("prepared_at") or ""

    extra_class = "pending" if is_pending else ""
    if is_pending and (b.get("apply_mode") or "").lower() == "external":
        extra_class += " external"
    mode_badge = ""
    if is_pending:
        am = (b.get("apply_mode") or "wellfound").lower()
        if am == "external":
            mode_badge = '<span class="mode-badge external-badge">🔗 external apply</span>'
        else:
            mode_badge = '<span class="mode-badge wellfound-badge">📝 wellfound apply</span>'

    if is_pending:
        original_section = ""
        blurb_label = "Prepared blurb (edit before triggering — bookmarklet will paste this):"
        actions = f"""
        <button onclick="saveBlurb('{listing_id}')">💾 Save edits</button>
        <button onclick="regen('{listing_id}')">🔄 Re-generate</button>
        <button class="primary" onclick="openWellfound('{listing_id}', '{_esc(b.get('url'))}')">
          🪟 Open Wellfound
        </button>
        <button class="success" onclick="confirmSubmitted('{listing_id}')">
          ✅ I submitted it
        </button>
        <button onclick="markExternal('{listing_id}')">🔗 Mark external</button>
        <button class="danger" onclick="deletePending('{listing_id}')">🗑️ Delete</button>
        """
    else:
        original_section = f"""
      <div class="qa-section">
        <div class="qa-label">Original blurb (sent earlier):</div>
        <div class="qa-original">{_esc(original_answer) or '<i>none</i>'}</div>
      </div>"""
        blurb_label = "Re-submit blurb (edit before re-sending):"
        actions = f"""
        <button onclick="saveBlurb('{listing_id}')">💾 Save edits</button>
        <button onclick="regen('{listing_id}')">🔄 Re-generate</button>
        <button class="danger" onclick="markWithdrawn('{listing_id}')">🛑 Mark withdrawn (local)</button>
        """

    return f"""
    <div class="card {extra_class}" data-listing-id="{listing_id}">
      <div class="card-head">
        <div>
          <div class="title">{_esc(b.get('title'))} {mode_badge}</div>
          <div class="company">{_esc(b.get('company'))}</div>
        </div>
        <div class="status" style="background:{color}">{_esc(status)}</div>
      </div>
      <div class="meta">
        <span>listing #{listing_id}</span>
        <span>{_esc(ts)}</span>
        <a href="{_esc(b.get('url'))}" target="_blank">job page →</a>
      </div>
{original_section}
      <div class="qa-section">
        <div class="qa-label">{blurb_label}</div>
        <textarea class="resubmit-text" rows="10">{_esc(new_blurb)}</textarea>
      </div>
      <div class="actions">{actions}</div>
      <div class="result" id="result-{listing_id}"></div>
    </div>
    """


def _render_full(bundles: list[dict]) -> str:
    counts: dict[str, int] = {}
    for b in bundles:
        s = b.get("status", "?")
        counts[s] = counts.get(s, 0) + 1
    stats = " · ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    pending = [b for b in bundles if b.get("_is_pending")]
    submitted = [b for b in bundles if not b.get("_is_pending")]
    pending_cards = "\n".join(_render_card(b) for b in pending)
    submitted_cards = "\n".join(_render_card(b) for b in submitted)
    proj_count = len(load_full_pool())
    profile = load_profile()
    profile_name = profile.get("name", "<NOT SET>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>wellfound-apply-helper — dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f8fafc; color: #0f172a; margin: 0 auto; padding: 24px;
        max-width: 1100px; }}
h1 {{ margin: 0 0 4px; font-size: 24px; }}
h2 {{ margin: 28px 0 8px; font-size: 17px; color: #334155; }}
.stats {{ color: #64748b; font-size: 13px; margin-bottom: 8px; }}
.banner {{ background: #fefce8; border: 1px solid #fde68a; border-radius: 8px; padding: 10px 14px; margin: 14px 0 24px; font-size: 13px; }}
.banner summary {{ cursor: pointer; font-weight: 600; color: #854d0e; }}
.banner-body {{ padding-top: 8px; }}
.banner ol {{ margin: 6px 0 10px 18px; padding: 0; }}
.banner li {{ margin: 4px 0; }}
.banner code {{ background: #fef3c7; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
.banner-howto {{ margin-top: 10px; padding: 8px 10px; background: white; border-radius: 5px; font-size: 12px; color: #475569; }}
.bookmarklet-drag {{ display: inline-block; padding: 14px 24px; background: linear-gradient(135deg, #f97316, #ea580c); color: white; font-weight: 700; font-size: 16px; border-radius: 8px; text-decoration: none; cursor: grab; user-select: none; box-shadow: 0 3px 8px rgba(234,88,12,0.3); }}
.bookmarklet-drag:hover {{ transform: translateY(-1px); }}
.bookmarklet-drag-secondary {{ background: linear-gradient(135deg, #94a3b8, #64748b); padding: 8px 16px; font-size: 13px; box-shadow: none; }}
kbd {{ background: white; border: 1px solid #cbd5e1; border-bottom-width: 2px; padding: 1px 6px; border-radius: 4px; font-family: ui-monospace,monospace; font-size: 12px; }}
.filter-bar {{ display: flex; gap: 12px; align-items: center; margin: 12px 0 8px; padding: 10px 14px; background: #f1f5f9; border-radius: 8px; }}
#filter-box {{ flex: 1; padding: 8px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; outline: none; }}
#filter-box:focus {{ border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.15); }}
.filter-counts {{ font-size: 12px; color: #64748b; min-width: 140px; text-align: right; }}
.card {{ background: white; border-radius: 10px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
.card.pending {{ background: #fffbeb; border: 1px solid #fde68a; }}
.card.external {{ background: #fef2f2; border: 1px solid #fecaca; }}
.card.filtered-out {{ display: none; }}
.card-head {{ display: flex; justify-content: space-between; gap: 12px; }}
.title {{ font-weight: 600; font-size: 16px; }}
.company {{ color: #475569; font-size: 14px; margin-top: 2px; }}
.status {{ color: white; padding: 4px 10px; border-radius: 999px; font-size: 11px; text-transform: uppercase; font-weight: 600; }}
.mode-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; margin-left: 8px; vertical-align: middle; }}
.external-badge {{ background: #fee2e2; color: #b91c1c; }}
.wellfound-badge {{ background: #dbeafe; color: #1d4ed8; }}
.meta {{ display: flex; gap: 14px; font-size: 12px; color: #64748b; margin: 8px 0 12px; flex-wrap: wrap; }}
.meta a {{ color: #2563eb; text-decoration: none; }}
.qa-section {{ margin: 12px 0; }}
.qa-label {{ font-size: 12px; font-weight: 600; color: #475569; margin-bottom: 6px; }}
.qa-original {{ background: #f1f5f9; border-radius: 6px; padding: 10px 12px; font-size: 13px; white-space: pre-wrap; line-height: 1.5; max-height: 100px; overflow-y: auto; }}
.resubmit-text {{ width: 100%; box-sizing: border-box; font-family: inherit; font-size: 13px; line-height: 1.5; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 6px; resize: vertical; }}
.actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
.actions button {{ padding: 7px 12px; border-radius: 6px; border: 1px solid #cbd5e1; background: white; cursor: pointer; font-size: 13px; }}
.actions button:hover {{ background: #f1f5f9; }}
.actions button.primary {{ background: #2563eb; color: white; border-color: #2563eb; }}
.actions button.danger {{ background: #fee2e2; color: #b91c1c; border-color: #fecaca; }}
.actions button.success {{ background: #dcfce7; color: #15803d; border-color: #bbf7d0; font-weight: 600; }}
.result {{ margin-top: 10px; font-size: 12px; color: #475569; min-height: 1em; }}
.result.ok {{ color: #16a34a; }}
.result.err {{ color: #dc2626; }}
.section-empty {{ font-size: 13px; color: #94a3b8; padding: 8px 4px; }}
</style>
</head>
<body>
  <h1>wellfound-apply-helper</h1>
  <div class="stats">User: <b>{_esc(profile_name)}</b> · {len(pending)} pending · {len(submitted)} submitted · {proj_count} projects loaded · refreshed {datetime.now().isoformat(timespec='seconds')}</div>
  {_render_bookmarklet_banner()}
  <div class="filter-bar">
    <input id="filter-box" type="search" placeholder="🔍 Search company or title (space-separated tokens AND together)" autocomplete="off">
    <span class="filter-counts" id="filter-counts">showing all</span>
  </div>
  <h2>📌 Pending ({len(pending)})</h2>
  {pending_cards or '<div class="section-empty">No pending bundles. Use the bookmarklet on a Wellfound job — it generates fresh per click.</div>'}
  <h2>📂 Submitted history ({len(submitted)})</h2>
  {submitted_cards or '<div class="section-empty">No submitted bundles yet.</div>'}

<script>
(function() {{
  const box = document.getElementById('filter-box');
  const countsEl = document.getElementById('filter-counts');
  if (!box) return;
  const cards = [...document.querySelectorAll('.card')].map(c => {{
    const company = (c.querySelector('.company')?.textContent || '').toLowerCase();
    const title = (c.querySelector('.title')?.textContent || '').toLowerCase();
    const lid = (c.getAttribute('data-listing-id') || '').toLowerCase();
    return {{ el: c, hay: company + ' || ' + title + ' || ' + lid }};
  }});
  const total = cards.length;
  const updateCounts = (shown) => {{
    countsEl.textContent = shown === total ? `showing all ${{total}}` : `showing ${{shown}} of ${{total}}`;
  }};
  let timer = null;
  box.addEventListener('input', () => {{
    clearTimeout(timer);
    timer = setTimeout(() => {{
      const q = box.value.trim().toLowerCase();
      const tokens = q ? q.split(/\\s+/).filter(t => t.length > 0) : [];
      let shown = 0;
      for (const {{el, hay}} of cards) {{
        const match = tokens.length === 0 || tokens.every(t => hay.includes(t));
        el.classList.toggle('filtered-out', !match);
        if (match) shown++;
      }}
      updateCounts(shown);
    }}, 100);
  }});
  box.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') {{ box.value = ''; box.dispatchEvent(new Event('input')); box.blur(); }}
  }});
}})();

async function _post(url, body) {{
  const r = await fetch(url, {{method: 'POST', headers: {{'Content-Type':'application/json'}}, body: body ? JSON.stringify(body) : null}});
  return await r.json().catch(()=>({{ok:false,error:'bad response'}}));
}}
function _show(id, text, ok) {{
  const el = document.getElementById('result-' + id);
  if (el) {{ el.textContent = text; el.className = 'result ' + (ok===true?'ok':ok===false?'err':''); }}
}}
function _blurb(id) {{
  const card = document.querySelector(`[data-listing-id="${{id}}"]`);
  return card.querySelector('.resubmit-text').value;
}}
async function saveBlurb(id) {{ _show(id, 'saving...'); const r = await _post('/api/save_blurb/' + id, {{blurb: _blurb(id)}}); _show(id, r.ok?'saved':'error: '+(r.error||'?'), r.ok); }}
async function regen(id) {{ _show(id, 'regenerating...'); const r = await _post('/api/regen/' + id); if (r.ok) {{ document.querySelector(`[data-listing-id="${{id}}"] .resubmit-text`).value = r.blurb; _show(id, 'new blurb generated.', true); }} else {{ _show(id, 'error: '+(r.error||'?'), false); }} }}
async function markWithdrawn(id) {{ _show(id, 'marking...'); const r = await _post('/api/withdraw/' + id); _show(id, r.ok?'marked withdrawn locally':'error: '+(r.error||'?'), r.ok); }}
async function openWellfound(id, url) {{ await _post('/api/save_blurb/' + id, {{blurb: _blurb(id)}}); window.open(url, '_blank'); _show(id, 'tab opened. Click ⚡ Fill smart bookmarklet on it.', true); }}
async function deletePending(id) {{ if (!confirm('Delete pending bundle?')) return; const r = await _post('/api/delete_pending/' + id); if (r.ok) document.querySelector(`[data-listing-id="${{id}}"]`).remove(); else _show(id, 'error: '+(r.error||'?'), false); }}
async function confirmSubmitted(id) {{ if (!confirm('Mark as submitted?')) return; const r = await _post('/api/confirm_submitted/' + id); if (r.ok) {{ _show(id, 'marked submitted', true); setTimeout(()=>window.location.reload(), 700); }} else {{ _show(id, 'error: '+(r.error||'?'), false); }} }}
async function markExternal(id) {{ const r = await _post('/api/mark_external/' + id); if (r.ok) {{ _show(id, 'marked external', true); setTimeout(()=>window.location.reload(), 600); }} else {{ _show(id, 'error: '+(r.error||'?'), false); }} }}
</script>
</body>
</html>
"""


# ---------- FastAPI app ----------------------------------------------------

app = FastAPI(title="wellfound-apply-helper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index():
    return _render_full(_load_all())


@app.get("/api/blurb/{listing_id}")
def get_blurb(listing_id: str):
    """Used by the legacy Fill blurb bookmarklet."""
    try:
        d = _bundle_for(listing_id)
    except HTTPException:
        return JSONResponse({"ok": False, "error": "no bundle for this listing"}, status_code=404)
    data = json.loads((d / "result.json").read_text())
    blurb = data.get("resubmit_blurb") or data.get("blurb") or ""
    if not blurb:
        for ans in (data.get("answers") or []):
            blurb = ans.get("answer") or ans.get("value") or ""
            if blurb:
                break
    return {
        "ok": bool(blurb),
        "listing_id": listing_id,
        "blurb": scrub_ai_tells(blurb),
        "title": data.get("title"),
        "company": data.get("company"),
    }


class BlurbBody(BaseModel):
    blurb: str | None = None


@app.post("/api/save_blurb/{listing_id}")
def save_blurb(listing_id: str, body: BlurbBody):
    if not body.blurb:
        return {"ok": False, "error": "blurb empty"}
    _save_bundle(listing_id, {"resubmit_blurb": scrub_ai_tells(body.blurb)})
    return {"ok": True}


@app.post("/api/regen/{listing_id}")
def regen(listing_id: str):
    d = _bundle_for(listing_id)
    data = json.loads((d / "result.json").read_text())
    blurb = compose_blurb(
        jd=data.get("jd_excerpt") or "",
        title=data.get("title", ""),
        company=data.get("company", ""),
    )
    _save_bundle(listing_id, {"resubmit_blurb": blurb})
    return {"ok": True, "blurb": blurb}


@app.post("/api/withdraw/{listing_id}")
def withdraw(listing_id: str):
    _save_bundle(listing_id, {
        "status": "withdrawn",
        "withdrawn_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    })
    return {"ok": True}


@app.post("/api/confirm_submitted/{listing_id}")
def confirm_submitted(listing_id: str):
    if _promote_pending_to_submitted(str(listing_id), banner_text="manually confirmed"):
        return {"ok": True}
    return {"ok": False, "error": "no pending bundle to promote"}


@app.post("/api/mark_external/{listing_id}")
def mark_external(listing_id: str):
    if not PENDING_ROOT.exists():
        return {"ok": False, "error": "no _pending root"}
    for d in PENDING_ROOT.iterdir():
        if not d.is_dir():
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        if str(data.get("listing_id")) == str(listing_id):
            data["apply_mode"] = "external"
            rj.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            return {"ok": True}
    return {"ok": False, "error": "not found"}


@app.post("/api/delete_pending/{listing_id}")
def delete_pending(listing_id: str):
    if not PENDING_ROOT.exists():
        return {"ok": False, "error": "no _pending root"}
    for d in PENDING_ROOT.iterdir():
        if not d.is_dir():
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        if str(data.get("listing_id")) == str(listing_id):
            shutil.rmtree(d)
            return {"ok": True}
    return {"ok": False, "error": "not found"}


# ---------- Smart bookmarklet endpoint: answer all questions in a modal -----

_INTEREST_RE = re.compile(
    r"(what\s+interests?\s+you|why\s+(this|us|do you want to work|are you (excited|interested))|"
    r"tell\s+(us|me)\s+(about\s+yourself|why)|"
    r"about\s+(this\s+(role|company|position)|the\s+role)|"
    r"motivat(ion|ed))",
    re.IGNORECASE,
)


@app.post("/api/answer_questions")
async def answer_questions(payload: dict):
    """Generate answers for every textarea in a Wellfound apply modal in one
    LLM call. Reuses existing bundle blurb for 'interest in company' questions
    when the bundle is found, saving an LLM call."""
    listing_id = payload.get("listing_id")
    company = (payload.get("company") or "").strip()
    title = (payload.get("title") or "").strip()
    jd = (payload.get("jd") or "").strip()
    questions = payload.get("questions") or []
    if not isinstance(questions, list) or not questions:
        return {"ok": False, "error": "no questions in payload"}

    # Enrich from bundle if we have one (gives us pre-prepped blurb + better context)
    bundle_blurb = ""
    if listing_id:
        try:
            d = _bundle_for(str(listing_id))
            bdata = json.loads((d / "result.json").read_text())
            company = company or bdata.get("company") or ""
            title = title or bdata.get("title") or ""
            jd = jd or bdata.get("jd_excerpt") or ""
            bundle_blurb = bdata.get("resubmit_blurb") or bdata.get("blurb") or ""
        except Exception:
            pass

    # Route: "interest" questions get answered from the bundle's pre-prepped
    # blurb (higher-quality, no LLM cost). Everything else goes through LLM.
    preanswered: dict[str, str] = {}
    remaining_qs = []
    for q in questions:
        prompt = (q.get("prompt") or "").lower()
        if bundle_blurb and _INTEREST_RE.search(prompt):
            preanswered[q.get("name")] = bundle_blurb
        else:
            remaining_qs.append(q)

    if not remaining_qs:
        return {
            "ok": True,
            "answers": [{"name": n, "answer": scrub_ai_tells(a)} for n, a in preanswered.items()],
            "count": len(preanswered),
            "source": "bundle",
            "bundle_used": True,
        }

    profile = load_profile()
    pool = format_pool_for_prompt()
    questions_dump = "\n".join(
        f"Q{i+1} (name={(q.get('name') or 'q'+str(i))!r}): {q.get('prompt') or '(no prompt)'}"
        for i, q in enumerate(remaining_qs)
    )
    user = f"""COMPANY: {company or '(unknown)'}
ROLE: {title or '(unknown)'}

JD (first 3500 chars):
{jd[:3500]}

QUESTIONS to answer ({len(remaining_qs)}):
{questions_dump}

PROJECT POOL (ground every concrete claim in these):
{pool[:6000]}

JSON only: {{"answers": [{{"name": "<verbatim>", "answer": "..."}}, ...]}}"""

    try:
        client = openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2500,
            temperature=0.55,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": answer_question_prompt(profile)},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    out = [{"name": n, "answer": scrub_ai_tells(a), "source": "bundle"} for n, a in preanswered.items()]
    for a in parsed.get("answers", []):
        ans = scrub_ai_tells(a.get("answer", "") or "")
        out.append({"name": a.get("name"), "answer": ans, "source": "llm"})
    return {
        "ok": True,
        "answers": out,
        "count": len(out),
        "source": "mixed" if preanswered and remaining_qs else ("bundle" if preanswered else "llm"),
        "bundle_used": bool(preanswered),
    }


# ---------- Ingest endpoints (for external scrapers / batch import) --------

@app.post("/api/ingest_candidates")
async def ingest_candidates(payload: dict):
    cands = payload.get("candidates") or payload.get("jobs") or []
    if not isinstance(cands, list):
        return {"ok": False, "error": "expected {'candidates': [...]}"}
    out_dir = Path("/tmp/wf_ingest.json")
    out_dir.write_text(json.dumps(cands, ensure_ascii=False))
    return {"ok": True, "saved_to": str(out_dir), "count": len(cands)}


@app.post("/api/ingest_pool_chunk")
async def ingest_pool_chunk(payload: dict):
    chunk = payload.get("chunk") or payload.get("jobs") or []
    if not isinstance(chunk, list):
        return {"ok": False, "error": "expected {'chunk': [...]}"}
    path = Path("/tmp/wf_pool_all.json")
    existing: dict[str, dict] = {}
    if path.exists():
        try:
            arr = json.loads(path.read_text())
            for j in arr:
                if j.get("listing_id"):
                    existing[str(j["listing_id"])] = j
        except Exception:
            pass
    added = 0
    for j in chunk:
        lid = str(j.get("listing_id") or "")
        if not lid or lid in existing:
            continue
        existing[lid] = j
        added += 1
    path.write_text(json.dumps(list(existing.values()), ensure_ascii=False))
    return {"ok": True, "added": added, "total_pool": len(existing), "path": str(path)}


# ---------- Entry ----------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9876)
    args = ap.parse_args()
    print(f"wellfound-apply-helper dashboard: http://localhost:{args.port}/")
    print(f"Chrome binary: {CHROME_APP or '(not found; set WELLFOUND_HELPER_CHROME_PATH if you need Auto Open)'}")
    print(f"Bundle dir: {BUNDLE_ROOT}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
