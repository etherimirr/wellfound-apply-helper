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
    "const findApplyBtn=()=>[...document.querySelectorAll('button,a')].find(b=>{const t=(b.textContent||'').trim();return /^(apply|apply now)$/i.test(t)&&b.offsetParent;});"
    "let tas=findTextareas();"
    "if(!tas.length){let applyBtn=findApplyBtn();let waited=0;while(!applyBtn&&waited<8000){await new Promise(r=>setTimeout(r,500));waited+=500;applyBtn=findApplyBtn();}if(applyBtn){applyBtn.click();let twait=0;while(!tas.length&&twait<8000){await new Promise(r=>setTimeout(r,500));twait+=500;tas=findTextareas();}}}"
    "if(!tas.length){const ext=[...document.querySelectorAll('button,a')].find(el=>/apply on (website|company website)/i.test(el.textContent));if(ext){try{const r=await fetch('http://localhost:9876/api/blurb/'+lid);if(r.ok){const j=await r.json();await navigator.clipboard.writeText(j.blurb||'');alert('External apply detected. Blurb copied to clipboard. Click Apply on website + paste into the company form.');return;}}catch(e){}alert('External apply (no Wellfound textarea). Click Apply on website.');return;}}"
    "const modalRoot=([...document.querySelectorAll('.ReactModal__Content, [role=\"dialog\"]')].find(m=>m.querySelector('input,textarea'))||document);"
    "const radioGroups={};"
    "if(modalRoot){for(const radio of modalRoot.querySelectorAll('input[type=\"radio\"]')){if(!radio.name)continue;(radioGroups[radio.name]=radioGroups[radio.name]||[]).push(radio);}}"
    "let inputsFilled=0;"
    "try{const pf=await(await fetch('http://localhost:9876/api/profile_fields')).json();if(pf.ok&&modalRoot){const setInp=(el,val)=>{if(!el||!val||el.value)return;const s=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el),'value').set;s.call(el,val);el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));inputsFilled++;};const getInputLabel=(inp)=>{if(inp.labels&&inp.labels.length){const l=(inp.labels[0].textContent||'').trim();if(l)return l.toLowerCase();}const aria=inp.getAttribute('aria-label')||'';if(aria)return aria.toLowerCase();const lbId=inp.getAttribute('aria-labelledby');if(lbId){const ref=document.getElementById(lbId);if(ref){const t=(ref.textContent||'').trim();if(t)return t.toLowerCase();}}if(inp.placeholder)return inp.placeholder.toLowerCase();const nameId=((inp.name||'')+' '+(inp.id||'')).toLowerCase();let prev=inp.previousElementSibling;for(let i=0;i<3&&prev;i++){const t=(prev.textContent||'').trim();if(t&&t.length>1&&t.length<100)return (t+' '+nameId).toLowerCase();prev=prev.previousElementSibling;}let p=inp.parentElement;for(let d=0;d<2&&p;d++){const children=Array.from(p.children);const idx=children.findIndex(c=>c===inp||c.contains(inp));if(idx>0){for(let i=idx-1;i>=Math.max(0,idx-3);i--){const c=children[i];const tg=c.tagName;if(tg==='LABEL'||tg==='H3'||tg==='H4'||tg==='P'||tg==='SPAN'||tg==='STRONG'||tg==='B'){const t=(c.textContent||'').trim();if(t&&t.length>1&&t.length<100)return (t+' '+nameId).toLowerCase();}}}p=p.parentElement;}return nameId;};for(const inp of modalRoot.querySelectorAll('input[type=\"url\"],input[type=\"text\"],input[type=\"email\"],input[type=\"tel\"],input:not([type])')){if(!inp.offsetParent)continue;const l=getInputLabel(inp);if(/current\\s*(company|employer|title|role|position)|cover\\s*letter|previous|salary|expected|why|describe|introduce/.test(l))continue;if(/\\blinkedin\\b/.test(l))setInp(inp,pf.linkedin);else if(/\\bgithub\\b/.test(l))setInp(inp,pf.github);else if(/\\bemail\\b/.test(l))setInp(inp,pf.email);else if(/\\bphone\\b|\\btel\\b|\\bmobile\\b/.test(l))setInp(inp,pf.phone);else if(/\\b(website|portfolio|personal\\s*site)\\b/.test(l)||(/\\bother\\b/.test(l)&&/urls?/.test(l)))setInp(inp,pf.website);else if(/\\b(first\\s*name|given\\s*name|preferred\\s*name)\\b/.test(l))setInp(inp,pf.firstName);else if(/\\b(last\\s*name|surname|family\\s*name)\\b/.test(l))setInp(inp,pf.lastName);else if(/\\bcity\\b|current\\s*location/.test(l)&&!inp.value)setInp(inp,pf.city);}}}catch(e){console.warn('input fill err',e);}"
    "let radiosFilled=0;"
    "if(Object.keys(radioGroups).length){for(const name of Object.keys(radioGroups)){const radios=radioGroups[name];if(radios.length<2)continue;let qtext='';let p=radios[0].closest('label')?.parentElement||radios[0].parentElement;for(let d=0;d<6&&p;d++){const t=(p.textContent||'').trim();if(t&&t.length>10&&t.length<400){qtext=t.split(/\\n/)[0].trim();break;}p=p.parentElement;}const ql=qtext.toLowerCase();let pick=null;if(/visa|sponsor|h[\\s-]?1b|opt|cpt/.test(ql))pick='yes';else if(/citizen|green card|permanent resident/.test(ql))pick='no';else if(/willing to travel/.test(ql))pick='yes';else if(/relocate|live in.*new york|nyc/.test(ql))pick='yes';else if(/authorized to work|legally allowed/.test(ql))pick='yes';else if(/start.*immediate|available.*start/.test(ql))pick='yes';else if(/over 18|at least 18/.test(ql))pick='yes';else if(/felony|criminal|convicted/.test(ql))pick='no';else if(/(remote|hybrid|onsite|in[- ]?office)/.test(ql))pick='__skip__';if(pick&&pick!=='__skip__'){const target=radios.find(r=>{const lbl=r.closest('label')?.textContent||r.value||'';return new RegExp('^\\\\s*'+pick+'\\\\b','i').test(lbl);});if(target&&!target.checked){target.click();radiosFilled++;}}}}"
    "if(!tas.length){if(radiosFilled){alert('Filled '+radiosFilled+' yes/no question(s).');return;}alert('No textareas or radios found. Click Apply first.');return;}"
    "const getPrompt=(ta)=>{if(ta.labels&&ta.labels.length){const l=(ta.labels[0].textContent||'').trim();if(l&&l.length<400)return l;}const aria=ta.getAttribute('aria-label')||'';if(aria&&aria.length<400)return aria.trim();const ph=ta.placeholder||'';if(ph&&ph.length<400)return ph.trim();let prev=ta.previousElementSibling;for(let i=0;i<3&&prev;i++){const t=(prev.textContent||'').trim();if(t&&t.length>3&&t.length<400)return t.split(/\\n+/)[0].trim().slice(0,400);prev=prev.previousElementSibling;}let p=ta.parentElement;for(let d=0;d<3&&p;d++){const lbl=p.querySelector('label,h3,h4');if(lbl&&lbl.textContent){const t=lbl.textContent.trim();if(t&&t.length>3&&t.length<400)return t.split(/\\n+/)[0].trim().slice(0,400);}const ownText=Array.from(p.childNodes).filter(n=>n.nodeType===3).map(n=>(n.textContent||'').trim()).filter(Boolean).join(' ');if(ownText&&ownText.length>3&&ownText.length<400)return ownText.slice(0,400);p=p.parentElement;}return '';};"
    "const questions=tas.map((ta,i)=>({name:ta.name||'textarea_'+i,prompt:getPrompt(ta),idx:i}));"
    "const jdEl=document.querySelector('[class*=\"jdBody\"],[class*=\"description\"],main article,main section');"
    "const jd=((jdEl&&jdEl.innerText)||document.body.innerText||'').slice(0,4500);"
    "const getSkills=()=>{const heads=document.querySelectorAll('h1,h2,h3,h4,h5,h6,strong,b,dt');let chips=[];for(const h of heads){const tx=(h.textContent||'').trim();if(!/^skills?$/i.test(tx))continue;let p=h.parentElement;for(let d=0;d<4&&p;d++){const cands=Array.from(p.querySelectorAll('span,div,a,li,button')).filter(e=>{const t=(e.textContent||'').trim();if(!t||t.length<2||t.length>200)return false;if(/^skills?$/i.test(t))return false;if(e.children.length>2)return false;if(e.contains(h)||h.contains(e))return false;return true;}).map(e=>(e.textContent||'').trim());if(cands.length>=2&&cands.length<=40){chips=cands;break;}p=p.parentElement;}if(chips.length)break;}return [...new Set(chips)].slice(0,30).join(' | ');};"
    "const skillsText=getSkills();"
    "const company=document.querySelector('a[href^=\"/company/\"]')?.textContent?.trim()||document.title.split(' at ')[1]?.split(' \\u2022 ')[0]||'';"
    "const title=document.querySelector('h1,h2')?.textContent?.trim()||document.title.split(' at ')[0]||'';"
    "const pd=document.createElement('div');"
    "pd.style.cssText='position:fixed;top:20px;right:20px;background:#fef3c7;border:2px solid #f59e0b;padding:12px 16px;border-radius:8px;font-family:system-ui;font-size:14px;z-index:99999;box-shadow:0 4px 12px rgba(0,0,0,0.15);max-width:320px;';"
    "pd.innerHTML='Generating '+questions.length+' answer(s)...';"
    "document.body.appendChild(pd);"
    "let resp;"
    "try{resp=await(await fetch('http://localhost:9876/api/answer_questions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({listing_id:lid,company,title,jd,skills_text:skillsText,questions})})).json();}catch(e){pd.innerHTML='Backend unreachable (localhost:9876).';setTimeout(()=>pd.remove(),5000);return;}"
    "if(!resp.ok){pd.innerHTML='Error: '+(resp.error||'failed');setTimeout(()=>pd.remove(),5000);return;}"
    "let filled=0;"
    "for(const ans of resp.answers||[]){const ta=tas.find(t=>(t.name||'')===ans.name)||tas[(ans.idx||0)];if(!ta||!ans.answer)continue;const setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ta),'value').set;setter.call(ta,ans.answer);ta.dispatchEvent(new Event('input',{bubbles:true}));ta.dispatchEvent(new Event('change',{bubbles:true}));filled++;}"
    "let srcLabel;"
    "if(resp.source==='bundle'){srcLabel='\\u267b\\ufe0f Reused dashboard blurb (no LLM cost)';pd.style.background='#dcfce7';pd.style.borderColor='#16a34a';}"
    "else if(resp.source==='bundle+patch'){srcLabel='\\u267b\\ufe0f Bundle + JD-keyword patch (gpt-4o-mini)';pd.style.background='#d1fae5';pd.style.borderColor='#059669';}"
    "else if(resp.source==='fitpitch'){srcLabel='\\ud83d\\udcdd Full FIT-PITCH (gpt-4o, 4-paragraph)';pd.style.background='#e0e7ff';pd.style.borderColor='#6366f1';}"
    "else if(resp.source==='fitpitch+patch'){srcLabel='\\ud83d\\udcdd FIT-PITCH + JD-keyword patch';pd.style.background='#ddd6fe';pd.style.borderColor='#7c3aed';}"
    "else if(resp.source==='mixed'){srcLabel='\\u267b\\ufe0f Mixed: bundle + LLM';pd.style.background='#fef3c7';pd.style.borderColor='#f59e0b';}"
    "else{srcLabel='\\ud83e\\udd16 Fresh gpt-4o-mini (short answer)';pd.style.background='#dbeafe';pd.style.borderColor='#2563eb';}"
    "let kwLine='';if(resp.jd_keyword_total){kwLine='<br><span style=\"font-size:11px;color:#475569;\">JD keywords used: '+resp.jd_keyword_hits+'/'+resp.jd_keyword_total+' ('+(resp.jd_keywords||[]).slice(0,5).join(', ')+')</span>';}"
    "pd.innerHTML='Filled '+filled+'/'+tas.length+' textarea(s)'+(radiosFilled?' + '+radiosFilled+' radio(s)':'')+(inputsFilled?' + '+inputsFilled+' input(s)':'')+'<br><b>'+srcLabel+'</b>'+kwLine+'<br>Review + Send. <span style=\"font-size:11px;color:#475569;\">(watching for Send banner...)</span>';"
    "const SUCCESS_RE=/Congrats!?\\s*Your application has been submitted|SUCCESS!?\\s*YOUR APPLICATION HAS BEEN SENT/i;"
    "const watchStart=Date.now();"
    "const watcher=setInterval(async()=>{"
    "if(Date.now()-watchStart>5*60*1000){clearInterval(watcher);pd.innerHTML='Watcher timed out. Click I-submitted-it on dashboard if you sent.';setTimeout(()=>pd.remove(),5000);return;}"
    "if(SUCCESS_RE.test(document.body.innerText)){clearInterval(watcher);if(lid){try{await fetch('http://localhost:9876/api/confirm_submitted/'+lid,{method:'POST'});}catch(e){}}pd.style.background='#dcfce7';pd.style.borderColor='#16a34a';pd.innerHTML='Submitted detected! '+(lid?'Moved to history on dashboard.':'(no listing_id detected, mark manually).');setTimeout(()=>pd.remove(),5000);}"
    "},1500);"
    "})();"
)


# Auto-loop bookmarklet: same Fill-smart logic + queue management.
# After Send detected, pops current job from queue and navigates to next URL.
_AUTO_LOOP_BOOKMARKLET_JS = (
    "javascript:(async()=>{"
    "let queue,fresh;"
    "try{const r=await fetch('http://localhost:9876/api/auto_queue');const d=await r.json();if(d.paused){alert('Auto loop is PAUSED on dashboard. Click Resume first.');return;}fresh=d.queue||[];if(!fresh.length){alert('Queue empty.');return;}}catch(e){alert('Cannot reach dashboard at localhost:9876.');return;}"
    "const cached=JSON.parse(sessionStorage.getItem('wfLoopQueue')||'null');"
    "if(cached&&cached.length&&cached[0].listing_id===fresh[0].listing_id){queue=cached;}else{queue=fresh;sessionStorage.setItem('wfLoopQueue',JSON.stringify(queue));sessionStorage.setItem('wfLoopActive','1');sessionStorage.setItem('wfLoopStartedAt',Date.now().toString());}"
    "let currentLid=null;"
    "const m1=location.pathname.match(/jobs\\/(\\d+)-/);"
    "const m2=location.search.match(/job_listing_slug=(\\d+)-/);"
    "if(m1)currentLid=m1[1];else if(m2)currentLid=m2[1];"
    "if(!currentLid||currentLid!==queue[0].listing_id){location.href=queue[0].url;return;}"
    "let lid=currentLid;"
    "const findTextareas=()=>{let arr=[...document.querySelectorAll('.ReactModal__Content textarea,[role=\"dialog\"] textarea,textarea[name*=\"customQuestionAnswers\"],textarea[name=\"userNote\"]')];if(!arr.length){arr=[...document.querySelectorAll('textarea')].filter(t=>t.offsetParent&&t.offsetHeight>20);}return [...new Set(arr)];};"
    "const findApplyBtn=()=>[...document.querySelectorAll('button,a')].find(b=>{const t=(b.textContent||'').trim();return /^(apply|apply now)$/i.test(t)&&b.offsetParent;});"
    "let tas=findTextareas();"
    "if(!tas.length){let applyBtn=findApplyBtn();let waited=0;while(!applyBtn&&waited<8000){await new Promise(r=>setTimeout(r,500));waited+=500;applyBtn=findApplyBtn();}if(applyBtn){applyBtn.click();let twait=0;while(!tas.length&&twait<8000){await new Promise(r=>setTimeout(r,500));twait+=500;tas=findTextareas();}}}"
    "if(!tas.length){const ext=[...document.querySelectorAll('button,a')].find(el=>/apply on (website|company website)/i.test(el.textContent));if(ext){alert('External apply detected. Skipping to next in queue.');queue.shift();sessionStorage.setItem('wfLoopQueue',JSON.stringify(queue));if(queue.length){location.href=queue[0].url;}else{sessionStorage.removeItem('wfLoopQueue');sessionStorage.removeItem('wfLoopActive');alert('Auto loop complete.');}return;}}"
    "const modalRoot=([...document.querySelectorAll('.ReactModal__Content, [role=\"dialog\"]')].find(m=>m.querySelector('input,textarea'))||document);"
    "const radioGroups={};"
    "if(modalRoot){for(const radio of modalRoot.querySelectorAll('input[type=\"radio\"]')){if(!radio.name)continue;(radioGroups[radio.name]=radioGroups[radio.name]||[]).push(radio);}}"
    "let inputsFilled=0;"
    "try{const pf=await(await fetch('http://localhost:9876/api/profile_fields')).json();if(pf.ok&&modalRoot){const setInp=(el,val)=>{if(!el||!val||el.value)return;const s=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el),'value').set;s.call(el,val);el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));inputsFilled++;};const getInputLabel=(inp)=>{if(inp.labels&&inp.labels.length){const l=(inp.labels[0].textContent||'').trim();if(l)return l.toLowerCase();}const aria=inp.getAttribute('aria-label')||'';if(aria)return aria.toLowerCase();const lbId=inp.getAttribute('aria-labelledby');if(lbId){const ref=document.getElementById(lbId);if(ref){const t=(ref.textContent||'').trim();if(t)return t.toLowerCase();}}if(inp.placeholder)return inp.placeholder.toLowerCase();const nameId=((inp.name||'')+' '+(inp.id||'')).toLowerCase();let prev=inp.previousElementSibling;for(let i=0;i<3&&prev;i++){const t=(prev.textContent||'').trim();if(t&&t.length>1&&t.length<100)return (t+' '+nameId).toLowerCase();prev=prev.previousElementSibling;}let p=inp.parentElement;for(let d=0;d<2&&p;d++){const children=Array.from(p.children);const idx=children.findIndex(c=>c===inp||c.contains(inp));if(idx>0){for(let i=idx-1;i>=Math.max(0,idx-3);i--){const c=children[i];const tg=c.tagName;if(tg==='LABEL'||tg==='H3'||tg==='H4'||tg==='P'||tg==='SPAN'||tg==='STRONG'||tg==='B'){const t=(c.textContent||'').trim();if(t&&t.length>1&&t.length<100)return (t+' '+nameId).toLowerCase();}}}p=p.parentElement;}return nameId;};for(const inp of modalRoot.querySelectorAll('input[type=\"url\"],input[type=\"text\"],input[type=\"email\"],input[type=\"tel\"],input:not([type])')){if(!inp.offsetParent)continue;const l=getInputLabel(inp);if(/current\\s*(company|employer|title|role|position)|cover\\s*letter|previous|salary|expected|why|describe|introduce/.test(l))continue;if(/\\blinkedin\\b/.test(l))setInp(inp,pf.linkedin);else if(/\\bgithub\\b/.test(l))setInp(inp,pf.github);else if(/\\bemail\\b/.test(l))setInp(inp,pf.email);else if(/\\bphone\\b|\\btel\\b|\\bmobile\\b/.test(l))setInp(inp,pf.phone);else if(/\\b(website|portfolio|personal\\s*site)\\b/.test(l)||(/\\bother\\b/.test(l)&&/urls?/.test(l)))setInp(inp,pf.website);else if(/\\b(first\\s*name|given\\s*name|preferred\\s*name)\\b/.test(l))setInp(inp,pf.firstName);else if(/\\b(last\\s*name|surname|family\\s*name)\\b/.test(l))setInp(inp,pf.lastName);else if(/\\bcity\\b|current\\s*location/.test(l)&&!inp.value)setInp(inp,pf.city);}}}catch(e){console.warn('input fill err',e);}"
    "let radiosFilled=0;"
    "if(Object.keys(radioGroups).length){for(const name of Object.keys(radioGroups)){const radios=radioGroups[name];if(radios.length<2)continue;let qtext='';let p=radios[0].closest('label')?.parentElement||radios[0].parentElement;for(let d=0;d<6&&p;d++){const t=(p.textContent||'').trim();if(t&&t.length>10&&t.length<400){qtext=t.split(/\\n/)[0].trim();break;}p=p.parentElement;}const ql=qtext.toLowerCase();let pick=null;if(/visa|sponsor|h[\\s-]?1b|opt|cpt/.test(ql))pick='yes';else if(/citizen|green card|permanent resident/.test(ql))pick='no';else if(/willing to travel/.test(ql))pick='yes';else if(/relocate|live in.*new york|nyc/.test(ql))pick='yes';else if(/authorized to work|legally allowed/.test(ql))pick='yes';else if(/start.*immediate|available.*start/.test(ql))pick='yes';else if(/over 18|at least 18/.test(ql))pick='yes';else if(/felony|criminal|convicted/.test(ql))pick='no';else if(/(remote|hybrid|onsite|in[- ]?office)/.test(ql))pick='__skip__';if(pick&&pick!=='__skip__'){const target=radios.find(r=>{const lbl=r.closest('label')?.textContent||r.value||'';return new RegExp('^\\\\s*'+pick+'\\\\b','i').test(lbl);});if(target&&!target.checked){target.click();radiosFilled++;}}}}"
    "const getPrompt=(ta)=>{if(ta.labels&&ta.labels.length){const l=(ta.labels[0].textContent||'').trim();if(l&&l.length<400)return l;}const aria=ta.getAttribute('aria-label')||'';if(aria&&aria.length<400)return aria.trim();const ph=ta.placeholder||'';if(ph&&ph.length<400)return ph.trim();let prev=ta.previousElementSibling;for(let i=0;i<3&&prev;i++){const t=(prev.textContent||'').trim();if(t&&t.length>3&&t.length<400)return t.split(/\\n+/)[0].trim().slice(0,400);prev=prev.previousElementSibling;}let p=ta.parentElement;for(let d=0;d<3&&p;d++){const lbl=p.querySelector('label,h3,h4');if(lbl&&lbl.textContent){const t=lbl.textContent.trim();if(t&&t.length>3&&t.length<400)return t.split(/\\n+/)[0].trim().slice(0,400);}const ownText=Array.from(p.childNodes).filter(n=>n.nodeType===3).map(n=>(n.textContent||'').trim()).filter(Boolean).join(' ');if(ownText&&ownText.length>3&&ownText.length<400)return ownText.slice(0,400);p=p.parentElement;}return '';};"
    "const questions=tas.map((ta,i)=>({name:ta.name||'textarea_'+i,prompt:getPrompt(ta),idx:i}));"
    "const jdEl=document.querySelector('[class*=\"jdBody\"],[class*=\"description\"],main article,main section');"
    "const jd=((jdEl&&jdEl.innerText)||document.body.innerText||'').slice(0,4500);"
    "const getSkills=()=>{const heads=document.querySelectorAll('h1,h2,h3,h4,h5,h6,strong,b,dt');let chips=[];for(const h of heads){const tx=(h.textContent||'').trim();if(!/^skills?$/i.test(tx))continue;let p=h.parentElement;for(let d=0;d<4&&p;d++){const cands=Array.from(p.querySelectorAll('span,div,a,li,button')).filter(e=>{const t=(e.textContent||'').trim();if(!t||t.length<2||t.length>200)return false;if(/^skills?$/i.test(t))return false;if(e.children.length>2)return false;if(e.contains(h)||h.contains(e))return false;return true;}).map(e=>(e.textContent||'').trim());if(cands.length>=2&&cands.length<=40){chips=cands;break;}p=p.parentElement;}if(chips.length)break;}return [...new Set(chips)].slice(0,30).join(' | ');};"
    "const skillsText=getSkills();"
    "const company=document.querySelector('a[href^=\"/company/\"]')?.textContent?.trim()||document.title.split(' at ')[1]?.split(' \\u2022 ')[0]||'';"
    "const title=document.querySelector('h1,h2')?.textContent?.trim()||document.title.split(' at ')[0]||'';"
    "const pd=document.createElement('div');"
    "pd.style.cssText='position:fixed;top:20px;right:20px;background:#fef3c7;border:2px solid #f59e0b;padding:14px 18px;border-radius:8px;font-family:system-ui;font-size:14px;z-index:99999;box-shadow:0 4px 12px rgba(0,0,0,0.15);max-width:340px;';"
    "const progress=queue.length;"
    "pd.innerHTML='\\ud83d\\udd01 Auto loop \\u2014 '+progress+' job(s) left<br>Generating '+questions.length+' answer(s)...';"
    "document.body.appendChild(pd);"
    "let resp;"
    "try{resp=await(await fetch('http://localhost:9876/api/answer_questions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({listing_id:lid,company,title,jd,skills_text:skillsText,questions})})).json();}catch(e){pd.innerHTML='Backend unreachable.';setTimeout(()=>pd.remove(),5000);return;}"
    "if(!resp.ok){pd.innerHTML='Error: '+(resp.error||'failed');return;}"
    "let filled=0;"
    "for(const ans of resp.answers||[]){const ta=tas.find(t=>(t.name||'')===ans.name)||tas[(ans.idx||0)];if(!ta||!ans.answer)continue;const setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ta),'value').set;setter.call(ta,ans.answer);ta.dispatchEvent(new Event('input',{bubbles:true}));ta.dispatchEvent(new Event('change',{bubbles:true}));filled++;}"
    "let srcLabel='';if(resp.source==='bundle'){srcLabel='\\u267b\\ufe0f Reused bundle (free)';pd.style.background='#dcfce7';pd.style.borderColor='#16a34a';}else if(resp.source==='bundle+patch'){srcLabel='\\u267b\\ufe0f Bundle + JD patch (mini)';pd.style.background='#d1fae5';pd.style.borderColor='#059669';}else if(resp.source==='fitpitch'){srcLabel='\\ud83d\\udcdd Full FIT-PITCH (gpt-4o)';pd.style.background='#e0e7ff';pd.style.borderColor='#6366f1';}else if(resp.source==='fitpitch+patch'){srcLabel='\\ud83d\\udcdd FIT-PITCH + patch';pd.style.background='#ddd6fe';pd.style.borderColor='#7c3aed';}else if(resp.source==='mixed'){srcLabel='\\u267b\\ufe0f Mixed';pd.style.background='#fef3c7';}else{srcLabel='\\ud83e\\udd16 gpt-4o-mini';pd.style.background='#dbeafe';pd.style.borderColor='#2563eb';}"
    "let kwLine='';if(resp.jd_keyword_total){kwLine='<br><span style=\"font-size:11px;color:#475569;\">JD keywords: '+resp.jd_keyword_hits+'/'+resp.jd_keyword_total+' ('+(resp.jd_keywords||[]).slice(0,6).join(', ')+')</span>';}"
    "pd.innerHTML='\\ud83d\\udd01 Loop ('+progress+' left) - <b>'+(queue[0].company||'?')+'</b><br>Filled '+filled+'/'+tas.length+(radiosFilled?' + '+radiosFilled+' radio(s)':'')+(inputsFilled?' + '+inputsFilled+' input(s)':'')+' '+srcLabel+kwLine+'<br><b>Review + click Send</b>. <span style=\"font-size:11px;color:#475569;\">(then auto-advances)</span>';"
    "const SUCCESS_RE=/Congrats!?\\s*Your application has been submitted|SUCCESS!?\\s*YOUR APPLICATION HAS BEEN SENT/i;"
    "const watchStart=Date.now();"
    "const watcher=setInterval(async()=>{"
    "if(Date.now()-watchStart>10*60*1000){clearInterval(watcher);pd.innerHTML='\\u23f1\\ufe0f Watcher timed out. Loop paused.';setTimeout(()=>pd.remove(),5000);return;}"
    "let pauseStatus=false;try{pauseStatus=(await(await fetch('http://localhost:9876/api/auto_queue')).json()).paused;}catch(e){}"
    "if(pauseStatus){clearInterval(watcher);sessionStorage.removeItem('wfLoopActive');pd.style.background='#fee2e2';pd.style.borderColor='#b91c1c';pd.innerHTML='\\u23f8\\ufe0f Loop paused via dashboard.';setTimeout(()=>pd.remove(),6000);return;}"
    "if(SUCCESS_RE.test(document.body.innerText)){clearInterval(watcher);if(lid){try{await fetch('http://localhost:9876/api/confirm_submitted/'+lid,{method:'POST'});}catch(e){}}let q=JSON.parse(sessionStorage.getItem('wfLoopQueue')||'[]');q.shift();sessionStorage.setItem('wfLoopQueue',JSON.stringify(q));pd.style.background='#dcfce7';pd.style.borderColor='#16a34a';if(q.length){pd.innerHTML='\\u2705 Sent! Loading next: <b>'+q[0].company+'</b> ('+q.length+' left)...';setTimeout(()=>{location.href=q[0].url;},1500);}else{sessionStorage.removeItem('wfLoopActive');sessionStorage.removeItem('wfLoopQueue');pd.innerHTML='\\ud83c\\udf89 Loop complete!';setTimeout(()=>pd.remove(),8000);}}"
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
            ⚡ Fill smart  ← drag to bookmark bar (single job)
          </a>
        </div>

        <div style="margin:18px 0;padding:12px 14px;background:#fef9c3;border-radius:8px;border:1px dashed #ca8a04;">
          <div style="font-size:13px;color:#854d0e;font-weight:600;margin-bottom:8px;">
            🔁 Auto loop — process N pending jobs in sequence
          </div>
          <a href="{_esc(_AUTO_LOOP_BOOKMARKLET_JS)}" class="bookmarklet-drag" style="background:linear-gradient(135deg,#a855f7,#7c3aed);"
             onclick="alert('Drag to bookmark bar.\\n\\nUsage:\\n1. Click on ANY Wellfound page (or this dashboard).\\n2. Auto-opens first queued job.\\n3. Click bookmarklet again → auto-fills, watches for Send.\\n4. You review + click Send.\\n5. Auto-advances to next job in queue.\\n6. Repeats until queue empty or you click Pause on dashboard.'); return false;">
            🔁 Auto smart loop  ← drag to bookmark bar (batch)
          </a>
          <div style="margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <label style="font-size:12px;color:#475569;">Cap to:</label>
            <input id="loop-limit" type="number" min="1" max="500" value="5" style="width:60px;padding:5px 8px;border-radius:5px;border:1px solid #fde047;font-size:12px;">
            <button onclick="loopSetLimit()" style="padding:6px 12px;border-radius:5px;border:1px solid #fde047;background:white;cursor:pointer;font-size:12px;">📏 Set limit</button>
            <button onclick="loopSetLimit(null)" style="padding:6px 12px;border-radius:5px;border:1px solid #fde047;background:white;cursor:pointer;font-size:12px;">∞ No cap</button>
            <button onclick="loopPause()" style="padding:6px 12px;border-radius:5px;border:1px solid #fde047;background:white;cursor:pointer;font-size:12px;">⏸️ Pause</button>
            <button onclick="loopResume()" style="padding:6px 12px;border-radius:5px;border:1px solid #fde047;background:white;cursor:pointer;font-size:12px;">▶️ Resume</button>
            <span id="loop-status" style="font-size:11px;color:#475569;width:100%;margin-top:4px;"></span>
          </div>
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

async function loopPause() {{ await _post('/api/loop_pause'); _initLoopStatus(); }}
async function loopResume() {{ await _post('/api/loop_resume'); _initLoopStatus(); }}
async function loopSetLimit(forceVal) {{
  let limit = forceVal;
  if (limit === undefined) {{
    const el = document.getElementById('loop-limit');
    limit = parseInt(el.value, 10);
    if (!limit || limit < 1) {{ alert('Enter a number ≥ 1, or click ∞ No cap'); return; }}
  }}
  const r = await _post('/api/loop_set_limit', {{limit: limit}});
  if (r.ok) {{ _initLoopStatus(); alert((limit === null ? '∞ Cap removed' : '📏 Capped to ' + limit + ' jobs') + '.\\n\\nIf a loop is already running on a Wellfound tab, close + reopen to apply.'); }}
}}
async function _initLoopStatus() {{
  try {{
    const r = await fetch('/api/auto_queue');
    const j = await r.json();
    const el = document.getElementById('loop-status');
    if (!el) return;
    const limitStr = j.limit ? ` (capped at ${{j.limit}})` : ' (no cap)';
    el.textContent = (j.paused ? '⏸️ paused · ' : '▶️ ') + j.count + ' jobs queued' + limitStr + ' · total available: ' + j.total_available;
    const inp = document.getElementById('loop-limit');
    if (inp && j.limit) inp.value = j.limit;
  }} catch(e){{}}
}}
_initLoopStatus();
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


@app.get("/api/profile_fields")
def get_profile_fields():
    """Return the user's contact / identity fields from profile.json so the
    bookmarklet can auto-fill LinkedIn / GitHub / email / phone / first-last
    name / city / website inputs in apply modals.

    Personal data never leaves the local machine — this endpoint is bound
    to 127.0.0.1 only via the dashboard's CORS config."""
    try:
        p = load_profile()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    full_name = (p.get("name") or "").strip()
    first, last = "", ""
    if full_name:
        bits = full_name.split()
        first = bits[0]
        last = " ".join(bits[1:]) if len(bits) > 1 else ""
    return {
        "ok": True,
        "linkedin": p.get("linkedin") or "",
        "github": (
            p.get("github") if str(p.get("github") or "").startswith("http")
            else (f"https://github.com/{p.get('github')}" if p.get("github") else "")
        ),
        "email": p.get("email") or "",
        "phone": p.get("phone") or "",
        "website": p.get("portfolio") or p.get("website") or "",
        "firstName": p.get("firstName") or p.get("first_name") or first,
        "lastName": p.get("lastName") or p.get("last_name") or last,
        "city": p.get("city") or p.get("location") or "",
    }


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


# ---------- Auto-loop endpoints --------------------------------------------

_LOOP_FLAG_FILE = Path("/tmp/wf_loop_state.json")


def _read_loop_state() -> dict:
    if not _LOOP_FLAG_FILE.exists():
        return {"paused": False, "limit": None}
    try:
        d = json.loads(_LOOP_FLAG_FILE.read_text())
        return {"paused": bool(d.get("paused")), "limit": d.get("limit")}
    except Exception:
        return {"paused": False, "limit": None}


def _write_loop_state(**patch) -> dict:
    state = _read_loop_state()
    state.update(patch)
    _LOOP_FLAG_FILE.write_text(json.dumps(state))
    return state


@app.get("/api/auto_queue")
def get_auto_queue():
    """Return the (possibly limited) ordered list of pending wellfound-apply
    bundles for the auto-loop bookmarklet to process."""
    state = _read_loop_state()
    out = []
    if PENDING_ROOT.exists():
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
            if (data.get("apply_mode") or "wellfound").lower() == "external":
                continue
            if data.get("url") and data.get("listing_id"):
                out.append({
                    "listing_id": str(data["listing_id"]),
                    "url": data["url"],
                    "company": data.get("company") or "",
                    "title": data.get("title") or "",
                })
    out.sort(key=lambda b: (b.get("company") or "").lower())
    limit = state.get("limit")
    total = len(out)
    if isinstance(limit, int) and limit > 0:
        out = out[:limit]
    return {"queue": out, "count": len(out), "total_available": total,
            "paused": state.get("paused", False), "limit": limit}


@app.post("/api/loop_pause")
def loop_pause():
    _write_loop_state(paused=True)
    return {"ok": True, "paused": True}


@app.post("/api/loop_resume")
def loop_resume():
    _write_loop_state(paused=False)
    return {"ok": True, "paused": False}


@app.post("/api/loop_set_limit")
def loop_set_limit(payload: dict):
    """Body: {"limit": 5} caps at 5; {"limit": null} removes cap."""
    raw = payload.get("limit") if isinstance(payload, dict) else None
    if raw in (None, "", 0, "0"):
        limit = None
    else:
        try:
            limit = int(raw)
            if limit < 0:
                limit = None
        except (TypeError, ValueError):
            return {"ok": False, "error": f"invalid limit: {raw!r}"}
    _write_loop_state(limit=limit)
    return {"ok": True, "limit": limit}


# ---------- Smart bookmarklet endpoint: answer all questions in a modal -----

_INTEREST_RE = re.compile(
    r"(what\s+interests?\s+you|"
    r"why\s+[\w.\-]{1,30}\s*\??|"     # "Why Yuzu?" / "Why X." / short company names
    r"why\s+(this|us|do you want to work|are you (excited|interested))|"
    r"tell\s+(us|me)\s+(about\s+yourself|why)|"
    r"about\s+(this\s+(role|company|position)|the\s+role)|"
    r"motivat(ion|ed)|"
    r"cover\s*letter|"                # "Cover Letter" textarea label
    r"^\s*pitch\s*$|"                 # bare "Pitch"
    r"introduce\s+yourself|"
    r"personal\s+statement|"
    r"what.{0,20}makes\s+you|"        # "What makes you a good fit"
    r"why\s+(are\s+)?you\s+(the\s+)?(right|good|best)\s+fit)",
    re.IGNORECASE,
)


# ---------- Skills chip extraction & keyword merging -----------------------

_SKILL_NOISE = {
    "skills", "skill", "tags", "tag", "view all", "see more",
    "show more", "and", "or", "the", "a", "an", "stack",
    "experience", "knowledge", "proficiency",
}


def _split_skill_chips(skills_text: str) -> list[str]:
    """Wellfound's Skills sidebar has chips like:
        'DevOps', 'AWS/EC2/ELB/S3/DynamoDB', 'Docker',
        'MERN Stack - Javascript (ES5 & ES6), MongoDB, Express.Js, React'
    These are RECRUITER-TAGGED keywords — they MUST hit. Split compound
    chips into atomic tokens so each individual tech becomes a target."""
    if not skills_text:
        return []
    out: list[str] = []
    raw_chips = [c.strip() for c in skills_text.split("|") if c.strip()]
    for chip in raw_chips:
        # Unwrap parens first so "Javascript (ES5 & ES6)" doesn't split mid-paren.
        unwrapped = re.sub(r"\(([^)]{1,40})\)", r" \1 ", chip)
        unwrapped = re.sub(r"\s+", " ", unwrapped).strip()
        has_slash = "/" in unwrapped
        has_separator = bool(re.search(r"[,&]| - ", unwrapped))
        # Atomic chip → keep as-is.
        if not has_slash and not has_separator and 2 <= len(unwrapped) <= 60:
            if unwrapped.lower() not in (c.lower() for c in out):
                out.append(unwrapped)
            continue
        # Compound → split. Drop the verbatim compound; atoms cover it.
        parts = re.split(r"[\\/,&]| - ", unwrapped)
        for p in parts:
            p = re.sub(r"\s+", " ", p).strip(" .;:-")
            if not p or len(p) > 50 or p.lower() in _SKILL_NOISE:
                continue
            if p.lower() not in (c.lower() for c in out):
                out.append(p)
            # Multi-word part with digit-bearing token (e.g. "Javascript ES5")
            # → also atomize. Skip pure phrases ("MERN Stack", "Generative AI").
            tokens = p.split()
            if len(tokens) >= 2 and any(re.search(r"\d", t) for t in tokens):
                for tok in tokens:
                    tok = tok.strip(" .;:-")
                    if not (2 <= len(tok) <= 30):
                        continue
                    if tok.lower() in _SKILL_NOISE:
                        continue
                    if not re.match(r"^[A-Za-z][A-Za-z0-9.+#-]*$", tok):
                        continue
                    if tok.lower() not in (c.lower() for c in out):
                        out.append(tok)
    seen, deduped = set(), []
    for k in out:
        kl = k.lower()
        if kl in seen:
            continue
        seen.add(kl)
        deduped.append(k)
    return deduped[:35]


def _merge_skills_into_keywords(jd_keywords: list[str],
                                  skill_chips: list[str]) -> list[str]:
    """Skill chips go FIRST — they're recruiter-tagged."""
    if not skill_chips:
        return jd_keywords
    lower_seen = {k.lower() for k in skill_chips}
    out = list(skill_chips)
    for k in jd_keywords:
        if k.lower() not in lower_seen:
            out.append(k)
            lower_seen.add(k.lower())
    return out


# ---------- Keyword patch (short addendum, ~120 words, ~3s) ----------------

_PATCH_SYSTEM = """You write a SHORT addendum paragraph (60-100 words) to be
appended to an existing job-application blurb. The paragraph's job is to
land specific MISSED keywords from the JD/Skills tags.

Rules:
- Output 60-100 words. ONE paragraph only.
- Every keyword on the MISSED list MUST appear in your paragraph (verify
  before output).
- Frame as honest familiarity: "I have hands-on experience with X, Y, Z
  from side-projects and coursework." Or "My stack also includes X (for A),
  Y (for B), Z." Do NOT fabricate employers, dates, or measurable results.
- Sound natural and applicant-confident, not like a keyword dump.
- NEVER use: em dashes (—), leverage, robust, cutting-edge, synergy,
  harness, transformative, passionate, perfect fit, eager to contribute,
  modernization, innovative approach, honed, paramount, intricacies.

Output ONLY the paragraph. No JSON, no preamble, no explanation."""


def patch_blurb_for_keywords(blurb: str, missed_kws: list[str],
                              jd: str, company: str, title: str) -> str:
    """Generate a SHORT (~120-token) addendum covering missed keywords and
    append to existing blurb. Faster than rewriting the entire blurb. NO
    retry — falls back to a deterministic familiarity tail if the model
    still misses any keyword. ~$0.001/job."""
    if not missed_kws:
        return blurb
    try:
        from blurb import scrub_ai_tells
    except Exception:
        scrub_ai_tells = lambda s: s
    user = f"""COMPANY: {company}
ROLE: {title}

MISSED KEYWORDS (must ALL appear in your paragraph):
{', '.join(missed_kws)}

CONTEXT (the existing blurb — for tone reference only, do NOT rewrite it):
{blurb[:800]}

Write the 60-100 word addendum paragraph now. Cover every missed keyword."""
    addendum = ""
    try:
        client = openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            temperature=0.4,
            messages=[
                {"role": "system", "content": _PATCH_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        addendum = (resp.choices[0].message.content or "").strip()
        if addendum.startswith("```"):
            addendum = re.sub(r"^```[a-z]*\s*", "", addendum).rstrip("`").strip()
        addendum = scrub_ai_tells(addendum) if addendum else ""
    except Exception as e:
        print(f"[patch_blurb_for_keywords] err: {e}", file=sys.stderr)
        addendum = ""
    combined = (blurb.rstrip() + "\n\n" + addendum) if addendum else blurb
    # Safety net: deterministic tail for any still-missing keywords.
    low = combined.lower()
    still_missed = [k for k in missed_kws if k.lower() not in low]
    if still_missed:
        tail = (" Also familiar with " + ", ".join(still_missed)
                + " from coursework and personal side-projects.")
        combined = combined.rstrip() + tail
    return combined


@app.post("/api/answer_questions")
async def answer_questions(payload: dict):
    """Generate answers for every textarea in a Wellfound apply modal in one
    LLM call. Reuses existing bundle blurb for 'interest in company' questions
    when the bundle is found, saving an LLM call."""
    listing_id = payload.get("listing_id")
    company = (payload.get("company") or "").strip()
    title = (payload.get("title") or "").strip()
    jd = (payload.get("jd") or "").strip()
    skills_text = (payload.get("skills_text") or "").strip()
    questions = payload.get("questions") or []
    if not isinstance(questions, list) or not questions:
        return {"ok": False, "error": "no questions in payload"}

    skill_chips = _split_skill_chips(skills_text)

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

    # Three buckets:
    #   1. preanswered = bundle blurb covers it (interest q + bundle exists)
    #   2. fitpitch_qs = interest q but NO bundle → run full FIT-PITCH pipeline
    #      (multi-project, 4 paragraphs, 280-450 words, same quality as dashboard)
    #   3. remaining_qs = everything else → short gpt-4o-mini answer
    preanswered: dict[str, str] = {}
    fitpitch_qs: list[dict] = []
    remaining_qs: list[dict] = []
    for q in questions:
        prompt = (q.get("prompt") or "").lower()
        is_interest = bool(_INTEREST_RE.search(prompt))
        if is_interest and bundle_blurb:
            preanswered[q.get("name")] = bundle_blurb
        elif is_interest:
            fitpitch_qs.append(q)
        else:
            remaining_qs.append(q)

    # Run full FIT-PITCH pipeline for interest questions that lack a bundle.
    fitpitch_answers: dict[str, str] = {}
    if fitpitch_qs:
        try:
            from blurb import compose_blurb
            blurb_text = compose_blurb(jd=jd, title=title, company=company,
                                       min_projects=3, max_projects=4)
            for q in fitpitch_qs:
                fitpitch_answers[q.get("name")] = blurb_text
        except Exception as e:
            print(f"[answer_questions] compose_blurb failed: {e}", file=sys.stderr)
            remaining_qs.extend(fitpitch_qs)
            fitpitch_qs = []

    profile = load_profile()
    pool = format_pool_for_prompt()

    # JD keyword extraction. Skill chips ≥ 6 → skip the LLM call (saves ~2s).
    if skill_chips and len(skill_chips) >= 6:
        jd_keywords = list(skill_chips)
    else:
        from blurb import extract_jd_keywords
        llm_kws = extract_jd_keywords(jd, title, company) if jd else []
        jd_keywords = _merge_skills_into_keywords(llm_kws, skill_chips)

    kw_block = ""
    if jd_keywords:
        kw_block = (
            f"\nJD KEYWORDS (mirror these in answers where they naturally fit):\n"
            f"  {', '.join(jd_keywords)}\n"
        )

    # If everything pre-answered, skip gpt-4o-mini batch. Run the keyword
    # patch on bundle + fitpitch outputs so they reflect THIS JD's vocabulary.
    if not remaining_qs:
        patched_bundle = False
        if preanswered and jd_keywords:
            for n, b in list(preanswered.items()):
                missed = [k for k in jd_keywords if k.lower() not in b.lower()]
                if len(missed) >= 2:
                    p = patch_blurb_for_keywords(b, missed, jd, company, title)
                    if p and p != b:
                        preanswered[n] = p
                        patched_bundle = True
        patched_fitpitch = False
        if fitpitch_answers and jd_keywords:
            for n, b in list(fitpitch_answers.items()):
                missed = [k for k in jd_keywords if k.lower() not in b.lower()]
                if len(missed) >= 2:
                    p = patch_blurb_for_keywords(b, missed, jd, company, title)
                    if p and p != b:
                        fitpitch_answers[n] = p
                        patched_fitpitch = True
        bundle_src = "bundle+patch" if patched_bundle else "bundle"
        fitpitch_src = "fitpitch+patch" if patched_fitpitch else "fitpitch"
        out = []
        out.extend({"name": n, "answer": scrub_ai_tells(a), "source": bundle_src} for n, a in preanswered.items())
        out.extend({"name": n, "answer": scrub_ai_tells(a), "source": fitpitch_src} for n, a in fitpitch_answers.items())
        kw_hits = 0
        if jd_keywords:
            text = " ".join(a["answer"].lower() for a in out)
            kw_hits = sum(1 for k in jd_keywords if k.lower() in text)
        if preanswered and not fitpitch_answers:
            top_src = bundle_src
        elif fitpitch_answers and not preanswered:
            top_src = fitpitch_src
        else:
            top_src = "mixed"
        return {
            "ok": True,
            "answers": out,
            "count": len(out),
            "source": top_src,
            "bundle_used": bool(preanswered),
            "bundle_patched": patched_bundle,
            "fitpitch_used": bool(fitpitch_answers),
            "fitpitch_patched": patched_fitpitch,
            "jd_keywords": jd_keywords,
            "jd_keyword_hits": kw_hits,
            "jd_keyword_total": len(jd_keywords),
        }

    # Mixed case: also patch bundle / fitpitch in-place before the LLM batch.
    bundle_patched_names: set = set()
    if preanswered and jd_keywords:
        for n, b in list(preanswered.items()):
            missed = [k for k in jd_keywords if k.lower() not in b.lower()]
            if len(missed) >= 2:
                p = patch_blurb_for_keywords(b, missed, jd, company, title)
                if p and p != b:
                    preanswered[n] = p
                    bundle_patched_names.add(n)
    fitpitch_patched_names: set = set()
    if fitpitch_answers and jd_keywords:
        for n, b in list(fitpitch_answers.items()):
            missed = [k for k in jd_keywords if k.lower() not in b.lower()]
            if len(missed) >= 2:
                p = patch_blurb_for_keywords(b, missed, jd, company, title)
                if p and p != b:
                    fitpitch_answers[n] = p
                    fitpitch_patched_names.add(n)

    questions_dump = "\n".join(
        f"Q{i+1} (name={(q.get('name') or 'q'+str(i))!r}): {q.get('prompt') or '(no prompt)'}"
        for i, q in enumerate(remaining_qs)
    )
    user = f"""COMPANY: {company or '(unknown)'}
ROLE: {title or '(unknown)'}

JD (first 3500 chars):
{jd[:3500]}
{kw_block}
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

    # Retry batch up to 2x if JD keyword coverage below target
    for _ in range(2):
        if not jd_keywords:
            break
        llm_text = " ".join((a.get("answer") or "").lower() for a in (parsed.get("answers") or []))
        all_text = llm_text + " " + " ".join(a.lower() for a in fitpitch_answers.values()) + " " + " ".join(a.lower() for a in preanswered.values())
        missed = [k for k in jd_keywords if k.lower() not in all_text]
        if len(missed) <= 1:
            break
        kw_feedback = (
            f"\n\nYou missed JD keywords: {', '.join(missed)}. Re-output ALL answers, "
            f"naturally incorporating these by paraphrasing existing project facts. "
            f"Output JSON only."
        )
        try:
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=2500, temperature=0.55,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": answer_question_prompt(profile)},
                    {"role": "user", "content": user + kw_feedback},
                ],
            )
            parsed = json.loads(resp2.choices[0].message.content or "{}")
        except Exception:
            break

    out = []
    out.extend({
        "name": n,
        "answer": scrub_ai_tells(a),
        "source": "bundle+patch" if n in bundle_patched_names else "bundle",
    } for n, a in preanswered.items())
    out.extend({
        "name": n,
        "answer": scrub_ai_tells(a),
        "source": "fitpitch+patch" if n in fitpitch_patched_names else "fitpitch",
    } for n, a in fitpitch_answers.items())
    for a in parsed.get("answers", []):
        ans = scrub_ai_tells(a.get("answer", "") or "")
        out.append({"name": a.get("name"), "answer": ans, "source": "llm"})
    kw_hits = 0
    if jd_keywords:
        text = " ".join(a["answer"].lower() for a in out)
        kw_hits = sum(1 for k in jd_keywords if k.lower() in text)
    sources = {a["source"] for a in out}
    if sources == {"bundle"}: source_label = "bundle"
    elif sources == {"bundle+patch"}: source_label = "bundle+patch"
    elif sources == {"fitpitch"}: source_label = "fitpitch"
    elif sources == {"fitpitch+patch"}: source_label = "fitpitch+patch"
    elif sources == {"llm"}: source_label = "llm"
    else: source_label = "mixed"
    return {
        "ok": True,
        "answers": out,
        "count": len(out),
        "source": source_label,
        "bundle_used": bool(preanswered),
        "bundle_patched": bool(bundle_patched_names),
        "fitpitch_used": bool(fitpitch_answers),
        "fitpitch_patched": bool(fitpitch_patched_names),
        "jd_keywords": jd_keywords,
        "jd_keyword_hits": kw_hits,
        "jd_keyword_total": len(jd_keywords),
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
