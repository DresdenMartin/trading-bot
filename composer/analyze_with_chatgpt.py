import os
import json
import re
from typing import Dict
from datetime import datetime, timezone
from openai import OpenAI
import openai_helpers as _openai_helpers

from openai_helpers import call_chat_completion, call_web_search
from project_config import WEB_RESEARCH_CACHE, WEB_RESEARCH_AUDIT

_WEB_RESEARCH_CACHE_PATH = WEB_RESEARCH_CACHE
try:
    with open(_WEB_RESEARCH_CACHE_PATH, 'r') as _f:
        _WEB_RESEARCH_CACHE = json.load(_f)
except Exception:
    _WEB_RESEARCH_CACHE = {}

_WEB_RESEARCH_CACHE_TTL = int(os.getenv('WEB_RESEARCH_CACHE_TTL', 60 * 60 * 6))
_WEB_RESEARCH_MIN_SOURCES = int(os.getenv('WEB_RESEARCH_MIN_SOURCES', 2))
_WEB_RESEARCH_MIN_DIVERSITY = int(os.getenv('WEB_RESEARCH_MIN_DIVERSITY', 2))
_WEB_RESEARCH_TARGET_SOURCES = int(os.getenv('WEB_RESEARCH_TARGET_SOURCES', 4))
_WEB_RESEARCH_RECENCY_DAYS = int(os.getenv('WEB_RESEARCH_RECENCY_DAYS', 3))
_WEB_RESEARCH_RECENCY_TARGET_RATIO = float(os.getenv('WEB_RESEARCH_RECENCY_TARGET_RATIO', 0.5))
_WEB_RESEARCH_MIN_CONF = float(os.getenv('WEB_RESEARCH_MIN_CONFIDENCE', 0.35))
_WEB_RESEARCH_TECH_WEIGHT = float(os.getenv('WEB_RESEARCH_TECH_WEIGHT', 5.0))


def _save_web_research_cache():
    try:
        with open(_WEB_RESEARCH_CACHE_PATH, 'w') as _f:
            json.dump(_WEB_RESEARCH_CACHE, _f)
    except Exception:
        pass


def _append_web_research_audit(entry: dict):
    try:
        with open(WEB_RESEARCH_AUDIT, 'a') as fh:
            fh.write(json.dumps(entry) + '\n')
    except Exception:
        pass


def _component_weights() -> dict:
    raw = {
        'news': os.getenv('WEB_RESEARCH_WEIGHT_NEWS', '0.35'),
        'analyst': os.getenv('WEB_RESEARCH_WEIGHT_ANALYST', '0.20'),
        'filings': os.getenv('WEB_RESEARCH_WEIGHT_FILINGS', '0.15'),
        'macro': os.getenv('WEB_RESEARCH_WEIGHT_MACRO', '0.15'),
        'product': os.getenv('WEB_RESEARCH_WEIGHT_PRODUCT', '0.10'),
        'legal': os.getenv('WEB_RESEARCH_WEIGHT_LEGAL', '0.05'),
    }
    weights = {}
    total = 0.0
    for key, val in raw.items():
        try:
            weights[key] = float(val)
        except Exception:
            weights[key] = 0.0
        total += weights[key]
    if total <= 0:
        return {k: 1.0 / len(raw) for k in raw}
    return {k: v / total for k, v in weights.items()}


def _parse_date(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    txt = str(value).strip()
    if not txt:
        return None
    txt = txt.replace('Z', '+00:00') if txt.endswith('Z') else txt
    try:
        return datetime.fromisoformat(txt)
    except Exception:
        pass
    try:
        return datetime.strptime(txt, '%Y-%m-%d')
    except Exception:
        return None


def _classify_source(name: str | None, url: str | None, declared_type: str | None) -> str:
    if declared_type:
        t = declared_type.strip().lower()
        if t in ('news', 'filings', 'analyst', 'macro', 'product', 'legal'):
            return t
    blob = ' '.join([name or '', url or '']).lower()
    if any(x in blob for x in ('sec.gov', '10-k', '10-q', '8-k', 'form 4', 'filing')):
        return 'filings'
    if any(x in blob for x in ('upgrade', 'downgrade', 'analyst', 'rating', 'price target', 'bank', 'broker')):
        return 'analyst'
    if any(x in blob for x in ('inflation', 'cpi', 'jobs report', 'fed', 'rates', 'macro')):
        return 'macro'
    if any(x in blob for x in ('lawsuit', 'doj', 'sec probe', 'regulatory', 'ftc', 'legal')):
        return 'legal'
    if any(x in blob for x in ('launch', 'product', 'release', 'model', 'chip', 'update')):
        return 'product'
    return 'news'


def _normalize_sources(raw_sources: list | None) -> list:
    sources = []
    if not raw_sources:
        return sources
    for src in raw_sources:
        if isinstance(src, str):
            name = src.strip()
            if not name:
                continue
            sources.append({
                'name': name[:160],
                'url': None,
                'date': None,
                'type': _classify_source(name, None, None),
            })
            continue
        if not isinstance(src, dict):
            continue
        name = (src.get('name') or src.get('title') or src.get('source') or src.get('url') or '').strip()
        url = (src.get('url') or src.get('link') or '').strip() or None
        date_raw = src.get('date') or src.get('published_at') or src.get('published') or src.get('timestamp')
        dt = _parse_date(date_raw)
        declared_type = src.get('type') or src.get('category')
        sources.append({
            'name': name[:160] if name else (url[:160] if url else None),
            'url': url,
            'date': dt.date().isoformat() if dt else None,
            'type': _classify_source(name, url, declared_type),
        })
    return sources


def _summarize_source_quality(sources: list) -> dict:
    source_count = len(sources)
    types = [s.get('type') for s in sources if s.get('type')]
    diversity_count = len(set(types)) if types else 0
    now = datetime.now(timezone.utc)
    fresh_count = 0
    dated_count = 0
    for src in sources:
        dt = _parse_date(src.get('date'))
        if dt:
            dated_count += 1
            delta_days = abs((now - dt.replace(tzinfo=timezone.utc)).total_seconds()) / 86400.0
            if delta_days <= _WEB_RESEARCH_RECENCY_DAYS:
                fresh_count += 1
    fresh_ratio = fresh_count / max(1, source_count)
    count_score = min(1.0, source_count / max(1, _WEB_RESEARCH_TARGET_SOURCES))
    if _WEB_RESEARCH_MIN_DIVERSITY > 0:
        diversity_score = min(1.0, diversity_count / _WEB_RESEARCH_MIN_DIVERSITY)
    else:
        diversity_score = 1.0
    if dated_count == 0:
        recency_score = 0.35
    else:
        target_ratio = max(0.1, _WEB_RESEARCH_RECENCY_TARGET_RATIO)
        recency_score = min(1.0, fresh_ratio / target_ratio)
    quality_score = (count_score + diversity_score + recency_score) / 3.0
    return {
        'source_count': source_count,
        'diversity_count': diversity_count,
        'fresh_count': fresh_count,
        'fresh_ratio': round(fresh_ratio, 3),
        'count_score': round(count_score, 3),
        'diversity_score': round(diversity_score, 3),
        'recency_score': round(recency_score, 3),
        'quality_score': round(quality_score, 3),
        'dated_sources': dated_count,
    }


def _normalize_components(raw_components: dict | None) -> dict:
    buckets = ['news', 'analyst', 'filings', 'macro', 'product', 'legal']
    out = {k: 0.0 for k in buckets}
    if not raw_components or not isinstance(raw_components, dict):
        return out
    aliases = {
        'analysts': 'analyst',
        'filing': 'filings',
        'regulatory': 'legal',
    }
    for key, val in raw_components.items():
        base = aliases.get(str(key).lower(), str(key).lower())
        if base not in out:
            continue
        try:
            out[base] = _clamp(float(val), -1.0, 1.0)
        except Exception:
            continue
    return out


def _score_from_components(components: dict) -> float:
    weights = _component_weights()
    total = 0.0
    for key, weight in weights.items():
        total += weight * float(components.get(key, 0.0))
    return 50.0 + (total * 50.0)

def _heuristic_scores(items: list) -> list:
    """Deterministic fallback scoring when the model is unavailable or returns empty results."""
    out = []
    for it in items:
        sym = it.get('symbol')
        if not sym:
            continue

        mets = it.get('metrics', {}) or {}
        news = it.get('news') or []
        analysts = it.get('analysts') or {}

        news_sent = float(mets.get('news_aggregate_sentiment') or 0.0)
        news_count = len(news)
        news_strength = min(1.0, news_count / 5.0)
        fresh_articles = 0
        now_utc = datetime.now(timezone.utc)
        for art in news:
            if not isinstance(art, dict):
                continue
            ts = art.get('published_at') or art.get('published_utc')
            dt = None
            if isinstance(ts, datetime):
                dt = ts
            elif isinstance(ts, str):
                try:
                    val = ts.replace('Z', '+00:00') if ts.endswith('Z') else ts
                    dt = datetime.fromisoformat(val)
                except Exception:
                    dt = None
            if dt is None:
                continue
            delta = abs((now_utc - dt).total_seconds())
            if delta <= 6 * 3600:
                fresh_articles += 1
        recency = min(1.0, fresh_articles / 3.0)

        conf = mets.get('confidence')
        try:
            conf = float(conf) if conf is not None else 0.5
        except Exception:
            conf = 0.5

        rsi = None
        try:
            rsi = float(mets.get('rsi_14')) if mets.get('rsi_14') is not None else None
        except Exception:
            rsi = None

        tech_score = 0.0
        if rsi is not None:
            if rsi < 30:
                tech_score = 0.6
            elif rsi > 70:
                tech_score = -0.6
        else:
            tech_score = (conf - 0.5)

        close_price = mets.get('close') or mets.get('latest_price')
        sma = mets.get('sma_14') or mets.get('sma')
        ema = mets.get('ema_14') or mets.get('ema')
        momentum = 0.0
        try:
            if close_price and sma:
                momentum += (float(close_price) - float(sma)) / max(1e-6, float(sma))
            if close_price and ema:
                momentum += (float(close_price) - float(ema)) / max(1e-6, float(ema))
        except Exception:
            momentum = 0.0
        momentum = max(-1.0, min(1.0, momentum))

        buy_ratio = 0.5
        if isinstance(analysts, dict):
            counts = analysts.get('counts') or {}
            total = sum(counts.values())
            if total:
                buy_ratio = counts.get('buy', 0) / float(total)

        w_news_sent = 0.30
        w_news_strength = 0.15
        w_recency = 0.10
        w_momentum = 0.25
        w_tech = 0.10
        w_analyst = 0.10


        comp_news_sent = max(-1.0, min(1.0, news_sent))
        comp_news_strength = (news_strength - 0.5)
        comp_recency = (recency - 0.5) * 2.0
        comp_tech = max(-1.0, min(1.0, tech_score))
        comp_momentum = momentum
        comp_analyst = max(-1.0, min(1.0, (buy_ratio - 0.5) * 2.0))

        combined = (
            w_news_sent * comp_news_sent
            + w_news_strength * comp_news_strength
            + w_recency * comp_recency
            + w_momentum * comp_momentum
            + w_tech * comp_tech
            + w_analyst * comp_analyst
        )
        raw_score = 50 + (combined * 50)
        raw_score += min(5, news_count * 0.5)
        score = int(max(1, min(100, round(raw_score))))

        rationale = (
            f"news_sent={news_sent:.2f}, news_count={news_count}, recency={recency:.2f},"
            f" momentum={comp_momentum:.2f}, analyst={buy_ratio:.2f}"
        )
        components = {
            'news': comp_news_sent,
            'analyst': comp_analyst,
            'filings': 0.0,
            'macro': 0.0,
            'product': 0.0,
            'legal': 0.0,
        }
        out.append(
            {
                'symbol': sym,
                'score': score,
                'score_raw': raw_score,
                'rationale': rationale,
                'components': components,
                'weights': _component_weights(),
                'details': {
                    'news_sent': news_sent,
                    'news_count': news_count,
                    'news_recency': recency,
                    'tech': comp_tech,
                    'momentum': comp_momentum,
                    'analyst_buy_ratio': buy_ratio,
                },
            }
        )

    out = sorted(out, key=lambda x: x['score'], reverse=True)
    for i, obj in enumerate(out, 1):
        obj['rank'] = i
    return out


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def _parse_json_block(txt: str) -> dict | None:
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _technical_bias(metrics: dict) -> float:
    indicators = metrics.get('indicators') if isinstance(metrics, dict) else None
    if not isinstance(indicators, dict):
        return 0.0
    score = 0.0
    count = 0
    close = indicators.get('close')
    ema21 = indicators.get('ema21')
    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')
    rsi = indicators.get('rsi14')
    if close is not None and ema21 is not None:
        count += 1
        score += 1.0 if close > ema21 else -1.0
    if macd is not None and macd_signal is not None:
        count += 1
        score += 1.0 if macd > macd_signal else -1.0
    if rsi is not None:
        count += 1
        if rsi >= 60:
            score += 1.0
        elif rsi <= 40:
            score -= 1.0
    if count == 0:
        return 0.0
    return _clamp(score / count, -1.0, 1.0)


def _apply_confidence_adjustments(score: float, confidence: float, sources_detail: list, quality: dict) -> tuple[int, float, str]:
    note_parts = []
    source_count = len(sources_detail)
    diversity_count = quality.get('diversity_count', 0) if isinstance(quality, dict) else 0
    fresh_ratio = quality.get('fresh_ratio', 0.0) if isinstance(quality, dict) else 0.0
    quality_score = quality.get('quality_score', 0.0) if isinstance(quality, dict) else 0.0

    if source_count < _WEB_RESEARCH_MIN_SOURCES:
        note_parts.append('low source count')
    if diversity_count < _WEB_RESEARCH_MIN_DIVERSITY:
        note_parts.append('low source diversity')
    if fresh_ratio < (_WEB_RESEARCH_RECENCY_TARGET_RATIO * 0.6):
        note_parts.append('stale sources')

    score = 50 + (score - 50) * (0.6 + 0.4 * _clamp(quality_score, 0.0, 1.0))
    confidence = min(confidence, 0.3 + 0.7 * _clamp(quality_score, 0.0, 1.0))

    if confidence < _WEB_RESEARCH_MIN_CONF:
        score = 50 + (score - 50) * _clamp(confidence, 0.0, 1.0)
        note_parts.append('low confidence')

    score = int(round(_clamp(score, 1, 100)))
    confidence = _clamp(confidence, 0.0, 1.0)
    return score, confidence, ', '.join(dict.fromkeys(note_parts))


def _web_research_collect(sym: str, metrics: dict) -> dict:
    system_prompt = 'You are a trading analyst who uses web search to research symbols.'
    prompt = (
        f"Research {sym} using web search. Focus on the last 7 days of news, filings, upgrades/downgrades, product launches, "
        "legal/regulatory actions, and notable macro impacts. Return ONLY a JSON object with keys: "
        "symbol (string), summary (string <= 450 chars), "
        "sources (array of 3-8 objects with keys: name, url, date (YYYY-MM-DD), type (news|filings|analyst|macro|product|legal)), "
        "catalysts (array of up to 4 short strings), risks (array of up to 4 short strings), "
        "snippets (array of up to 6 short excerpts/paraphrases <= 120 chars).\n\n"
        f"Known metrics (optional context): {json.dumps(metrics)}"
    )
    raw = call_web_search(system_prompt, prompt, max_output_tokens=1000)
    data = _parse_json_block(raw) or {}
    sources_raw = data.get('sources') if isinstance(data.get('sources'), list) else []
    return {
        'symbol': data.get('symbol') or sym,
        'summary': str(data.get('summary') or '').strip()[:450],
        'sources_raw': sources_raw,
        'sources_detail': _normalize_sources(sources_raw),
        'catalysts': [str(c)[:120] for c in (data.get('catalysts') or [])][:4],
        'risks': [str(r)[:120] for r in (data.get('risks') or [])][:4],
        'snippets': [str(s)[:140] for s in (data.get('snippets') or [])][:6],
        'raw_text': raw[:2000],
    }


def _web_research_score_from_facts(sym: str, metrics: dict, research: dict) -> dict:
    system_prompt = 'You are a disciplined short-term trading analyst.'
    prompt = (
        "Using ONLY the provided research summary and sources, score the symbol for the next 1-5 days. "
        "Return ONLY JSON with keys: symbol (string), confidence (float 0-1), rationale (string <= 200 chars), "
        "component_scores (object with keys news, analyst, filings, macro, product, legal; values -1..1). "
        "Do not invent facts or browse again.\n\n"
        f"Symbol: {sym}\n"
        f"Research summary: {research.get('summary')}\n"
        f"Sources: {json.dumps(research.get('sources_detail') or [])}\n"
        f"Catalysts: {json.dumps(research.get('catalysts') or [])}\n"
        f"Risks: {json.dumps(research.get('risks') or [])}\n"
        f"Known metrics (for context only): {json.dumps(metrics)}"
    )
    raw = call_chat_completion(system_prompt, prompt, max_output_tokens=700)
    data = _parse_json_block(raw) or {}
    return {
        'symbol': data.get('symbol') or sym,
        'confidence': data.get('confidence'),
        'rationale': data.get('rationale') or '',
        'component_scores': data.get('component_scores') or data.get('components') or {},
        'raw_text': raw[:2000],
    }


def _web_research_score(item: dict) -> dict | None:
    sym = item.get('symbol')
    if not sym:
        return None
    metrics = item.get('metrics', {}) or {}
    cache_key = sym.upper()
    now_ts = int(datetime.now(timezone.utc).timestamp())
    cached = _WEB_RESEARCH_CACHE.get(cache_key)
    if cached and isinstance(cached, dict):
        try:
            if now_ts - int(cached.get('ts', 0)) < _WEB_RESEARCH_CACHE_TTL:
                return cached.get('value')
        except Exception:
            pass

    research = _web_research_collect(sym, metrics)
    sources_detail = research.get('sources_detail') or []
    source_quality = _summarize_source_quality(sources_detail)

    scoring = _web_research_score_from_facts(sym, metrics, research)
    components = _normalize_components(scoring.get('component_scores') if isinstance(scoring, dict) else {})
    score_raw = _score_from_components(components)
    try:
        confidence = float(scoring.get('confidence'))
    except Exception:
        confidence = 0.5
    rationale = (scoring.get('rationale') or '').strip()

    score = float(_clamp(score_raw, 1, 100))
    confidence = _clamp(confidence, 0.0, 1.0)
    score, confidence, note = _apply_confidence_adjustments(score, confidence, sources_detail, source_quality)

    tech_bias = _technical_bias(metrics)
    score_before_tech = score
    if tech_bias:
        score = int(round(_clamp(score + (tech_bias * _WEB_RESEARCH_TECH_WEIGHT), 1, 100)))

    sources_label = []
    for src in sources_detail:
        label = src.get('url') or src.get('name')
        if label:
            sources_label.append(label)

    if note:
        rationale = (rationale + f" | {note}").strip()

    result = {
        'symbol': sym,
        'score': score,
        'score_raw': score_raw,
        'score_before_tech': score_before_tech,
        'confidence': confidence,
        'rationale': rationale,
        'sources': sources_label[:8],
        'sources_detail': sources_detail,
        'source_quality': source_quality,
        'components': components,
        'weights': _component_weights(),
        'catalysts': research.get('catalysts') or [],
        'risks': research.get('risks') or [],
        'research_summary': research.get('summary') or '',
        'tech_bias': tech_bias,
    }
    try:
        _WEB_RESEARCH_CACHE[cache_key] = {'ts': now_ts, 'value': result}
        _save_web_research_cache()
    except Exception:
        pass
    try:
        _append_web_research_audit({
            'ts': datetime.now(timezone.utc).isoformat(),
            'symbol': sym,
            'research': {
                'summary': research.get('summary'),
                'sources': sources_detail,
                'catalysts': research.get('catalysts'),
                'risks': research.get('risks'),
                'snippets': research.get('snippets'),
                'raw_text': research.get('raw_text'),
            },
            'scoring': {
                'components': components,
                'weights': _component_weights(),
                'confidence': confidence,
                'rationale': rationale,
                'raw_text': scoring.get('raw_text'),
                'score_raw': score_raw,
            },
            'quality': source_quality,
            'final': {
                'score': score,
                'score_before_tech': score_before_tech,
                'tech_bias': tech_bias,
            },
        })
    except Exception:
        pass
    return result


def _web_research_scores(items: list) -> list:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    workers = int(os.getenv('WEB_RESEARCH_WORKERS', '3'))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_web_research_score, it): it for it in items}
        for fut in as_completed(futs):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception:
                pass
    results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    for i, obj in enumerate(results, 1):
        obj['rank'] = i
    return results


def analyze_summary(symbol: str, metrics: Dict) -> str:
    use_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))
    api_key = os.getenv('OPENAI_API_KEY')
    if not use_anthropic and not api_key:
        raise RuntimeError('OPENAI_API_KEY not set')
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    system_prompt = 'You are a concise financial assistant.'
    # stronger, explicit prompt. require non-empty 'summary' or 'suggested_action'
    prompt = (
        f"You are a concise financial assistant. Given the symbol {symbol} and the metrics below, RETURN ONLY A SINGLE JSON OBJECT (no surrounding text) with EXACTLY the following keys: summary (string, <=150 words), signals (object), suggested_action (string: one of 'buy','sell','hold' or null), confidence (number 0.0-1.0 or null), recommended_size (string or null), rationale (string or null).\n"
        f"Fill every key; if you don't have information put null, but always include a non-empty 'summary' when possible.\n"
        f"Input metrics: {json.dumps(metrics)}\n"
        "Return only pure JSON."
    )
    use_helper = use_anthropic or (call_chat_completion is not _openai_helpers.call_chat_completion)
    if model.strip().lower().startswith('gpt-5') or use_helper:
        text = call_chat_completion(system_prompt, prompt, max_output_tokens=600)
    else:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ],
            max_completion_tokens=600,
            temperature=0.0,
        )
        try:
            text = resp.choices[0].message.content
        except Exception:
            text = ''

    # Try direct JSON parse
    def _parse_and_validate(txt):
        try:
            j = json.loads(txt)
        except Exception:
            import re

            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                try:
                    j = json.loads(m.group(0))
                except Exception:
                    return None
            else:
                return None

        # Validate minimal content: prefer non-empty summary or suggested_action
        summary = (j.get('summary') or '') if isinstance(j, dict) else ''
        suggested = j.get('suggested_action') if isinstance(j, dict) else None
        if (not summary or str(summary).strip() == '') and not suggested:
            return None
        # ensure keys exist
        for k in ('summary', 'signals', 'suggested_action', 'confidence', 'recommended_size', 'rationale'):
            if k not in j:
                j[k] = None if k != 'signals' else {}
        return j

    parsed = _parse_and_validate(text)
    # If parsing failed or model returned empty fields, retry once with a short re-prompt
    if parsed is None:
        if os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes'):
            print('[DEBUG] initial model response invalid or empty; retrying with clarifying prompt')
        retry_prompt = (
            "You returned empty or incomplete fields. Please produce a valid JSON object now with the same schema. If minimal information is available, write a short conservative summary sentence and set suggested_action to null. Return only JSON."
        )
        text2 = call_chat_completion(system_prompt, retry_prompt + '\n\nPrevious attempt:\n' + text, max_output_tokens=600)
        parsed = _parse_and_validate(text2)

    if parsed is None:
        # last-resort heuristic fallback using metrics
        summary = f"No model summary available. Metrics provided: {json.dumps(metrics)}"
        return {
            'summary': summary,
            'signals': {},
            'suggested_action': None,
            'confidence': None,
            'recommended_size': None,
            'rationale': None,
        }

    return parsed


def analyze_portfolio_short_term(items: list) -> list:
    """Given a list of items describing symbols and recent bulletins, ask the model to score each symbol 1-100

    items: list of dicts, each must include at least: {'symbol': str, 'metrics': {...}, 'news_articles': [...], 'analyst_entries': [...]}

    Returns list of dicts: [{'symbol': str, 'score': int, 'rationale': str}, ...] sorted by score desc.
    If OPENAI_API_KEY is not set, returns a heuristic-based score.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    # prepare a compact payload for the model and heuristic fallback
    payload = []
    for it in items:
        payload.append(
            {
                'symbol': it.get('symbol'),
                'metrics': it.get('metrics', {}),
                'news': it.get('news_articles', []),
                'analysts': it.get('analyst_entries') or it.get('analyst_sentiment') or [],
            }
        )

    prompt = (
        "You are an expert short-term trading analyst. For each symbol and its recent bulletins (news headlines, timestamps, and analyst notes),"
        " evaluate the expected price/momentum direction over the NEXT 24 HOURS and assign a numeric score from 1 to 100 (100 very favorable, 1 very unfavorable)."
        " Focus on freshness (articles in last 4 hours), analyst consensus (buy/hold/sell counts), and confirmed market-moving language (upgrade, downgrade, guidance cut, offering)."
        " When in doubt prefer conservative scores; prefer diversity in the returned scores so small signals change rankings."
        " Return ONLY valid JSON: an array of objects with keys: symbol (string), score (integer 1-100), rationale (short string <= 120 chars)."
        f"\n\nInput: {json.dumps(payload)}\n"
    )

    if _env_bool('USE_WEB_RESEARCH', False):
        if not api_key:
            return _heuristic_scores(payload)
        try:
            out = _web_research_scores(payload)
            if out:
                return out
        except Exception:
            return _heuristic_scores(payload)

    if not api_key:
        return _heuristic_scores(payload)

    system_prompt = 'You are an expert short-term trading analyst.'
    enhanced_prompt = (
        "For each symbol return JSON objects with: symbol, score (1-100), rationale (<=120 chars), and a weighted_breakdown dict with components: news_sentiment, news_strength, technical, analyst. "
        "Provide numeric values for breakdown components in range -1..1 and a short textual rationale. Return only a JSON array.\n\n" + prompt
    )
    try:
        txt = call_chat_completion(system_prompt, enhanced_prompt, max_output_tokens=1200)
        try:
            arr = json.loads(txt)
        except Exception:
            import re

            m = re.search(r"\[\s*\{[\s\S]*\}\s*\]", txt)
            if m:
                arr = json.loads(m.group(0))
            else:
                arr = []
        # sanitize and sort
        out = []
        for a in arr:
            try:
                s = int(a.get('score') or a.get('score', 0))
            except Exception:
                try:
                    s = int(float(a.get('score') or 0))
                except Exception:
                    s = 0
            s = max(1, min(100, s))
            sym = a.get('symbol')
            if not sym:
                continue
            out.append({'symbol': sym, 'score': s, 'rationale': a.get('rationale') or a.get('reason') or ''})
        out = sorted(out, key=lambda x: x['score'], reverse=True)
        for i, o in enumerate(out, 1):
            o['rank'] = i
        if out:
            return out
        raise ValueError('model returned no scores')
    except Exception:
        if os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes'):
            print('[DEBUG] analyze_portfolio_short_term falling back to heuristic scoring')
        return _heuristic_scores(payload)
