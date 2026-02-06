"""Trade alerts via email."""
import os
import logging

logger = logging.getLogger(__name__)


def _email_send(subject: str, body: str) -> bool:
    """Send message via SMTP. Returns True on success."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
    except ImportError:
        return False
    host = os.getenv("ALERT_SMTP_HOST", "").strip()
    port = int(os.getenv("ALERT_SMTP_PORT", "587"))
    user = os.getenv("ALERT_SMTP_USER", "").strip()
    password = os.getenv("ALERT_SMTP_PASSWORD", "").strip()
    to_addr = os.getenv("ALERT_EMAIL_TO", "").strip()
    if not all([host, user, password, to_addr]):
        return False
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
        return True
    except Exception as e:
        logger.warning("Email alert failed: %s", e)
        return False


def _build_stock_summary_lines(result: dict) -> list:
    """Build summary lines for each stock from candidates or scores, including analysis summary."""
    lines = []
    # invest_flow uses 'candidates'; analyze_mag7_and_invest uses 'scores'
    items = result.get("candidates") or result.get("scores") or []
    if not items:
        return lines

    lines.append("Stock ratings:")
    for item in items:
        sym = item.get("symbol", "?")
        score = item.get("score")
        suggested = (item.get("suggested") or item.get("decision") or "hold").lower()
        price = item.get("latest_price") or item.get("details", {}).get("latest_price")
        analysis = item.get("analysis") or {}
        rationale = analysis.get("rationale") if isinstance(analysis, dict) else None
        summary = analysis.get("summary") if isinstance(analysis, dict) else None
        rationale_bullets = item.get("rationale_bullets") or []
        conf = item.get("confidence")
        parts = [f"  {sym}: {suggested}"]
        if score is not None:
            try:
                sc = float(score)
                parts[0] += f" (score {sc:.2f})"
            except (TypeError, ValueError):
                pass
        if conf is not None:
            try:
                parts[0] += f", conf {int(conf)}%"
            except (TypeError, ValueError):
                pass
        if price is not None:
            try:
                parts[0] += f" @ ${float(price):.2f}"
            except (TypeError, ValueError):
                pass
        lines.append(parts[0])
        # Quick analysis summary: prefer summary, then rationale, then bullets
        if summary and isinstance(summary, str) and summary.strip():
            short = summary.strip()[:200] + ("..." if len(summary) > 200 else "")
            lines.append(f"    Summary: {short}")
        if rationale and isinstance(rationale, str) and rationale.strip():
            short = rationale.strip()[:120] + ("..." if len(rationale) > 120 else "")
            lines.append(f"    → {short}")
        elif rationale_bullets and isinstance(rationale_bullets, list):
            for bullet in rationale_bullets[:3]:
                if isinstance(bullet, str) and bullet.strip():
                    lines.append(f"    • {bullet.strip()[:100]}")
    return lines


def send_trade_alert(flow: str, result: dict) -> None:
    """Send a trade alert via email with stock ratings summary and any placed/closed orders.

    Sends when ALERT_EMAIL_TO is set and result has candidates/scores and/or orders.
    Sends even when the outcome is all hold (no orders placed)—so you get analysis every run.
    Includes a per-stock rating and quick analysis summary for each symbol.
    """
    if not os.getenv("ALERT_EMAIL_TO"):
        return

    placed = result.get("placed") or result.get("orders") or []
    closed = result.get("closed") or []
    stock_lines = _build_stock_summary_lines(result)
    note = result.get("note", "")

    # Send if we have stock ratings and/or orders
    if not stock_lines and not placed and not closed:
        return

    lines = [f"Trading Bot Alert ({flow})", ""]

    if stock_lines:
        lines.extend(stock_lines)
        lines.append("")

    if note:
        lines.append(f"Note: {note}")
        lines.append("")

    if placed:
        lines.append("Orders placed:")
        for p in placed:
            action = p.get("action", "buy")
            sym = p.get("symbol", "?")
            qty = p.get("qty", "?")
            price = p.get("reference_price")
            resp = p.get("resp") or {}
            status = resp.get("status") or ("error" if resp.get("error") else "ok")
            err = resp.get("error", "")
            line = f"  {action.upper()} {sym} x{qty}"
            if price is not None:
                line += f" @ ${price:.2f}"
            line += f" - {status}"
            if err:
                line += f" ({err})"
            lines.append(line)
        lines.append("")

    if closed:
        lines.append("Positions closed:")
        for c in closed:
            sym = c.get("symbol", "?")
            qty = c.get("qty", "?")
            lines.append(f"  SELL {sym} x{qty}")
        lines.append("")

    acct = result.get("account") or result.get("account_info") or {}
    pv = acct.get("portfolio_value") or acct.get("equity")
    if pv is not None:
        try:
            lines.append(f"Portfolio value: ${float(pv):,.2f}")
        except (TypeError, ValueError):
            pass

    body = "\n".join(lines).strip()
    if _email_send(f"Trading Bot: {flow}", body):
        logger.info("Trade alert sent via email")
