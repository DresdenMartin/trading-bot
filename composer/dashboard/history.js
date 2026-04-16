const refreshEl = document.getElementById("last-refresh");
const refreshStatus = document.getElementById("refresh-status");
const refreshBtn = document.getElementById("refresh-btn");
const canvas = document.getElementById("history-chart");
const metaEl = document.getElementById("history-meta");

const ctx = canvas.getContext("2d");

function setLastRefresh(ts) {
  const d = new Date(ts);
  refreshEl.textContent = d.toLocaleString();
}

function drawChart(points) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#111824";
  ctx.fillRect(0, 0, w, h);

  if (points.length < 2) {
    ctx.fillStyle = "#a5b4c4";
    ctx.font = "14px IBM Plex Mono, monospace";
    ctx.fillText("Not enough data yet.", 20, 40);
    return;
  }

  const values = points.map((p) => p.equity).filter((v) => v !== null && v !== undefined);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min) * 0.15 || 1;
  const lo = min - pad;
  const hi = max + pad;

  const stepX = w / (points.length - 1);
  const scaleY = (val) => h - ((val - lo) / (hi - lo)) * (h - 40) - 20;

  ctx.strokeStyle = "#2a3648";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = 20 + (i * (h - 40)) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#62d0ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, idx) => {
    const x = idx * stepX;
    const y = scaleY(p.equity ?? lo);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#ff7a50";
  const last = points[points.length - 1];
  const lastX = (points.length - 1) * stepX;
  const lastY = scaleY(last.equity ?? lo);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#a5b4c4";
  ctx.font = "12px IBM Plex Mono, monospace";
  ctx.fillText(`Low: ${min.toFixed(2)}`, 16, h - 10);
  ctx.fillText(`High: ${max.toFixed(2)}`, w - 120, h - 10);
}

async function fetchHistory() {
  refreshStatus.textContent = "Refreshing...";
  try {
    const res = await fetch("/api/history?limit=240");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    const points = (data.history || []).filter((p) => p.equity !== null);
    drawChart(points);
    metaEl.textContent = points.length
      ? `Points: ${points.length} | Latest equity: ${points[points.length - 1].equity?.toFixed(2)}`
      : "No history yet. Refresh the main dashboard to record snapshots.";
    setLastRefresh(new Date().toISOString());
    refreshStatus.textContent = "Updated";
  } catch (err) {
    refreshStatus.textContent = "Failed";
    metaEl.textContent = `Error: ${err.message || err}`;
  }
}

refreshBtn.addEventListener("click", fetchHistory);
fetchHistory();
setInterval(fetchHistory, 60000);
