const refreshEl = document.getElementById("last-refresh");
const refreshStatus = document.getElementById("refresh-status");
const watchlistEl = document.getElementById("watchlist");
const openOrdersEl = document.getElementById("open-orders");
const portfolioEl = document.getElementById("portfolio");
const rankingEl = document.getElementById("ranking");
const analysisTsEl = document.getElementById("analysis-ts");
const refreshBtn = document.getElementById("refresh-btn");
const cancelOrdersBtn = document.getElementById("cancel-orders-btn");
const openOrdersStatus = document.getElementById("open-orders-status");
const reallocateBtn = document.getElementById("reallocate-btn");
const reallocateStatus = document.getElementById("reallocate-status");

const fmt = (n) => (n === null || n === undefined ? "--" : Number(n).toFixed(2));
const reallocateFrames = ["Running", "Running.", "Running..", "Running..."];
let reallocateTicker = null;
let reallocateFrameIdx = 0;

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function setLastRefresh(ts) {
  const d = new Date(ts);
  refreshEl.textContent = d.toLocaleString();
}

function renderWatchlist(data) {
  const rows = (data.symbols || []).map((item) => {
    return `
      <div class="row">
        <strong>${item.symbol}</strong>
        <span>Last: ${fmt(item.price)}</span>
        <span>${item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : "no live tick"}</span>
      </div>
    `;
  });
  watchlistEl.innerHTML = rows.join("");
}

function renderOpenOrders(data) {
  const orders = data.orders || [];
  if (!orders.length) {
    openOrdersEl.innerHTML = `<p class="status">No open orders.</p>`;
    if (openOrdersStatus) openOrdersStatus.textContent = "0 open orders";
    return;
  }
  if (openOrdersStatus) openOrdersStatus.textContent = `${orders.length} open orders`;
  const rows = orders.map((order) => {
    const symbol = order.symbol || "--";
    const side = (order.side || "--").toUpperCase();
    const qty = order.qty || order.qty_requested || order.filled_qty || "--";
    const type = (order.type || order.order_type || "--").toUpperCase();
    const limit = order.limit_price ? `Limit: ${fmt(order.limit_price)}` : null;
    const notional = order.notional ? `Notional: ${fmt(order.notional)}` : null;
    const priceLabel = limit || notional || "--";
    const status = order.status || "--";
    const submitted = order.submitted_at ? new Date(order.submitted_at).toLocaleTimeString() : "--";
    return `
      <div class="row orders">
        <strong>${symbol}</strong>
        <span>${side} ${qty}</span>
        <span>${type}</span>
        <span>${priceLabel}</span>
        <span>${status} @ ${submitted}</span>
      </div>
    `;
  });
  openOrdersEl.innerHTML = rows.join("");
}

function renderPortfolio(data) {
  const acct = data.account || {};
  const positions = data.positions || [];
  const header = `
    <div class="row highlight">
      <strong>Equity</strong>
      <span>${fmt(acct.equity || acct.portfolio_value)}</span>
      <span>Cash: ${fmt(acct.cash)}</span>
    </div>
  `;
  const rows = positions.map((p) => {
    return `
      <div class="row">
        <strong>${p.symbol}</strong>
        <span>Qty: ${fmt(p.qty)}</span>
        <span>MV: ${fmt(p.market_value)}</span>
      </div>
    `;
  });
  portfolioEl.innerHTML = header + rows.join("");
}

function renderRanking(data) {
  const analysis = data.analysis || {};
  const scores = analysis.scores || [];
  if (!scores.length) {
    rankingEl.innerHTML = `<p class="status">No ranking data yet.</p>`;
    analysisTsEl.textContent = "--";
    return;
  }
  analysisTsEl.textContent = analysis.timestamp ? new Date(analysis.timestamp).toLocaleString() : "--";
  const rows = scores.slice(0, 7).map((item, idx) => {
    return `
      <div class="row ${idx < 3 ? "highlight" : ""}">
        <strong>${item.symbol}</strong>
        <span>Score: ${item.score}</span>
        <span>${item.rationale || ""}</span>
      </div>
    `;
  });
  rankingEl.innerHTML = rows.join("");
}

async function refreshAll() {
  if (refreshStatus) refreshStatus.textContent = "Refreshing...";
  const results = await Promise.allSettled([
    fetchJson("/api/watchlist"),
    fetchJson("/api/portfolio"),
    fetchJson("/api/analysis"),
    fetchJson("/api/open_orders"),
  ]);
  const [watchlist, portfolio, analysis, openOrders] = results;
  let failures = 0;
  if (watchlist.status === "fulfilled") {
    renderWatchlist(watchlist.value);
  } else {
    failures += 1;
    console.error(watchlist.reason);
  }
  if (portfolio.status === "fulfilled") {
    renderPortfolio(portfolio.value);
  } else {
    failures += 1;
    console.error(portfolio.reason);
  }
  if (analysis.status === "fulfilled") {
    renderRanking(analysis.value);
  } else {
    failures += 1;
    console.error(analysis.reason);
  }
  if (openOrders.status === "fulfilled") {
    renderOpenOrders(openOrders.value);
  } else {
    failures += 1;
    if (openOrdersStatus) openOrdersStatus.textContent = "Failed to load orders";
    console.error(openOrders.reason);
  }
  setLastRefresh(new Date().toISOString());
  if (refreshStatus) {
    refreshStatus.textContent = failures ? `Updated with ${failures} error(s)` : "Updated";
  }
}

async function triggerReallocate() {
  startReallocateAnimation();
  try {
    const res = await fetchJson("/api/reallocate", { method: "POST" });
    stopReallocateAnimation(`Done. ${res?.summary?.orders ? res.summary.orders.length : 0} orders.`);
    await refreshAll();
  } catch (err) {
    stopReallocateAnimation(`Error: ${err.message || err}`);
  }
}

async function cancelOpenOrders() {
  if (!openOrdersStatus) {
    return;
  }
  openOrdersStatus.textContent = "Cancelling...";
  try {
    await fetchJson("/api/cancel_open_orders", { method: "POST" });
    openOrdersStatus.textContent = "Cancelled open orders.";
    await refreshAll();
  } catch (err) {
    openOrdersStatus.textContent = `Cancel failed: ${err.message || err}`;
  }
}

function startReallocateAnimation() {
  if (!reallocateStatus || reallocateTicker) {
    return;
  }
  reallocateFrameIdx = 0;
  reallocateStatus.textContent = reallocateFrames[reallocateFrameIdx];
  reallocateStatus.classList.add("running");
  if (reallocateBtn) {
    reallocateBtn.disabled = true;
    reallocateBtn.classList.add("disabled");
  }
  reallocateTicker = setInterval(() => {
    reallocateFrameIdx = (reallocateFrameIdx + 1) % reallocateFrames.length;
    reallocateStatus.textContent = reallocateFrames[reallocateFrameIdx];
  }, 600);
}

function stopReallocateAnimation(message) {
  if (reallocateTicker) {
    clearInterval(reallocateTicker);
    reallocateTicker = null;
  }
  if (reallocateStatus) {
    reallocateStatus.textContent = message;
    reallocateStatus.classList.remove("running");
  }
  if (reallocateBtn) {
    reallocateBtn.disabled = false;
    reallocateBtn.classList.remove("disabled");
  }
}

if (refreshBtn) {
  refreshBtn.addEventListener("click", refreshAll);
}
if (cancelOrdersBtn) {
  cancelOrdersBtn.addEventListener("click", () => {
    const ok = confirm("Cancel all open orders?");
    if (ok) {
      cancelOpenOrders();
    }
  });
}
reallocateBtn.addEventListener("click", () => {
  const ok = confirm("Run manual reallocation now?");
  if (ok) {
    triggerReallocate();
  }
});

refreshAll();
setInterval(refreshAll, 60000);
