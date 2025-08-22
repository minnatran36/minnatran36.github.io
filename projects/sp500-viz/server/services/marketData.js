// server/services/marketData.js
// Yahoo Finance "chart" endpoint (no API key) -> Adjusted Close daily bars
// Output: [{ date: "YYYY-MM-DD", adjClose: number, volume: number }]
async function fetchDailyAdjCloses(ticker, start, end) {
  // Yahoo expects seconds since epoch
  const fromSec = Math.floor(new Date(start + "T00:00:00Z").getTime() / 1000);
  // add one day so the end date is inclusive
  const endDate = new Date(end + "T00:00:00Z").getTime() + 24 * 3600 * 1000;
  const toSec = Math.floor(endDate / 1000);

  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}` +
    `?period1=${fromSec}&period2=${toSec}&interval=1d&events=div%2Csplit&includeAdjustedClose=true`;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Yahoo HTTP ${res.status}`);
  const j = await res.json();

  const result = j?.chart?.result?.[0];
  if (!result) {
    const msg = j?.chart?.error?.description || "Yahoo: missing result";
    throw new Error(msg);
  }

  const ts = result.timestamp || [];
  const adj = result.indicators?.adjclose?.[0]?.adjclose || [];
  const vol = result.indicators?.quote?.[0]?.volume || [];

  const rows = [];
  for (let i = 0; i < ts.length; i++) {
    const t = ts[i];
    const a = adj[i];
    if (a == null) continue;                         // skip null bars
    const d = new Date(t * 1000).toISOString().slice(0, 10);
    rows.push({
      date: d,
      adjClose: Number(a),
      volume: Number(vol[i] ?? 0),
    });
  }
  // Already in order, but sort to be safe
  rows.sort((a, b) => (a.date < b.date ? -1 : 1));
  // Filter to [start, end] inclusively (defensive)
  return rows.filter(r => r.date >= start && r.date <= end);
}

module.exports = { fetchDailyAdjCloses };
