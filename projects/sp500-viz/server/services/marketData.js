//const fetch = require("node-fetch");
const { CONFIG } = require("../config");

async function fetchDailyAdjCloses(ticker, start, end) {
  const from = Math.floor(new Date(start).getTime() / 1000);
  const to = Math.floor(new Date(end).getTime() / 1000);

  const url = `https://finnhub.io/api/v1/stock/candle?symbol=${ticker}&resolution=D&from=${from}&to=${to}&token=${CONFIG.MARKET_API_KEY}`;
  const res = await fetch(url);
  const json = await res.json();

  if (json.s !== "ok") throw new Error("Finnhub candle error: " + JSON.stringify(json));

  return json.t.map((t, i) => ({
    date: new Date(t * 1000).toISOString().slice(0, 10),
    adjClose: json.c[i],
    volume: json.v[i],
  }));
}

module.exports = { fetchDailyAdjCloses };
