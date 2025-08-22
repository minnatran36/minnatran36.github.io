//const fetch = require("node-fetch");
const { CONFIG } = require("../config");

async function fetchHeadlines(ticker, start, end) {
  const url = `https://finnhub.io/api/v1/company-news?symbol=${ticker}&from=${start}&to=${end}&token=${CONFIG.NEWS_API_KEY}`;
  const res = await fetch(url);
  const json = await res.json();

  return (json || []).map((n) => ({
    date: n.datetime ? new Date(n.datetime * 1000).toISOString().slice(0, 10) : "",
    title: n.headline,
    source: n.source,
    url: n.url,
  }));
}

module.exports = { fetchHeadlines };
