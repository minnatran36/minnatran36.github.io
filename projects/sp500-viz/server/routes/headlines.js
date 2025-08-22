const express = require("express");
const router = express.Router();
const { fetchHeadlines } = require("../services/news");
const { validateTicker } = require("./sp500");

function normalize(s) {
  return s.toLowerCase().replace(/[^\w\s]/g, "").replace(/\s+/g, " ").trim();
}

function dedupeHeadlines(items) {
  const seen = new Map();
  for (const h of items) {
    const key = normalize(h.title);
    if (!seen.has(key)) {
      seen.set(key, { ...h, syndicatedCount: 0 });
    } else {
      const prev = seen.get(key);
      prev.syndicatedCount++;
    }
  }
  return Array.from(seen.values());
}

router.get("/:ticker", async (req, res) => {
  try {
    const { ticker } = req.params;
    const { start, end } = req.query;
    validateTicker(ticker);

    const items = await fetchHeadlines(ticker.toUpperCase(), start, end);
    res.json({ ticker: ticker.toUpperCase(), start, end, headlines: dedupeHeadlines(items) });
  } catch (e) {
    res.status(e.status || 500).json(e.body || { error: e.message });
  }
});

module.exports = { headlines: router };
