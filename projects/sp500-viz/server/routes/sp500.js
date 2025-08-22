const express = require("express");
const router = express.Router();

// Hardcoded minimal S&P set for demo; replace with full list later
const CURRENT = new Set(["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]);

router.get("/", (_req, res) => {
  res.json({ tickers: Array.from(CURRENT).sort() });
});

function validateTicker(t) {
  if (!CURRENT.has(t.toUpperCase())) {
    const err = { error: "Ticker not in current S&P 500" };
    const e = new Error(err.error);
    e.status = 400;
    e.body = err;
    throw e;
  }
}

module.exports = { sp500: router, validateTicker };
