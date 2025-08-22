const express = require("express");
const router = express.Router();
const { fetchDailyAdjCloses } = require("../services/marketData");
const { computeDailyReturns, computeEndpointReturns } = require("../utils/returns");
const { computeWeeklyEndpoints, computeMonthlyEndpoints } = require("../utils/endpoints");
const { validateTicker } = require("./sp500");

router.get("/:ticker", async (req, res) => {
  try {
    const { ticker } = req.params;
    const year = Number(req.query.year);
    validateTicker(ticker);

    const start = `${year}-01-01`;
    const end = `${year}-12-31`;
    const rows = await fetchDailyAdjCloses(ticker.toUpperCase(), start, end);

    const dailyR = computeDailyReturns(rows);
    const weeklyEnds = computeWeeklyEndpoints(rows);
    const monthlyEnds = computeMonthlyEndpoints(rows);
    const weekR = computeEndpointReturns(rows, weeklyEnds);
    const monthR = computeEndpointReturns(rows, monthlyEnds);

    res.json({
      ticker: ticker.toUpperCase(),
      year,
      daily: rows.map((r) => ({ date: r.date, adjClose: r.adjClose, dailyReturn: dailyR[r.date] })),
      weeklyEndpoints: weeklyEnds,
      weeklyReturns: weekR,
      monthlyEndpoints: monthlyEnds,
      monthlyReturns: monthR,
    });
  } catch (e) {
    res.status(e.status || 500).json(e.body || { error: e.message });
  }
});

module.exports = { returns: router };
