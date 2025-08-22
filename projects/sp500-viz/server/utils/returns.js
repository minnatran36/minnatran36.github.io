function computeDailyReturns(rows) {
  const out = {};
  rows.forEach((r, i) => {
    if (i === 0) out[r.date] = null;
    else out[r.date] = round4((r.adjClose - rows[i - 1].adjClose) / rows[i - 1].adjClose);
  });
  return out;
}

function computeEndpointReturns(rows, endpoints) {
  const byDate = new Map(rows.map((r) => [r.date, r.adjClose]));
  const out = {};
  endpoints.forEach((d, i) => {
    if (i === 0) out[d] = null;
    else {
      const a = byDate.get(endpoints[i - 1]);
      const b = byDate.get(d);
      out[d] = round4((b - a) / a);
    }
  });
  return out;
}

function round4(x) {
  return Math.round(x * 1e4) / 1e4;
}

module.exports = { computeDailyReturns, computeEndpointReturns };
