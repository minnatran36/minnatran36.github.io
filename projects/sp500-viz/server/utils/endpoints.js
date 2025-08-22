function computeWeeklyEndpoints(rows) {
  const out = [];
  let currentWeek = -1;
  let lastDate = "";
  rows.forEach((r, i) => {
    const d = new Date(r.date);
    const jan4 = new Date(Date.UTC(d.getUTCFullYear(), 0, 4));
    const diff = (d - jan4) / 86400000;
    const isoWeek = Math.floor((diff + ((jan4.getUTCDay() + 6) % 7)) / 7);

    if (isoWeek !== currentWeek) {
      if (lastDate) out.push(lastDate);
      currentWeek = isoWeek;
    }
    lastDate = r.date;
    if (i === rows.length - 1) out.push(lastDate);
  });
  return out;
}

function computeMonthlyEndpoints(rows) {
  const out = [];
  let currentYM = "";
  rows.forEach((r, i) => {
    const ym = r.date.slice(0, 7);
    if (ym !== currentYM && i > 0) out.push(rows[i - 1].date);
    currentYM = ym;
  });
  if (rows.length) out.push(rows[rows.length - 1].date);
  return out;
}

module.exports = { computeWeeklyEndpoints, computeMonthlyEndpoints };
