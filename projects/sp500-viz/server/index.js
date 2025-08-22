const express = require("express");
const cors = require("cors");
const path = require("path");
const { CONFIG } = require("./config");
const { sp500 } = require("./routes/sp500");
const { returns } = require("./routes/returns");
const { headlines } = require("./routes/headlines");

const app = express();

// If you ever need to hit this API from another origin during dev, keep cors():
// app.use(cors());
// Since we’re serving the client from the same origin now, CORS is optional.
app.use(express.json());

// ---- API routes (same as before) ----
app.use("/api/sp500", sp500);
app.use("/api/returns", returns);
app.use("/api/headlines", headlines);

// ---- Static client ----
const clientDir = path.join(__dirname, "..", "client");
app.use(express.static(clientDir));

// Root -> index.html
app.get("/", (_req, res) => {
  res.sendFile(path.join(clientDir, "index.html"));
});

// Optional: SPA fallback (if you add more client routes later)
// app.get("*", (_req, res) => res.sendFile(path.join(clientDir, "index.html")));

app.listen(CONFIG.PORT, () => {
  console.log(`Server running on http://localhost:${CONFIG.PORT}`);
});
