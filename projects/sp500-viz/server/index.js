const express = require("express");
const cors = require("cors");
const { CONFIG } = require("./config");
const { sp500 } = require("./routes/sp500");
const { returns } = require("./routes/returns");
const { headlines } = require("./routes/headlines");

const app = express();
app.use(cors());
app.use(express.json());

app.use("/api/sp500", sp500);
app.use("/api/returns", returns);
app.use("/api/headlines", headlines);

app.listen(CONFIG.PORT, () => {
  console.log(`Server running on http://localhost:${CONFIG.PORT}`);
});
