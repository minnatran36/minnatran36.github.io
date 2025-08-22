require("dotenv").config();

module.exports.CONFIG = {
  PORT: process.env.PORT || 4000,
  MARKET_API_KEY: process.env.MARKET_API_KEY || "",
  NEWS_API_KEY: process.env.NEWS_API_KEY || "",
  CACHE_TTL_SEC: 3600,
};
