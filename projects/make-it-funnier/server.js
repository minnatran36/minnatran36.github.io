// server.js
import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import fetch from "node-fetch";
import cors from "cors";

const app = express();
// Allow JSON bodies
app.use(express.json({ limit: '1mb' }));
// CORS: allow your Pages origin and localhost
const corsOptions = {
origin: [
   'https://minnatran36.github.io',
   'http://localhost:3000',
    'http://localhost:5173'
  ],
  methods: ['GET', 'POST', 'OPTIONS'],
  // Let the library reflect whatever headers the browser asks for.
  // (If you MUST pin later, include Authorization, Accept, X-Requested-With, and UA-CH headers.)
  optionsSuccessStatus: 200
};
app.use(cors(corsOptions));
app.options('*', cors(corsOptions)); // handle all preflights

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(__dirname)); // serve index.html + assets from root

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
// Choose a small, fast model name you have access to:
const MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini"; // or "gpt-4o"

app.post("/api/funny", async (req, res) => {
  try {
    const { text, style = "wry", spice = 1, remixSeed = "" } = req.body || {};
    if (!OPENAI_API_KEY) return res.status(500).send("Missing OPENAI_API_KEY");
    if (!text || typeof text !== "string") return res.status(400).json({ error: "text required" });

    // Guardrails: keep it one-liner, same meaning-ish, funny twist allowed
    const system = [
      "You punch-up sentences into one-liners. Keep it concise (max ~25 words).",
      "Keep core meaning recognizable unless spice is high.",
      "No slurs, hate, or NSFW. Avoid doxxing or personal data.",
      "Prefer clever misdirection, contrast, escalation, and tags.",
      "Return ONLY the rewritten line. No quotes. No prefaces."
    ].join(" ");

    // Style directives
    const styles = {
      "deadpan": "Tone: dry, understated. Minimal adjectives. Let contrast deliver humor.",
      "wry": "Tone: sly, lightly ironic. Quick tag at the end.",
      "absurdist": "Tone: surreal twist but keep sentence coherent.",
      "self-roast": "Tone: self-deprecating, good-natured.",
      "wholesome": "Tone: playful and kind; avoid cynicism."
    };

    const spiceHints = [
      "Spice 0: tiny nudge, keep structure, swap a word or add a short tag.",
      "Spice 1: mild twist, small exaggeration, short tag.",
      "Spice 2: noticeable rewrite, playful escalation, stronger punchline.",
      "Spice 3: big swing allowed, but keep it one sentence and intelligible."
    ];

    const user = `
Original: ${text}
Style: ${style}
${styles[style] || ""}
${spiceHints[Math.max(0, Math.min(3, Number(spice)))]}
Remix seed: ${remixSeed}
Respond with only the funnier single-sentence rewrite, no quotes.`;

    // Chat Completions
    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: MODEL,
        temperature: Math.min(0.2 + 0.25 * Number(spice || 1), 1.0),
        top_p: 1,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user }
        ]
      })
    });

    if (!resp.ok) {
      const t = await resp.text();
      return res.status(500).send(t);
    }

    const data = await resp.json();
    const output = data?.choices?.[0]?.message?.content ?? "";
    res.json({ output });
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

app.get('/api/health', (_req, res) => res.json({ ok: true }));


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`> http://localhost:${PORT}`));
