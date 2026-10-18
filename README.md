# Majsoul-Agent
Vision-based auto-play Agent for Mahjong Soul three-player East. It reads the Steam client from screenshots, tracks the table state, picks moves with an LLM (DeepSeek) and falls back to local rules, then clicks and checks the result. A watchdog restarts the Agent if it fails, and it plays matches back to back.
