Frontend plan (Next.js)

Summary
- Single page app that captures a 4th-down situation and calls the backend.
- Shows three result blocks: Win Probabilities (wp), Component probabilities (comp), and Coach Policy (coach).
- Option to compare all three at once via `/predict/all`.

Data flow
- Collect raw inputs: quarter, clock (min/sec), yardline_100 (or side+yardline), ydstogo, score diff, timeouts, season type, roof/surface, temperature (F), wind (mph).
- Build a `PlayInput` JSON payload and POST to the backend.
- Render formatted results and a recommended action from wp.

Implementation Notes
- Create a model selector or call `/predict/all`.
- Validate ranges client-side; mirror backend types in `types.ts`.
- Configure API base via `NEXT_PUBLIC_API_BASE`.

Run locally
- In one terminal, start the backend API:
  - `uvicorn backend.app.main:app --reload --port 8000`
- In another terminal, start the Next.js dev server:
  - `cd web && npm install && npm run dev`
  - Optionally create `web/.env.local` and set `NEXT_PUBLIC_API_BASE=http://localhost:8000`

