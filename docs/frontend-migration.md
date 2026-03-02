# Frontend Migration PRD + Implementation Plan (Vite + React + TypeScript)

Last updated: 2026-03-02

## Context
The current “frontend” is served from the Python app run with `uvicorn`. This project will rebuild that UI as a separate Vite + React + TypeScript app under `frontend/`, while preserving current behavior (“works the same”).

---

## Implementation Plan (Phases + Steps)

**Frontend implementation note:** When building the new React UI (primarily Phase 1 and Phase 3), follow the `frontend-design` skill guidance to ensure the UI is production-grade and visually polished (without changing functional behavior/flows required for parity).

### Phase 0 — Audit current “uvicorn frontend”
- Inventory how the UI is served today (templates, static files, HTML routes, API calls, auth/session, uploads, websockets, etc.).
- Produce a **parity checklist** (routes/pages + key behaviors + screenshots) that defines “works the same”.

**Dev Notes (after Phase 0):** Append `Dev Notes — Phase 0` to this file with findings, route map, and any unknowns/risks discovered.

### Phase 1 — Create new Vite + React + TypeScript app in `frontend/`
- Scaffold `frontend/` with Vite (React + TS), project structure, and build output.
- Add dev ergonomics: lint/format, env handling, path aliases, and a dev-server proxy to the uvicorn backend (avoid CORS).
- Add a root `Makefile` with `make frontend` (run Vite dev server from `frontend/`) and `make backend` (run uvicorn backend from repo root).
- Implement app shell (routing, layout, error boundaries, API client wrapper).

**Dev Notes (after Phase 1):** Append what was scaffolded, key package choices, dev/prod assumptions, and how to run it.

### Phase 2 — Integrate with the existing uvicorn backend (serve built UI)
- Decide and implement the production topology:
  - Dev: Vite dev server + uvicorn API.
  - Prod: uvicorn serves `frontend/dist` (static) + SPA fallback for client routes.
- Ensure API routes remain unchanged and frontend routing doesn’t break deep links.

**Dev Notes (after Phase 2):** Append backend integration details, any route-prefix decisions (e.g., `/api`), and fallback behavior.

### Phase 3 — Port features to match current UI behavior
- Rebuild each existing page/flow in React to match the parity checklist (forms, validations, errors, loading states).
- Recreate any JS behaviors currently embedded in templates (file upload, streaming, websockets, etc.).
- Keep a running **parity status** table (Done/Blocked/Differs) per route/feature.

**Dev Notes (after Phase 3):** Append what was ported, diffs vs old UI (if any), and follow-ups.

### Phase 4 — Verification (functional parity)
- Manual QA against the checklist; optionally add Playwright/Cypress smoke tests for the most important flows.
- Fix gaps until the checklist is green.

**Dev Notes (after Phase 4):** Append test notes, known issues, and final parity confirmation.

### Phase 5 — Cutover + cleanup
- Update docs/scripts so the default dev workflow is clear (backend + frontend).
- If approved, retire old template/static frontend code paths once the new frontend is the source of truth.

**Dev Notes (after Phase 5):** Append cutover steps, anything removed, and rollback notes.

---

## PRD — “Migrate uvicorn-served frontend to Vite React + TypeScript”

### Problem
The current UI is served directly from the Python/uvicorn app, making modern frontend iteration (HMR, modular UI, typed client code, clearer separation) harder.

### Goal
Rebuild the existing frontend as a Vite + React + TypeScript app in a separate `frontend/` folder while preserving user-visible behavior (“works the same”).

### Non-Goals
- No product redesign.
- No new features.
- No backend API redesign (unless strictly required for parity).

### Success Criteria (Acceptance)
- All current UI routes/flows work with equivalent behavior (per the Phase 0 parity checklist).
- Dev workflow: run frontend with HMR + backend via uvicorn, with a dev proxy to avoid CORS issues.
- Root shortcuts exist for local dev: `make frontend` and `make backend`.
- Prod workflow: a single deploy can serve the built frontend (from uvicorn) and the existing API without breaking deep links.

### Key Requirements
- **Parity:** Same routes, forms, validations, auth handling, and error states as today.
- **Folder structure:** New `frontend/` at repo root (or agreed location), isolated Node tooling.
- **API connectivity:** Frontend calls the same backend endpoints; dev proxy in development.
- **Routing:** Client-side routing with SPA fallback in production serving.
- **Configuration:** Environment variables for API base / WS base as needed.
- **DX commands:** Root `Makefile` must provide `make frontend` and `make backend` for starting each service from repo root.

### Deliverables
- `frontend/` Vite React TS application.
- Backend changes (if needed) to serve `frontend/dist` and provide SPA fallback.
- Root `Makefile` targets for `frontend` and `backend` development runs.
- A parity checklist + screenshots (created in Phase 0).
- Phase-by-phase dev notes appended to this document.

### Risks / Open Questions (to resolve in Phase 0)
- Whether the current UI relies on server-rendered template logic that must be replicated client-side.
- Authentication/session mechanics (cookies vs tokens) and CSRF behavior.
- Any streaming, SSE, or websocket usage that requires special handling.
- Any implicit URL structure assumptions (trailing slashes, redirects, query param conventions).

---

## Dev Notes (append after each phase)

### Dev Notes — Phase 0
TBD

### Dev Notes — Phase 1
TBD

### Dev Notes — Phase 2
TBD

### Dev Notes — Phase 3
TBD

### Dev Notes — Phase 4
TBD

### Dev Notes — Phase 5
TBD

---

## Appendix — `frontend-design` skill (verbatim)

```md
---
name: frontend-design
description: Create distinctive, production-grade frontend interfaces with high design quality. Use this skill when the user asks to build web components, pages, artifacts, posters, or applications (examples include websites, landing pages, dashboards, React components, HTML/CSS layouts, or when styling/beautifying any web UI). Generates creative, polished code and UI design that avoids generic AI aesthetics.
license: Complete terms in LICENSE.txt
---

This skill guides creation of distinctive, production-grade frontend interfaces that avoid generic "AI slop" aesthetics. Implement real working code with exceptional attention to aesthetic details and creative choices.

The user provides frontend requirements: a component, page, application, or interface to build. They may include context about the purpose, audience, or technical constraints.

## Design Thinking

Before coding, understand the context and commit to a BOLD aesthetic direction:
- **Purpose**: What problem does this interface solve? Who uses it?
- **Tone**: Pick an extreme: brutally minimal, maximalist chaos, retro-futuristic, organic/natural, luxury/refined, playful/toy-like, editorial/magazine, brutalist/raw, art deco/geometric, soft/pastel, industrial/utilitarian, etc. There are so many flavors to choose from. Use these for inspiration but design one that is true to the aesthetic direction.
- **Constraints**: Technical requirements (framework, performance, accessibility).
- **Differentiation**: What makes this UNFORGETTABLE? What's the one thing someone will remember?

**CRITICAL**: Choose a clear conceptual direction and execute it with precision. Bold maximalism and refined minimalism both work - the key is intentionality, not intensity.

Then implement working code (HTML/CSS/JS, React, Vue, etc.) that is:
- Production-grade and functional
- Visually striking and memorable
- Cohesive with a clear aesthetic point-of-view
- Meticulously refined in every detail

## Frontend Aesthetics Guidelines

Focus on:
- **Typography**: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics; unexpected, characterful font choices. Pair a distinctive display font with a refined body font.
- **Color & Theme**: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes.
- **Motion**: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions. Use scroll-triggering and hover states that surprise.
- **Spatial Composition**: Unexpected layouts. Asymmetry. Overlap. Diagonal flow. Grid-breaking elements. Generous negative space OR controlled density.
- **Backgrounds & Visual Details**: Create atmosphere and depth rather than defaulting to solid colors. Add contextual effects and textures that match the overall aesthetic. Apply creative forms like gradient meshes, noise textures, geometric patterns, layered transparencies, dramatic shadows, decorative borders, custom cursors, and grain overlays.

NEVER use generic AI-generated aesthetics like overused font families (Inter, Roboto, Arial, system fonts), cliched color schemes (particularly purple gradients on white backgrounds), predictable layouts and component patterns, and cookie-cutter design that lacks context-specific character.

Interpret creatively and make unexpected choices that feel genuinely designed for the context. No design should be the same. Vary between light and dark themes, different fonts, different aesthetics. NEVER converge on common choices (Space Grotesk, for example) across generations.

**IMPORTANT**: Match implementation complexity to the aesthetic vision. Maximalist designs need elaborate code with extensive animations and effects. Minimalist or refined designs need restraint, precision, and careful attention to spacing, typography, and subtle details. Elegance comes from executing the vision well.

Remember: Claude is capable of extraordinary creative work. Don't hold back, show what can truly be created when thinking outside the box and committing fully to a distinctive vision.
```
