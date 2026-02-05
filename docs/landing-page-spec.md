# TribalMemory Landing Page — Design Spec

> **Goal:** A landing page so visually striking that UI designers screenshot it and post "stealing this." White-on-black, terminal aesthetic, animated ASCII art. Not a template — a signature.

---

## Design Philosophy

**The Cursor Effect.** When Cursor's agent boots up and that ASCII logo animates into existence in the terminal, developers pause. It's a 2-second moment that says "this is crafted." We want that feeling — but as an entire landing page.

**References:**
- Cursor agent ASCII boot animation (the inspiration)
- charm.sh — playful terminal identity, developer personality
- Linear — centered hero, breathing room, zero fat
- Evil Martians analysis — centered layout, dual CTA, trust block

**Anti-references (what we're NOT):**
- SaaS templates with gradient hero sections
- Anything with stock photography
- Dashboard screenshots in browser mockups

---

## 1. Global Aesthetic

### Color Palette

```
Background:   #000000 (pure black)
Primary text: #FFFFFF (pure white)
Muted text:   #666666 (subdued gray — timestamps, secondary info)
Accent:       #00FF00 (terminal green — used SPARINGLY: cursor blink, active states)
Hover:        #333333 (dark gray — subtle button/link hover states)
Code blocks:  #111111 (barely-there gray background)
```

No gradients. No colors beyond these. The constraint IS the identity.

### Typography

```
Monospace hero:     JetBrains Mono (or Berkeley Mono if licensed)
Body text:          Inter (clean, disappears — lets content breathe)
ASCII art:          Raw monospace, pre-formatted
Code snippets:      JetBrains Mono, slightly smaller
```

**Key rule:** Headlines and the hero section use monospace. Body copy uses Inter. This creates a push-pull between "terminal" and "polished" that makes the whole page feel intentional.

### Spacing

- Generous vertical whitespace between sections (120px+)
- Max-width container: 960px (narrower than typical — reads like a document, not a dashboard)
- Let the black breathe. Empty space IS the design.

---

## 2. The Hero — "The Boot Sequence"

This is the signature moment. When the page loads, the user sees a terminal boot animation.

### Animation Sequence (2.5 seconds total)

**Phase 1: Cursor blink (0.0s–0.4s)**
- Black screen. Single blinking green cursor `▋` in the center.
- The user thinks "is this loading?"

**Phase 2: ASCII logo typewriter (0.4s–1.8s)**
- Characters appear rapid-fire, like being typed by a fast terminal:

```
  ████████╗██████╗ ██╗██████╗  █████╗ ██╗     
  ╚══██╔══╝██╔══██╗██║██╔══██╗██╔══██╗██║     
     ██║   ██████╔╝██║██████╔╝███████║██║     
     ██║   ██╔══██╗██║██╔══██╗██╔══██╗██║     
     ██║   ██║  ██║██║██████╔╝██║  ██║███████╗
     ╚═╝   ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝╚══════╝
  ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
  ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
  ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝ 
  ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝  
  ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║   
  ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   
```

- Characters render left-to-right, top-to-bottom
- Speed: ~800 chars/second (fast but readable)
- Each character "materializes" from dim gray (#333) to white (#fff) with a 100ms fade
- The just-typed character has a brief green glow (#0f0) that fades to white

**Phase 3: Tagline fade-in (1.8s–2.3s)**
- Below the ASCII art, fades in smoothly:

```
  Your AI agents don't share a brain. Now they do.
```

- Inter font, ~20px, muted gray → white fade over 500ms

**Phase 4: CTAs materialize (2.3s–2.5s)**
- Two buttons fade in side by side:

```
  [ pip install tribalmemory ]     [ View on GitHub → ]
```

- Primary CTA: white text, white 1px border, fills white-on-hover (inverts to black text). Links to [pypi.org/project/tribalmemory](https://pypi.org/project/tribalmemory/)
- Secondary CTA: muted gray text, no border, brightens on hover. Links to [github.com/abbudjoe/TribalMemory](https://github.com/abbudjoe/TribalMemory)

### Brain-Chip ASCII Icon (optional flourish)

Small ASCII art version of the TribalMemory brain-circuit logo rendered above or beside the text logo:

```
       ╭──────────╮
      ╱   ┌─┬─┐    ╲
     │   ─┤ ● ├─   │
     │    └─┼─┘    │
      ╲   ──┼──   ╱
       ╰──────────╯
```

This sits to the left of the main ASCII wordmark, or centered above it. Small. Recognizable. Converted from the existing vector logo into monospace art.

### "Skip animation" behavior
- Click/scroll/keypress instantly completes the animation
- `prefers-reduced-motion` → show final state immediately
- On revisit (sessionStorage flag) → skip to final state

---

## 3. The Architecture Diagram — "The Live Wire"

This replaces the traditional "product screenshot" block. It's an ASCII architecture diagram that animates.

```
  ┌─────────────┐
  │  Claude Code │──── MCP ────┐
  └─────────────┘              │
  ┌─────────────┐              ▼
  │  Codex CLI   │──── MCP ──▶ ████████████████████
  └─────────────┘              ▲ Tribal Memory Server
  ┌─────────────┐              │   localhost:18790
  │  OpenClaw    │── plugin ───┘
  └─────────────┘
```

**The animation:** Data packets (represented as `·` or `▸` characters) visually travel along the connection lines. They pulse from agents toward the server and back, creating a living circuit feel:

- Small dots (`·`) flow along the `────` lines toward the server
- On arrival, the server box briefly brightens
- Return dots flow back to the agents
- Continuous subtle loop, ~3s cycle
- Speed is relaxed, not frantic — like a heartbeat

**Below the diagram**, in muted gray:

```
  One memory store. Every agent connected. Zero repetition.
```

---

## 4. The Demo — "The Proof"

Instead of a static code snippet, embed an actual asciinema-style terminal replay showing the "One Brain, Two Agents" demo. We already have the `.cast` files.

### Layout

Two terminal windows side by side (on desktop) or stacked (mobile):

```
┌─── Claude Code ─────────────────┐  ┌─── Codex CLI ──────────────────┐
│ > "Remember: auth uses JWT RS256"│  │ > "How does auth work?"        │
│ ✅ Stored in tribal memory       │  │ Based on tribal memory:        │
│                                  │  │ Auth service uses JWT with     │
│                                  │  │ RS256 signing...               │
└──────────────────────────────────┘  └──────────────────────────────────┘
```

- Styled as real terminal windows with title bars
- Content types out live (asciinema player or custom JS)
- Timing: Claude Code types first → brief pause → Codex recalls
- The "aha moment" is visceral — they SHARE a brain
- Loops seamlessly

### Below the demo

```
  tribal_store → one agent learns
  tribal_recall → every agent knows
```

---

## 5. Feature Grid — "The Manifest"

Not cards. Not icons. A terminal-style manifest that reads like a `--help` output.

```
  FEATURES
  ────────────────────────────────────────────────────

  --semantic-search      Find memories by meaning, not keywords
  --cross-agent          Memories from one agent → all agents  
  --graph-search         Entity extraction + relationship traversal
  --hybrid-retrieval     Vector + BM25 keyword search combined
  --local-first          FastEmbed ONNX: zero cloud, zero API keys
  --session-indexing     Index conversation transcripts for search
  --deduplication        Won't store the same thing twice
  --temporal-reasoning   Date extraction and time-based filtering
  --import-export        Portable JSON bundles across instances
  --mcp-native           Works with Claude Code, Codex, and more
```

Each line fades in sequentially (50ms stagger) as the user scrolls into view. On hover, the description text brightens and the flag slides slightly right.

---

## 6. The Install Block — "Three Lines"

```
  GET STARTED
  ────────────────────────────────────────────────────

  $ pip install tribalmemory
  $ tribalmemory init
  $ tribalmemory serve

  That's it. Server running. Memory shared. No config needed.
```

- The `$` prompts are green (#0f0)
- Commands are white
- `pip install tribalmemory` links to [pypi.org/project/tribalmemory](https://pypi.org/project/tribalmemory/) (subtle underline on hover)
- On scroll-in, each line types out with ~200ms delay between lines
- The "That's it." line fades in after the commands complete

### Provider Options (collapsible)

```
  # Want OpenAI embeddings instead?
  $ tribalmemory init --openai

  # Already running Ollama?
  $ tribalmemory init --ollama
```

Shown in a `<details>` or expanding section, muted by default.

---

## 7. Trust / Stats Block

Minimal. Numbers, not logos (we're early-stage).

```
  ────────────────────────────────────────────────────

  735 tests passing    v0.4.2 on PyPI    Apache 2.0
  100% LoCoMo recall   3 providers       10+ MCP tools

  ────────────────────────────────────────────────────
```

- **LoCoMo recall note:** 100% accuracy across 1986 questions (open-domain, adversarial, temporal, single-hop, multi-hop) on the [LoCoMo benchmark](https://github.com/snap-research/locomo). Update this number once the full run completes — currently tracking 100% at 885/1986.
- Numbers in white, bold monospace
- Labels in muted gray below each number
- Horizontal layout, evenly spaced
- Numbers animate up (0 → 735) with a satisfying counter tick when scrolled into view
- "v0.4.2 on PyPI" links to [pypi.org/project/tribalmemory](https://pypi.org/project/tribalmemory/)

---

## 8. The Integrations — "Plug In"

Three code blocks showing how each agent connects:

```
  CLAUDE CODE                   CODEX CLI                   OPENCLAW
  ──────────────                ──────────────              ──────────────
  {                             [mcp_servers.               $ openclaw
    "mcpServers": {               tribal-memory]              plugins install
      "tribal-memory": {        command =                     ./extensions/
        "command":                "tribalmemory-mcp"            memory-tribal
          "tribalmemory-mcp"    
      }                         # ~/.codex/config.toml
    }
  }
```

Three columns on desktop, stacked on mobile. Each with a small animated cursor that blinks in the code, like an editor just pasted the config.

---

## 9. Privacy Block

Short, punchy, in a bordered box:

```
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  LOCAL MODE = ZERO DATA LEAVES YOUR MACHINE      │
  │                                                  │
  │  ▸ Embeddings computed locally (ONNX runtime)    │
  │  ▸ Memories stored locally (LanceDB)             │
  │  ▸ No API keys. No cloud. No telemetry.          │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

The border draws itself on scroll-in (animated border, like the box is being typed).

---

## 10. Cloud Teaser — "Coming Soon"

A subtle, understated teaser for the upcoming cloud sync feature. Not a full section — more of a whisper between the privacy block and the closer.

```
  COMING SOON
  ────────────────────────────────────────────────────

  ☁ Cloud Sync — Share memories across machines.
    Same privacy-first approach. Your keys, your data.
    Encrypted at rest. Self-hostable.

  [ Join the waitlist → ]
```

- Muted gray text, dimmer than other sections — it's a preview, not a pitch
- "Join the waitlist" CTA is ghost-style (no border, underline on hover)
- Links to a simple email capture form or GitHub Discussions thread
- The `☁` icon renders in terminal green (#0f0) — the only accent in this section
- This section is **optional at launch** — can be added once the cloud spec is further along

---

## 11. Final CTA — "The Closer"

```
  Your agents are forgetting everything.
  Fix that.

  $ pip install tribalmemory


  ★ Star on GitHub          📖 Read the Docs          💬 Join Discord
```

- "Your agents are forgetting everything." — large monospace, white
- "Fix that." — after 500ms delay, punchy
- The pip command has a blinking cursor at the end
- Footer links in muted gray, brighten on hover

**Footer link targets:**
- ★ Star on GitHub → [github.com/abbudjoe/TribalMemory](https://github.com/abbudjoe/TribalMemory)
- 📖 Read the Docs → docs site (TBD — `/docs` on the landing page domain, or separate subdomain)
- 💬 Join Discord → [discord.gg/Rzk3E8g2s5](https://discord.gg/Rzk3E8g2s5)

---

## Technical Implementation

### Stack

```
Next.js 15 (App Router)  — or Astro if we want static-only
Tailwind CSS             — utility-first, dark mode native
Framer Motion            — scroll-triggered animations, typewriter
Custom TypeScript        — ASCII typewriter engine, packet animation
asciinema-player         — embedded terminal replay (demo section)
```

### Performance Targets

- **First Contentful Paint:** < 1.0s (black screen + cursor is instant)
- **Total page weight:** < 200KB (no images besides the favicon)
- **Lighthouse score:** 95+ across all categories
- **Zero JavaScript required for content** — animations are progressive enhancement

### Responsive Behavior

- **Desktop (1024px+):** Full ASCII art, side-by-side demos, 3-col integrations
- **Tablet (768px):** Slightly smaller ASCII art, stacked demos
- **Mobile (< 768px):** Simplified ASCII (smaller wordmark variant), single column, reduced animations

### ASCII Art Scaling

The large box-drawing ASCII logo needs a mobile variant:

```
  ╔╦╗┬─┐┬┌┐ ┌─┐┬  
   ║ ├┬┘│├┴┐├─┤│  
   ╩ ┴└─┴└─┘┴ ┴┴─┘
  ╔╦╗┌─┐┌┬┐┌─┐┬─┐┬ ┬
  ║║║├┤ ││││ │├┬┘└┬┘
  ╩ ╩└─┘┴ ┴└─┘┴└─ ┴ 
```

Smaller, still recognizable, works at 320px width.

---

## Micro-Interactions (The "Copy That" Moments)

These are the details that make designers screenshot the page:

1. **Cursor trail:** The mouse cursor leaves a faint, fading trail of dots on the black background — like a terminal trace. Subtle. Disappears in 500ms.

2. **Scanline overlay:** A barely-visible scanline texture (CSS only, repeating 2px gradient) gives the entire page a CRT monitor feel. Opacity: 3-5%. Just enough to feel, not enough to notice consciously.

3. **Code block copy:** Hover any code block → a `[ copy ]` button appears. Click → the button text changes to `[ copied ✓ ]` with a brief green flash.

4. **Scroll progress:** A thin 1px white line at the very top of the viewport grows left-to-right as you scroll down. Terminal feel: it's a progress bar.

5. **ASCII logo easter egg:** Press `Ctrl+/` or type "memory" on the page → the ASCII logo re-plays its boot animation.

6. **Link underlines:** No underlines by default. On hover, a single-pixel underline types itself in from left to right (width animation, not opacity).

7. **Section transitions:** Each section has a thin horizontal rule (`────────`) that draws itself as you scroll into view, left to right.

---

## Page Structure Summary

```
┌──────────────────────────────────────────────┐
│  Nav: [Logo]              [GitHub] [Docs]    │  ← minimal, fixed
│  GitHub → github.com/abbudjoe/TribalMemory   │
│  Docs → docs.tribalmemory.dev (or /docs)     │
├──────────────────────────────────────────────┤
│                                              │
│           [ASCII BOOT ANIMATION]             │  ← the hero
│     Your AI agents don't share a brain.      │
│               Now they do.                   │
│                                              │
│  [ pip install tribalmemory ]  [ GitHub → ]  │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│         [ANIMATED ARCHITECTURE]              │  ← the live wire
│  Claude Code ── MCP ──▶ Tribal Memory        │
│  Codex CLI   ── MCP ──▶    Server            │
│  OpenClaw    ── plug ──▶                     │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│    [LIVE TERMINAL DEMO - TWO AGENTS]         │  ← the proof
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│           FEATURES (--help style)            │  ← the manifest
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│              GET STARTED                     │  ← three lines
│  $ pip install tribalmemory                  │
│  $ tribalmemory init                         │
│  $ tribalmemory serve                        │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│    735 tests   v0.4.2   Apache 2.0           │  ← trust
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│  CLAUDE CODE    CODEX CLI    OPENCLAW        │  ← integrations
│  {...}          [...]        $ ...           │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│  ┌ LOCAL MODE = ZERO DATA LEAVES ──────────┐ │  ← privacy
│  └─────────────────────────────────────────┘ │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│  ☁ Cloud Sync — coming soon.                 │  ← cloud teaser
│  [ Join the waitlist → ]                     │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│     Your agents are forgetting everything.   │  ← closer
│     Fix that.                                │
│                                              │
│     $ pip install tribalmemory▋              │
│                                              │
│     [GitHub]  [Docs]  [Discord]              │
│                                              │
└──────────────────────────────────────────────┘
```

---

## What Makes This "Copy-Worthy"

1. **The boot animation** — nobody does ASCII typewriter on landing pages. It's the Cursor terminal moment, but for a whole page.
2. **Living architecture diagram** — data packets flowing through ASCII art is genuinely novel.
3. **The constraint** — pure black and white with one accent color. The discipline makes it feel premium.
4. **CRT scanline overlay** — subconscious nostalgia hit. Designers will notice this.
5. **Terminal `--help` as feature grid** — turns a boring features section into something developers actually want to read.
6. **Drawing borders on scroll** — the box in the privacy section literally types itself. Small moment, big impact.
7. **Zero images** — the entire page is text and CSS. That's a statement.
8. **The cursor trail** — tiny touch that creates spatial memory. "That page where the mouse left traces."
