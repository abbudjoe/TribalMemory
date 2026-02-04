import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  staticFile,
  Img,
} from "remotion";

// ─── Colors ────────────────────────────────────────────
const BG = "#ffffff";
const BG_CARD = "#1e1e2e";
const BORDER = "#313244";
const TEXT_BODY = "#1e1e2e";
const TEXT_BODY_DIM = "#6c6f85";
const TEXT = "#cdd6f4";
const TEXT_DIM = "#a6adc8";
const GREEN = "#40a02b";
const BLUE = "#1e66f5";
const PURPLE = "#8839ef";
const YELLOW = "#df8e1d";
const CYAN = "#179299";

// ─── Helpers ───────────────────────────────────────────
function typewriter(text: string, frame: number, charsPerFrame = 0.8): string {
  const chars = Math.floor(frame * charsPerFrame);
  return text.slice(0, chars);
}

function fadeIn(frame: number, start: number, dur = 15): number {
  return interpolate(frame, [start, start + dur], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
}

function slideUp(frame: number, start: number, fps: number): number {
  return interpolate(
    spring({ frame: frame - start, fps, config: { damping: 15, mass: 0.8 } }),
    [0, 1],
    [60, 0]
  );
}

// ─── Logo Component ────────────────────────────────────
const Logo: React.FC<{ size?: number }> = ({ size = 160 }) => (
  <img
    src={staticFile("logo.jpg")}
    style={{
      width: size,
      height: size,
      borderRadius: size * 0.15,
      objectFit: "cover",
    }}
  />
);

// ─── Terminal Card (wider for Twitter) ─────────────────
const TerminalCard: React.FC<{
  children: React.ReactNode;
  title: string;
  agent: string;
  color: string;
  opacity?: number;
  translateY?: number;
}> = ({ children, title, agent, color, opacity = 1, translateY = 0 }) => (
  <div
    style={{
      background: BG_CARD,
      border: `2px solid ${BORDER}`,
      borderRadius: 20,
      padding: "0",
      width: 1400,
      overflow: "hidden",
      opacity,
      transform: `translateY(${translateY}px)`,
      boxShadow: `0 12px 48px rgba(0,0,0,0.18), 0 0 0 1px ${BORDER}`,
    }}
  >
    {/* Title bar */}
    <div
      style={{
        display: "flex",
        alignItems: "center",
        padding: "16px 24px",
        borderBottom: `2px solid ${BORDER}`,
        gap: 12,
      }}
    >
      <div style={{ display: "flex", gap: 10 }}>
        <div style={{ width: 18, height: 18, borderRadius: "50%", background: "#ff5f57" }} />
        <div style={{ width: 18, height: 18, borderRadius: "50%", background: "#febc2e" }} />
        <div style={{ width: 18, height: 18, borderRadius: "50%", background: "#28c840" }} />
      </div>
      <span style={{ color: TEXT_DIM, fontSize: 26, fontFamily: "monospace", marginLeft: 12 }}>
        {title}
      </span>
      <span style={{ marginLeft: "auto", color, fontSize: 26, fontFamily: "monospace", fontWeight: 600 }}>
        {agent}
      </span>
    </div>
    {/* Content */}
    <div
      style={{
        padding: "28px 36px",
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 28,
        lineHeight: 1.7,
        color: TEXT,
        minHeight: 300,
      }}
    >
      {children}
    </div>
  </div>
);

// ─── Tool Call Line ────────────────────────────────────
const ToolCall: React.FC<{ tool: string; args: string; visible: boolean }> = ({
  tool,
  args,
  visible,
}) =>
  visible ? (
    <div style={{ marginBottom: 8, fontSize: 26 }}>
      <span style={{ color: PURPLE }}>tool </span>
      <span style={{ color: YELLOW, fontWeight: 600 }}>{tool}</span>
      <span style={{ color: TEXT_DIM }}>({args})</span>
    </div>
  ) : null;

const ToolResult: React.FC<{
  content: string;
  source: string;
  similarity: string;
  visible: boolean;
}> = ({ content, source, similarity, visible }) =>
  visible ? (
    <div
      style={{
        background: "#1c2128",
        borderLeft: `4px solid ${GREEN}`,
        padding: "14px 20px",
        margin: "8px 0 20px 0",
        borderRadius: "0 12px 12px 0",
      }}
    >
      <div style={{ color: GREEN, fontSize: 26 }}>✓ found</div>
      <div style={{ color: TEXT, fontWeight: 600, marginTop: 6, fontSize: 28 }}>
        "{content}"
      </div>
      <div style={{ color: TEXT_DIM, fontSize: 24, marginTop: 4 }}>
        similarity: {similarity} │ source:{" "}
        <span style={{ color: YELLOW }}>{source}</span>
      </div>
    </div>
  ) : null;

// ─── Memory Stored Line ────────────────────────────────
const MemoryStored: React.FC<{
  content: string;
  tags: string[];
  visible: boolean;
}> = ({ content, tags, visible }) =>
  visible ? (
    <div
      style={{
        background: "#1c2128",
        borderLeft: `4px solid ${CYAN}`,
        padding: "14px 20px",
        margin: "8px 0 14px 0",
        borderRadius: "0 12px 12px 0",
      }}
    >
      <div style={{ fontSize: 28 }}>
        <span style={{ color: CYAN }}>✓ </span>
        <span style={{ color: TEXT }}>"{content}"</span>
      </div>
      <div style={{ color: TEXT_DIM, fontSize: 24, marginTop: 4 }}>
        tags: {tags.map((t) => `#${t}`).join(" ")}
      </div>
    </div>
  ) : null;

// ─── Scenes (frame ranges) ────────────────────────────
// 30 fps → 30 frames = 1 second
const SCENES = {
  titleIn: 0,
  titleHold: 69,
  titleOut: 121,
  // Problem — sequential reveals
  problemLine1: 138,       // "Claude doesn't know what you told Codex"
  problemLine2: 213,       // +2s (60f): "Codex doesn't know what you told Claude"
  problemUntilNow: 318,    // +3s (90f): "Until now."
  problemOut: 378,          // +2s (60f): fade out
  // Agent A
  agentALabel: 411,
  agentAIn: 445,
  agentAPrompt: 480,
  agentAMem1: 526,
  agentAMem2: 572,
  agentAMem3: 618,
  agentADone: 652,
  agentAOut: 675,
  // Transition
  divider: 687,
  // Agent B
  agentBLabel: 721,
  agentBIn: 756,
  agentBPrompt: 790,
  agentBTool1: 825,
  agentBResult1: 848,
  agentBTool2: 894,
  agentBResult2: 917,
  agentBSummary: 963,
  agentBOut: 993,
  // Blank gap (screen goes white)
  blankStart: 993,
  // Finale — sequential reveals
  finaleLogoIn: 1035,      // Logo + "One memory server."
  finaleAgents: 1065,      // +1s: "Any number of agents."
  finalePip: 1125,         // +2s: "pip install tribalmemory"
  finaleGithub: 1155,      // +1s: GitHub link
  finaleEnd: 1230,         // hold ~2.5s
};

// ─── Main Component ────────────────────────────────────
export const TribalMemoryDemo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(
    frame,
    [SCENES.titleIn, SCENES.titleIn + 20, SCENES.titleOut, SCENES.titleOut + 15],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const titleScale = interpolate(
    spring({ frame, fps, config: { damping: 12, mass: 0.8 } }),
    [0, 1],
    [0.9, 1]
  );

  // Problem scene — each line fades in independently
  const problemLine1Op = fadeIn(frame, SCENES.problemLine1);
  const problemLine2Op = fadeIn(frame, SCENES.problemLine2);
  const problemUntilNowOp = fadeIn(frame, SCENES.problemUntilNow, 20);
  const problemFadeOut = interpolate(
    frame,
    [SCENES.problemOut, SCENES.problemOut + 15],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const agentALabelOp = fadeIn(frame, SCENES.agentALabel);
  const agentAOp = fadeIn(frame, SCENES.agentAIn);
  const agentAY = slideUp(frame, SCENES.agentAIn, fps);
  const agentAOutOp = interpolate(
    frame,
    [SCENES.agentAOut, SCENES.agentAOut + 15],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const agentBLabelOp = fadeIn(frame, SCENES.agentBLabel);
  const agentBOp = fadeIn(frame, SCENES.agentBIn);
  const agentBY = slideUp(frame, SCENES.agentBIn, fps);
  const agentBOutOp = interpolate(
    frame,
    [SCENES.agentBOut, SCENES.agentBOut + 15],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Finale — each element fades in sequentially
  const finaleLogoOp = fadeIn(frame, SCENES.finaleLogoIn, 20);
  const finaleOneMemOp = fadeIn(frame, SCENES.finaleLogoIn, 20);
  const finaleAgentsOp = fadeIn(frame, SCENES.finaleAgents);
  const finalePipOp = fadeIn(frame, SCENES.finalePip);
  const finaleGithubOp = fadeIn(frame, SCENES.finaleGithub);
  const finaleScale = interpolate(
    spring({ frame: Math.max(0, frame - SCENES.finaleLogoIn), fps, config: { damping: 12 } }),
    [0, 1],
    [0.95, 1]
  );

  return (
    <AbsoluteFill
      style={{
        background: BG,
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0, left: 0, right: 0, bottom: 0,
          background: "radial-gradient(ellipse at 50% 0%, rgba(30,102,245,0.04) 0%, transparent 60%)",
        }}
      />

      {/* ─── Title Scene ─── */}
      {frame < SCENES.problemLine1 + 30 && (
        <AbsoluteFill
          style={{ justifyContent: "center", alignItems: "center", opacity: titleOpacity, transform: `scale(${titleScale})` }}
        >
          <div style={{ textAlign: "center" }}>
            <Logo size={180} />
            <h1 style={{ fontSize: 120, fontWeight: 800, color: TEXT_BODY, margin: "20px 0 0 0", letterSpacing: -2 }}>
              Tribal Memory
            </h1>
            <p style={{ fontSize: 52, color: TEXT_BODY_DIM, marginTop: 16, fontWeight: 400 }}>
              One brain, many agents.
            </p>
            <div style={{ marginTop: 40, padding: "14px 32px", background: BG_CARD, borderRadius: 12, display: "inline-block" }}>
              <code style={{ color: "#a6e3a1", fontSize: 38 }}>pip install tribalmemory</code>
            </div>
          </div>
        </AbsoluteFill>
      )}

      {/* ─── Problem Statement (sequential reveals) ─── */}
      {frame >= SCENES.problemLine1 && frame < SCENES.problemOut + 30 && (
        <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", opacity: problemFadeOut }}>
          <div style={{ textAlign: "center", maxWidth: 1600, padding: 60 }}>
            <p style={{
              fontSize: 64,
              color: TEXT_BODY,
              lineHeight: 1.5,
              fontWeight: 500,
              opacity: problemLine1Op,
            }}>
              Claude doesn't know what you told Codex.
            </p>
            <p style={{
              fontSize: 64,
              color: TEXT_BODY_DIM,
              lineHeight: 1.5,
              fontWeight: 500,
              marginTop: 16,
              opacity: problemLine2Op,
            }}>
              Codex doesn't know what you told Claude.
            </p>
            <p style={{
              fontSize: 80,
              color: BLUE,
              marginTop: 48,
              fontWeight: 700,
              opacity: problemUntilNowOp,
            }}>
              Until now.
            </p>
          </div>
        </AbsoluteFill>
      )}

      {/* ─── Agent A Section ─── */}
      {frame >= SCENES.agentALabel && frame < SCENES.divider + 30 && (
        <AbsoluteFill
          style={{ justifyContent: "center", alignItems: "center", opacity: Math.min(agentALabelOp, agentAOutOp) }}
        >
          <div style={{ position: "absolute", top: 50, textAlign: "center", opacity: agentALabelOp }}>
            <span style={{ fontSize: 36, color: GREEN, fontWeight: 600, textTransform: "uppercase", letterSpacing: 4 }}>
              Step 1 — Store
            </span>
          </div>
          <div style={{ opacity: agentAOp, transform: `translateY(${agentAY}px)` }}>
            <TerminalCard title="~/my-project" agent="Claude Code" color={GREEN}>
              <div style={{ color: CYAN, fontSize: 28 }}>user</div>
              <div style={{ marginBottom: 16, fontSize: 28 }}>
                {typewriter("Remember these architecture decisions:", Math.max(0, frame - SCENES.agentAPrompt), 1.2)}
              </div>
              <MemoryStored content="Auth uses JWT with RS256 signing" tags={["auth", "jwt"]} visible={frame >= SCENES.agentAMem1} />
              <MemoryStored content="Postgres 16 with pgvector" tags={["database", "postgres"]} visible={frame >= SCENES.agentAMem2} />
              <MemoryStored content="Next.js 15 with App Router" tags={["frontend", "nextjs"]} visible={frame >= SCENES.agentAMem3} />
              {frame >= SCENES.agentADone && (
                <div style={{ color: GREEN, marginTop: 12, fontSize: 28 }}>
                  All 3 stored. Source: <span style={{ color: YELLOW }}>claude-code</span>
                </div>
              )}
            </TerminalCard>
          </div>
        </AbsoluteFill>
      )}

      {/* ─── Transition Arrow ─── */}
      {frame >= SCENES.divider && frame < SCENES.agentBIn + 15 && (
        <AbsoluteFill
          style={{
            justifyContent: "center",
            alignItems: "center",
            opacity: interpolate(
              frame,
              [SCENES.divider, SCENES.divider + 10, SCENES.agentBIn, SCENES.agentBIn + 15],
              [0, 1, 1, 0],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 80, color: BLUE }}>↓</div>
            <div style={{ fontSize: 40, color: TEXT_BODY_DIM, marginTop: 12, fontStyle: "italic" }}>
              Same server, different agent
            </div>
          </div>
        </AbsoluteFill>
      )}

      {/* ─── Agent B Section ─── */}
      {frame >= SCENES.agentBLabel && frame < SCENES.agentBOut + 15 && (
        <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", opacity: Math.min(agentBLabelOp, agentBOutOp) }}>
          <div style={{ position: "absolute", top: 50, textAlign: "center", opacity: agentBLabelOp }}>
            <span style={{ fontSize: 36, color: BLUE, fontWeight: 600, textTransform: "uppercase", letterSpacing: 4 }}>
              Step 2 — Recall
            </span>
          </div>
          <div style={{ opacity: agentBOp, transform: `translateY(${agentBY}px)` }}>
            <TerminalCard title="~/my-project" agent="Codex" color={BLUE}>
              <div style={{ color: CYAN, fontSize: 28 }}>user</div>
              <div style={{ marginBottom: 16, fontSize: 28 }}>
                {typewriter("How does auth work? What database?", Math.max(0, frame - SCENES.agentBPrompt), 1.2)}
              </div>
              <ToolCall tool="tribal_recall" args='"auth"' visible={frame >= SCENES.agentBTool1} />
              <ToolResult content="Auth uses JWT with RS256 signing" source="mcp-claude-code" similarity="0.50" visible={frame >= SCENES.agentBResult1} />
              <ToolCall tool="tribal_recall" args='"database"' visible={frame >= SCENES.agentBTool2} />
              <ToolResult content="Postgres 16 with pgvector" source="mcp-claude-code" similarity="0.39" visible={frame >= SCENES.agentBResult2} />
              {frame >= SCENES.agentBSummary && (
                <div style={{ marginTop: 12, fontSize: 28 }}>
                  <span style={{ color: BLUE }}>codex </span>
                  <span style={{ color: TEXT }}>
                    Found memories from <span style={{ color: YELLOW, fontWeight: 600 }}>Claude Code</span>. 🧠
                  </span>
                </div>
              )}
            </TerminalCard>
          </div>
        </AbsoluteFill>
      )}

      {/* ─── Finale (sequential reveals) ─── */}
      {frame >= SCENES.finaleLogoIn && (
        <AbsoluteFill
          style={{ justifyContent: "center", alignItems: "center", transform: `scale(${finaleScale})` }}
        >
          <div style={{ textAlign: "center" }}>
            <div style={{ opacity: finaleLogoOp }}>
              <Logo size={160} />
            </div>
            <h2 style={{ fontSize: 88, fontWeight: 800, color: TEXT_BODY, margin: "24px 0 0 0", opacity: finaleOneMemOp }}>
              One memory server.
            </h2>
            <h2 style={{ fontSize: 88, fontWeight: 800, color: BLUE, margin: "8px 0 0 0", opacity: finaleAgentsOp }}>
              Any number of agents.
            </h2>
            <div style={{ marginTop: 48, display: "flex", gap: 24, justifyContent: "center", opacity: finalePipOp }}>
              <div style={{ padding: "16px 36px", background: BG_CARD, borderRadius: 12, border: `2px solid ${BORDER}` }}>
                <code style={{ color: "#a6e3a1", fontSize: 36 }}>pip install tribalmemory</code>
              </div>
            </div>
            <p style={{ color: TEXT_BODY_DIM, fontSize: 32, marginTop: 28, opacity: finaleGithubOp }}>
              github.com/abbudjoe/TribalMemory
            </p>
          </div>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};
