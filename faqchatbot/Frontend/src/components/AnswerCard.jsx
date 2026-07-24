import { useState } from "react";
import SourceList from "./SourceList";
import FeedbackButton from "./FeedbackButton";

// Splits the answer on ```lang\ncode\n``` fences and renders each code
// block as a highlighted box (colored by language) instead of raw text
// with literal backticks.
function renderAnswer(answer) {
  const parts = answer.split(/```(\w+)?\n?([\s\S]*?)```/g);
  const nodes = [];

  for (let i = 0; i < parts.length; i += 3) {
    const text = parts[i];
    const lang = (parts[i + 1] || "").toLowerCase();
    const code = parts[i + 2];

    if (text && text.trim()) {
      nodes.push(
        <p key={`t-${i}`} className="text-[15px] leading-relaxed whitespace-pre-line mb-3">
          {text.trim()}
        </p>
      );
    }

    if (code !== undefined) {
      const isPython = lang === "python" || lang === "py";
      nodes.push(
        <div key={`c-${i}`} className="mb-3">
          <span
            className={
              isPython
                ? "inline-block text-[11px] uppercase tracking-wider px-2 py-0.5 rounded mb-1.5 font-mono text-accent2 bg-accent/10"
                : "inline-block text-[11px] uppercase tracking-wider px-2 py-0.5 rounded mb-1.5 font-mono text-edge bg-edge/10"
            }
          >
            {lang || "code"}
          </span>
          <pre
            className={
              isPython
                ? "bg-surface2 border-l-2 border-accent rounded-lg px-4 py-3 overflow-x-auto"
                : "bg-surface2 border-l-2 border-edge rounded-lg px-4 py-3 overflow-x-auto"
            }
          >
            <code className="text-[13px] font-mono text-[#dfe4e8] whitespace-pre">
              {code.trim()}
            </code>
          </pre>
        </div>
      );
    }
  }

  return nodes;
}

function AnswerCard({ chatLogId, question, answer, sources, fromCache }) {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className="bg-surface border border-border rounded-2xl px-6 py-5">
      <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1.5">You asked</p>
      <p className="text-[15.5px] text-[#dfe4e8] mb-4 pb-4 border-b border-border">{question}</p>

      <p className="text-[11px] uppercase tracking-wider text-accent2 font-mono mb-1.5">Answer</p>
      <div>{renderAnswer(answer)}</div>

      <div className="flex items-center justify-between mt-4 pt-3.5 border-t border-border">
        <div className="flex items-center gap-3">
          {sources?.length > 0 && (
            <button
              onClick={() => setShowSources((prev) => !prev)}
              className="text-[13px] text-edge hover:underline"
            >
              {showSources ? "Hide sources" : `View sources (${sources.length})`}
            </button>
          )}
          {fromCache && (
            <span className="text-xs text-muted italic">⚡ cached answer</span>
          )}
        </div>

        <FeedbackButton chatLogId={chatLogId} />
      </div>

      {showSources && <SourceList sources={sources} />}
    </div>
  );
}

export default AnswerCard;