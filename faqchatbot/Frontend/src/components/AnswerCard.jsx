import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import SourceList from "./SourceList";
import FeedbackButton from "./FeedbackButton";

// Custom renderers so markdown output (tables, code blocks, bold, lists, etc.)
// matches the app's existing dark theme instead of default browser styling.
const markdownComponents = {
  p: ({ children }) => (
    <p className="text-[15px] leading-relaxed mb-3">{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-[#f2f5f7]">{children}</strong>
  ),
  ul: ({ children }) => (
    <ul className="list-disc pl-5 mb-3 space-y-1 text-[15px]">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal pl-5 mb-3 space-y-1 text-[15px]">{children}</ol>
  ),
  li: ({ children }) => <li className="text-[#dfe4e8]">{children}</li>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-edge hover:underline">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto mb-3">
      <table className="w-full border-collapse text-[14px]">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-surface2">{children}</thead>,
  th: ({ children }) => (
    <th className="border border-border px-3 py-2 text-left text-accent2 font-mono text-[12px] uppercase tracking-wide">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-border px-3 py-2 align-top text-[#dfe4e8]">{children}</td>
  ),
  code: ({ inline, className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className || "");
    const lang = (match?.[1] || "").toLowerCase();

    if (inline) {
      return (
        <code className="bg-surface2 px-1.5 py-0.5 rounded text-[13px] font-mono text-[#dfe4e8]" {...props}>
          {children}
        </code>
      );
    }

    const isPython = lang === "python" || lang === "py";
    return (
      <div className="mb-3">
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
            {String(children).replace(/\n$/, "")}
          </code>
        </pre>
      </div>
    );
  },
};

function AnswerCard({ chatLogId, question, answer, sources, fromCache }) {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className="bg-surface border border-border rounded-2xl px-6 py-5">
      <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1.5">You asked</p>
      <p className="text-[15.5px] text-[#dfe4e8] mb-4 pb-4 border-b border-border">{question}</p>

      <p className="text-[11px] uppercase tracking-wider text-accent2 font-mono mb-1.5">Answer</p>
      <div>
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
          {answer}
        </ReactMarkdown>
      </div>

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