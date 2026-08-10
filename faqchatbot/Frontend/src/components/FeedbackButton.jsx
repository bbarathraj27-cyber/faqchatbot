import { useState } from "react";

const getAPIBaseURL = () => {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;

  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://localhost:8000";
  }
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  if (hostname.includes("app.github.dev")) {
    return `${protocol}//${hostname.replace(/-5173\./, "-8000.")}`;
  }
  const origin = window.location.origin;
  if (origin.includes(":5173")) {
    return origin.replace(":5173", ":8000");
  }
  return "http://localhost:8000";
};

const API_BASE_URL = getAPIBaseURL();

function FeedbackButton({ chatLogId }) {
  const [submitted, setSubmitted] = useState(null); // null | "up" | "down"

  const sendFeedback = async (isHelpful) => {
    setSubmitted(isHelpful ? "up" : "down");
    try {
      await fetch(`${API_BASE_URL}/question/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_log_id: chatLogId, is_helpful: isHelpful }),
      });
    } catch (err) {
      console.error("Failed to send feedback:", err);
    }
  };

  return (
    <div className="flex items-center gap-1.5">
      <button
        onClick={() => sendFeedback(true)}
        disabled={submitted !== null}
        className={`text-sm px-2 py-1 rounded-md border border-border ${
          submitted === "up" ? "bg-emerald-500/10 border-emerald-500/40" : "hover:border-borderStrong"
        }`}
        aria-label="Mark answer as helpful"
      >
        👍
      </button>
      <button
        onClick={() => sendFeedback(false)}
        disabled={submitted !== null}
        className={`text-sm px-2 py-1 rounded-md border border-border ${
          submitted === "down" ? "bg-red-500/10 border-red-500/40" : "hover:border-borderStrong"
        }`}
        aria-label="Mark answer as not helpful"
      >
        👎
      </button>
    </div>
  );
}

export default FeedbackButton;