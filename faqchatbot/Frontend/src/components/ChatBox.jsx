import { useState, useEffect } from "react";
import AnswerCard from "./AnswerCard";

// Determine API base URL based on environment
const getAPIBaseURL = () => {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;

  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://localhost:8000";
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

function ChatBox() {
  const [question, setQuestion] = useState("");
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);

  // Pull suggested questions from what's actually in the DB, instead of a
  // hardcoded list that could reference content that was never scraped.
  useEffect(() => {
    fetch(`${API_BASE_URL}/sources/topics`)
      .then((res) => (res.ok ? res.json() : { suggestions: [] }))
      .then((data) => {
        const questions = (data.suggestions || []).map((s) => `What is ${s.label}?`);
        setSuggestions(questions);
      })
      .catch(() => setSuggestions([]));
  }, []);

  const askQuestion = async (raw) => {
    const trimmed = raw.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);

    try {
      const url = `${API_BASE_URL}/question/ask`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Request failed with status ${response.status}: ${errorText}`);
      }

      const data = await response.json();

      setConversation((prev) => [
        {
          id: crypto.randomUUID(),
          chatLogId: data.chat_log_id,
          question: trimmed,
          answer: data.answer,
          sources: data.sources,
          fromCache: data.from_cache,
        },
        ...prev,
      ]);
      setQuestion("");
    } catch (err) {
      console.error("Error details:", err);
      setError("Something went wrong getting an answer. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    askQuestion(question);
  };

  return (
    <div>
      {/* Search bar */}
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2.5 bg-surface border border-borderStrong rounded-2xl pl-5 pr-2 py-2 max-w-[680px] mx-auto shadow-[0_20px_60px_-20px_rgba(0,0,0,0.6)] focus-within:border-accent transition"
      >
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about tf.data, Keras, GPU setup, model deployment…"
          disabled={loading}
          className="flex-1 bg-transparent outline-none text-[15.5px] py-3 placeholder:text-[#5c6672]"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="bg-accent text-[#0a0e12] font-semibold text-sm px-5 py-3 rounded-xl disabled:opacity-40 disabled:hover:translate-y-0 disabled:hover:shadow-none hover:-translate-y-0.5 hover:shadow-[0_8px_22px_rgba(255,122,26,0.35)] transition whitespace-nowrap"
        >
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      {error && <p className="text-red-400 text-sm text-center mt-3">{error}</p>}

      {/* Suggestions - only show before any conversation starts, and only
          once we actually have DB-backed topics to suggest */}
      {conversation.length === 0 && suggestions.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-w-[760px] mx-auto mt-5">
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => askQuestion(s)}
              className="bg-surface2 border border-border text-left text-[13.5px] px-4 py-3.5 rounded-xl hover:border-edge hover:bg-[#182029] hover:-translate-y-0.5 transition"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Conversation */}
      {conversation.length > 0 && (
        <div className="max-w-[760px] mx-auto mt-14 flex flex-col gap-4 text-left">
          {conversation.map((entry) => (
            <AnswerCard
              key={entry.id}
              chatLogId={entry.chatLogId}
              question={entry.question}
              answer={entry.answer}
              sources={entry.sources}
              fromCache={entry.fromCache}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default ChatBox;