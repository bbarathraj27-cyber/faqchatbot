import { useState, useEffect, useRef } from "react";

const getAPIBaseURL = () => {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  if (hostname === "localhost" || hostname === "127.0.0.1") return "http://localhost:8000";
  if (hostname.includes("app.github.dev")) return `${protocol}//${hostname.replace(/-5173\./, "-8000.")}`;
  const origin = window.location.origin;
  if (origin.includes(":5173")) return origin.replace(":5173", ":8000");
  return "http://localhost:8000";
};
const API_BASE_URL = getAPIBaseURL();

function Admin() {
  const [urls, setUrls] = useState("");
  const [questionSelector, setQuestionSelector] = useState("");
  const [answerSelector, setAnswerSelector] = useState("");
  const [range, setRange] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [sources, setSources] = useState([]);
  const [sourcesLoading, setSourcesLoading] = useState(true);
  const [deletingId, setDeletingId] = useState(null);
  const [deletingAll, setDeletingAll] = useState(false);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);

  const [dbStats, setDbStats] = useState(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [analyzeError, setAnalyzeError] = useState(null);
  const [deepAnalyzing, setDeepAnalyzing] = useState(false);
  const [deepResult, setDeepResult] = useState(null);
  const [deepError, setDeepError] = useState(null);
  const [topicsModalPage, setTopicsModalPage] = useState(null);

  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const pollRef = useRef(null);

  // CHANGE 1: Added state
  const [verificationStatus, setVerificationStatus] = useState(null);
  const [pausing, setPausing] = useState(false);
  const [runningNow, setRunningNow] = useState(false);
  const verificationPollRef = useRef(null);

  const fetchSources = async () => {
    setSourcesLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/sources/`);
      const data = await response.json();
      setSources(data);
    } catch (err) {
      console.error("Failed to load sources:", err);
    } finally {
      setSourcesLoading(false);
    }
  };

  const fetchDbStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/sources/stats`);
      const data = await response.json();
      setDbStats(data);
    } catch (err) {
      console.error("Failed to load db stats:", err);
    }
  };

  // CHANGE 2: Added fetch/trigger functions
  const fetchVerificationStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/admin/verification-status`);
      if (!response.ok) return;
      const data = await response.json();
      setVerificationStatus((prev) => {
        if (data.running_pass || (prev && prev.running_pass && !data.running_pass)) {
          fetchSources();
          fetchDbStats();
        }
        return data;
      });
    } catch (err) {
      console.error("Failed to load verification status:", err);
    }
  };

  const handleRunVerificationNow = async () => {
    setRunningNow(true);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/verification-run-now`, { method: "POST" });
      if (response.ok) setVerificationStatus(await response.json());
    } catch (err) {
      console.error("Failed to trigger verification run:", err);
    } finally {
      setRunningNow(false);
    }
  };

  // CHANGE 2b: Pause/Resume handlers for the self-healing agent
  const handlePauseVerification = async () => {
    setPausing(true);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/verification-pause`, { method: "POST" });
      if (response.ok) setVerificationStatus(await response.json());
    } catch (err) {
      console.error("Failed to pause verification agent:", err);
    } finally {
      setPausing(false);
    }
  };

  const handleResumeVerification = async () => {
    setPausing(true);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/verification-resume`, { method: "POST" });
      if (response.ok) setVerificationStatus(await response.json());
    } catch (err) {
      console.error("Failed to resume verification agent:", err);
    } finally {
      setPausing(false);
    }
  };

  // CHANGE 3: Replaced existing useEffect
  useEffect(() => {
    fetchSources();
    fetchDbStats();
    fetchVerificationStatus();
    verificationPollRef.current = setInterval(() => {
      fetchVerificationStatus();
    }, 10000);
    return () => {
      clearInterval(pollRef.current);
      clearInterval(verificationPollRef.current);
    };
  }, []);

  const handleDelete = async (sourceId) => {
    setDeletingId(sourceId);
    try {
      const response = await fetch(`${API_BASE_URL}/sources/${sourceId}`, { method: "DELETE" });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      await fetchSources();
      await fetchDbStats();
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteAll = async () => {
    if (!confirmDeleteAll) {
      setConfirmDeleteAll(true);
      return;
    }
    setDeletingAll(true);
    try {
      const response = await fetch(`${API_BASE_URL}/sources/`, { method: "DELETE" });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      await fetchSources();
      await fetchDbStats();
      setConfirmDeleteAll(false);
    } catch (err) {
      console.error("Delete all failed:", err);
    } finally {
      setDeletingAll(false);
    }
  };

  const firstUrl = () => urls.split("\n").map((u) => u.trim()).filter(Boolean)[0] || "";

  const pollJobStatus = (id) => {
    pollRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/admin/scrape-status/${id}`);
        if (!response.ok) return;
        const data = await response.json();
        setJobStatus(data);
        if (data.status === "completed") {
          clearInterval(pollRef.current);
          fetchSources();
          fetchDbStats();
        }
      } catch (err) {
        console.error("Polling failed:", err);
      }
    }, 2000);
  };

  const handleScrapeIndex = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setJobId(null);
    setJobStatus(null);

    const rangeNum = parseInt(range, 10);

    if (rangeNum && rangeNum > 0) {
      const site = firstUrl();
      if (!site) return;

      setLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/admin/scrape-range-async`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: site,
            limit: rangeNum,
            question_selector: questionSelector || null,
            answer_selector: answerSelector || null,
          }),
        });
        if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
        const data = await response.json();
        setJobId(data.job_id);
        setJobStatus({ status: "running", total_pages: data.total_pages, pages_done: 0, chunks_created: 0, total_words: 0 });
        pollJobStatus(data.job_id);
      } catch (err) {
        setError("Range scrape failed. Check the URL and try again.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    } else {
      const urlList = urls.split("\n").map((u) => u.trim()).filter(Boolean);
      if (urlList.length === 0) return;

      setLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/admin/scrape`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            urls: urlList,
            question_selector: questionSelector || null,
            answer_selector: answerSelector || null,
          }),
        });
        if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
        const data = await response.json();
        setResult(data);
        fetchSources();
        fetchDbStats();
      } catch (err) {
        setError("Scrape failed. Check the URLs and selectors, then try again.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleAnalyze = async () => {
    const site = firstUrl();
    if (!site) return;

    setAnalyzing(true);
    setAnalyzeError(null);
    setAnalyzeResult(null);
    setDeepResult(null);
    setDeepError(null);
    setJobId(null);
    setJobStatus(null);

    try {
      const response = await fetch(`${API_BASE_URL}/admin/analyze-site`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: site }),
      });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      const data = await response.json();
      setAnalyzeResult(data);
    } catch (err) {
      setAnalyzeError("Couldn't analyze that site. Check the URL and try again.");
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleDeepAnalyze = async () => {
    const site = firstUrl();
    if (!site) return;
    setDeepAnalyzing(true);
    setDeepError(null);
    setDeepResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/admin/analyze-site-deep`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: site }),
      });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      const data = await response.json();
      setDeepResult(data);
    } catch (err) {
      setDeepError("Deep analysis failed. Try again.");
      console.error(err);
    } finally {
      setDeepAnalyzing(false);
    }
  };

  const handleStoreAll = async () => {
    if (!analyzeResult?.all_urls) return;
    try {
      const response = await fetch(`${API_BASE_URL}/admin/scrape-bulk-async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: analyzeResult.all_urls }),
      });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      const data = await response.json();
      setJobId(data.job_id);
      setJobStatus({ status: "running", total_pages: data.total_pages, pages_done: 0, chunks_created: 0, total_words: 0 });
      pollJobStatus(data.job_id);
    } catch (err) {
      console.error("Failed to start bulk scrape:", err);
    }
  };

  const handleResetAnalysis = () => {
    setAnalyzeResult(null);
    setDeepResult(null);
    setJobId(null);
    setJobStatus(null);
    clearInterval(pollRef.current);
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-10 flex flex-col gap-8">
      <div>
        <h1 className="font-display text-2xl font-bold mb-1">Admin</h1>
        <p className="text-muted mb-6 text-sm">Scrape pages, analyze a full site, or manage what's already stored.</p>
      </div>

      <div className="bg-surface border border-border rounded-2xl p-5 flex flex-col gap-4">
        <div>
          <label className="text-sm font-medium text-[#dfe4e8] block mb-1.5">
            Website / FAQ URL(s) — one per line
          </label>
          <textarea
            value={urls}
            onChange={(e) => setUrls(e.target.value)}
            placeholder="https://www.tensorflow.org"
            rows={3}
            className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm outline-none focus:border-accent placeholder:text-[#5c6672]"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-[#dfe4e8] block mb-1.5">Question CSS selector (optional)</label>
            <input
              type="text"
              value={questionSelector}
              onChange={(e) => setQuestionSelector(e.target.value)}
              placeholder="h3.faq-question"
              className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm outline-none focus:border-accent placeholder:text-[#5c6672]"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-[#dfe4e8] block mb-1.5">Answer CSS selector (optional)</label>
            <input
              type="text"
              value={answerSelector}
              onChange={(e) => setAnswerSelector(e.target.value)}
              placeholder="div.faq-answer"
              className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm outline-none focus:border-accent placeholder:text-[#5c6672]"
            />
          </div>
        </div>

        <div>
          <label className="text-sm font-medium text-[#dfe4e8] block mb-1.5">
            Range — number of pages (optional)
          </label>
          <input
            type="number"
            min="1"
            value={range}
            onChange={(e) => setRange(e.target.value)}
            placeholder="e.g. 5 — leave blank to scrape only the pasted URL(s) above"
            className="w-full max-w-[220px] bg-surface2 border border-border rounded-lg px-3 py-2 text-sm outline-none focus:border-accent placeholder:text-[#5c6672]"
          />
          <p className="text-[11px] text-muted mt-1">
            If set, scrapes this many pages from the site's sitemap. If empty, scrapes exactly the URL(s) above.
          </p>
        </div>

        <div className="flex gap-3 flex-wrap">
          <button
            onClick={handleScrapeIndex}
            disabled={loading || !urls.trim()}
            className="bg-accent text-[#0a0e12] font-semibold px-5 py-2 rounded-lg disabled:opacity-40 hover:-translate-y-0.5 transition text-sm"
          >
            {loading ? "Scraping..." : "Scrape & Index"}
          </button>

          <button
            onClick={handleAnalyze}
            disabled={analyzing || !urls.trim()}
            className="bg-edge/90 text-[#0a0e12] font-semibold px-5 py-2 rounded-lg disabled:opacity-40 hover:-translate-y-0.5 transition text-sm"
          >
            {analyzing ? "Counting..." : "Analyze & Bulk Import"}
          </button>
        </div>

        {error && <p className="text-red-400 text-sm">{error}</p>}

        {result && (
          <div className="bg-surface2 border border-border rounded-lg p-4">
            <p className="text-emerald-400 text-sm font-semibold mb-3">
              ✅ Scraped {result.pages_scraped} page(s), created {result.chunks_created} chunk(s).
            </p>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">Indexing pages</p>
                <p className="text-base font-semibold text-[#eef2f5]">{result.pages_scraped}</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">Chunking words</p>
                <p className="text-base font-semibold text-accent2">{result.total_words?.toLocaleString() || 0}</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">DB space used (total)</p>
                <p className="text-base font-semibold text-edge">{dbStats?.estimated_storage_mb ?? "—"} MB</p>
              </div>
            </div>
          </div>
        )}

        {analyzeError && <p className="text-red-400 text-sm">{analyzeError}</p>}

        {analyzeResult && analyzeResult.sitemap_found && (
          <div className="bg-surface2 border border-border rounded-lg p-4 flex flex-col gap-3">
            <p className="text-emerald-400 text-sm font-semibold">
              ✅ {analyzeResult.total_pages} indexing page(s) found in {analyzeResult.domain}'s sitemap
            </p>

            {!deepResult && (
              <button
                onClick={handleDeepAnalyze}
                disabled={deepAnalyzing}
                className="text-xs bg-accent/90 text-[#0a0e12] font-semibold px-3 py-1.5 rounded-lg disabled:opacity-40 hover:-translate-y-0.5 transition self-start"
              >
                {deepAnalyzing ? "Analyzing content…" : "Estimate word / chunk / storage size →"}
              </button>
            )}
            {deepAnalyzing && <p className="text-muted text-xs italic">Sampling pages — this takes under a minute…</p>}
            {deepError && <p className="text-red-400 text-sm">{deepError}</p>}

            {deepResult && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">Indexing pages</p>
                    <p className="text-lg font-semibold text-[#eef2f5]">{deepResult.total_pages_in_sitemap}</p>
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">Est. total words</p>
                    <p className="text-lg font-semibold text-[#eef2f5]">{deepResult.total_word_count.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">Estimated chunks</p>
                    <p className="text-lg font-semibold text-accent2">{deepResult.estimated_total_chunks.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">Estimated storage</p>
                    <p className="text-lg font-semibold text-edge">{deepResult.estimated_storage_mb} MB</p>
                  </div>
                </div>

                {deepResult.page_previews?.length > 0 && (
                  <div>
                    <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-2">
                      Sample page contents ({deepResult.page_previews.length} pages)
                    </p>
                    <div className="bg-surface border border-border rounded-lg overflow-hidden">
                      <table className="w-full text-sm">
                        <thead className="bg-surface2 text-muted">
                          <tr>
                            <th className="text-left px-3 py-2 font-medium text-xs">URL</th>
                            <th className="text-left px-3 py-2 font-medium text-xs">Topics</th>
                          </tr>
                        </thead>
                        <tbody>
                          {deepResult.page_previews.map((page) => (
                            <tr key={page.url} className="border-t border-border">
                              <td className="px-3 py-2 font-mono text-xs text-[#dfe4e8] break-all max-w-xs">{page.url}</td>
                              <td className="px-3 py-2">
                                <div className="flex flex-wrap gap-1.5 items-center">
                                  {page.topics.slice(0, 3).map((t) => (
                                    <span key={t} className="text-[11px] bg-surface2 text-edge px-2 py-0.5 rounded-full border border-edge/20">
                                      {t}
                                    </span>
                                  ))}
                                  {page.topics.length > 3 && (
                                    <button
                                      onClick={() => setTopicsModalPage(page)}
                                      className="text-[11px] text-accent2 hover:underline"
                                    >
                                      +{page.topics.length - 3} more →
                                    </button>
                                  )}
                                  {page.topics.length === 0 && (
                                    <span className="text-[11px] text-muted italic">no topics found</span>
                                  )}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {!jobId && (
                  <div className="flex gap-2 pt-2">
                    <button
                      onClick={handleResetAnalysis}
                      className="text-sm px-4 py-2 rounded-lg border border-border text-muted hover:text-[#dfe4e8] hover:border-borderStrong transition"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleStoreAll}
                      className="text-sm px-4 py-2 rounded-lg bg-accent text-[#0a0e12] font-semibold hover:-translate-y-0.5 transition"
                    >
                      Scrape & Store All {analyzeResult.total_pages} Pages
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {jobStatus && (
          <div className="bg-surface2 border border-edge/30 rounded-lg p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold text-[#eef2f5]">
                {jobStatus.status === "completed" ? "✅ Import complete" : "⏳ Importing in background…"}
              </p>
              <p className="text-xs text-muted">{jobStatus.pages_done} / {jobStatus.total_pages} pages</p>
            </div>
            <div className="w-full h-2 bg-surface rounded-full overflow-hidden">
              <div
                className="h-full bg-edge transition-all duration-500"
                style={{ width: `${(jobStatus.pages_done / jobStatus.total_pages) * 100}%` }}
              />
            </div>

            <div className="grid grid-cols-3 gap-4 mt-2 pt-2 border-t border-border">
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">Pages done</p>
                <p className="text-base font-semibold text-[#eef2f5]">{jobStatus.pages_done} / {jobStatus.total_pages}</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">Chunking words</p>
                <p className="text-base font-semibold text-accent2">{(jobStatus.total_words || 0).toLocaleString()}</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-1">DB space used (total)</p>
                <p className="text-base font-semibold text-edge">{dbStats?.estimated_storage_mb ?? "—"} MB</p>
              </div>
            </div>

            {jobStatus.current_url && jobStatus.status === "running" && (
              <p className="text-xs text-muted">currently on: <span className="font-mono">{jobStatus.current_url}</span></p>
            )}
            {jobStatus.status === "completed" && (
              <button onClick={handleResetAnalysis} className="text-xs text-edge hover:underline self-start mt-1">
                Start a new analysis
              </button>
            )}
          </div>
        )}
      </div>

      {/* CHANGE 4: Added verification status card, now with Pause/Resume */}
      {verificationStatus && (
        <div className="mb-6 rounded-xl border border-edge/40 bg-panel/40 p-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <h3 className="font-display text-sm font-semibold flex items-center gap-2">
                🔁 Self-healing verification agent
                {verificationStatus.paused && (
                  <span className="text-[10px] font-semibold uppercase tracking-wide px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30">
                    Paused
                  </span>
                )}
              </h3>
              <p className="text-xs text-muted mt-0.5">
                {verificationStatus.paused
                  ? `Paused` +
                    (verificationStatus.current_url ? ` — was checking ${verificationStatus.current_url}` : "")
                  : verificationStatus.running_pass
                  ? `Checking source ${verificationStatus.sources_checked}/${verificationStatus.sources_total}` +
                    (verificationStatus.current_url ? ` — ${verificationStatus.current_url}` : "")
                  : verificationStatus.last_pass_summary
                  ? verificationStatus.last_pass_summary
                  : "Waiting for its first pass to start."}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRunVerificationNow}
                disabled={runningNow || verificationStatus.running_pass || verificationStatus.paused}
                className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-edge/60 hover:border-accent2 transition disabled:opacity-40"
              >
                {verificationStatus.running_pass ? "Running…" : "Run now"}
              </button>
              {verificationStatus.paused ? (
                <button
                  onClick={handleResumeVerification}
                  disabled={pausing}
                  className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-emerald-500/90 text-[#0a0e12] hover:-translate-y-0.5 transition disabled:opacity-40"
                >
                  {pausing ? "Resuming…" : "Resume"}
                </button>
              ) : (
                <button
                  onClick={handlePauseVerification}
                  disabled={pausing || !verificationStatus.running_pass}
                  className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-amber-500/50 text-amber-400 hover:bg-amber-500/10 transition disabled:opacity-40"
                >
                  {pausing ? "Pausing…" : "Pause"}
                </button>
              )}
            </div>
          </div>

          {verificationStatus.running_pass && !verificationStatus.paused && verificationStatus.sources_total > 0 && (
            <div className="mt-3 h-1.5 rounded-full bg-edge/20 overflow-hidden">
              <div
                className="h-full bg-accent2 transition-all"
                style={{
                  width: `${(verificationStatus.sources_checked / verificationStatus.sources_total) * 100}%`,
                }}
              />
            </div>
          )}

          <div className="flex flex-wrap gap-x-6 gap-y-1 mt-3 text-xs text-muted">
            <span>Corrected this pass: <span className="text-edge font-semibold">{verificationStatus.corrected_this_pass}</span></span>
            <span>Errors this pass: <span className="text-edge font-semibold">{verificationStatus.errors_this_pass}</span></span>
            {verificationStatus.next_pass_at && (
              <span>Next pass: <span className="text-edge font-semibold">{new Date(verificationStatus.next_pass_at).toLocaleString()}</span></span>
            )}
          </div>
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h2 className="font-display text-lg font-semibold">Ingested Sources</h2>
          <div className="flex items-center gap-3">
            {dbStats && (
              <p className="text-xs text-muted">
                {dbStats.total_sources} source(s) · {dbStats.total_vectors} vector(s) · {dbStats.estimated_storage_mb} MB used
              </p>
            )}
            {sources.length > 0 && (
              <button
                onClick={handleDeleteAll}
                disabled={deletingAll}
                onBlur={() => setConfirmDeleteAll(false)}
                className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition disabled:opacity-40 ${
                  confirmDeleteAll
                    ? "bg-red-500 text-white border-red-500 hover:bg-red-600"
                    : "text-red-400 border-red-500/30 hover:bg-red-500/10"
                }`}
              >
                {deletingAll ? "Deleting all…" : confirmDeleteAll ? "Click again to confirm — permanent" : "Delete All"}
              </button>
            )}
          </div>
        </div>

        {sourcesLoading ? (
          <p className="text-muted text-sm">Loading...</p>
        ) : sources.length === 0 ? (
          <p className="text-muted text-sm">No sources ingested yet.</p>
        ) : (
          <div className="bg-surface border border-border rounded-2xl overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-surface2 text-muted">
                <tr>
                  <th className="text-left px-4 py-2 font-medium">URL</th>
                  <th className="text-left px-4 py-2 font-medium">Status</th>
                  <th className="text-left px-4 py-2 font-medium">Chunks</th>
                  <th className="text-left px-4 py-2 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sources.map((source) => (
                  <tr key={source.id} className="border-t border-border">
                    <td className="px-4 py-2 break-all font-mono text-xs text-[#dfe4e8]">{source.url}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-0.5 rounded text-xs border ${
                        source.status === "success" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                        : source.status === "failed" ? "bg-red-500/10 text-red-400 border-red-500/30"
                        : "bg-white/5 text-muted border-border"
                      }`}>
                        {source.status}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-muted">{source.chunks_created}</td>
                    <td className="px-4 py-2">
                      <button
                        onClick={() => handleDelete(source.id)}
                        disabled={deletingId === source.id}
                        className="text-xs text-red-400 hover:text-red-300 hover:underline disabled:opacity-40"
                      >
                        {deletingId === source.id ? "Deleting…" : "Delete"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {topicsModalPage && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 px-4" onClick={() => setTopicsModalPage(null)}>
          <div className="bg-surface border border-border rounded-2xl p-6 max-w-md w-full" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display font-semibold text-[#eef2f5]">All topics</h3>
              <button onClick={() => setTopicsModalPage(null)} className="text-muted hover:text-[#eef2f5] text-xl leading-none">×</button>
            </div>
            <p className="text-xs font-mono text-muted break-all mb-4">{topicsModalPage.url}</p>
            <div className="flex flex-wrap gap-2">
              {topicsModalPage.topics.map((t) => (
                <span key={t} className="text-xs bg-surface2 text-edge px-2.5 py-1 rounded-full border border-edge/20">{t}</span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Admin;