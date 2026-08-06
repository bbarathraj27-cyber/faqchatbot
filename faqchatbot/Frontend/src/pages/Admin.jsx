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
  const [selectedSourceIds, setSelectedSourceIds] = useState(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [confirmBulkDelete, setConfirmBulkDelete] = useState(false);

  const [dbStats, setDbStats] = useState(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [analyzeError, setAnalyzeError] = useState(null);
  const [deepAnalyzing, setDeepAnalyzing] = useState(false);
  const [deepResult, setDeepResult] = useState(null);
  const [deepError, setDeepError] = useState(null);
  const [topicsModalPage, setTopicsModalPage] = useState(null);

  const [selectedLanguages, setSelectedLanguages] = useState(new Set(["en"]));
  const [filteredUrls, setFilteredUrls] = useState(null);
  const [filteringUrls, setFilteringUrls] = useState(false);

  // Range tab: "1-200", "201-400", etc. Also accepts a plain number (legacy
  // shorthand for "1-N"). rangeAnalysis holds the language breakdown + how
  // many of those pages are already ingested, scoped to just that slice.
  const [rangeAnalysis, setRangeAnalysis] = useState(null);
  const [rangeAnalyzing, setRangeAnalyzing] = useState(false);
  const [rangeAnalyzeError, setRangeAnalyzeError] = useState(null);
  const [rangeSelectedLanguages, setRangeSelectedLanguages] = useState(new Set(["en"]));

  // Single-scrape path: pasted URL(s), no range. Language breakdown of
  // exactly the pasted links (URL-pattern based, no sitemap fetch), so a
  // checklist shows up right under the textarea the moment URLs are typed.
  const [pastedLanguages, setPastedLanguages] = useState(null);
  const [pastedAnalyzing, setPastedAnalyzing] = useState(false);
  const [pastedSelectedLanguages, setPastedSelectedLanguages] = useState(new Set(["en"]));

  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const pollRef = useRef(null);
  const [bulkPausing, setBulkPausing] = useState(false);
  const [bulkResuming, setBulkResuming] = useState(false);
  const [bulkCancelling, setBulkCancelling] = useState(false);
  const [confirmCancelJob, setConfirmCancelJob] = useState(false);
  const [resumableJobs, setResumableJobs] = useState([]);

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

  const fetchVerificationStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/admin/verification-status`);
      if (!response.ok) return;
      const data = await response.json();
      setVerificationStatus((prev) => {
        const startedRun = prev && !prev.running_pass && data.running_pass;
        const finishedRun = prev && prev.running_pass && !data.running_pass;
        if (startedRun || finishedRun) {
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

  useEffect(() => {
    fetchSources();
    fetchDbStats();
    fetchVerificationStatus();
    fetchResumableJobs();
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
      setSelectedSourceIds((prev) => {
        if (!prev.has(sourceId)) return prev;
        const next = new Set(prev);
        next.delete(sourceId);
        return next;
      });
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
      setSelectedSourceIds(new Set());
    } catch (err) {
      console.error("Delete all failed:", err);
    } finally {
      setDeletingAll(false);
    }
  };

  const toggleSourceSelected = (id) => {
    setSelectedSourceIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAllSources = () => {
    setSelectedSourceIds((prev) =>
      prev.size === sources.length ? new Set() : new Set(sources.map((s) => s.id))
    );
  };

  const handleBulkDelete = async () => {
    if (selectedSourceIds.size === 0) return;
    if (!confirmBulkDelete) {
      setConfirmBulkDelete(true);
      return;
    }
    setBulkDeleting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/sources/delete-bulk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: Array.from(selectedSourceIds) }),
      });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      setSelectedSourceIds(new Set());
      setConfirmBulkDelete(false);
      await fetchSources();
      await fetchDbStats();
    } catch (err) {
      console.error("Bulk delete failed:", err);
    } finally {
      setBulkDeleting(false);
    }
  };

  const firstUrl = () => urls.split("\n").map((u) => u.trim()).filter(Boolean)[0] || "";

  // Accepts "1-200", "201-400", "1 - 200", or a plain number ("200", treated
  // as legacy shorthand for "1-200"). Returns { start, end } (1-indexed,
  // inclusive) or null if the field doesn't parse yet.
  const parseRange = (value) => {
    const trimmed = (value || "").trim();
    if (!trimmed) return null;

    const pairMatch = trimmed.match(/^(\d+)\s*-\s*(\d+)$/);
    if (pairMatch) {
      const start = parseInt(pairMatch[1], 10);
      const end = parseInt(pairMatch[2], 10);
      if (start > 0 && end >= start) return { start, end };
      return null;
    }

    const singleMatch = trimmed.match(/^(\d+)$/);
    if (singleMatch) {
      const end = parseInt(singleMatch[1], 10);
      if (end > 0) return { start: 1, end };
    }
    return null;
  };

  const pollJobStatus = (id) => {
    // Without this, every call (start / resume / cancel-then-still-running)
    // stacks a NEW interval on top of any still-running one instead of
    // replacing it, since pollRef.current was being overwritten without
    // clearing the old timer first. That leads to duplicate overlapping
    // status fetches and, worse, duplicate calls to fetchSources/fetchDbStats
    // once the job finishes.
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/admin/scrape-status/${id}`);
        if (!response.ok) return;
        const data = await response.json();
        setJobStatus(data);
        if (data.status === "completed" || data.status === "paused" || data.status === "cancelled") {
          clearInterval(pollRef.current);
          fetchSources();
          fetchDbStats();
          if (data.status === "paused") fetchResumableJobs();
        }
      } catch (err) {
        console.error("Polling failed:", err);
      }
    }, 2000);
  };

  const fetchResumableJobs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/admin/scrape-jobs`);
      if (!response.ok) return;
      const data = await response.json();
      setResumableJobs(data);
    } catch (err) {
      console.error("Failed to load resumable jobs:", err);
    }
  };

  const handlePauseJob = async () => {
    if (!jobId) return;
    setBulkPausing(true);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/scrape-pause/${jobId}`, { method: "POST" });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
    } catch (err) {
      console.error("Pause request failed:", err);
    } finally {
      setBulkPausing(false);
    }
  };

  const handleResumeJob = async (id) => {
    const targetId = id || jobId;
    if (!targetId) return;
    setBulkResuming(true);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/scrape-resume/${targetId}`, { method: "POST" });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      const data = await response.json();
      setJobId(targetId);
      setJobStatus(data);
      setResumableJobs((prev) => prev.filter((j) => j.job_id !== targetId));
      pollJobStatus(targetId);
    } catch (err) {
      console.error("Resume request failed:", err);
    } finally {
      setBulkResuming(false);
    }
  };

  const handleCancelJob = async (id) => {
    const targetId = id || jobId;
    if (!targetId) return;
    if (!confirmCancelJob) {
      setConfirmCancelJob(true);
      return;
    }
    setBulkCancelling(true);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/scrape-cancel/${targetId}`, { method: "POST" });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      const data = await response.json();
      if (targetId === jobId) {
        setJobStatus(data);
        if (data.status !== "cancelled") pollJobStatus(targetId);
      }
      setResumableJobs((prev) => prev.filter((j) => j.job_id !== targetId));
      setConfirmCancelJob(false);
    } catch (err) {
      console.error("Cancel request failed:", err);
    } finally {
      setBulkCancelling(false);
    }
  };

  const handleScrapeIndex = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setJobId(null);
    setJobStatus(null);

    const parsedRange = parseRange(range);

    if (parsedRange) {
      const site = firstUrl();
      if (!site) return;

      setLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/admin/scrape-range-async`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: site,
            start: parsedRange.start,
            end: parsedRange.end,
            question_selector: questionSelector || null,
            answer_selector: answerSelector || null,
            // Only send a language filter once we actually know what
            // languages are in this range (rangeAnalysis loaded) - avoids
            // accidentally restricting to the stale default ["en"] before
            // the checklist has had a chance to load/be adjusted.
            languages:
              rangeAnalysis && rangeAnalysis.languages.length > 0
                ? Array.from(rangeSelectedLanguages)
                : null,
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
      let urlList = urls.split("\n").map((u) => u.trim()).filter(Boolean);
      if (urlList.length === 0) return;

      // Apply the pasted-URL language checklist (if it loaded in time and
      // the user unchecked something) before sending the final list off.
      if (pastedLanguages && pastedLanguages.length > 0 && pastedSelectedLanguages.size < pastedLanguages.length) {
        try {
          const filterResponse = await fetch(`${API_BASE_URL}/admin/filter-urls-by-language`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls: urlList, languages: Array.from(pastedSelectedLanguages) }),
          });
          if (filterResponse.ok) {
            const filterData = await filterResponse.json();
            urlList = filterData.urls;
          }
        } catch (err) {
          console.error("Language filter failed, scraping unfiltered list:", err);
        }
        if (urlList.length === 0) {
          setError("No pasted URL(s) match the selected language(s).");
          return;
        }
      }

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
      const codes = (data.languages || []).map((l) => l.code);
      setSelectedLanguages(new Set(codes.includes("en") ? ["en"] : codes));
      setFilteredUrls(null);
    } catch (err) {
      setAnalyzeError("Couldn't analyze that site. Check the URL and try again.");
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };

  useEffect(() => {
    if (!analyzeResult?.all_urls || selectedLanguages.size === 0) {
      if (analyzeResult?.all_urls) setFilteredUrls([]);
      return;
    }
    let cancelled = false;
    setFilteringUrls(true);
    (async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/admin/filter-urls-by-language`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            urls: analyzeResult.all_urls,
            languages: Array.from(selectedLanguages),
          }),
        });
        if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
        const data = await response.json();
        if (!cancelled) setFilteredUrls(data.urls);
      } catch (err) {
        console.error("Failed to filter URLs by language:", err);
        if (!cancelled) setFilteredUrls([]);
      } finally {
        if (!cancelled) setFilteringUrls(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyzeResult, selectedLanguages]);

  const toggleLanguage = (code) => {
    setSelectedLanguages((prev) => {
      const next = new Set(prev);
      if (next.has(code)) next.delete(code);
      else next.add(code);
      return next;
    });
    setDeepResult(null);
  };

  // Debounced: whenever the range field (or the URL) changes, ask the
  // backend what languages appear in that specific page slice, and how
  // many of those pages are already ingested - so entering "201-400" right
  // after "1-200" shows a fresh checklist scoped to just the new batch.
  useEffect(() => {
    const parsed = parseRange(range);
    const site = firstUrl();
    if (!parsed || !site) {
      setRangeAnalysis(null);
      setRangeAnalyzeError(null);
      setRangeAnalyzing(false);
      return;
    }

    let cancelled = false;
    setRangeAnalyzing(true);
    setRangeAnalyzeError(null);

    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/admin/analyze-range`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: site, start: parsed.start, end: parsed.end }),
        });
        if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
        const data = await response.json();
        if (cancelled) return;
        setRangeAnalysis(data);
        const codes = (data.languages || []).map((l) => l.code);
        setRangeSelectedLanguages(new Set(codes.includes("en") ? ["en"] : codes));
      } catch (err) {
        if (cancelled) return;
        console.error("Failed to analyze range:", err);
        setRangeAnalysis(null);
        setRangeAnalyzeError("Couldn't check that range. Check the URL and try again.");
      } finally {
        if (!cancelled) setRangeAnalyzing(false);
      }
    }, 500);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [range, urls]);

  const toggleRangeLanguage = (code) => {
    setRangeSelectedLanguages((prev) => {
      const next = new Set(prev);
      if (next.has(code)) next.delete(code);
      else next.add(code);
      return next;
    });
  };

  // Debounced: whenever the pasted URL list changes AND no range is set
  // (i.e. the "single scrape - exact URL(s)" path), fetch a language
  // breakdown of just those pasted links so a checklist shows up under the
  // textarea without requiring any extra click.
  useEffect(() => {
    const parsedRangeActive = parseRange(range);
    const urlList = urls.split("\n").map((u) => u.trim()).filter(Boolean);

    if (parsedRangeActive || urlList.length === 0) {
      setPastedLanguages(null);
      setPastedAnalyzing(false);
      return;
    }

    let cancelled = false;
    setPastedAnalyzing(true);

    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/admin/detect-languages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls: urlList }),
        });
        if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
        const data = await response.json();
        if (cancelled) return;
        setPastedLanguages(data.languages || []);
        const codes = (data.languages || []).map((l) => l.code);
        setPastedSelectedLanguages(new Set(codes.includes("en") ? ["en"] : codes));
      } catch (err) {
        if (cancelled) return;
        console.error("Failed to detect languages for pasted URLs:", err);
        setPastedLanguages(null);
      } finally {
        if (!cancelled) setPastedAnalyzing(false);
      }
    }, 500);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urls, range]);

  const togglePastedLanguage = (code) => {
    setPastedSelectedLanguages((prev) => {
      const next = new Set(prev);
      if (next.has(code)) next.delete(code);
      else next.add(code);
      return next;
    });
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
        body: JSON.stringify({ url: site, urls: filteredUrls || [] }),
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
    if (!filteredUrls || filteredUrls.length === 0) return;
    try {
      const response = await fetch(`${API_BASE_URL}/admin/scrape-bulk-async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: filteredUrls }),
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
    setFilteredUrls(null);
    setSelectedLanguages(new Set(["en"]));
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

          {!parseRange(range) && (pastedAnalyzing || (pastedLanguages && pastedLanguages.length > 0)) && (
            <div className="bg-surface2 border border-border rounded-lg p-3 mt-2">
              {pastedAnalyzing && !pastedLanguages && (
                <p className="text-muted text-xs italic">Checking pasted URL(s) for language…</p>
              )}
              {pastedLanguages && pastedLanguages.length > 0 && (
                <>
                  <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-2">
                    Languages found in the pasted URL(s) — tick which to scrape
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {pastedLanguages.map((lang) => {
                      const checked = pastedSelectedLanguages.has(lang.code);
                      return (
                        <label
                          key={lang.code}
                          className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border cursor-pointer transition ${
                            checked
                              ? "bg-accent/15 border-accent text-[#eef2f5]"
                              : "bg-surface2 border-border text-muted hover:border-borderStrong"
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => togglePastedLanguage(lang.code)}
                            className="accent-accent"
                          />
                          {lang.name}
                          <span className="text-[10px] opacity-70">({lang.count.toLocaleString()})</span>
                        </label>
                      );
                    })}
                  </div>
                  <p className="text-[11px] text-muted mt-2">
                    This only applies to "Scrape & Index" below (exact URL(s), no range). Unticking a language drops
                    matching pasted URL(s) before scraping.
                  </p>
                </>
              )}
            </div>
          )}
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
            Range — page range (optional)
          </label>
          <input
            type="text"
            value={range}
            onChange={(e) => setRange(e.target.value)}
            placeholder="e.g. 1-200, then 201-400 next time — leave blank to scrape only the pasted URL(s) above"
            className="w-full max-w-[420px] bg-surface2 border border-border rounded-lg px-3 py-2 text-sm outline-none focus:border-accent placeholder:text-[#5c6672]"
          />
          <p className="text-[11px] text-muted mt-1">
            Enter a range like "1-200" to scrape that slice of the site's sitemap in order; run "201-400" next to
            continue where it left off. Pages already stored in the DB are skipped automatically. A plain number
            (e.g. "200") still works as shorthand for "1-200". Leave blank to scrape exactly the URL(s) above.
          </p>

          {rangeAnalyzing && <p className="text-muted text-xs italic mt-2">Checking that range…</p>}
          {rangeAnalyzeError && <p className="text-red-400 text-xs mt-2">{rangeAnalyzeError}</p>}

          {rangeAnalysis && !rangeAnalyzing && (
            <div className="bg-surface2 border border-border rounded-lg p-3 mt-2">
              {rangeAnalysis.past_end_of_sitemap ? (
                <p className="text-amber-400 text-xs font-semibold">
                  ⚠ The sitemap only has {rangeAnalysis.total_pages_in_sitemap.toLocaleString()} page(s) — this range
                  starts past the end of it.
                </p>
              ) : (
                <>
                  <p className="text-emerald-400 text-xs font-semibold">
                    {rangeAnalysis.count_in_range.toLocaleString()} page(s) in range {rangeAnalysis.start}–
                    {rangeAnalysis.end} of {rangeAnalysis.total_pages_in_sitemap.toLocaleString()} total —{" "}
                    {rangeAnalysis.already_ingested_count.toLocaleString()} already ingested,{" "}
                    {rangeAnalysis.new_count.toLocaleString()} new
                  </p>
                  {rangeAnalysis.languages && rangeAnalysis.languages.length > 0 && (
                    <div className="mt-2">
                      <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-2">
                        Languages found in this range — tick which to scrape
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {rangeAnalysis.languages.map((lang) => {
                          const checked = rangeSelectedLanguages.has(lang.code);
                          return (
                            <label
                              key={lang.code}
                              className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border cursor-pointer transition ${
                                checked
                                  ? "bg-accent/15 border-accent text-[#eef2f5]"
                                  : "bg-surface2 border-border text-muted hover:border-borderStrong"
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => toggleRangeLanguage(lang.code)}
                                className="accent-accent"
                              />
                              {lang.name}
                              <span className="text-[10px] opacity-70">({lang.count.toLocaleString()})</span>
                            </label>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
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

            {analyzeResult.languages && analyzeResult.languages.length > 0 && (
              <div className="bg-surface border border-border rounded-lg p-3">
                <p className="text-[11px] uppercase tracking-wider text-muted font-mono mb-2">
                  Languages found — tick which to scrape
                </p>
                <div className="flex flex-wrap gap-2">
                  {analyzeResult.languages.map((lang) => {
                    const checked = selectedLanguages.has(lang.code);
                    return (
                      <label
                        key={lang.code}
                        className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border cursor-pointer transition ${
                          checked
                            ? "bg-accent/15 border-accent text-[#eef2f5]"
                            : "bg-surface2 border-border text-muted hover:border-borderStrong"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleLanguage(lang.code)}
                          className="accent-accent"
                        />
                        {lang.name}
                        <span className="text-[10px] opacity-70">({lang.count.toLocaleString()})</span>
                      </label>
                    );
                  })}
                </div>
                <p className="text-[11px] text-muted mt-2">
                  {filteringUrls
                    ? "Filtering…"
                    : filteredUrls !== null
                    ? `${filteredUrls.length.toLocaleString()} page(s) match the selected language(s)`
                    : ""}
                </p>
              </div>
            )}

            {!deepResult && (
              <button
                onClick={handleDeepAnalyze}
                disabled={deepAnalyzing || filteringUrls || !filteredUrls || filteredUrls.length === 0}
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
                      disabled={!filteredUrls || filteredUrls.length === 0}
                      className="text-sm px-4 py-2 rounded-lg bg-accent text-[#0a0e12] font-semibold hover:-translate-y-0.5 transition disabled:opacity-40"
                    >
                      Scrape & Store {(filteredUrls || []).length.toLocaleString()} Page(s)
                      {selectedLanguages.size === 1 ? ` (${Array.from(selectedLanguages)[0]})` : ""}
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {jobStatus && (
          <div className="bg-surface2 border border-edge/30 rounded-lg p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <p className="text-sm font-semibold text-[#eef2f5]">
                {jobStatus.status === "completed" && "✅ Import complete"}
                {jobStatus.status === "running" && "⏳ Importing in background…"}
                {jobStatus.status === "paused" && "⏸️ Paused"}
                {jobStatus.status === "cancelled" && "🛑 Cancelled"}
              </p>
              <div className="flex items-center gap-3">
                <p className="text-xs text-muted">{jobStatus.pages_done} / {jobStatus.total_pages} pages</p>
                {jobStatus.status === "running" && (
                  <>
                    <button
                      onClick={handlePauseJob}
                      disabled={bulkPausing}
                      className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-edge/60 hover:border-accent2 transition disabled:opacity-40"
                    >
                      {bulkPausing ? "Pausing…" : "Pause"}
                    </button>
                    <button
                      onClick={() => handleCancelJob(jobId)}
                      onBlur={() => setConfirmCancelJob(false)}
                      disabled={bulkCancelling}
                      className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition disabled:opacity-40 ${
                        confirmCancelJob
                          ? "bg-red-500 text-white border-red-500 hover:bg-red-600"
                          : "text-red-400 border-red-500/30 hover:bg-red-500/10"
                      }`}
                    >
                      {bulkCancelling ? "Cancelling…" : confirmCancelJob ? "Click again to confirm" : "Cancel"}
                    </button>
                  </>
                )}
                {jobStatus.status === "paused" && (
                  <>
                    <button
                      onClick={() => handleResumeJob(jobId)}
                      disabled={bulkResuming}
                      className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-accent text-[#0a0e12] hover:-translate-y-0.5 transition disabled:opacity-40"
                    >
                      {bulkResuming ? "Resuming…" : "Resume"}
                    </button>
                    <button
                      onClick={() => handleCancelJob(jobId)}
                      onBlur={() => setConfirmCancelJob(false)}
                      disabled={bulkCancelling}
                      className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition disabled:opacity-40 ${
                        confirmCancelJob
                          ? "bg-red-500 text-white border-red-500 hover:bg-red-600"
                          : "text-red-400 border-red-500/30 hover:bg-red-500/10"
                      }`}
                    >
                      {bulkCancelling ? "Cancelling…" : confirmCancelJob ? "Click again to confirm" : "Cancel"}
                    </button>
                  </>
                )}
              </div>
            </div>
            <div className="w-full h-2 bg-surface rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${jobStatus.status === "paused" ? "bg-amber-500" : "bg-edge"}`}
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
            {jobStatus.status === "paused" && jobStatus.error && (
              <p className="text-xs text-amber-400">{jobStatus.error}</p>
            )}
            {jobStatus.status === "paused" && jobStatus.last_resume_check && (
              <p className="text-xs text-muted">last resume check: {jobStatus.last_resume_check}</p>
            )}
            {jobStatus.status === "paused" && (
              <p className="text-xs text-muted">
                Progress is saved — hit Resume any time (even after closing this tab or restarting the server) to
                continue from page {jobStatus.pages_done + 1}.
              </p>
            )}
            {jobStatus.status === "cancelled" && (
              <p className="text-xs text-muted">
                Cancelled — {jobStatus.pages_done} page(s) already scraped are kept; the rest were dropped.
              </p>
            )}
            {(jobStatus.status === "completed" || jobStatus.status === "cancelled") && (
              <button onClick={handleResetAnalysis} className="text-xs text-edge hover:underline self-start mt-1">
                Start a new analysis
              </button>
            )}
          </div>
        )}

        {!jobStatus && resumableJobs.length > 0 && (
          <div className="bg-surface2 border border-amber-500/30 rounded-lg p-4 flex flex-col gap-2">
            <p className="text-sm font-semibold text-amber-400">⏸️ You have {resumableJobs.length} paused import{resumableJobs.length > 1 ? "s" : ""}</p>
            {resumableJobs.map((j) => (
              <div key={j.job_id} className="flex items-center justify-between flex-wrap gap-2 text-xs">
                <span className="text-muted">
                  {j.pages_done} / {j.total_pages} pages done
                  {j.error ? ` — ${j.error}` : ""}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleResumeJob(j.job_id)}
                    disabled={bulkResuming}
                    className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-accent text-[#0a0e12] hover:-translate-y-0.5 transition disabled:opacity-40"
                  >
                    {bulkResuming ? "Resuming…" : "Resume"}
                  </button>
                  <button
                    onClick={() => handleCancelJob(j.job_id)}
                    onBlur={() => setConfirmCancelJob(false)}
                    disabled={bulkCancelling}
                    className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition disabled:opacity-40 ${
                      confirmCancelJob
                        ? "bg-red-500 text-white border-red-500 hover:bg-red-600"
                        : "text-red-400 border-red-500/30 hover:bg-red-500/10"
                    }`}
                  >
                    {bulkCancelling ? "Cancelling…" : confirmCancelJob ? "Confirm" : "Cancel"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

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
            {sources.length > 0 && selectedSourceIds.size > 0 && (
              <button
                onClick={handleBulkDelete}
                disabled={bulkDeleting}
                onBlur={() => setConfirmBulkDelete(false)}
                className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition disabled:opacity-40 ${
                  confirmBulkDelete
                    ? "bg-red-500 text-white border-red-500 hover:bg-red-600"
                    : "text-red-400 border-red-500/30 hover:bg-red-500/10"
                }`}
              >
                {bulkDeleting
                  ? "Deleting…"
                  : confirmBulkDelete
                  ? "Click again to confirm — permanent"
                  : `Delete Selected (${selectedSourceIds.size})`}
              </button>
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
                  <th className="text-left px-4 py-2 font-medium w-8">
                    <input
                      type="checkbox"
                      checked={sources.length > 0 && selectedSourceIds.size === sources.length}
                      onChange={toggleSelectAllSources}
                      className="accent-accent"
                    />
                  </th>
                  <th className="text-left px-4 py-2 font-medium">S.No</th>
                  <th className="text-left px-4 py-2 font-medium">URL</th>
                  <th className="text-left px-4 py-2 font-medium">Status</th>
                  <th className="text-left px-4 py-2 font-medium">Chunks</th>
                  <th className="text-left px-4 py-2 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sources.map((source, index) => (
                  <tr
                    key={source.id}
                    className={`border-t border-border ${selectedSourceIds.has(source.id) ? "bg-accent/5" : ""}`}
                  >
                    <td className="px-4 py-2">
                      <input
                        type="checkbox"
                        checked={selectedSourceIds.has(source.id)}
                        onChange={() => toggleSourceSelected(source.id)}
                        className="accent-accent"
                      />
                    </td>
                    <td className="px-4 py-2 text-muted">{index + 1}</td>
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