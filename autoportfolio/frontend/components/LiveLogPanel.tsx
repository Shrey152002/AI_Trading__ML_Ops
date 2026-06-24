"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Terminal } from "lucide-react";

import { getPipelineLogs } from "@/lib/api/endpoints";
import type { PipelineLogEntry } from "@/lib/api/types";

const LEVEL_COLORS: Record<string, string> = {
  INFO: "text-slate-300",
  WARNING: "text-amber-400",
  ERROR: "text-red-400",
  CRITICAL: "text-red-400",
};

const POLL_INTERVAL_MS = 1500;
const MAX_ENTRIES = 500;

export function LiveLogPanel({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [entries, setEntries] = useState<PipelineLogEntry[]>([]);
  const sinceRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;

    async function poll() {
      try {
        const { entries: newEntries } = await getPipelineLogs(sinceRef.current);
        if (cancelled || newEntries.length === 0) return;
        sinceRef.current = newEntries[newEntries.length - 1].seq;
        setEntries((prev) => [...prev, ...newEntries].slice(-MAX_ENTRIES));
      } catch {
        // best-effort live tail — a transient fetch failure just means we try again next tick
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [open]);

  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <button
        onClick={() => onOpenChange(!open)}
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-semibold text-slate-800"
      >
        <span className="flex items-center gap-2">
          <Terminal className="h-4 w-4" />
          Live backend activity
          {open && (
            <span className="flex h-1.5 w-1.5 rounded-full bg-emerald-500">
              <span className="h-full w-full animate-ping rounded-full bg-emerald-500" />
            </span>
          )}
        </span>
        {open ? (
          <ChevronUp className="h-4 w-4 text-slate-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-slate-400" />
        )}
      </button>
      {open && (
        <div className="border-t border-slate-200 p-3">
          <p className="mb-2 text-xs text-slate-500">
            Raw log output from the data ingestion, validation, feature, drift-check, and
            training steps — shared across all portfolios, so this is most meaningful right
            after you trigger a pipeline run.
          </p>
          <div
            ref={scrollRef}
            onScroll={(e) => {
              const el = e.currentTarget;
              autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
            }}
            className="h-64 overflow-y-auto rounded-md bg-slate-950 p-3 font-mono text-[11px] leading-5"
          >
            {entries.length === 0 ? (
              <p className="text-slate-500">Waiting for activity...</p>
            ) : (
              entries.map((entry) => (
                <div key={entry.seq} className="whitespace-pre-wrap break-all">
                  <span className="text-slate-500">
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </span>{" "}
                  <span className={LEVEL_COLORS[entry.level] ?? "text-slate-300"}>
                    {entry.level}
                  </span>{" "}
                  <span className="text-slate-500">{entry.logger}</span>{" "}
                  <span className="text-slate-100">{entry.message}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
