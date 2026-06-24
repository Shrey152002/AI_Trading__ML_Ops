"use client";

import { useEffect, useId, useRef, useState } from "react";
import { Info } from "lucide-react";

export function InfoTooltip({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const containerRef = useRef<HTMLSpanElement>(null);

  // Click-to-toggle only (no hover) — combining both led to a state race where a
  // click immediately followed by mouseleave closed the tooltip before it could be
  // read. Click-away (via this listener) and Escape both close it.
  useEffect(() => {
    if (!open) return;
    function handlePointerDown(event: PointerEvent) {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  return (
    <span ref={containerRef} className="relative inline-flex">
      <button
        type="button"
        aria-describedby={id}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex h-3.5 w-3.5 items-center justify-center text-slate-400 hover:text-slate-600"
      >
        <Info className="h-3.5 w-3.5" />
      </button>
      {open && (
        <div
          id={id}
          role="tooltip"
          className="absolute bottom-full left-1/2 z-20 mb-2 w-64 -translate-x-1/2 rounded-lg border border-slate-200 bg-white p-3 text-xs leading-relaxed text-slate-600 shadow-lg"
        >
          <p className="font-semibold text-slate-800">{title}</p>
          <div className="mt-1">{children}</div>
          <div className="absolute left-1/2 top-full -mt-1 h-2 w-2 -translate-x-1/2 rotate-45 border-b border-r border-slate-200 bg-white" />
        </div>
      )}
    </span>
  );
}
