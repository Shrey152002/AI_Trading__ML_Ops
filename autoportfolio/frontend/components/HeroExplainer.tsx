import { Database, Cpu, MessageSquareText, ShieldCheck, Trophy } from "lucide-react";

const STEPS = [
  {
    icon: Database,
    title: "1. Ingest & engineer features",
    description:
      "Daily prices for each ticker get pulled in and turned into return, volatility, and momentum features.",
  },
  {
    icon: Cpu,
    title: "2. Train four RL agents",
    description:
      "PPO, A2C, SAC, and DDPG each learn an allocation policy by trial and error on historical data.",
  },
  {
    icon: Trophy,
    title: "3. Benchmark & promote",
    description:
      "Whichever agent performs best on held-out test data gets registered and put into production.",
  },
  {
    icon: MessageSquareText,
    title: "4. Serve recommendations",
    description:
      "Ask for an allocation any time — the live model responds with weights and a plain-English reason.",
  },
];

export function HeroExplainer() {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6 text-white sm:p-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">AutoPortfolio</h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-300">
            A deep reinforcement learning system that recommends how to allocate capital
            across a basket of stocks — trained, benchmarked, monitored for drift, and
            served through the API and this dashboard.
          </p>
        </div>
        <span className="flex items-center gap-1.5 whitespace-nowrap rounded-full border border-amber-400/40 bg-amber-400/10 px-3 py-1 text-xs font-medium text-amber-300">
          <ShieldCheck className="h-3.5 w-3.5" />
          Not a trading bot — no orders are ever placed
        </span>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step) => (
          <div
            key={step.title}
            className="rounded-lg border border-white/10 bg-white/5 p-3 backdrop-blur-sm"
          >
            <step.icon className="h-5 w-5 text-slate-300" />
            <p className="mt-2 text-sm font-semibold text-white">{step.title}</p>
            <p className="mt-1 text-xs leading-relaxed text-slate-400">{step.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
