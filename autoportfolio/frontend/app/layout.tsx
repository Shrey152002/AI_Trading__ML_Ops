import type { Metadata } from "next";
import Link from "next/link";
import { LineChart } from "lucide-react";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { HealthBanner } from "@/components/HealthBanner";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AutoPortfolio Dashboard",
  description: "Live portfolio allocation recommendations from AutoPortfolio's RL agents.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900">
        <Providers>
          <HealthBanner />
          <header className="border-b border-slate-200 bg-white">
            <div className="mx-auto flex max-w-6xl items-center gap-3 px-6 py-3">
              <Link href="/" className="flex items-center gap-2">
                <span className="flex h-7 w-7 items-center justify-center rounded-md bg-slate-900">
                  <LineChart className="h-4 w-4 text-white" />
                </span>
                <span className="text-sm font-semibold tracking-tight text-slate-900">
                  AutoPortfolio
                </span>
              </Link>
              <span className="hidden text-xs text-slate-400 sm:inline">
                RL portfolio allocation — recommendations, not trades
              </span>
            </div>
          </header>
          <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
