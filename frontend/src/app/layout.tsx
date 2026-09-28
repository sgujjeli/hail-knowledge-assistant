import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hail Knowledge Assistant — hailoop.co.uk",
  description:
    "AI-powered document knowledge assistant: Azure OpenAI GPT-4o, Azure AI Search hybrid retrieval, FastAPI, PostgreSQL, Redis, RAGAS evaluation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
