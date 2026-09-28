/**
 * Main page — Hail Knowledge Assistant.
 *
 * Layout:
 *   Left panel: DocumentUploader + document list
 *   Right panel: ChatWindow + message input + VoiceInput
 *
 * State:
 *   activeDocId — restricts retrieval to a single document (optional)
 *   documents   — list of uploaded docs
 *   messages    — from useRAGChat hook
 */
"use client";

import { useState, useRef, useCallback } from "react";
import DocumentUploader from "@/components/DocumentUploader";
import ChatWindow from "@/components/ChatWindow";
import VoiceInput from "@/components/VoiceInput";
import { useRAGChat } from "@/hooks/useRAGChat";

interface Doc {
  doc_id: string;
  filename: string;
  chunk_count: number;
  page_count: number;
}

export default function Home() {
  const [documents, setDocuments] = useState<Doc[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | undefined>(undefined);
  const [inputText, setInputText] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const { messages, isLoading, sendMessage, clearMessages } = useRAGChat(activeDocId);

  const handleUploadComplete = useCallback((docId: string, filename: string) => {
    setDocuments((prev) => {
      if (prev.some((d) => d.doc_id === docId)) return prev;
      return [{ doc_id: docId, filename, chunk_count: 0, page_count: 0 }, ...prev];
    });
    setActiveDocId(docId);
    clearMessages();
  }, [clearMessages]);

  const handleSend = useCallback(() => {
    const q = inputText.trim();
    if (!q || isLoading) return;
    setInputText("");
    sendMessage(q);
  }, [inputText, isLoading, sendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      {/* ── Left panel: documents ── */}
      <aside className="w-80 flex-shrink-0 bg-white border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-100">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-2xl">🤖</span>
            <h1 className="font-bold text-gray-900 text-lg">Hail</h1>
          </div>
          <p className="text-xs text-gray-500">Knowledge Assistant · hailoop.co.uk</p>
          <p className="text-xs text-gray-400 mt-0.5">
            GPT-4o · Azure AI Search · FastAPI · hailoop.co.uk
          </p>
        </div>

        {/* Uploader */}
        <div className="p-4 border-b border-gray-100">
          <DocumentUploader onUploadComplete={handleUploadComplete} />
        </div>

        {/* Document list */}
        <div className="flex-1 overflow-y-auto p-4">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Uploaded Documents
          </p>
          {documents.length === 0 ? (
            <p className="text-sm text-gray-400">No documents yet.</p>
          ) : (
            <ul className="space-y-2">
              {documents.map((doc) => (
                <li key={doc.doc_id}>
                  <button
                    onClick={() => {
                      setActiveDocId(doc.doc_id === activeDocId ? undefined : doc.doc_id);
                      clearMessages();
                    }}
                    className={`w-full text-left rounded-lg px-3 py-2 text-sm transition-colors
                      ${activeDocId === doc.doc_id
                        ? "bg-blue-50 text-blue-700 border border-blue-200"
                        : "text-gray-700 hover:bg-gray-100"
                      }`}
                  >
                    <div className="font-medium truncate">📄 {doc.filename}</div>
                    <div className="text-xs text-gray-400 mt-0.5">
                      {doc.chunk_count ? `${doc.chunk_count} chunks` : "ready"}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Active filter indicator */}
        <div className="p-3 border-t border-gray-100 text-xs text-gray-400">
          {activeDocId
            ? `🔍 Searching: ${documents.find((d) => d.doc_id === activeDocId)?.filename}`
            : "🔍 Searching: all documents"}
        </div>
      </aside>

      {/* ── Right panel: chat ── */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Chat header */}
        <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-gray-900">Document Q&A</h2>
            <p className="text-xs text-gray-500">
              Hybrid search · Grounded responses · RAGAS evaluation
            </p>
          </div>
          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              className="text-xs text-gray-400 hover:text-gray-600 px-3 py-1 rounded border border-gray-200 hover:border-gray-300"
            >
              Clear chat
            </button>
          )}
        </header>

        {/* Messages */}
        <ChatWindow messages={messages} isLoading={isLoading} />

        {/* Input bar */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="flex items-end gap-3 max-w-4xl mx-auto">
            <VoiceInput
              onTranscript={(text) => {
                setInputText(text);
                setTimeout(() => inputRef.current?.focus(), 50);
              }}
              disabled={isLoading}
            />
            <textarea
              ref={inputRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your document… (Enter to send, Shift+Enter for new line)"
              rows={2}
              disabled={isLoading}
              className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-2.5 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                         disabled:bg-gray-50 disabled:cursor-not-allowed"
            />
            <button
              onClick={handleSend}
              disabled={!inputText.trim() || isLoading}
              className="px-5 py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium
                         hover:bg-blue-700 active:scale-95 transition-all
                         disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100"
            >
              {isLoading ? "…" : "Send →"}
            </button>
          </div>
          <p className="text-center text-xs text-gray-400 mt-2">
            Powered by Azure OpenAI GPT-4o · Azure AI Search hybrid · FastAPI · RAGAS · hailoop.co.uk
          </p>
        </div>
      </main>
    </div>
  );
}
