/**
 * ChatWindow — renders streaming messages and source citations.
 *
 * Receives messages from useRAGChat hook.
 * Auto-scrolls to latest message.
 * Renders [CITATIONS] as clickable source cards below each assistant message.
 */
"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/hooks/useRAGChat";

interface Props {
  messages: Message[];
  isLoading: boolean;
}

export default function ChatWindow({ messages, isLoading }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400">
        <div className="text-center">
          <div className="text-5xl mb-4">🤖</div>
          <p className="font-medium text-lg">Ask Hail anything about your document</p>
          <p className="text-sm mt-1">Upload a PDF or Word file to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg) => (
        <div key={msg.id}>
          {/* Message bubble */}
          <div
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-2xl rounded-2xl px-4 py-3 text-sm leading-relaxed
                ${msg.role === "user"
                  ? "bg-blue-600 text-white rounded-br-sm"
                  : "bg-white border border-gray-200 text-gray-800 rounded-bl-sm shadow-sm"
                }`}
            >
              {/* Role label */}
              {msg.role === "assistant" && (
                <div className="text-xs font-semibold text-blue-600 mb-1 uppercase tracking-wide">
                  Hail
                </div>
              )}

              {/* Message text — pre-wrap preserves line breaks */}
              <div className="whitespace-pre-wrap">
                {msg.content}
                {msg.isStreaming && (
                  <span className="inline-block w-2 h-4 ml-1 bg-blue-400 animate-pulse rounded-sm align-middle" />
                )}
              </div>
            </div>
          </div>

          {/* Citations */}
          {msg.role === "assistant" && !msg.isStreaming && msg.citations && msg.citations.length > 0 && (
            <div className="mt-2 ml-1 flex flex-wrap gap-2">
              {msg.citations.map((cite, i) => (
                <div
                  key={i}
                  className="flex items-center gap-1.5 bg-amber-50 border border-amber-200 text-amber-800
                             text-xs rounded-full px-3 py-1"
                >
                  <span>📄</span>
                  <span className="font-medium">{cite.filename}</span>
                  <span className="text-amber-500">·</span>
                  <span>Page {cite.page_number}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Typing indicator when loading but no partial text yet */}
      {isLoading && messages[messages.length - 1]?.role !== "assistant" && (
        <div className="flex justify-start">
          <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
