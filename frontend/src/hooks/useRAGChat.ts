/**
 * useRAGChat — React hook for streaming RAG responses via SSE.
 *
 * Connects to POST /api/chat and reads Server-Sent Events:
 *   data: <token>           → appended to message
 *   data: [CITATIONS][...]  → parsed as source cards
 *   data: [DONE]            → stream complete
 */
import { useState, useCallback, useRef } from "react";

export interface Citation {
  filename: string;
  page_number: number;
  chunk_id: string | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useRAGChat(docId?: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (query: string) => {
      if (!query.trim()) return;

      const userMsg: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: query,
      };

      const assistantMsgId = `assistant-${Date.now()}`;
      const assistantMsg: Message = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsLoading(true);
      setError(null);

      // Abort any in-flight request
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      try {
        const response = await fetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, doc_id: docId }),
          signal: abortRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let citations: Citation[] = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });
          const lines = text.split("\n");

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6); // strip "data: "

            if (data === "[DONE]") {
              // Mark streaming complete
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId
                    ? { ...m, isStreaming: false, citations }
                    : m
                )
              );
            } else if (data.startsWith("[CITATIONS]")) {
              // Parse citation JSON
              try {
                citations = JSON.parse(data.slice(11));
              } catch {
                citations = [];
              }
            } else {
              // Append token to assistant message
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId
                    ? { ...m, content: m.content + data }
                    : m
                )
              );
            }
          }
        }
      } catch (err: any) {
        if (err.name === "AbortError") return;
        setError(err.message || "Something went wrong");
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: "Sorry, an error occurred. Please try again.", isStreaming: false }
              : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [docId]
  );

  const clearMessages = useCallback(() => setMessages([]), []);

  return { messages, isLoading, error, sendMessage, clearMessages };
}
