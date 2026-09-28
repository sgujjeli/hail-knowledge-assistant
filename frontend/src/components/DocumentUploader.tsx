/**
 * DocumentUploader — drag-and-drop PDF/DOCX upload with progress polling.
 *
 * Flow:
 *   1. User drops/selects file
 *   2. POST /api/documents/upload → returns { doc_id, status: "processing" }
 *   3. Poll GET /api/documents/{doc_id}/status every 2s (from Redis cache)
 *   4. On status=ready → call onUploadComplete(doc_id)
 */
"use client";

import { useState, useCallback, useRef } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Props {
  onUploadComplete: (docId: string, filename: string) => void;
}

interface IngestionStatus {
  status: "processing" | "parsing" | "chunking" | "embedding" | "indexing" | "ready" | "failed";
  progress: number;
  chunk_count?: number;
  page_count?: number;
  error?: string;
}

export default function DocumentUploader({ onUploadComplete }: Props) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState<IngestionStatus | null>(null);
  const [filename, setFilename] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.match(/\.(pdf|docx)$/i)) {
      alert("Only PDF and DOCX files are supported.");
      return;
    }

    setUploading(true);
    setFilename(file.name);
    setStatus({ status: "processing", progress: 5 });

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/api/documents/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const { doc_id } = await res.json();

      // Poll ingestion status
      const pollInterval = setInterval(async () => {
        const statusRes = await fetch(`${API_BASE}/api/documents/${doc_id}/status`);
        const s: IngestionStatus = await statusRes.json();
        setStatus(s);

        if (s.status === "ready") {
          clearInterval(pollInterval);
          setUploading(false);
          onUploadComplete(doc_id, file.name);
        } else if (s.status === "failed") {
          clearInterval(pollInterval);
          setUploading(false);
          alert(`Ingestion failed: ${s.error}`);
        }
      }, 2000);

    } catch (err: any) {
      setUploading(false);
      setStatus(null);
      alert(`Error: ${err.message}`);
    }
  }, [onUploadComplete]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const statusLabel: Record<string, string> = {
    processing: "Uploading…",
    parsing:    "Parsing document…",
    chunking:   "Splitting into chunks…",
    embedding:  "Generating embeddings…",
    indexing:   "Indexing into Azure AI Search…",
    ready:      "Ready!",
    failed:     "Failed",
  };

  return (
    <div className="w-full">
      <div
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
          ${isDragOver ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400"}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        <div className="text-4xl mb-3">📄</div>
        <p className="text-gray-600 font-medium">
          Drop a PDF or Word document here, or click to browse
        </p>
        <p className="text-gray-400 text-sm mt-1">Max 20 MB</p>
      </div>

      {/* Progress */}
      {uploading && status && (
        <div className="mt-4">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>📑 {filename}</span>
            <span>{statusLabel[status.status] || status.status}</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${status.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Ready state */}
      {status?.status === "ready" && (
        <div className="mt-3 text-green-600 text-sm font-medium">
          ✅ {filename} — {status.chunk_count} chunks across {status.page_count} pages indexed
        </div>
      )}
    </div>
  );
}
