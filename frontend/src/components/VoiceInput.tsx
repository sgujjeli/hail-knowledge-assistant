/**
 * VoiceInput — browser MediaRecorder → Azure Speech STT → query.
 *
 * Flow:
 *   1. User clicks mic button → start recording (MediaRecorder API)
 *   2. User clicks again → stop recording, get WAV blob
 *   3. POST /api/voice/transcribe → { transcript }
 *   4. Call onTranscript(transcript) → parent sends to chat
 */
"use client";

import { useState, useRef, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Props {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}

export default function VoiceInput({ onTranscript, disabled }: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Prefer ogg/opus — supported by Azure Speech SDK (OGG_OPUS container)
      // Fall back to webm (backend will attempt WAV path as last resort)
      const mimeType = MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")
        ? "audio/ogg;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: mimeType });

        setIsTranscribing(true);
        try {
          const formData = new FormData();
          const ext = mimeType.includes("ogg") ? "ogg" : mimeType.includes("wav") ? "wav" : "webm";
          formData.append("file", blob, `voice.${ext}`);

          const res = await fetch(`${API_BASE}/api/voice/transcribe`, {
            method: "POST",
            body: formData,
          });

          if (res.ok) {
            const { transcript } = await res.json();
            if (transcript) onTranscript(transcript);
          } else {
            console.error("Transcription failed:", await res.text());
          }
        } finally {
          setIsTranscribing(false);
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Microphone access is required for voice input.");
    }
  }, [onTranscript]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
  }, []);

  const toggle = useCallback(() => {
    if (isRecording) stopRecording();
    else startRecording();
  }, [isRecording, startRecording, stopRecording]);

  return (
    <button
      onClick={toggle}
      disabled={disabled || isTranscribing}
      title={isRecording ? "Stop recording" : "Start voice input"}
      className={`p-2 rounded-full transition-colors
        ${isRecording
          ? "bg-red-500 text-white animate-pulse"
          : isTranscribing
          ? "bg-yellow-400 text-white cursor-wait"
          : "bg-gray-100 text-gray-600 hover:bg-gray-200"
        }
        disabled:opacity-50 disabled:cursor-not-allowed`}
    >
      {isTranscribing ? "⏳" : isRecording ? "⏹️" : "🎙️"}
    </button>
  );
}
