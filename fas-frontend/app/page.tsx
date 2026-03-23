"use client";

import { useEffect, useRef, useState } from "react";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    startCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error(err);
    }
  };

  const captureAndSend = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx?.drawImage(video, 0, 0);

    setLoading(true);

    canvas.toBlob(async (blob) => {
      try {
        const formData = new FormData();
        formData.append("file", blob as Blob);

        const res = await fetch(
          "https://fas-tgop.onrender.com/predict",
          {
            method: "POST",
            body: formData,
          }
        );

        const data = await res.json();
        setResult(data.result || "error");
      } catch (err) {
        console.error(err);
        setResult("error");
      } finally {
        setLoading(false);
      }
    }, "image/jpeg");
  };

  const startDetection = () => {
    setRunning(true);
    const interval = setInterval(() => {
      captureAndSend();
    }, 2000);

    (window as any).stopDetection = () => {
      clearInterval(interval);
      setRunning(false);
    };
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Face Anti-Spoofing</h1>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ width: "320px", borderRadius: "10px" }}
      />

      <canvas ref={canvasRef} style={{ display: "none" }} />

      <br /><br />

      {!running ? (
        <button onClick={startDetection}>Start Detection</button>
      ) : (
        <button onClick={() => (window as any).stopDetection()}>
          Stop
        </button>
      )}

      <p>{loading && "Processing..."}</p>

      {result && (
        <div
          style={{
            marginTop: "20px",
            padding: "10px",
            color: "white",
            backgroundColor:
              result === "real" ? "green" : result === "fake" ? "red" : "gray",
          }}
        >
          {result.toUpperCase()}
        </div>
      )}
    </div>
  );
}