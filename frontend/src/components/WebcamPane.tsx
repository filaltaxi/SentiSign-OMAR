import { useCallback, useEffect, useRef } from 'react';
import type { EmotionType } from './EmotionStrip';

interface WebcamPaneProps {
    isActive: boolean;
    onEmotionDetected: (emotion: EmotionType) => void;
    onSignDetected: (word: string | null, cls: string | null, confidence: number) => void;
    currentEmotion: EmotionType;
    wordLabel: string;
    confidence: number;
}

// MediaPipe utilities will be loaded via script tags in index.html,
// we access them via window object in a real implementation.
declare global {
    interface Window {
        Hands: any;
        Camera: any;
        drawConnectors: any;
        drawLandmarks: any;
        HAND_CONNECTIONS: any;
    }
}

const RECOGNISE_DELAY = 80;

export function WebcamPane({
    isActive,
    onEmotionDetected,
    onSignDetected,
    currentEmotion,
    wordLabel,
    confidence,
}: WebcamPaneProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Keep latest callback props without forcing MediaPipe to re-subscribe.
    const onEmotionDetectedRef = useRef(onEmotionDetected);
    const onSignDetectedRef = useRef(onSignDetected);

    useEffect(() => {
        onEmotionDetectedRef.current = onEmotionDetected;
    }, [onEmotionDetected]);

    useEffect(() => {
        onSignDetectedRef.current = onSignDetected;
    }, [onSignDetected]);

    // Ref to hold the MediaPipe instances
    const mpRef = useRef<{
        camera: any | null;
        hands: any | null;
        emotionInterval: ReturnType<typeof setInterval> | null;
        lastRecogniseTime: number;
    }>({
        camera: null,
        hands: null,
        emotionInterval: null,
        lastRecogniseTime: 0,
    });

    const onHandResults = useCallback(async (results: any) => {
        const videoEl = videoRef.current;
        const canvas = canvasRef.current;
        if (!videoEl || !canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = videoEl.videoWidth || 640;
        canvas.height = videoEl.videoHeight || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
            onSignDetectedRef.current(null, null, 0);
            return;
        }

        // Draw
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const lm = results.multiHandLandmarks[i];
            let isPhysicalRight;
            if (results.multiHandLandmarks.length >= 2) {
                const w0 = results.multiHandLandmarks[0][0].x;
                const w1 = results.multiHandLandmarks[1][0].x;
                isPhysicalRight = i === (w0 > w1 ? 0 : 1);
            } else {
                const label = results.multiHandedness?.[i]?.classification?.[0]?.label ?? 'Left';
                isPhysicalRight = label === 'Left';
            }

            const color = isPhysicalRight ? '#00d4aa' : '#ff9a3c';

            window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: color + '55', lineWidth: 2 });
            window.drawLandmarks(ctx, lm, { color, lineWidth: 1, radius: 3 });

            const xs = lm.map((p: any) => p.x * canvas.width);
            const ys = lm.map((p: any) => p.y * canvas.height);
            const x1 = Math.max(0, Math.min(...xs) - 20);
            const y1 = Math.max(0, Math.min(...ys) - 20);
            const x2 = Math.min(canvas.width, Math.max(...xs) + 20);
            const y2 = Math.min(canvas.height, Math.max(...ys) + 20);

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = color;
            ctx.font = '13px "DM Sans"';
            ctx.fillText(isPhysicalRight ? 'L' : 'R', x1 + 4, y1 - 4);
        }

        // Build features
        const right = new Array(63).fill(0);
        const left = new Array(63).fill(0);

        if (results.multiHandLandmarks.length === 1) {
            const label = results.multiHandedness?.[0]?.classification?.[0]?.label ?? 'Left';
            const flat = results.multiHandLandmarks[0].flatMap((p: any) => [p.x, p.y, p.z]);
            if (label === 'Right') flat.forEach((v: number, j: number) => right[j] = v);
            else flat.forEach((v: number, j: number) => left[j] = v);
        } else {
            const w0 = results.multiHandLandmarks[0][0].x;
            const w1 = results.multiHandLandmarks[1][0].x;
            const rIdx = w0 > w1 ? 0 : 1;
            const lIdx = w0 > w1 ? 1 : 0;
            results.multiHandLandmarks[rIdx].flatMap((p: any) => [p.x, p.y, p.z]).forEach((v: number, j: number) => right[j] = v);
            results.multiHandLandmarks[lIdx].flatMap((p: any) => [p.x, p.y, p.z]).forEach((v: number, j: number) => left[j] = v);
        }
        const features = [...right, ...left];

        // Throttle API calls
        const now = Date.now();
        if (now - mpRef.current.lastRecogniseTime < RECOGNISE_DELAY) {
            return;
        }
        mpRef.current.lastRecogniseTime = now;

        try {
            const res = await fetch('/api/recognise', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ landmarks: features })
            });
            const data = await res.json();
            onSignDetectedRef.current(data.word, data.class, data.confidence);
        } catch (err) {
            console.error("Recognise API failed", err);
        }
    }, []);

    // Lifecycle for Camera / Session
    useEffect(() => {
        let ignore = false;

        function stopSession(opts?: { resetEmotion?: boolean }) {
            const resetEmotion = opts?.resetEmotion ?? true;

            mpRef.current.lastRecogniseTime = 0;

            if (mpRef.current.emotionInterval) {
                clearInterval(mpRef.current.emotionInterval);
                mpRef.current.emotionInterval = null;
            }
            if (mpRef.current.camera) {
                try {
                    mpRef.current.camera.stop();
                } catch { }
                mpRef.current.camera = null;
            }
            if (mpRef.current.hands) {
                try {
                    mpRef.current.hands.close();
                } catch { }
                mpRef.current.hands = null;
            }
            if (videoRef.current?.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach((t) => t.stop());
                videoRef.current.srcObject = null;
            }
            if (videoRef.current) {
                try {
                    videoRef.current.pause();
                } catch { }
            }
            if (canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
            if (resetEmotion) {
                onEmotionDetectedRef.current('neutral');
            }
        }

        async function startSession() {
            if (!videoRef.current || !canvasRef.current) return;
            if (!window.Hands || !window.Camera) {
                console.error("MediaPipe scripts not loaded yet");
                return;
            }

            try {
                const videoEl = videoRef.current;

                // Clean slate if a previous session wasn't fully torn down.
                stopSession({ resetEmotion: false });
                mpRef.current.lastRecogniseTime = 0;

                const handsDetector = new window.Hands({
                    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`
                });

                handsDetector.setOptions({
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.7,
                    minTrackingConfidence: 0.6,
                });

                handsDetector.onResults(onHandResults);
                mpRef.current.hands = handsDetector;

                const camera = new window.Camera(videoEl, {
                    onFrame: async () => {
                        if (ignore) return;
                        if (mpRef.current.hands && videoEl) {
                            await mpRef.current.hands.send({ image: videoEl });
                        }
                    },
                    width: 640,
                    height: 480,
                });

                mpRef.current.camera = camera;
                await camera.start();

                if (ignore) {
                    stopSession({ resetEmotion: false });
                    return;
                }

                // Emotion Loop
                mpRef.current.emotionInterval = setInterval(async () => {
                    if (ignore) return;
                    if (!videoEl.videoWidth) return;
                    try {
                        const tmpCanvas = document.createElement('canvas');
                        tmpCanvas.width = videoEl.videoWidth;
                        tmpCanvas.height = videoEl.videoHeight;
                        const tmpCtx = tmpCanvas.getContext('2d');
                        if (!tmpCtx) return;

                        tmpCtx.translate(tmpCanvas.width, 0);
                        tmpCtx.scale(-1, 1);
                        tmpCtx.drawImage(videoEl, 0, 0);

                        const base64 = tmpCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
                        const res = await fetch('/api/emotion', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ image: base64 })
                        });
                        const data = await res.json();

                        if (data.emotion && !ignore) {
                            onEmotionDetectedRef.current(data.emotion);
                        }
                    } catch { }
                }, 500);

            } catch (err) {
                console.error("Camera failed:", err);
                stopSession({ resetEmotion: false });
            }
        }

        if (isActive) {
            startSession();
        } else {
            stopSession();
        }

        return () => {
            ignore = true;
            stopSession();
        };
    }, [isActive, onHandResults]);

    return (
        <div className="bg-surface border border-border-color rounded-xl overflow-hidden relative shadow-lg">
            <div className="relative aspect-4/3 bg-black">
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="w-full h-full object-cover block scale-x-[-1]"
                />
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none scale-x-[-1]"
                />

                {/* Top Left Status */}
                <div className="absolute top-3 left-3 bg-black/70 border border-border-color rounded-lg px-3 py-1.5 text-[0.75rem] flex items-center gap-1.5 backdrop-blur-sm">
                    <div className={`w-2 h-2 rounded-full transition-colors duration-300 ${isActive ? (confidence > 0 ? 'bg-amber shadow-[0_0_8px_var(--color-amber)]' : 'bg-brand shadow-[0_0_8px_var(--color-brand)]') : 'bg-muted'}`} />
                    <span>{isActive ? 'Session active' : 'Camera off'}</span>
                </div>

                {/* Top Right Emotion */}
                <div
                    className={`absolute top-3 right-3 bg-black/70 border rounded-lg px-3.5 py-1.5 text-[0.8rem] font-medium capitalize transition-colors duration-400 backdrop-blur-sm`}
                    style={{
                        borderColor: getEmotionColor(currentEmotion),
                        color: getEmotionColor(currentEmotion)
                    }}
                >
                    {isActive ? currentEmotion : '—'}
                </div>
            </div>

            {/* Sign Confidence Strip */}
            <div className="px-5 py-2.5 border-t border-border-color flex items-center gap-3 text-[0.8rem] text-muted bg-[#080d14]">
                <span className="min-w-[140px] font-medium text-text truncate">
                    {wordLabel}
                </span>
                <div className="flex-1 h-1 bg-border-color rounded-full overflow-hidden">
                    <div
                        className="h-full bg-brand rounded-full transition-all duration-200"
                        style={{ width: `${Math.round(confidence * 100)}%` }}
                    />
                </div>
                <span className="min-w-[38px] text-right font-mono text-[0.75rem]">
                    {Math.round(confidence * 100)}%
                </span>
            </div>
        </div>
    );
}

function getEmotionColor(e: string) {
    const map: Record<string, string> = {
        happy: '#ffd700', sad: '#6aa3d5', angry: '#ff5f5f',
        fear: '#c084fc', disgust: '#86efac', surprise: '#ff9a3c', neutral: '#00d4aa'
    };
    return map[e] || '#00d4aa';
}
