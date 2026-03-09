import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera } from 'lucide-react';
import type { EmotionType } from './EmotionStrip';
import type { SignModel } from '../model/ModelContext';
import { extractTemporalFeatures, hasTemporalSignal } from '../lib/handFeatures';

interface WebcamPaneProps {
    model: SignModel;
    isActive: boolean;
    commitResetNonce: number;
    onEmotionDetected: (emotion: EmotionType) => void;
    onSignDetected: (
        word: string | null,
        cls: string | null,
        confidence: number,
        meta?: SignDetectionMeta
    ) => void;
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
const LSTM_N_FRAMES = 60;
const LSTM_FEATURE_DIM = 126;
const LSTM_MIN_INFERENCE_FRAMES = 10;
const LSTM_NO_SIGNAL_FRAMES = 10;
const LSTM_STRIDE_FRAMES = 5;
const LSTM_COOLDOWN_MS = 500;
const EMOTION_AURA_COLORS: Record<EmotionType, string> = {
    neutral: '51,153,255',
    happy: '255,213,79',
    sad: '130,100,255',
    angry: '255,75,75',
    fear: '180,80,255',
    disgust: '80,210,120',
    surprise: '255,160,50',
};

export type SignDetectionMeta = {
    margin?: number;
    phase?: 'preview' | 'final' | 'reset';
};

type TemporalSegmentPhase = 'idle' | 'collecting' | 'cooldown';

type TemporalHudState = {
    windowFrames: number;
    signalFrames: number;
    strideFrames: number;
    noSignalFrames: number;
    cooldownRemainingMs: number;
    hasSignal: boolean;
    inFlight: boolean;
    phase: TemporalSegmentPhase;
};

function padTemporalSequence(sequence: number[][]): number[][] {
    if (sequence.length >= LSTM_N_FRAMES) {
        return sequence.slice(-LSTM_N_FRAMES);
    }

    const padLength = LSTM_N_FRAMES - sequence.length;
    const padding = Array.from({ length: padLength }, () => new Array(LSTM_FEATURE_DIM).fill(0));
    return [...sequence, ...padding];
}

export function WebcamPane({
    model,
    isActive,
    commitResetNonce,
    onEmotionDetected,
    onSignDetected,
    currentEmotion,
    wordLabel,
    confidence,
}: WebcamPaneProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [temporalHud, setTemporalHud] = useState<TemporalHudState>({
        windowFrames: 0,
        signalFrames: 0,
        strideFrames: 0,
        noSignalFrames: 0,
        cooldownRemainingMs: 0,
        hasSignal: false,
        inFlight: false,
        phase: 'idle',
    });

    // Keep latest callback props without forcing MediaPipe to re-subscribe.
    const onEmotionDetectedRef = useRef(onEmotionDetected);
    const onSignDetectedRef = useRef(onSignDetected);

    useEffect(() => {
        onEmotionDetectedRef.current = onEmotionDetected;
    }, [onEmotionDetected]);

    useEffect(() => {
        onSignDetectedRef.current = onSignDetected;
    }, [onSignDetected]);

    const syncTemporalHud = useCallback((next: TemporalHudState) => {
        setTemporalHud((prev) => {
            if (
                prev.windowFrames === next.windowFrames &&
                prev.signalFrames === next.signalFrames &&
                prev.strideFrames === next.strideFrames &&
                prev.noSignalFrames === next.noSignalFrames &&
                prev.cooldownRemainingMs === next.cooldownRemainingMs &&
                prev.hasSignal === next.hasSignal &&
                prev.inFlight === next.inFlight &&
                prev.phase === next.phase
            ) {
                return prev;
            }
            return next;
        });
    }, []);

    const trimTemporalWindowAfterCommit = useCallback(() => {
        const temporal = mpRef.current;
        if (model !== 'lstm' || !isActive) return;

        temporal.frameBuffer = [];
        temporal.signalFrameCount = 0;
        temporal.strideFrames = 0;
        temporal.noSignalFrames = 0;

        syncTemporalHud({
            windowFrames: 0,
            signalFrames: 0,
            strideFrames: 0,
            noSignalFrames: temporal.noSignalFrames,
            cooldownRemainingMs: Math.max(0, temporal.cooldownUntilMs - Date.now()),
            hasSignal: temporal.temporalPhase === 'cooldown',
            inFlight: temporal.temporalRequestInFlight,
            phase: temporal.temporalPhase,
        });
    }, [isActive, model, syncTemporalHud]);

    // Ref to hold the MediaPipe instances
    const mpRef = useRef<{
        camera: any | null;
        hands: any | null;
        emotionInterval: ReturnType<typeof setInterval> | null;
        lastRecogniseTime: number;
        frameBuffer: number[][];
        signalFrameCount: number;
        noSignalFrames: number;
        strideFrames: number;
        cooldownUntilMs: number;
        temporalPhase: TemporalSegmentPhase;
        temporalRequestInFlight: boolean;
        inputCanvas: HTMLCanvasElement | null;
        inputCtx: CanvasRenderingContext2D | null;
    }>({
        camera: null,
        hands: null,
        emotionInterval: null,
        lastRecogniseTime: 0,
        frameBuffer: [],
        signalFrameCount: 0,
        noSignalFrames: 0,
        strideFrames: 0,
        cooldownUntilMs: 0,
        temporalPhase: 'idle',
        temporalRequestInFlight: false,
        inputCanvas: null,
        inputCtx: null,
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

        // Draw
        const handLandmarks = results.multiHandLandmarks ?? [];
        for (let i = 0; i < handLandmarks.length; i++) {
            const lm = results.multiHandLandmarks[i];
            const label = results.multiHandedness?.[i]?.classification?.[0]?.label ?? 'Left';
            const isRight = label === 'Right';
            const color = isRight ? '#007FFF' : '#FF7F40';
            const mirrorOverlay = model === 'mlp';

            if (mirrorOverlay) {
                ctx.save();
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
            }

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

            if (mirrorOverlay) {
                ctx.restore();
            }

            const labelX = mirrorOverlay ? canvas.width - x2 : x1;
            ctx.fillStyle = color;
            ctx.font = '13px "Manrope"';
            ctx.fillText(isRight ? 'R' : 'L', labelX + 4, y1 - 4);
        }

        const features = extractTemporalFeatures(results);
        const hasSignal = hasTemporalSignal(features);
        const now = Date.now();

        if (model === 'mlp') {
            if (!hasSignal) {
                onSignDetectedRef.current(null, null, 0);
                return;
            }
            if (now - mpRef.current.lastRecogniseTime < RECOGNISE_DELAY) return;
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
                console.error('Recognise API failed', err);
            }
            return;
        }

        const runTemporalRecognition = async (
            sequence: number[][],
            phase: 'preview' | 'final'
        ) => {
            if (mpRef.current.temporalRequestInFlight) return;
            mpRef.current.temporalRequestInFlight = true;
            syncTemporalHud({
                windowFrames: mpRef.current.frameBuffer.length,
                signalFrames: mpRef.current.signalFrameCount,
                strideFrames: mpRef.current.strideFrames,
                noSignalFrames: mpRef.current.noSignalFrames,
                cooldownRemainingMs: Math.max(0, mpRef.current.cooldownUntilMs - Date.now()),
                hasSignal,
                inFlight: true,
                phase: mpRef.current.temporalPhase,
            });

            try {
                const res = await fetch('/api/temporal/recognise', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequence })
                });
                const data = await res.json();
                const cls = typeof data.class === 'string' ? data.class : null;
                const word = typeof data.word === 'string' ? data.word : null;
                const conf = Number.isFinite(Number(data.confidence)) ? Number(data.confidence) : 0;
                const top5 = Array.isArray(data.top5) ? data.top5 : [];
                const top1 = Number.isFinite(Number(top5[0]?.[1])) ? Number(top5[0]?.[1]) : conf;
                const top2 = Number.isFinite(Number(top5[1]?.[1])) ? Number(top5[1]?.[1]) : 0;
                onSignDetectedRef.current(word, cls, conf, {
                    margin: Math.max(0, top1 - top2),
                    phase,
                });
            } catch (err) {
                console.error('Temporal recognise API failed', err);
            } finally {
                mpRef.current.temporalRequestInFlight = false;
                syncTemporalHud({
                    windowFrames: mpRef.current.frameBuffer.length,
                    signalFrames: mpRef.current.signalFrameCount,
                    strideFrames: mpRef.current.strideFrames,
                    noSignalFrames: mpRef.current.noSignalFrames,
                    cooldownRemainingMs: Math.max(0, mpRef.current.cooldownUntilMs - Date.now()),
                    hasSignal,
                    inFlight: false,
                    phase: mpRef.current.temporalPhase,
                });
            }
        };

        const temporal = mpRef.current;

        if (hasSignal) {
            temporal.noSignalFrames = 0;
            if (temporal.temporalPhase === 'cooldown') {
                if (now < temporal.cooldownUntilMs) {
                    syncTemporalHud({
                        windowFrames: 0,
                        signalFrames: 0,
                        strideFrames: 0,
                        noSignalFrames: 0,
                        cooldownRemainingMs: Math.max(0, temporal.cooldownUntilMs - now),
                        hasSignal: true,
                        inFlight: temporal.temporalRequestInFlight,
                        phase: temporal.temporalPhase,
                    });
                    return;
                }

                temporal.temporalPhase = 'collecting';
                temporal.cooldownUntilMs = 0;
                temporal.frameBuffer = [];
                temporal.signalFrameCount = 0;
                temporal.strideFrames = 0;

                syncTemporalHud({
                    windowFrames: 0,
                    signalFrames: 0,
                    strideFrames: 0,
                    noSignalFrames: 0,
                    cooldownRemainingMs: 0,
                    hasSignal: true,
                    inFlight: temporal.temporalRequestInFlight,
                    phase: temporal.temporalPhase,
                });
            }

            if (temporal.temporalPhase === 'idle') {
                temporal.temporalPhase = 'collecting';
                temporal.frameBuffer = [];
                temporal.signalFrameCount = 0;
                temporal.strideFrames = 0;
                temporal.cooldownUntilMs = 0;
            }

            temporal.signalFrameCount += 1;
            temporal.strideFrames += 1;
            temporal.frameBuffer.push(features);

            const segmentReachedMax = temporal.frameBuffer.length >= LSTM_N_FRAMES;
            const readyForPreview =
                temporal.signalFrameCount >= LSTM_MIN_INFERENCE_FRAMES &&
                temporal.strideFrames >= LSTM_STRIDE_FRAMES;

            if (segmentReachedMax) {
                const sequence = padTemporalSequence(temporal.frameBuffer);
                temporal.temporalPhase = 'cooldown';
                temporal.cooldownUntilMs = Date.now() + LSTM_COOLDOWN_MS;
                temporal.frameBuffer = [];
                temporal.signalFrameCount = 0;
                temporal.strideFrames = 0;
                await runTemporalRecognition(sequence, 'final');
                syncTemporalHud({
                    windowFrames: 0,
                    signalFrames: 0,
                    strideFrames: 0,
                    noSignalFrames: 0,
                    cooldownRemainingMs: Math.max(0, temporal.cooldownUntilMs - Date.now()),
                    hasSignal: true,
                    inFlight: temporal.temporalRequestInFlight,
                    phase: temporal.temporalPhase,
                });
                return;
            }

            if (readyForPreview) {
                temporal.strideFrames = 0;
                const sequence = padTemporalSequence(temporal.frameBuffer);
                await runTemporalRecognition(sequence, 'preview');
            } else {
                syncTemporalHud({
                    windowFrames: temporal.frameBuffer.length,
                    signalFrames: temporal.signalFrameCount,
                    strideFrames: temporal.strideFrames,
                    noSignalFrames: temporal.noSignalFrames,
                    cooldownRemainingMs: 0,
                    hasSignal: true,
                    inFlight: temporal.temporalRequestInFlight,
                    phase: temporal.temporalPhase,
                });
            }
            return;
        }

        temporal.noSignalFrames += 1;
        temporal.strideFrames = 0;

        if (temporal.temporalPhase === 'collecting' && temporal.noSignalFrames >= LSTM_NO_SIGNAL_FRAMES) {
            const sequence =
                temporal.frameBuffer.length >= LSTM_MIN_INFERENCE_FRAMES
                    ? padTemporalSequence(temporal.frameBuffer)
                    : null;
            temporal.frameBuffer = [];
            temporal.signalFrameCount = 0;
            temporal.noSignalFrames = 0;
            temporal.cooldownUntilMs = 0;
            temporal.temporalPhase = 'idle';

            if (sequence) {
                await runTemporalRecognition(sequence, 'final');
            } else {
                onSignDetectedRef.current(null, null, 0, { phase: 'reset' });
            }

            syncTemporalHud({
                windowFrames: 0,
                signalFrames: 0,
                strideFrames: 0,
                noSignalFrames: temporal.noSignalFrames,
                cooldownRemainingMs: 0,
                hasSignal: false,
                inFlight: temporal.temporalRequestInFlight,
                phase: temporal.temporalPhase,
            });
            return;
        }

        if (temporal.temporalPhase === 'cooldown' && now >= temporal.cooldownUntilMs) {
            temporal.temporalPhase = 'idle';
            temporal.noSignalFrames = 0;
            temporal.cooldownUntilMs = 0;
            onSignDetectedRef.current(null, null, 0, { phase: 'reset' });
            syncTemporalHud({
                windowFrames: 0,
                signalFrames: 0,
                strideFrames: 0,
                noSignalFrames: temporal.noSignalFrames,
                cooldownRemainingMs: 0,
                hasSignal: false,
                inFlight: temporal.temporalRequestInFlight,
                phase: temporal.temporalPhase,
            });
            return;
        }

        syncTemporalHud({
            windowFrames: temporal.frameBuffer.length,
            signalFrames: temporal.signalFrameCount,
            strideFrames: 0,
            noSignalFrames: temporal.noSignalFrames,
            cooldownRemainingMs: Math.max(0, temporal.cooldownUntilMs - now),
            hasSignal: false,
            inFlight: temporal.temporalRequestInFlight,
            phase: temporal.temporalPhase,
        });
    }, [model, syncTemporalHud]);

    // Lifecycle for Camera / Session
    useEffect(() => {
        if (commitResetNonce === 0) return;
        trimTemporalWindowAfterCommit();
    }, [commitResetNonce, trimTemporalWindowAfterCommit]);

    useEffect(() => {
        let ignore = false;

        function stopSession(opts?: { resetEmotion?: boolean }) {
            const resetEmotion = opts?.resetEmotion ?? true;

            mpRef.current.lastRecogniseTime = 0;
            mpRef.current.frameBuffer = [];
            mpRef.current.signalFrameCount = 0;
            mpRef.current.noSignalFrames = 0;
            mpRef.current.strideFrames = 0;
            mpRef.current.cooldownUntilMs = 0;
            mpRef.current.temporalPhase = 'idle';
            mpRef.current.temporalRequestInFlight = false;
            mpRef.current.inputCanvas = null;
            mpRef.current.inputCtx = null;
            syncTemporalHud({
                windowFrames: 0,
                signalFrames: 0,
                strideFrames: 0,
                noSignalFrames: 0,
                cooldownRemainingMs: 0,
                hasSignal: false,
                inFlight: false,
                phase: 'idle',
            });

            if (mpRef.current.emotionInterval) {
                clearInterval(mpRef.current.emotionInterval);
                mpRef.current.emotionInterval = null;
            }
            if (mpRef.current.camera) {
                try {
                    mpRef.current.camera.stop();
                } catch {
                    // Camera shutdown can race with teardown during rapid toggles.
                }
                mpRef.current.camera = null;
            }
            if (mpRef.current.hands) {
                try {
                    mpRef.current.hands.close();
                } catch {
                    // MediaPipe may already be disposed by the time cleanup runs.
                }
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
                } catch {
                    // Safe to ignore if the element was never fully initialized.
                }
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
                mpRef.current.frameBuffer = [];
                mpRef.current.signalFrameCount = 0;
                mpRef.current.noSignalFrames = 0;
                mpRef.current.strideFrames = 0;
                mpRef.current.cooldownUntilMs = 0;
                mpRef.current.temporalPhase = 'idle';
                mpRef.current.temporalRequestInFlight = false;

                const handsDetector = new window.Hands({
                    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`
                });

                const inputCanvas = document.createElement('canvas');
                const inputCtx = inputCanvas.getContext('2d');
                mpRef.current.inputCanvas = inputCanvas;
                mpRef.current.inputCtx = inputCtx;

                handsDetector.setOptions({
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });

                handsDetector.onResults(onHandResults);
                mpRef.current.hands = handsDetector;

                const camera = new window.Camera(videoEl, {
                    onFrame: async () => {
                        if (ignore) return;
                        if (mpRef.current.hands && videoEl) {
                            if (model === 'mlp') {
                                await mpRef.current.hands.send({ image: videoEl });
                                return;
                            }

                            const w = videoEl.videoWidth;
                            const h = videoEl.videoHeight;
                            const ctx = mpRef.current.inputCtx;
                            const canvas = mpRef.current.inputCanvas;
                            if (ctx && canvas && w && h) {
                                if (canvas.width !== w) canvas.width = w;
                                if (canvas.height !== h) canvas.height = h;
                                ctx.save();
                                ctx.clearRect(0, 0, w, h);
                                ctx.translate(w, 0);
                                ctx.scale(-1, 1);
                                ctx.drawImage(videoEl, 0, 0, w, h);
                                ctx.restore();
                                await mpRef.current.hands.send({ image: canvas });
                            } else {
                                await mpRef.current.hands.send({ image: videoEl });
                            }
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
                    } catch {
                        // Emotion polling should fail soft without interrupting the session.
                    }
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
    }, [isActive, model, onHandResults, syncTemporalHud]);

    const windowProgress = Math.min(1, temporalHud.windowFrames / LSTM_N_FRAMES);
    const readyProgress = temporalHud.phase === 'cooldown'
        ? Math.min(1, (LSTM_COOLDOWN_MS - temporalHud.cooldownRemainingMs) / LSTM_COOLDOWN_MS)
        : Math.min(1, temporalHud.signalFrames / LSTM_MIN_INFERENCE_FRAMES);
    const temporalPhaseLabel = temporalHud.inFlight
        ? 'Scanning'
        : temporalHud.phase === 'collecting'
            ? 'Collecting'
            : temporalHud.phase === 'cooldown'
                ? 'Cooldown'
                : 'Idle';
    const temporalPhaseDotClass = temporalHud.inFlight
        ? 'bg-[#ff8a50] shadow-[0_0_10px_rgba(255,138,80,0.55)]'
        : temporalHud.phase === 'collecting'
            ? 'bg-brand shadow-[0_0_10px_rgba(0,127,255,0.35)]'
            : temporalHud.phase === 'cooldown'
                ? 'bg-[#ffb357] shadow-[0_0_10px_rgba(255,179,87,0.4)]'
                : 'bg-[#a8bfdc]';
    const hudMetricLabel = temporalHud.phase === 'cooldown' ? 'Rearm' : 'Segment';
    const hudMetricValue = temporalHud.phase === 'cooldown'
        ? `${(temporalHud.cooldownRemainingMs / 1000).toFixed(1)}s`
        : `${Math.min(temporalHud.signalFrames, LSTM_MIN_INFERENCE_FRAMES)}/${LSTM_MIN_INFERENCE_FRAMES}`;
    const confidencePercent = Math.round(confidence * 100);
    const showPredictionHud = isActive;
    const predictionHudActive = wordLabel !== 'No sign detected' && confidencePercent > 0;
    const predictionLabel = predictionHudActive ? wordLabel : 'Awaiting sign';
    const auraRgb = EMOTION_AURA_COLORS[isActive ? currentEmotion : 'neutral'];

    return (
        <div
            className="relative h-full w-full overflow-hidden rounded-2xl border bg-[rgba(4,10,24,0.8)] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition-all duration-500"
            style={{
                borderColor: `rgba(${auraRgb},${isActive ? 0.28 : 0.13})`,
                boxShadow: isActive
                    ? `inset 0 1px 0 rgba(255,255,255,0.04), 0 0 0 1px rgba(${auraRgb},0.18), 0 18px 36px rgba(${auraRgb},0.16)`
                    : 'inset 0 1px 0 rgba(255,255,255,0.04)',
            }}
        >
            <div
                className="relative h-full w-full overflow-hidden bg-[rgba(8,16,36,0.6)]"
                style={{ borderRadius: 'inherit' }}
            >
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className={`block h-full w-full scale-x-[-1] object-cover transition-all duration-500 ${isActive ? 'opacity-100' : 'opacity-55 saturate-75'}`}
                    style={{ borderRadius: 'inherit' }}
                />
                {!isActive && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                        <div className="flex h-20 w-20 items-center justify-center rounded-full border border-[rgba(51,153,255,0.12)] bg-[rgba(8,16,36,0.32)]">
                            <Camera size={34} className="text-[rgba(51,153,255,0.15)]" strokeWidth={1.75} />
                        </div>
                    </div>
                )}
                <canvas
                    ref={canvasRef}
                    className="pointer-events-none absolute inset-0 h-full w-full"
                    style={{ borderRadius: 'inherit' }}
                />
                <div
                    className="pointer-events-none absolute inset-0 border transition-all duration-500"
                    style={{
                        borderColor: isActive ? `rgba(${auraRgb},0.48)` : 'transparent',
                    }}
                />

                {(showPredictionHud || (model === 'lstm' && isActive)) && (
                    <div className="absolute bottom-4 left-4 z-10 flex max-w-[calc(100%-2rem)] flex-col gap-2.5">
                        {showPredictionHud && (
                            <div
                                className="flex max-w-[260px] items-center gap-2 rounded-full px-4 py-2 text-white shadow-[0_10px_24px_rgba(0,0,0,0.22)] backdrop-blur-md transition-all duration-200"
                                style={{
                                    border: predictionHudActive ? '1px solid rgba(255,255,255,0.12)' : '1px solid rgba(255,255,255,0.08)',
                                    background: predictionHudActive ? 'rgba(3,18,40,0.72)' : 'rgba(3,18,40,0.52)',
                                }}
                            >
                                <span
                                    className="h-2.5 w-2.5 flex-shrink-0 rounded-full transition-all duration-200"
                                    style={{
                                        background: predictionHudActive ? '#44d9a0' : 'rgba(160,190,230,0.7)',
                                        boxShadow: predictionHudActive ? '0 0 10px rgba(68,217,160,0.7)' : 'none',
                                    }}
                                />
                                <span
                                    className="truncate text-[0.66rem] font-semibold uppercase tracking-[0.14em] transition-colors duration-200"
                                    style={{ color: predictionHudActive ? 'rgba(255,255,255,0.78)' : 'rgba(220,232,255,0.56)' }}
                                >
                                    {predictionLabel}
                                </span>
                                <span
                                    className="font-mono text-[0.74rem] font-bold transition-colors duration-200"
                                    style={{ color: predictionHudActive ? '#8cd2ff' : 'rgba(170,198,236,0.58)' }}
                                >
                                    {confidencePercent}%
                                </span>
                            </div>
                        )}

                        {model === 'lstm' && isActive && (
                            <div className="flex w-[min(240px,calc(100vw-3rem))] items-center gap-3 rounded-2xl border border-white/15 bg-[#031228]/72 px-4 py-3 text-white shadow-[0_14px_28px_rgba(0,0,0,0.24)] backdrop-blur-md">
                                <span className={`h-2 w-2 rounded-full transition-all duration-300 ${temporalPhaseDotClass}`} />
                                <div className="min-w-0 flex-1">
                                    <div className="flex items-center justify-between gap-2 text-[0.58rem] font-bold uppercase tracking-[0.18em] text-white/65">
                                        <span>{temporalPhaseLabel}</span>
                                        <span>{hudMetricLabel}</span>
                                    </div>
                                    <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-white/12">
                                        <div
                                            className="h-full rounded-full bg-gradient-to-r from-[#53b1ff] to-[#9ed7ff] transition-all duration-300"
                                            style={{ width: `${Math.max(windowProgress, readyProgress) * 100}%` }}
                                        />
                                    </div>
                                    <div className="mt-1 text-[0.64rem] font-medium tracking-[0.08em] text-white/78">
                                        {temporalHud.windowFrames}/{LSTM_N_FRAMES} · {hudMetricValue}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

            </div>
        </div>
    );
}
