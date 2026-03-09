import { useCallback, useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Play, Square } from 'lucide-react';

interface SentenceOutputProps {
    sentence: string | null;
    audioUrl?: string | null;
    audioFilename?: string | null;
    generationStage?: 'idle' | 'sentence' | 'audio';
    compact?: boolean;
    onPlaybackChange?: (speaking: boolean) => void;
}

export function SentenceOutput({
    sentence,
    audioUrl,
    audioFilename,
    generationStage = 'idle',
    compact = false,
    onPlaybackChange,
}: SentenceOutputProps) {
    const [typedSentence, setTypedSentence] = useState<string>('');
    const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
    const [autoplayBlocked, setAutoplayBlocked] = useState<boolean>(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const showAudioProcessing = generationStage === 'audio' && !audioUrl;

    useEffect(() => {
        onPlaybackChange?.(isSpeaking);
    }, [isSpeaking, onPlaybackChange]);

    useEffect(() => {
        if (!sentence) {
            Promise.resolve().then(() => setTypedSentence(''));
            return;
        }

        Promise.resolve().then(() => setTypedSentence(''));
        let charIndex = 0;
        const typeInterval = setInterval(() => {
            charIndex += 1;
            setTypedSentence(sentence.slice(0, charIndex));

            if (charIndex >= sentence.length) {
                clearInterval(typeInterval);
            }
        }, 18);

        return () => clearInterval(typeInterval);
    }, [sentence]);

    const playAudio = useCallback(async () => {
        const audioEl = audioRef.current;
        if (!audioEl) return;

        try {
            await audioEl.play();
            setAutoplayBlocked(false);
        } catch {
            setAutoplayBlocked(true);
        }
    }, []);

    useEffect(() => {
        const audioEl = audioRef.current;
        if (!audioEl) {
            setAutoplayBlocked(false);
            Promise.resolve().then(() => setIsSpeaking(false));
            return;
        }

        audioEl.pause();
        audioEl.currentTime = 0;
        setAutoplayBlocked(false);
        Promise.resolve().then(() => setIsSpeaking(false));

        if (!audioUrl) {
            return;
        }

        const handleCanPlay = () => {
            void playAudio();
        };

        if (audioEl.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
            void playAudio();
            return;
        }

        audioEl.addEventListener('canplay', handleCanPlay, { once: true });
        return () => {
            audioEl.removeEventListener('canplay', handleCanPlay);
        };
    }, [audioUrl, playAudio]);

    const stopAudio = () => {
        const audioEl = audioRef.current;
        if (!audioEl) return;
        audioEl.pause();
        audioEl.currentTime = 0;
        setIsSpeaking(false);
    };

    if (compact) {
        return (
            <div className="flex flex-col gap-2">
                <div
                    className={twMerge(
                        clsx(
                            "relative overflow-hidden rounded-2xl border border-[rgba(51,153,255,0.15)] bg-[rgba(51,153,255,0.05)] px-3 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition-all duration-300",
                            sentence ? "shadow-[0_10px_22px_rgba(0,0,0,0.18)]" : "text-muted"
                        )
                    )}
                    style={{ minHeight: 64, maxHeight: 90 }}
                >
                    <div className="pointer-events-none absolute inset-x-6 top-0 h-px bg-gradient-to-r from-transparent via-brand/30 to-transparent" />
                    {typedSentence ? (
                        <p
                            className="text-[0.98rem] font-semibold leading-[1.32] text-[rgba(220,235,255,0.92)]"
                            style={{
                                display: '-webkit-box',
                                WebkitLineClamp: 3,
                                WebkitBoxOrient: 'vertical',
                                overflow: 'hidden',
                            }}
                        >
                            {typedSentence}
                        </p>
                    ) : (
                        <p className="text-[0.86rem] font-medium text-[rgba(100,140,200,0.3)]">
                            Sentence will appear here…
                        </p>
                    )}
                </div>

                <audio
                    src={audioUrl ?? undefined}
                    ref={audioRef}
                    onPlay={() => setIsSpeaking(true)}
                    onPause={() => setIsSpeaking(false)}
                    onEnded={() => setIsSpeaking(false)}
                    className="hidden"
                />

                {showAudioProcessing && (
                    <div className="flex h-9 items-center justify-center gap-2 rounded-full border border-[rgba(51,153,255,0.16)] bg-[rgba(8,16,36,0.72)] px-4 text-[0.68rem] font-bold uppercase tracking-[0.14em] text-[rgba(140,170,220,0.75)] shadow-[0_10px_20px_rgba(0,0,0,0.18)]">
                        <span className="audio-processing-dot [animation-delay:0ms]" />
                        <span className="audio-processing-dot [animation-delay:180ms]" />
                        <span className="audio-processing-dot [animation-delay:360ms]" />
                        Processing audio
                    </div>
                )}

                {audioUrl && (
                    <div className="flex items-center gap-2">
                        <button
                            type="button"
                            onClick={() => void playAudio()}
                            className={twMerge(
                                clsx(
                                    "flex h-9 flex-1 items-center justify-center gap-1.5 rounded-xl border px-3 text-[0.74rem] font-bold transition-all duration-300",
                                    isSpeaking
                                        ? "border-[rgba(51,153,255,0.3)] bg-[rgba(51,153,255,0.16)] text-brand shadow-[0_10px_18px_rgba(0,127,255,0.12)]"
                                        : autoplayBlocked
                                            ? "border-[rgba(255,179,71,0.24)] bg-[rgba(255,179,71,0.12)] text-[#ffb347]"
                                            : "border-[rgba(51,153,255,0.18)] bg-[rgba(8,16,36,0.68)] text-brand hover:bg-[rgba(51,153,255,0.08)]"
                                )
                            )}
                        >
                            <Play size={13} fill="currentColor" /> Play
                        </button>
                        <button
                            type="button"
                            onClick={stopAudio}
                            className="flex h-9 flex-1 items-center justify-center gap-1.5 rounded-xl border border-[rgba(255,80,60,0.24)] bg-[rgba(255,80,60,0.1)] px-3 text-[0.74rem] font-bold text-[rgba(255,120,105,0.9)] transition-all duration-300 hover:bg-[rgba(255,80,60,0.16)]"
                        >
                            <Square size={13} fill="currentColor" /> Stop
                        </button>
                        <a
                            href={audioUrl}
                            download={audioFilename || 'sentisign.wav'}
                            className="flex h-9 flex-1 items-center justify-center rounded-xl border border-[rgba(51,153,255,0.18)] bg-[rgba(8,16,36,0.68)] px-3 text-[0.74rem] font-bold text-brand no-underline transition-all duration-300 hover:bg-[rgba(51,153,255,0.08)]"
                        >
                            Download
                        </a>
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-3">
            <div
                className={twMerge(
                    clsx(
                        "relative overflow-hidden rounded-[20px] border-2 border-[rgba(51,153,255,0.15)] bg-[rgba(51,153,255,0.05)] px-4 py-4 text-center shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition-all duration-300",
                        !sentence ? "text-muted" : "sentence-reveal text-text shadow-[0_12px_22px_rgba(0,0,0,0.18)]"
                    )
                )}
            >
                <div className="pointer-events-none absolute inset-x-8 top-0 h-px bg-gradient-to-r from-transparent via-brand/30 to-transparent" />
                {typedSentence ? (
                    <p className="mx-auto min-h-[76px] max-w-[20ch] text-balance text-[clamp(1.35rem,2.6vw,2rem)] font-bold leading-[1.18] text-[rgba(220,235,255,0.92)]">
                        {typedSentence}
                    </p>
                ) : (
                    <p className="mx-auto min-h-[76px] max-w-[22ch] text-balance text-[1rem] font-semibold leading-[1.35] text-[rgba(100,140,200,0.3)]">
                        Generated sentence appears here
                    </p>
                )}
            </div>

            {showAudioProcessing && (
                <div className="animate-in fade-in zoom-in-95 duration-300">
                    <div className="relative overflow-hidden rounded-[999px] border border-[rgba(51,153,255,0.16)] bg-[rgba(8,16,36,0.72)] px-5 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04),0_14px_24px_rgba(0,0,0,0.18)]">
                        <div className="pointer-events-none absolute inset-y-0 left-[-35%] w-[35%] bg-[linear-gradient(90deg,transparent,rgba(51,153,255,0.22),transparent)] animate-[audio-processing-sheen_1.8s_ease-in-out_infinite]" />
                        <div className="relative flex min-h-[46px] items-center justify-center gap-3">
                            <div className="flex items-center gap-1.5" aria-hidden="true">
                                <span className="audio-processing-dot [animation-delay:0ms]" />
                                <span className="audio-processing-dot [animation-delay:180ms]" />
                                <span className="audio-processing-dot [animation-delay:360ms]" />
                            </div>
                            <div className="flex flex-col items-start">
                                <span className="text-[0.72rem] font-extrabold uppercase tracking-[0.18em] text-[rgba(140,170,220,0.75)]">
                                    Processing Audio
                                </span>
                                <span className="text-[0.82rem] font-semibold text-[rgba(140,170,220,0.62)]">
                                    Preparing the spoken output...
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {audioUrl && (
                <div className="flex flex-col gap-2 animate-in fade-in zoom-in-95 duration-400">
                    {isSpeaking && (
                        <div className="flex items-end justify-center gap-1.5 rounded-xl border border-[rgba(51,153,255,0.16)] bg-[rgba(8,16,36,0.72)] py-2">
                            <span className="wave-bar [animation-delay:0ms]" />
                            <span className="wave-bar [animation-delay:120ms]" />
                            <span className="wave-bar [animation-delay:210ms]" />
                            <span className="wave-bar [animation-delay:300ms]" />
                            <span className="wave-bar [animation-delay:420ms]" />
                        </div>
                    )}
                    {autoplayBlocked && (
                        <div className="rounded-xl border border-[rgba(255,179,71,0.24)] bg-[rgba(255,179,71,0.12)] px-3 py-2 text-[0.76rem] font-semibold text-[#ffb347]">
                            Browser autoplay was blocked. Press play to start the audio.
                        </div>
                    )}
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
	                        <audio
	                            src={audioUrl}
	                            controls
	                            ref={audioRef}
	                            onPlay={() => setIsSpeaking(true)}
	                            onPause={() => setIsSpeaking(false)}
	                            onEnded={() => setIsSpeaking(false)}
	                            className="sentisign-audio h-11 w-full overflow-hidden rounded-xl border border-border-color bg-[#0b1220] accent-brand shadow-[0_8px_16px_rgba(15,34,68,0.08)] sm:flex-1"
	                        />
                        <div className="flex gap-2 sm:flex-none">
                            <button
                                type="button"
                                onClick={stopAudio}
                                className="flex min-w-[104px] items-center justify-center gap-2 rounded-xl border border-[rgba(255,80,60,0.24)] bg-[rgba(255,80,60,0.1)] px-4 py-2.5 text-[0.8rem] font-bold text-[rgba(255,120,105,0.9)] transition-all duration-300 hover:bg-[rgba(255,80,60,0.16)] hover:shadow-[0_10px_20px_rgba(255,59,48,0.14)]"
                            >
                                <Square size={14} fill="currentColor" /> Stop
                            </button>
                            {autoplayBlocked && (
                                <button
                                    type="button"
                                    onClick={() => void playAudio()}
                                    className="flex min-w-[104px] items-center justify-center gap-2 rounded-xl border border-[rgba(51,153,255,0.18)] bg-[rgba(8,16,36,0.68)] px-4 py-2.5 text-[0.8rem] font-bold text-brand transition-all duration-300 hover:bg-[rgba(51,153,255,0.08)] hover:shadow-[0_10px_20px_rgba(0,127,255,0.12)]"
                                >
                                    <Play size={14} fill="currentColor" /> Play
                                </button>
                            )}
                            <a
                                href={audioUrl}
                                download={audioFilename || 'sentisign.wav'}
                                className="flex min-w-[120px] items-center justify-center gap-2 rounded-xl border border-[rgba(51,153,255,0.18)] bg-[rgba(8,16,36,0.68)] px-4 py-2.5 text-[0.8rem] font-bold text-brand no-underline transition-all duration-300 hover:bg-[rgba(51,153,255,0.08)] hover:shadow-[0_10px_20px_rgba(0,127,255,0.12)]"
                            >
                                Download
                            </a>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
