import { useState, useRef, useCallback } from 'react';
import { WebcamPane } from '../components/WebcamPane';
import { WordBuffer } from '../components/WordBuffer';
import { SentenceOutput } from '../components/SentenceOutput';
import { EmotionStrip } from '../components/EmotionStrip';
import type { EmotionType } from '../components/EmotionStrip';
import { Play, Square, RotateCcw, XCircle } from 'lucide-react';

const HOLD_FRAMES = 10;
const MIN_CONFIDENCE = 0.60;

export function Communicate() {
    const [sessionActive, setSessionActive] = useState(false);

    // Buffers and emotion tracking
    const [wordBuffer, setWordBuffer] = useState<string[]>([]);
    const [currentEmotion, setCurrentEmotion] = useState<EmotionType>('neutral');

    // Live Recognition state
    const [signLabel, setSignLabel] = useState('No sign detected');
    const [confidence, setConfidence] = useState(0);

    // Audio / Sentence generation output
    const [sentence, setSentence] = useState<string | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [audioFilename, setAudioFilename] = useState<string | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);

    // Tracking frames to require a "hold" before accepting a word
    const trackingRef = useRef({
        holdCounter: 0,
        lastWord: '',
        currentClass: null as string | null,
    });

    const handleSignDetected = useCallback((word: string | null, cls: string | null, conf: number) => {
        setConfidence(conf);
        setSignLabel(word ? `${cls} \u2192 ${word}` : (cls || 'NOTHING'));

        const t = trackingRef.current;

        if (cls !== t.currentClass) {
            t.currentClass = cls;
            t.holdCounter = 0;
            return;
        }

        if (conf >= MIN_CONFIDENCE) {
            trackingRef.current.holdCounter++;
        } else {
            // Soft-decay so momentary dips don't instantly kill progress.
            trackingRef.current.holdCounter = Math.max(0, trackingRef.current.holdCounter - 1);
        }

        if (trackingRef.current.holdCounter >= HOLD_FRAMES && word && word !== trackingRef.current.lastWord) {
            setWordBuffer(prev => [...prev, word]);
            trackingRef.current.lastWord = word;
            trackingRef.current.holdCounter = 0;
        }
    }, []);

    const removeLastWord = () => setWordBuffer(prev => prev.slice(0, -1));
    const clearWords = () => {
        setWordBuffer([]);
        setSentence(null);
        setAudioUrl(null);
        setAudioFilename(null);
        trackingRef.current.lastWord = '';
        trackingRef.current.holdCounter = 0;
        trackingRef.current.currentClass = null;
    };

    const handleGenerateAndSpeak = async () => {
        if (wordBuffer.length === 0) return;
        setIsGenerating(true);

        try {
            const res = await fetch('/api/generate_and_speak_async', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ words: wordBuffer, emotion: currentEmotion })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Generation failed');

            setSentence(data.sentence);
            setAudioFilename(data.filename);

            // Polling loop
            const deadline = Date.now() + 5 * 60 * 1000;
            let jobDone = false;
            while (Date.now() < deadline) {
                const statusRes = await fetch(data.status_url);
                const statusData = await statusRes.json();
                if (statusData.state === 'done') {
                    setAudioUrl(`${data.audio_url}?t = ${Date.now()} `);
                    jobDone = true;
                    break;
                }
                if (statusData.state === 'error') {
                    throw new Error(statusData.error || 'TTS failed');
                }
                await new Promise(r => setTimeout(r, 700));
            }
            if (!jobDone) throw new Error('Timed out waiting for audio');
        } catch (err) {
            console.error(err);
            // Fallback for toast notification system would go here
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div className="px-10 py-12 max-w-[1200px] mx-auto animate-in fade-in zoom-in-95 duration-500 ease-out">
            <div className="mb-8 pl-1">
                <h1 className="font-heading font-extrabold text-[clamp(2rem,4vw,3.2rem)] leading-[1.1] mb-3 tracking-tight">
                    Sign. Feel. <em className="text-brand not-italic">Speak.</em>
                </h1>
                <p className="text-muted text-[1rem] max-w-[520px] leading-relaxed">
                    Sign ASL words at the camera. SentiSign reads your signs and emotion simultaneously, then speaks a natural sentence in your voice.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-[1fr_400px] gap-6 items-start">
                {/* LEFT COLUMN */}
                <div className="flex flex-col">
                    <WebcamPane
                        isActive={sessionActive}
                        onEmotionDetected={setCurrentEmotion}
                        onSignDetected={handleSignDetected}
                        currentEmotion={currentEmotion}
                        wordLabel={sessionActive ? signLabel : 'No sign detected'}
                        confidence={sessionActive ? confidence : 0}
                    />
                    <div className="mt-0 border-x border-b border-border-color rounded-b-xl bg-surface/50 backdrop-blur">
                        <WordBuffer words={wordBuffer} />
                    </div>
                </div>

                {/* RIGHT COLUMN */}
                <div className="flex flex-col gap-4">
                    {/* Session Card */}
                    <div className="bg-surface border border-border-color rounded-xl p-5 shadow-sm">
                        <h3 className="font-heading font-bold text-[0.85rem] uppercase tracking-widest text-muted mb-4">
                            Session
                        </h3>
                        <button
                            onClick={() => {
                                setSessionActive((prev) => {
                                    const next = !prev;
                                    if (next) {
                                        trackingRef.current.holdCounter = 0;
                                        trackingRef.current.currentClass = null;
                                    }
                                    return next;
                                });
                            }}
                            className={`w - full py - 3.5 mb - 2.5 rounded - lg font - semibold text - [1rem] flex items - center justify - center gap - 2 transition - all ${sessionActive
                                ? 'bg-[rgba(255,95,95,0.15)] text-red hover:bg-[rgba(255,95,95,0.25)] border border-[rgba(255,95,95,0.3)]'
                                : 'bg-brand text-black hover:bg-[#00f0c3] hover:-translate-y-px'
                                } `}
                        >
                            {sessionActive ? <><Square size={18} fill="currentColor" /> Stop Session</> : <><Play size={18} fill="currentColor" /> Start Session</>}
                        </button>
                        <div className="flex gap-2.5">
                            <button onClick={removeLastWord} className="btn btn-secondary flex-1 flex items-center justify-center gap-1.5 h-11 py-0">
                                <RotateCcw size={16} /> Undo
                            </button>
                            <button onClick={clearWords} className="btn btn-danger flex-1 flex items-center justify-center gap-1.5 h-11 py-0">
                                <XCircle size={16} /> Clear
                            </button>
                        </div>
                    </div>

                    {/* Emotion Card */}
                    <div className="bg-surface border border-border-color rounded-xl p-5 shadow-sm">
                        <h3 className="font-heading font-bold text-[0.85rem] uppercase tracking-widest text-muted mb-4">
                            Emotion Detected
                        </h3>
                        <EmotionStrip currentEmotion={currentEmotion} />
                    </div>

                    {/* Output Card */}
                    <div className="bg-surface border border-border-color rounded-xl p-5 shadow-sm">
                        <h3 className="font-heading font-bold text-[0.85rem] uppercase tracking-widest text-muted mb-4">
                            Output
                        </h3>
                        <SentenceOutput
                            sentence={sentence}
                            audioUrl={audioUrl}
                            audioFilename={audioFilename}
                            isGenerating={isGenerating}
                            onGenerateAndSpeak={handleGenerateAndSpeak}
                            canGenerate={wordBuffer.length > 0}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
