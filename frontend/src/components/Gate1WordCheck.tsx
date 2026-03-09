import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Gate1WordCheckProps {
    onWordChecked: (word: string) => void;
    onAddMoreSamples: (word: string) => void;
}

export function Gate1WordCheck({ onWordChecked, onAddMoreSamples }: Gate1WordCheckProps) {
    const [word, setWord] = useState('');
    const [isChecking, setIsChecking] = useState(false);
    const [result, setResult] = useState<{
        status: 'idle' | 'exists' | 'error';
        message?: string;
        gifUrl?: string;
    }>({ status: 'idle' });

    const handleCheck = async () => {
        const w = word.trim().toUpperCase();
        if (!w) return;

        setIsChecking(true);
        setResult({ status: 'idle' });

        try {
            const res = await fetch('/api/signs/check', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word: w })
            });
            const data = await res.json();

            if (data.exists) {
                setResult({
                    status: 'exists',
                    message: `${w} is already in our system.`,
                    gifUrl: data.gif_url
                });
            } else {
                // Success path: tell parent to advance
                onWordChecked(w);
            }
        } catch (err: any) {
            setResult({ status: 'error', message: err.message || 'Server error occurred' });
        } finally {
            setIsChecking(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="rounded-xl border border-border-color bg-[rgba(8,16,36,0.72)] p-7 shadow-[0_18px_40px_rgba(0,0,0,0.26)] backdrop-blur-[16px]"
        >
            <h2 className="mb-2 font-heading text-[1.2rem] font-bold tracking-tight text-text">Gate 1 &mdash; Word Check</h2>
            <p className="mb-6 text-[0.95rem] leading-relaxed text-muted">
                Enter the English word you want to add a sign for. We'll check instantly if it already exists.
            </p>

            <div className="flex gap-3">
                <input
                    type="text"
                    placeholder="e.g. AMBULANCE"
                    value={word}
                    onChange={(e) => setWord(e.target.value.toUpperCase())}
                    onKeyDown={(e) => e.key === 'Enter' && handleCheck()}
                    className="flex-1 rounded-lg border border-[rgba(51,153,255,0.2)] bg-[rgba(8,16,36,0.7)] px-4 py-3 font-sans font-medium tracking-wide text-text outline-none transition-colors placeholder:normal-case placeholder:text-[rgba(100,140,200,0.45)] focus:border-brand"
                    disabled={isChecking}
                />
                <button
                    className="btn btn-primary min-w-[120px] flex items-center justify-center gap-2"
                    onClick={handleCheck}
                    disabled={!word.trim() || isChecking}
                >
                    {isChecking ? (
                        <><span className="inline-block w-3.5 h-3.5 border-2 border-[rgba(0,0,0,0.3)] border-t-black rounded-full animate-[spin_0.7s_linear_infinite]" /> Checking</>
                    ) : (
                        <>Check &rarr;</>
                    )}
                </button>
            </div>

            <AnimatePresence>
                {result.status !== 'idle' && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-6 overflow-hidden"
                    >
                        {result.status === 'error' && (
                            <div className="bg-[rgba(255,95,95,0.08)] border border-[rgba(255,95,95,0.3)] text-red px-4 py-3.5 rounded-lg text-[0.95rem] flex items-start gap-3">
                                <span className="text-[1.1rem] leading-none">&#10005;</span>
                                <span className="flex-1">{result.message}</span>
                            </div>
                        )}

                        {result.status === 'exists' && (
                            <div className="flex flex-col gap-4">
                                <div className="bg-[rgba(255,179,71,0.08)] border border-[rgba(255,179,71,0.3)] text-amber px-4 py-3.5 rounded-lg text-[0.95rem] flex items-start gap-3">
                                    <span className="text-[1.1rem] leading-none">&#9888;</span>
                                    <span className="flex-1"><strong>{word}</strong> is already in our system. Here is the sign we use:</span>
                                </div>

                                <div className="flex flex-col items-center gap-5 rounded-lg border border-border-color bg-[rgba(4,10,26,0.55)] p-5 sm:flex-row sm:items-start">
                                    {result.gifUrl ? (
                                        <img src={result.gifUrl} alt={word} className="h-[120px] w-[120px] rounded-lg border border-border-color object-cover shadow-[0_12px_28px_rgba(0,0,0,0.2)]" />
                                    ) : (
                                        <div className="flex h-[120px] w-[120px] select-none items-center justify-center rounded-lg bg-[rgba(51,153,255,0.08)] text-[2.5rem] shadow-inner">&#9995;</div>
                                    )}
                                    <div className="flex-1">
                                        <h3 className="font-heading font-bold text-[1.2rem] mb-1.5">{word}</h3>
                                        <p className="text-muted text-[0.9rem] leading-relaxed mb-4">
                                            This word is already mapped to a sign in SentiSign. You can add more training samples to improve recognition accuracy.
                                        </p>
                                        <button
                                            onClick={() => onAddMoreSamples(word)}
                                            className="btn btn-secondary text-[0.8rem] py-2 px-3.5 border-border-color text-text hover:border-brand hover:text-brand"
                                        >
                                            + Add More Samples
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}
