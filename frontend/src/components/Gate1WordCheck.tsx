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
            className="bg-surface border border-border-color rounded-xl p-7 shadow-sm"
        >
            <h2 className="font-heading font-bold text-[1.2rem] mb-2 tracking-tight">Gate 1 &mdash; Word Check</h2>
            <p className="text-muted text-[0.95rem] mb-6 leading-relaxed">
                Enter the English word you want to add a sign for. We'll check instantly if it already exists.
            </p>

            <div className="flex gap-3">
                <input
                    type="text"
                    placeholder="e.g. AMBULANCE"
                    value={word}
                    onChange={(e) => setWord(e.target.value.toUpperCase())}
                    onKeyDown={(e) => e.key === 'Enter' && handleCheck()}
                    className="flex-1 bg-bg border border-border-color rounded-lg px-4 py-3 text-text font-sans outline-none focus:border-brand transition-colors placeholder:text-muted placeholder:normal-case font-medium tracking-wide"
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

                                <div className="flex flex-col sm:flex-row items-center sm:items-start gap-5 bg-bg border border-border-color rounded-lg p-5">
                                    {result.gifUrl ? (
                                        <img src={result.gifUrl} alt={word} className="w-[120px] h-[120px] object-cover rounded-lg border border-border-color shadow-sm" />
                                    ) : (
                                        <div className="w-[120px] h-[120px] bg-border-color rounded-lg flex items-center justify-center text-[2.5rem] shadow-inner select-none">&#9995;</div>
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
