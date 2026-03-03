import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface SentenceOutputProps {
    sentence: string | null;
    audioUrl?: string | null;
    audioFilename?: string | null;
    isGenerating?: boolean;
    onGenerateAndSpeak: () => void;
    canGenerate: boolean;
}

export function SentenceOutput({
    sentence,
    audioUrl,
    audioFilename,
    isGenerating,
    onGenerateAndSpeak,
    canGenerate,
}: SentenceOutputProps) {
    return (
        <div className="flex flex-col gap-4">
            <div
                className={twMerge(
                    clsx(
                        "bg-bg border border-border-color rounded-lg p-3.5 text-[1rem] leading-relaxed min-h-[60px]",
                        !sentence ? "text-muted italic" : "text-text italic"
                    )
                )}
            >
                {sentence ? `"${sentence}"` : "Sentence will appear here..."}
            </div>

            {audioUrl && (
                <div className="flex flex-col gap-2.5 animate-in fade-in duration-300">
                    <audio
                        src={audioUrl}
                        controls
                        autoPlay
                        className="w-full rounded-lg accent-brand"
                    />
                    <a
                        href={audioUrl}
                        download={audioFilename || 'sentisign.wav'}
                        className="flex items-center justify-center gap-1.5 text-brand no-underline text-[0.85rem] px-3 py-2 border border-[rgba(0,212,170,0.2)] rounded-lg transition-all duration-200 hover:bg-[rgba(0,212,170,0.1)]"
                    >
                        &darr; Download .wav
                    </a>
                </div>
            )}

            <div className="flex gap-2.5 mt-1">
                <button
                    className="btn btn-primary flex-1 flex items-center justify-center gap-2"
                    onClick={onGenerateAndSpeak}
                    disabled={!canGenerate || isGenerating}
                >
                    {isGenerating ? (
                        <>
                            <Spinner /> Synthesizing audio...
                        </>
                    ) : (
                        <>&#10022; Generate &amp; Speak</>
                    )}
                </button>
            </div>
        </div>
    );
}

function Spinner() {
    return (
        <span className="inline-block w-3.5 h-3.5 border-2 border-[rgba(0,0,0,0.3)] border-t-black rounded-full animate-[spin_0.7s_linear_infinite]" />
    );
}
