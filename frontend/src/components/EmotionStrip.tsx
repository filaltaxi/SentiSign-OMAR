import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'] as const;
export type EmotionType = typeof EMOTIONS[number];

interface EmotionStripProps {
    currentEmotion: EmotionType;
}

export function EmotionStrip({ currentEmotion }: EmotionStripProps) {
    return (
        <div className="flex gap-1.5 flex-wrap">
            {EMOTIONS.map((emo) => (
                <div
                    key={emo}
                    className={twMerge(
                        clsx(
                            "px-2.5 py-1 rounded-full text-[0.75rem] border transition-all duration-300",
                            emo === currentEmotion
                                ? "border-brand text-brand bg-[rgba(0,212,170,0.1)]"
                                : "border-border-color text-muted"
                        )
                    )}
                >
                    {emo}
                </div>
            ))}
        </div>
    );
}
