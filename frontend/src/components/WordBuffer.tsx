import { motion, AnimatePresence } from 'framer-motion';

interface WordBufferProps {
    words: string[];
}

export function WordBuffer({ words }: WordBufferProps) {
    return (
        <div className="p-4 border-t border-border-color min-h-[72px] flex flex-wrap gap-2 items-center">
            {words.length === 0 ? (
                <span className="text-muted text-[0.85rem] italic">
                    Words will appear here as you sign...
                </span>
            ) : (
                <AnimatePresence>
                    {words.map((word, i) => (
                        <motion.span
                            key={`${word}-${i}`}
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            transition={{ duration: 0.2 }}
                            className="bg-[rgba(0,212,170,0.1)] border border-[rgba(0,212,170,0.3)] text-brand px-3 py-1 rounded-full text-[0.85rem] font-medium block"
                        >
                            {word}
                        </motion.span>
                    ))}
                </AnimatePresence>
            )}
        </div>
    );
}
