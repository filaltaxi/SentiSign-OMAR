import { motion } from 'framer-motion';
import { NavLink } from 'react-router-dom';

interface Gate4SuccessProps {
    onReset: () => void;
    word: string;
}

export function Gate4Success({ onReset, word }: Gate4SuccessProps) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mx-auto max-w-[500px] rounded-xl border border-[rgba(68,217,160,0.2)] bg-[rgba(68,217,160,0.06)] p-8 text-center shadow-[0_18px_34px_rgba(0,127,255,0.12)] backdrop-blur-[16px]"
        >
            <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full border-2 border-[rgba(68,217,160,0.3)] bg-[rgba(68,217,160,0.12)] text-[2rem] text-[rgba(68,217,160,0.9)] shadow-[0_0_15px_rgba(68,217,160,0.18)]">
                &#10003;
            </div>
            <h2 className="font-heading font-extrabold text-[1.8rem] mb-2 tracking-tight text-text">Success!</h2>
            <p className="text-muted mb-8 text-[1.05rem]">
                The sign model for <strong>{word}</strong> has been successfully trained and added to the vocabulary.
            </p>

            <div className="flex flex-col gap-3 w-full">
                <NavLink to="/signs" className="btn btn-primary w-full text-center flex items-center justify-center py-3.5 text-[1rem]">
                    View Signs Gallery
                </NavLink>
                <button onClick={onReset} className="btn btn-secondary w-full py-3.5 text-[1rem] border-border-color">
                    Add Another Sign
                </button>
            </div>
        </motion.div>
    );
}
