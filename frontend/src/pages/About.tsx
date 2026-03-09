import { motion } from 'framer-motion';

export function About() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mx-auto max-w-[900px] px-5 py-12 md:px-10"
        >
            <h1 className="mb-4 pb-[0.08em] font-heading text-[clamp(2.2rem,5vw,3.5rem)] font-extrabold leading-[1.1] tracking-tight text-text">
                About <em className="text-brand not-italic">SentiSign</em>
            </h1>
            <p className="mb-12 max-w-[650px] text-[1.1rem] leading-relaxed text-muted">
                SentiSign is a multimodal assistive communication system that translates ASL hand signs and facial emotion into natural, emotionally-expressive speech &mdash; in real time, on a standard webcam.
            </p>

            <div className="mb-14">
                <h2 className="mb-6 font-mono text-[0.74rem] font-semibold uppercase tracking-[0.18em] text-[rgba(100,140,200,0.62)]">How It Works</h2>
                <div className="ml-2 flex flex-col gap-0 border-l border-[rgba(51,153,255,0.2)] pl-6">

                    <div className="group relative border-b border-[rgba(51,153,255,0.12)] py-6 last:border-0">
                        <div className="absolute -left-[37px] top-6 flex h-8 w-8 items-center justify-center rounded-full border border-[rgba(51,153,255,0.2)] bg-[rgba(8,16,36,0.7)] font-heading text-[1rem] font-bold text-muted shadow-[0_10px_24px_rgba(0,0,0,0.2)] transition-colors group-hover:text-brand">
                            1
                        </div>
                        <div className="mb-2 flex items-center gap-3">
                            <h3 className="font-semibold text-[1.1rem] text-text">Sign Recognition</h3>
                            <span className="text-[0.7rem] px-2 py-0.5 rounded-md bg-[rgba(51,153,255,0.12)] border border-brand/20 text-brand font-medium">MediaPipe + MLP</span>
                        </div>
                        <p className="text-muted text-[0.95rem] leading-relaxed max-w-[600px]">
                            MediaPipe Hands extracts 21 three-dimensional landmarks per hand (126 features for two hands). A lightweight MLP classifier maps normalised landmark coordinates to vocabulary words. Background, lighting, and skin tone are irrelevant &mdash; only hand geometry matters.
                        </p>
                    </div>

                    <div className="group relative border-b border-[rgba(51,153,255,0.12)] py-6 last:border-0">
                        <div className="absolute -left-[37px] top-6 flex h-8 w-8 items-center justify-center rounded-full border border-[rgba(51,153,255,0.2)] bg-[rgba(8,16,36,0.7)] font-heading text-[1rem] font-bold text-muted shadow-[0_10px_24px_rgba(0,0,0,0.2)] transition-colors group-hover:text-amber">
                            2
                        </div>
                        <div className="mb-2 flex items-center gap-3">
                            <h3 className="font-semibold text-[1.1rem] text-text">Emotion Recognition</h3>
                            <span className="text-[0.7rem] px-2 py-0.5 rounded-md bg-[rgba(255,179,71,0.1)] border border-amber/20 text-amber font-medium">ResNet CNN</span>
                        </div>
                        <p className="text-muted text-[0.95rem] leading-relaxed max-w-[600px]">
                            Simultaneously, a ResNet-based CNN detects the signer's facial emotion across seven universal categories: angry, disgust, fear, happy, neutral, sad, surprise. Both recognition tasks run in the same webcam session with no interruption.
                        </p>
                    </div>

                    <div className="group relative border-b border-[rgba(51,153,255,0.12)] py-6 last:border-0">
                        <div className="absolute -left-[37px] top-6 flex h-8 w-8 items-center justify-center rounded-full border border-[rgba(51,153,255,0.2)] bg-[rgba(8,16,36,0.7)] font-heading text-[1rem] font-bold text-muted shadow-[0_10px_24px_rgba(0,0,0,0.2)] transition-colors group-hover:text-brand">
                            3
                        </div>
                        <div className="mb-2 flex items-center gap-3">
                            <h3 className="font-semibold text-[1.1rem] text-text">Sentence Generation</h3>
                            <span className="text-[0.7rem] px-2 py-0.5 rounded-md bg-[rgba(51,153,255,0.12)] border border-brand/20 text-brand font-medium">Flan-T5-Large</span>
                        </div>
                        <p className="text-muted text-[0.95rem] leading-relaxed max-w-[600px]">
                            Recognised words are passed to Flan-T5-Large, an instruction-tuned language model that converts telegraphic word sequences into grammatically complete sentences &mdash; zero-shot, no fine-tuning required.
                        </p>
                    </div>

                    <div className="group relative border-b border-[rgba(51,153,255,0.12)] py-6 last:border-0">
                        <div className="absolute -left-[37px] top-6 flex h-8 w-8 items-center justify-center rounded-full border border-[rgba(51,153,255,0.2)] bg-[rgba(8,16,36,0.7)] font-heading text-[1rem] font-bold text-muted shadow-[0_10px_24px_rgba(0,0,0,0.2)] transition-colors group-hover:text-amber">
                            4
                        </div>
                        <div className="mb-2 flex items-center gap-3">
                            <h3 className="font-semibold text-[1.1rem] text-text">Emotion-Aware Speech</h3>
                            <span className="text-[0.7rem] px-2 py-0.5 rounded-md bg-[rgba(255,179,71,0.1)] border border-amber/20 text-amber font-medium">Cartesia default</span>
                        </div>
                        <p className="text-muted text-[0.95rem] leading-relaxed max-w-[600px]">
                            The sentence is synthesised with Cartesia Sonic-3 by default, with an optional Chatterbox fallback selectable through the backend environment. That keeps Cartesia as the main path while still allowing a local fallback engine.
                        </p>
                    </div>

                </div>
            </div>

            <div className="mb-14">
                <h2 className="mb-6 font-mono text-[0.74rem] font-semibold uppercase tracking-[0.18em] text-[rgba(100,140,200,0.62)]">Team</h2>
                <div className="mb-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
                    {[
                        { name: 'Omar Sheriff', role: 'Student, ECE' },
                        { name: 'P S Arjun', role: 'Student, ECE' },
                        { name: 'Sree Harinandan', role: 'Student, ECE' },
                        { name: 'Sudipto Bagchi', role: 'Student, ECE' },
                    ].map(member => (
                        <div key={member.name} className="flex flex-col justify-center rounded-[14px] border border-border-color bg-[rgba(8,16,36,0.7)] p-4 shadow-[0_14px_32px_rgba(0,0,0,0.22)]">
                            <span className="font-heading font-bold text-[0.95rem] text-text">{member.name}</span>
                            <span className="text-[0.8rem] text-muted">{member.role}</span>
                        </div>
                    ))}
                </div>
                <div className="mb-3 w-full rounded-[14px] border border-[rgba(51,153,255,0.35)] bg-[rgba(8,16,36,0.72)] p-4 shadow-[0_16px_36px_rgba(0,0,0,0.24)] sm:w-[calc(50%-8px)]">
                    <span className="font-heading font-bold text-[0.95rem] text-brand">Jasmin Sebastin</span>
                    <span className="block text-[0.8rem] text-muted">Guide, Asst. Professor</span>
                </div>
                <p className="text-muted text-[0.85rem] mt-4 opacity-80">Rajagiri School of Engineering and Technology, Kerala, India</p>
            </div>

            <div>
                <h2 className="mb-6 font-mono text-[0.74rem] font-semibold uppercase tracking-[0.18em] text-[rgba(100,140,200,0.62)]">Technology Stack</h2>
                <div className="flex flex-wrap gap-2.5">
                    {['Python 3.10', 'FastAPI', 'PyTorch 2.6 + CUDA', 'MediaPipe 0.10.9', 'MediaPipe.js', 'Flan-T5-Large', 'Cartesia + Chatterbox', 'ResNet CNN', 'scikit-learn', 'OpenCV'].map(tech => (
                        <div key={tech} className="rounded-lg border border-[rgba(51,153,255,0.15)] bg-[rgba(51,153,255,0.07)] px-3.5 py-1.5 text-[0.85rem] text-[rgba(160,190,240,0.75)] shadow-[0_10px_22px_rgba(0,0,0,0.16)]">
                            {tech}
                        </div>
                    ))}
                </div>
            </div>

        </motion.div>
    );
}
