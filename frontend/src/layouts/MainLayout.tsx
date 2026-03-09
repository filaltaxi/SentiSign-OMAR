import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, LayoutGroup, motion, useReducedMotion } from 'framer-motion';
import { useModel } from '../model/ModelContext';

export function MainLayout() {
    const location = useLocation();
    const prefersReducedMotion = useReducedMotion();
    const { model, availableModels, setModel } = useModel();

    const pillTransition = prefersReducedMotion
        ? { duration: 0 }
        : { type: 'spring' as const, stiffness: 560, damping: 44, mass: 0.7 };

    return (
        <>
            <nav className="sticky top-0 z-50 h-[var(--app-nav-h)] border-b border-[rgba(51,153,255,0.10)] bg-[rgba(4,10,26,0.92)] backdrop-blur-[12px]">
                <div className="mx-auto flex h-full w-full max-w-[1400px] items-center justify-between px-4 sm:px-8">
                    <div className="flex items-center gap-2 pb-[0.08em] font-heading text-[1.25rem] font-extrabold leading-[1.1] tracking-tight text-white">
                        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-brand to-brand-end shadow-[0_8px_18px_rgba(0,127,255,0.30)]">
                            <span className="h-2.5 w-2.5 rounded-full bg-white" />
                        </span>
                        Senti<span className="inline-block text-brand-end [text-shadow:0_1px_0_rgba(0,0,0,0.18)]">Sign</span>
                    </div>

                    <LayoutGroup id="top-nav">
                        <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1 rounded-full border border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.06)] p-1 shadow-[0_10px_22px_rgba(0,0,0,0.24)]">
                        <NavLink
                            to="/"
                            className={({ isActive }) =>
                                `relative flex h-10 items-center justify-center rounded-full px-3.5 text-[0.82rem] font-semibold tracking-wide transition-colors duration-200 active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#040c1e] ${isActive ? 'text-white' : 'text-[rgba(188,207,238,0.72)] hover:bg-[rgba(255,255,255,0.06)] hover:text-white'}`
                            }
                        >
                            {({ isActive }) => (
                                <>
                                    {isActive && (
                                        <motion.span
                                            layoutId="nav-active-pill"
                                            transition={pillTransition}
                                            className="pointer-events-none absolute inset-0 rounded-full bg-gradient-to-r from-brand to-brand-end shadow-[0_10px_22px_rgba(0,127,255,0.26)]"
                                        />
                                    )}
                                    <span className="relative z-10">Communicate</span>
                                </>
                            )}
                        </NavLink>
                        <NavLink
                            to="/signs"
                            className={({ isActive }) =>
                                `relative flex h-10 items-center justify-center rounded-full px-3.5 text-[0.82rem] font-semibold tracking-wide transition-colors duration-200 active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#040c1e] ${isActive ? 'text-white' : 'text-[rgba(188,207,238,0.72)] hover:bg-[rgba(255,255,255,0.06)] hover:text-white'}`
                            }
                        >
                            {({ isActive }) => (
                                <>
                                    {isActive && (
                                        <motion.span
                                            layoutId="nav-active-pill"
                                            transition={pillTransition}
                                            className="pointer-events-none absolute inset-0 rounded-full bg-gradient-to-r from-brand to-brand-end shadow-[0_10px_22px_rgba(0,127,255,0.26)]"
                                        />
                                    )}
                                    <span className="relative z-10">Signs</span>
                                </>
                            )}
                        </NavLink>
                        <NavLink
                            to="/contribute"
                            className={({ isActive }) =>
                                `relative flex h-10 items-center justify-center rounded-full px-3.5 text-[0.82rem] font-semibold tracking-wide transition-colors duration-200 active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#040c1e] ${isActive ? 'text-white' : 'text-[rgba(188,207,238,0.72)] hover:bg-[rgba(255,255,255,0.06)] hover:text-white'}`
                            }
                        >
                            {({ isActive }) => (
                                <>
                                    {isActive && (
                                        <motion.span
                                            layoutId="nav-active-pill"
                                            transition={pillTransition}
                                            className="pointer-events-none absolute inset-0 rounded-full bg-gradient-to-r from-brand to-brand-end shadow-[0_10px_22px_rgba(0,127,255,0.26)]"
                                        />
                                    )}
                                    <span className="relative z-10">Contribute</span>
                                </>
                            )}
                        </NavLink>
                        <NavLink
                            to="/about"
                            className={({ isActive }) =>
                                `relative flex h-10 items-center justify-center rounded-full px-3.5 text-[0.82rem] font-semibold tracking-wide transition-colors duration-200 active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#040c1e] ${isActive ? 'text-white' : 'text-[rgba(188,207,238,0.72)] hover:bg-[rgba(255,255,255,0.06)] hover:text-white'}`
                            }
                        >
                            {({ isActive }) => (
                                <>
                                    {isActive && (
                                        <motion.span
                                            layoutId="nav-active-pill"
                                            transition={pillTransition}
                                            className="pointer-events-none absolute inset-0 rounded-full bg-gradient-to-r from-brand to-brand-end shadow-[0_10px_22px_rgba(0,127,255,0.26)]"
                                        />
                                    )}
                                    <span className="relative z-10">About</span>
                                </>
                            )}
                        </NavLink>
                        </div>

                        <div className="flex items-center gap-1 rounded-full border border-[rgba(255,255,255,0.10)] bg-[rgba(255,255,255,0.06)] p-1 shadow-[0_10px_22px_rgba(0,0,0,0.24)]">
                            <button
                                type="button"
                                onClick={() => setModel('mlp')}
                                className={`h-8 rounded-full px-3 text-[0.72rem] font-extrabold uppercase tracking-[0.12em] transition-colors ${model === 'mlp' ? 'bg-[rgba(51,153,255,0.16)] text-[#8fcbff]' : 'text-[rgba(188,207,238,0.72)] hover:bg-[rgba(255,255,255,0.06)] hover:text-white'}`}
                            >
                                CSL
                            </button>
                            <button
                                type="button"
                                onClick={() => setModel('lstm')}
                                className={`h-8 rounded-full px-3 text-[0.72rem] font-extrabold uppercase tracking-[0.12em] transition-colors ${model === 'lstm' ? 'bg-[rgba(200,90,33,0.18)] text-[#ffb07c]' : 'text-[rgba(188,207,238,0.72)] hover:bg-[rgba(255,255,255,0.06)] hover:text-white'}`}
                                title={availableModels.lstm ? 'Temporal LSTM' : 'Temporal LSTM (train required)'}
                            >
                                ASL
                            </button>
                        </div>
                        </div>
                    </LayoutGroup>
                </div>
            </nav>

            <main>
                <AnimatePresence mode="wait" initial={false}>
                    <motion.div
                        key={location.pathname}
                        initial={prefersReducedMotion ? false : { opacity: 0, y: 8 }}
                        animate={prefersReducedMotion ? { opacity: 1, y: 0 } : { opacity: 1, y: 0 }}
                        exit={prefersReducedMotion ? { opacity: 1, y: 0 } : { opacity: 0, y: -6 }}
                        transition={
                            prefersReducedMotion
                                ? { duration: 0 }
                                : { duration: 0.18, ease: [0.2, 0.9, 0.2, 1] }
                        }
                        className="min-h-[calc(100dvh-var(--app-nav-h))]"
                    >
                        <Outlet />
                    </motion.div>
                </AnimatePresence>
            </main>
        </>
    );
}
