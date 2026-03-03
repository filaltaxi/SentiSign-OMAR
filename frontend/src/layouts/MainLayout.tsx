import { NavLink, Outlet } from 'react-router-dom';

export function MainLayout() {
    return (
        <>
            <nav className="flex items-center justify-between px-10 py-4.5 border-b border-border-color bg-[rgba(10,14,20,0.95)] sticky top-0 z-100 backdrop-blur-md">
                <div className="font-heading font-extrabold text-[1.4rem] text-brand tracking-tight">
                    Senti<span className="text-text">Sign</span>
                </div>
                <div className="flex gap-8">
                    <NavLink
                        to="/"
                        className={({ isActive }) =>
                            `text-[0.9rem] font-medium transition-colors duration-200 hover:text-brand ${isActive ? 'text-brand' : 'text-muted'}`
                        }
                    >
                        Communicate
                    </NavLink>
                    <NavLink
                        to="/signs"
                        className={({ isActive }) =>
                            `text-[0.9rem] font-medium transition-colors duration-200 hover:text-brand ${isActive ? 'text-brand' : 'text-muted'}`
                        }
                    >
                        Signs
                    </NavLink>
                    <NavLink
                        to="/contribute"
                        className={({ isActive }) =>
                            `text-[0.9rem] font-medium transition-colors duration-200 hover:text-brand ${isActive ? 'text-brand' : 'text-muted'}`
                        }
                    >
                        Contribute
                    </NavLink>
                    <NavLink
                        to="/about"
                        className={({ isActive }) =>
                            `text-[0.9rem] font-medium transition-colors duration-200 hover:text-brand ${isActive ? 'text-brand' : 'text-muted'}`
                        }
                    >
                        About
                    </NavLink>
                </div>
            </nav>

            <main>
                <Outlet />
            </main>
        </>
    );
}
