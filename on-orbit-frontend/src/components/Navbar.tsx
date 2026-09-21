/* eslint-disable @typescript-eslint/no-unused-vars */
"use client";

import Link from 'next/link';
import Image from 'next/image';
import { useSyncExternalStore } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { worksans } from '@/app/styles/font';
import { ChartPie, Satellite, User } from 'lucide-react';
import {
    clearTokens,
    getAuthServerSnapshot,
    getAuthSnapshot,
    getUsernameSnapshot,
    subscribeToAuth,
} from "@/lib/api";

export default function Navbar() {
    const router = useRouter();
    const pathname = usePathname();

    // Read straight from the auth store rather than copying localStorage into
    // state on mount. The old effect ran once and never again, so the navbar
    // kept showing a stale logged-in state after a session expired, and never
    // reacted to a logout in another tab. The server snapshot is "logged out",
    // so server and first client render agree and hydration stays clean.
    const token = useSyncExternalStore(
        subscribeToAuth,
        getAuthSnapshot,
        getAuthServerSnapshot,
    );
    const storedName = useSyncExternalStore(
        subscribeToAuth,
        getUsernameSnapshot,
        () => null,
    );
    const isLoggedIn = token !== null;
    const username = storedName ?? '';

    const handleLogout = () => {
        // Clears the refresh token too; leaving it behind meant a logged-out
        // browser still held a credential good for seven days.
        clearTokens();
        // No setIsLoggedIn here: clearTokens notifies the auth store and the
        // subscription above re-renders this component.
        router.push('/');
    };

    const isDashboardOrCesium = pathname === "/dashboard" || pathname && pathname.startsWith("/cesium-view") || pathname && pathname.startsWith("/maneuvering") || pathname && pathname.startsWith("/user");

    return (
        <div className={`bg-white ${worksans.className}`}>
            {/* Top navbar for other pages */}
            {!isDashboardOrCesium && (
                <nav className="px-5 py-3 shadow-sm flex justify-between items-center">
                    {isLoggedIn ? (
                        <Link href="/dashboard" className='flex items-center gap-2'>
                            <Image src="/logo.png" alt="logo" width={172.8} height={36} />
                        </Link>
                    ) : (
                        <Link href="/" className='flex items-center gap-2'>
                            <Image src="/logo.png" alt="logo" width={172.8} height={36} />
                        </Link>
                    )}

                    <div className="flex items-center gap-5 text-black">
                        <Link href="/about">About</Link>
                        {isLoggedIn ? (
                            <>
                                <button onClick={() => router.push('/dashboard')}>Dashboard</button>
                                <button onClick={handleLogout}>Logout</button>
                            </>
                        ) : (
                            <>
                                {pathname !== "/login" && (
                                    <Link href="/login"><button>Login</button></Link>
                                )}
                                {pathname !== "/signup" && (
                                    <Link href="/signup"><button>Sign Up</button></Link>
                                )}
                            </>
                        )}
                    </div>
                </nav>
            )}

            {/* Sidebar navbar for dashboard + cesium-view */}
            {isDashboardOrCesium && (
                <div className='flex h-screen'>
                    <nav className="w-[250px] flex-shrink-0 fixed left-0 top-0 h-screen border-r pl-5 pr-10 py-5 bg-white z-10 flex flex-col justify-between">
                        {isLoggedIn ? (
                            <div className='flex flex-col gap-10'>
                                <Link href="/dashboard" className='flex items-center'>
                                    <Image src="/logo.png" alt="logo" width={172.8} height={36} />
                                </Link>

                                <div className='flex flex-col gap-2'>
                                    <Link href="/dashboard">
                                        <span className={`py-2 px-5 rounded-xl flex gap-2 items-center ${
                                            pathname === "/dashboard" ? "bg-[#f9f9fa] shadow-sm" : ""
                                        }`}>
                                            <ChartPie className='h-4 w-4' />
                                            Overview
                                        </span>
                                    </Link>

                                    <Link href="/cesium-view">
                                        <span className={`py-2 px-5 rounded-xl flex gap-2 items-center ${
                                            pathname && pathname.startsWith("/cesium-view") ? "bg-[#f9f9fa] shadow-sm" : ""
                                        }`}>
                                            <Satellite className='h-4 w-4' />
                                            Visualization
                                        </span>
                                    </Link>
                                    
                                    <Link href="/ml">
                                        <span className={`py-2 px-5 rounded-xl flex gap-2 items-center ${
                                            pathname === "/ml" ? "bg-[#f9f9fa] shadow-sm" : ""
                                        }`}>
                                            <svg className='h-4 w-4' xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                            </svg>
                                            ML Models
                                        </span>
                                    </Link>
                                    
                                    <Link href="/user">
                                        <span className={`py-2 px-5 rounded-xl flex gap-2 items-center ${
                                            pathname === "/user" ? "bg-[#f9f9fa] shadow-sm" : ""
                                        }`}>
                                            <User className='h-4 w-4' />
                                            Profile
                                        </span>
                                    </Link>
                                </div>
                            </div>
                        ) : (
                            <Link href="/" className='flex items-center gap-2'>
                                <Image src="/logo.png" alt="logo" width={172.8} height={36} />
                            </Link>
                        )}

                        <div className="flex flex-col w-full items-start justify-center gap-5 text-black">
                            <Link href="/about">About</Link>
                            {isLoggedIn ? (
                                <button onClick={handleLogout}>Logout</button>
                            ) : (
                                <Link href="/signup"><button>Sign Up</button></Link>
                            )}
                        </div>
                    </nav>
                </div>
            )}
        </div>
    );
}
