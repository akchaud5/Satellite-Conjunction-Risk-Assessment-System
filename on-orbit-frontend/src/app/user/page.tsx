"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "../../components/Navbar";

interface User {
  id: string;
  email: string;
  role: string;
  created_at: string;
}

export default function UserPage() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        setError("Please login to access this page.");
        router.push("/login");
        return;
      }

      try {
        const response = await fetch("http://localhost:8000/api/users/current_user/", {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          if (response.status === 401) {
            localStorage.removeItem("token");
            router.push("/login");
            return;
          }
          throw new Error("Failed to fetch user info.");
        }

        const data = await response.json();
        setUser(data);
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("An unknown error occurred.");
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, [router]);

  return (
    <div className="p-10 ml-[250px] w-full">
      <section>
        <span>
          <span className="text-gray-400">Dashboards&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp; </span>
          <span className="text-black">Overview</span>
        </span>
      </section>
      <div className="flex-1 p-10 bg-white min-h-screen">
        {loading ? (
          <div className="flex flex-col items-center justify-center">
            <div className="text-center">
              <div className="mb-4 text-xl font-medium">Loading user info...</div>
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
            </div>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center">
            <div className="text-center text-red-500 max-w-lg">
              <div className="mb-4 text-xl font-medium">Error</div>
              <p>{error}</p>
              <button
                onClick={() => router.push("/login")}
                className="mt-6 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Go to Login
              </button>
            </div>
          </div>
        ) : user ? (
          <div className="max-w-4xl mx-auto w-full">
            <div className="rounded-3xl bg-gray-50 flex flex-col w-full p-8 shadow-md">
              <h1 className="text-2xl text-black mb-6 text-center">
                User Profile
              </h1>
              <div className="space-y-4">
                <p>
                  <strong>Email:</strong> {user.email}
                </p>
                <p>
                  <strong>Role:</strong> {user.role}
                </p>
                <p>
                  <strong>Created At:</strong>{" "}
                  {new Date(user.created_at).toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center">
            <p>No user data available.</p>
          </div>
        )}
      </div>
    </div>
  );
}