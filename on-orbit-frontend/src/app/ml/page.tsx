"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import MLModelsList from "@/components/ml/MLModelsList";
import TrainModelForm from "@/components/ml/TrainModelForm";
import MLPredictionHistory from "@/components/ml/MLPredictionHistory";

interface User {
  id: string;
  email: string;
  role: string;
}

export default function MLPage() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("models");
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

  if (loading) {
    return (
      <>
        <Navbar />
        <div className="pl-[250px] flex-1 h-full bg-white w-screen m-10">
          <section>
            <span>
              <span className="text-gray-400">Machine Learning&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp; </span>
              <span className="text-black">Loading...</span>
            </span>
          </section>
          <div className="flex items-center justify-center h-64">
            <p className="text-gray-500">Loading machine learning dashboard...</p>
          </div>
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <Navbar />
        <div className="pl-[250px] flex-1 h-full bg-white w-screen m-10">
          <section>
            <span>
              <span className="text-gray-400">Machine Learning&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp; </span>
              <span className="text-black">Error</span>
            </span>
          </section>
          <div className="flex items-center justify-center h-64">
            <p className="text-red-500">{error}</p>
          </div>
        </div>
      </>
    );
  }

  // Check if user has required permissions
  const isAnalyst = user?.role === "admin" || user?.role === "analyst";

  return (
    <>
      <Navbar />
      <div className="pl-[250px] flex flex-1 flex-col h-full bg-white w-screen m-10 gap-5">
        <section>
          <span>
            <span className="text-gray-400">Machine Learning&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp; </span>
            <span className="text-black">Models</span>
          </span>
        </section>

        {/* Tab navigation */}
        <div className="flex mb-4 border-b">
          <button
            className={`py-2 px-4 ${activeTab === "models" ? "border-b-2 border-blue-500 font-medium" : "text-gray-500"}`}
            onClick={() => setActiveTab("models")}
          >
            ML Models
          </button>
          <button
            className={`py-2 px-4 ${activeTab === "predictions" ? "border-b-2 border-blue-500 font-medium" : "text-gray-500"}`}
            onClick={() => setActiveTab("predictions")}
          >
            Prediction History
          </button>
          {isAnalyst && (
            <button
              className={`py-2 px-4 ${activeTab === "train" ? "border-b-2 border-blue-500 font-medium" : "text-gray-500"}`}
              onClick={() => setActiveTab("train")}
            >
              Train Model
            </button>
          )}
        </div>

        {/* Tab content */}
        {activeTab === "models" && <MLModelsList />}
        {activeTab === "predictions" && <MLPredictionHistory />}
        {activeTab === "train" && isAnalyst && <TrainModelForm />}
      </div>
    </>
  );
}
