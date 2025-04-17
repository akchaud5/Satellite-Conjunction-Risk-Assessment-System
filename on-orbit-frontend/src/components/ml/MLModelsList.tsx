"use client";

import { useState, useEffect } from "react";
import { LucideLoader, AlertCircle, Check, X } from "lucide-react";

interface MLModel {
  id: string;
  name: string;
  description: string;
  model_type: string;
  algorithm: string;
  version: string;
  status: string;
  created_at: string;
  updated_at: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  mae?: number;
  rmse?: number;
}

export default function MLModelsList() {
  const [models, setModels] = useState<MLModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const token = localStorage.getItem("token");
        if (!token) throw new Error("Authentication required");

        const response = await fetch("http://localhost:8000/api/ml/models/", {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.statusText}`);
        }

        const data = await response.json();
        setModels(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch models");
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  const filteredModels = models.filter(model => {
    if (filter === "all") return true;
    return model.model_type === filter;
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "active":
        return (
          <span className="px-2 py-1 rounded-full text-xs bg-green-100 text-green-800 flex items-center gap-1">
            <Check size={12} />
            Active
          </span>
        );
      case "training":
        return (
          <span className="px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800 flex items-center gap-1">
            <LucideLoader size={12} className="animate-spin" />
            Training
          </span>
        );
      case "failed":
        return (
          <span className="px-2 py-1 rounded-full text-xs bg-red-100 text-red-800 flex items-center gap-1">
            <X size={12} />
            Failed
          </span>
        );
      default:
        return (
          <span className="px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800">
            {status}
          </span>
        );
    }
  };

  const getModelTypeName = (type: string) => {
    switch (type) {
      case "collision_probability":
        return "Collision Probability";
      case "conjunction_risk":
        return "Risk Classification";
      case "miss_distance":
        return "Miss Distance";
      default:
        return type;
    }
  };

  if (loading) {
    return (
      <div className="rounded-3xl bg-[#f9f9fa] w-full px-8 py-8">
        <div className="flex items-center justify-center h-64">
          <LucideLoader className="mr-2 h-6 w-6 animate-spin" />
          <p>Loading models...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-3xl bg-[#f9f9fa] w-full px-8 py-8">
        <div className="flex items-center justify-center h-64 text-red-500">
          <AlertCircle className="mr-2 h-6 w-6" />
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-3xl bg-[#f9f9fa] w-full px-8 py-8">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-xl font-medium">Machine Learning Models</h1>
        <div className="flex gap-2">
          <button
            className={`px-3 py-1 rounded-lg ${filter === 'all' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setFilter("all")}
          >
            All
          </button>
          <button
            className={`px-3 py-1 rounded-lg ${filter === 'collision_probability' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setFilter("collision_probability")}
          >
            Probability
          </button>
          <button
            className={`px-3 py-1 rounded-lg ${filter === 'conjunction_risk' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setFilter("conjunction_risk")}
          >
            Risk
          </button>
          <button
            className={`px-3 py-1 rounded-lg ${filter === 'miss_distance' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setFilter("miss_distance")}
          >
            Miss Distance
          </button>
        </div>
      </div>

      {filteredModels.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64">
          <p className="text-gray-500 mb-4">No models found</p>
          <p className="text-sm text-gray-400">Train a new model to get started</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white rounded-lg overflow-hidden">
            <thead className="bg-gray-100">
              <tr>
                <th className="py-3 px-4 text-left">Name</th>
                <th className="py-3 px-4 text-left">Type</th>
                <th className="py-3 px-4 text-left">Algorithm</th>
                <th className="py-3 px-4 text-left">Version</th>
                <th className="py-3 px-4 text-left">Status</th>
                <th className="py-3 px-4 text-left">Performance</th>
                <th className="py-3 px-4 text-left">Created</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((model) => (
                <tr key={model.id} className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">
                    <div className="font-medium">{model.name}</div>
                    {model.description && (
                      <div className="text-xs text-gray-500">{model.description}</div>
                    )}
                  </td>
                  <td className="py-3 px-4">{getModelTypeName(model.model_type)}</td>
                  <td className="py-3 px-4">{model.algorithm}</td>
                  <td className="py-3 px-4">{model.version}</td>
                  <td className="py-3 px-4">{getStatusBadge(model.status)}</td>
                  <td className="py-3 px-4">
                    {model.model_type === "collision_probability" ? (
                      <div className="text-sm">
                        {model.mae && <div>MAE: {model.mae.toFixed(4)}</div>}
                        {model.rmse && <div>RMSE: {model.rmse.toFixed(4)}</div>}
                      </div>
                    ) : model.model_type === "conjunction_risk" ? (
                      <div className="text-sm">
                        {model.accuracy && <div>Accuracy: {(model.accuracy * 100).toFixed(2)}%</div>}
                        {model.f1_score && <div>F1: {model.f1_score.toFixed(4)}</div>}
                      </div>
                    ) : (
                      <div className="text-sm text-gray-500">Not available</div>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    {new Date(model.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
