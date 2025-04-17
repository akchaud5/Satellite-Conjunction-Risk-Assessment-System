"use client";

import { useState, useEffect } from "react";
import { LucideLoader, AlertCircle } from "lucide-react";

interface ModelPrediction {
  id: string;
  ml_model: {
    id: string;
    name: string;
    model_type: string;
  };
  cdm: {
    id: number;
    message_id: string;
    sat1_object_designator: string;
    sat2_object_designator: string;
  };
  predicted_probability: number | null;
  predicted_miss_distance: number | null;
  risk_score: number | null;
  risk_category: string | null;
  prediction_time: string;
}

export default function MLPredictionHistory() {
  const [predictions, setPredictions] = useState<ModelPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const token = localStorage.getItem("token");
        if (!token) throw new Error("Authentication required");

        const response = await fetch("http://localhost:8000/api/ml/predictions/", {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch predictions: ${response.statusText}`);
        }

        const data = await response.json();
        setPredictions(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch predictions");
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, []);

  const formatProbability = (value: number | null) => {
    if (value === null) return "N/A";
    if (value < 0.0001) {
      return value.toExponential(4);
    }
    return value.toFixed(6);
  };

  const getRiskColor = (category: string | null) => {
    if (!category) return "text-gray-500";
    switch (category.toLowerCase()) {
      case "high":
        return "text-red-600 font-medium";
      case "medium":
        return "text-orange-600";
      case "low":
        return "text-green-600";
      default:
        return "text-gray-500";
    }
  };

  if (loading) {
    return (
      <div className="rounded-3xl bg-[#f9f9fa] w-full px-8 py-8">
        <div className="flex items-center justify-center h-64">
          <LucideLoader className="mr-2 h-6 w-6 animate-spin" />
          <p>Loading prediction history...</p>
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
      <h1 className="text-xl font-medium mb-6">Prediction History</h1>

      {predictions.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64">
          <p className="text-gray-500 mb-4">No predictions found</p>
          <p className="text-sm text-gray-400">Make a prediction using one of the available models</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white rounded-lg overflow-hidden">
            <thead className="bg-gray-100">
              <tr>
                <th className="py-3 px-4 text-left">Conjunction</th>
                <th className="py-3 px-4 text-left">Model</th>
                <th className="py-3 px-4 text-left">Probability</th>
                <th className="py-3 px-4 text-left">Risk Score</th>
                <th className="py-3 px-4 text-left">Risk Category</th>
                <th className="py-3 px-4 text-left">Miss Distance</th>
                <th className="py-3 px-4 text-left">Prediction Time</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((prediction) => (
                <tr key={prediction.id} className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">
                    <div className="font-medium">{prediction.cdm.message_id || `CDM #${prediction.cdm.id}`}</div>
                    <div className="text-xs text-gray-500">
                      {prediction.cdm.sat1_object_designator} & {prediction.cdm.sat2_object_designator}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <div>{prediction.ml_model.name}</div>
                    <div className="text-xs text-gray-500">
                      {prediction.ml_model.model_type === "collision_probability"
                        ? "Probability Model"
                        : prediction.ml_model.model_type === "conjunction_risk"
                        ? "Risk Model"
                        : "Miss Distance Model"}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    {formatProbability(prediction.predicted_probability)}
                  </td>
                  <td className="py-3 px-4">
                    {prediction.risk_score !== null
                      ? (prediction.risk_score * 100).toFixed(2) + "%"
                      : "N/A"}
                  </td>
                  <td className={`py-3 px-4 ${getRiskColor(prediction.risk_category)}`}>
                    {prediction.risk_category || "N/A"}
                  </td>
                  <td className="py-3 px-4">
                    {prediction.predicted_miss_distance !== null
                      ? prediction.predicted_miss_distance.toFixed(3) + " km"
                      : "N/A"}
                  </td>
                  <td className="py-3 px-4">
                    {new Date(prediction.prediction_time).toLocaleString()}
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
