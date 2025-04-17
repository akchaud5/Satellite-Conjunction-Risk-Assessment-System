"use client";

import { useState, useEffect } from "react";

interface MLModel {
  id: string;
  name: string;
  model_type: string;
  status: string;
}

interface CDM {
  id: number;
  message_id: string;
  sat1_object_designator: string;
  sat2_object_designator: string;
}

interface MLPredictionFormProps {
  cdmId: number;
  onPredictionComplete?: (result: any) => void;
}

export default function MLPredictionForm({ cdmId, onPredictionComplete }: MLPredictionFormProps) {
  const [models, setModels] = useState<MLModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [predictionType, setPredictionType] = useState<string>("collision_probability");
  const [loading, setLoading] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [cdm, setCdm] = useState<CDM | null>(null);

  // Fetch ML models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const token = localStorage.getItem("token");
        if (!token) throw new Error("Authentication required");

        const response = await fetch("http://localhost:8000/api/ml/models/?status=active", {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.statusText}`);
        }

        const data = await response.json();
        setModels(data.filter((model: MLModel) => model.status === "active"));

        // Set default model if available
        if (data.length > 0) {
          // Find models matching the current prediction type
          const matchingModels = data.filter(
            (model: MLModel) => model.model_type === predictionType && model.status === "active"
          );
          
          if (matchingModels.length > 0) {
            setSelectedModel(matchingModels[0].id);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch models");
      } finally {
        setLoadingModels(false);
      }
    };

    fetchModels();
  }, [predictionType]);

  // Fetch CDM data
  useEffect(() => {
    const fetchCDM = async () => {
      try {
        const token = localStorage.getItem("token");
        if (!token) throw new Error("Authentication required");

        const response = await fetch(`http://localhost:8000/api/cdms/${cdmId}/`, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch CDM: ${response.statusText}`);
        }

        const data = await response.json();
        setCdm(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch CDM data");
      }
    };

    if (cdmId) {
      fetchCDM();
    }
  }, [cdmId]);

  const handlePredictionTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setPredictionType(e.target.value);
    setSelectedModel(""); // Reset selected model when prediction type changes
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const token = localStorage.getItem("token");
      if (!token) throw new Error("Authentication required");

      const response = await fetch("http://localhost:8000/api/ml/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          cdm_id: cdmId,
          model_id: selectedModel || undefined,
          prediction_type: predictionType,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to make prediction");
      }

      const data = await response.json();
      setResult(data);
      if (onPredictionComplete) {
        onPredictionComplete(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to make prediction");
    } finally {
      setLoading(false);
    }
  };

  // Filter models by selected prediction type
  const filteredModels = models.filter(model => model.model_type === predictionType);

  const renderResult = () => {
    if (!result) return null;

    // Format probability as exponential if very small
    const formatProbability = (value: number) => {
      if (value < 0.0001) {
        return value.toExponential(4);
      }
      return value.toFixed(6);
    };

    const getRiskColor = (category: string) => {
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

    return (
      <div className="mt-6 p-4 bg-white rounded-lg border">
        <h3 className="text-lg font-medium mb-2">Prediction Results</h3>
        <div className="space-y-1">
          {result.prediction_id && (
            <p className="text-sm text-gray-500">Prediction ID: {result.prediction_id}</p>
          )}
          
          {/* Probability result */}
          {result.predicted_probability !== undefined && (
            <div>
              <p className="flex justify-between">
                <span>Collision Probability:</span>
                <span className="font-medium">{formatProbability(result.predicted_probability)}</span>
              </p>
            </div>
          )}
          
          {/* Risk assessment result */}
          {result.risk_category && (
            <div>
              <p className="flex justify-between">
                <span>Risk Category:</span>
                <span className={getRiskColor(result.risk_category)}>{result.risk_category}</span>
              </p>
              {result.risk_score !== undefined && (
                <p className="flex justify-between">
                  <span>Risk Score:</span>
                  <span>{(result.risk_score * 100).toFixed(2)}%</span>
                </p>
              )}
            </div>
          )}
          
          {/* Miss distance result */}
          {result.predicted_miss_distance !== undefined && (
            <div>
              <p className="flex justify-between">
                <span>Predicted Miss Distance:</span>
                <span>{result.predicted_miss_distance.toFixed(3)} km</span>
              </p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full p-4 bg-[#f9f9fa] rounded-lg">
      <h2 className="text-lg font-medium mb-4">ML Prediction</h2>
      
      {cdm && (
        <div className="mb-4 text-sm">
          <p>
            <span className="text-gray-500">Conjunction:</span> {cdm.message_id || `CDM #${cdm.id}`}
          </p>
          <p>
            <span className="text-gray-500">Objects:</span> {cdm.sat1_object_designator} & {cdm.sat2_object_designator}
          </p>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-800 rounded-md text-sm">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="prediction_type" className="block text-sm font-medium mb-1">
            Prediction Type
          </label>
          <select
            id="prediction_type"
            value={predictionType}
            onChange={handlePredictionTypeChange}
            className="w-full p-2 border rounded-md"
          >
            <option value="collision_probability">Collision Probability</option>
            <option value="conjunction_risk">Risk Classification</option>
            <option value="miss_distance">Miss Distance</option>
          </select>
        </div>

        <div>
          <label htmlFor="model" className="block text-sm font-medium mb-1">
            Model
          </label>
          <select
            id="model"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full p-2 border rounded-md"
            disabled={loadingModels || filteredModels.length === 0}
          >
            <option value="">-- Select the best available model --</option>
            {filteredModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
          {loadingModels && <p className="text-sm text-gray-500 mt-1">Loading models...</p>}
          {!loadingModels && filteredModels.length === 0 && (
            <p className="text-sm text-red-500 mt-1">
              No active models available for this prediction type.
            </p>
          )}
        </div>

        <button
          type="submit"
          disabled={loading || !selectedModel}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300 flex items-center justify-center min-w-[120px]"
        >
          {loading ? (
            <>
              <span className="animate-spin mr-2 h-4 w-4 border-t-2 border-b-2 border-white rounded-full"></span>
              Processing...
            </>
          ) : (
            "Make Prediction"
          )}
        </button>
      </form>

      {renderResult()}
    </div>
  );
}
