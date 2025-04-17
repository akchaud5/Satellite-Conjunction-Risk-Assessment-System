"use client";

import { useState } from "react";

interface FormData {
  model_name: string;
  model_type: string;
  algorithm: string;
  description: string;
  version: string;
  test_size: number;
  tune: boolean;
}

export default function TrainModelForm() {
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<FormData>({
    model_name: "",
    model_type: "collision_probability",
    algorithm: "random_forest",
    description: "",
    version: "1.0.0",
    test_size: 0.2,
    tune: false,
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    const checked = (e.target as HTMLInputElement).checked;

    setFormData(prev => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setSuccess(false);
    setError(null);

    try {
      const token = localStorage.getItem("token");
      if (!token) throw new Error("Authentication required");

      const response = await fetch("http://localhost:8000/api/ml/training/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to start training job");
      }

      const data = await response.json();
      setSuccess(true);
      // Reset form after successful submission
      setFormData({
        model_name: "",
        model_type: "collision_probability",
        algorithm: "random_forest",
        description: "",
        version: "1.0.0",
        test_size: 0.2,
        tune: false,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to train model");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-3xl bg-[#f9f9fa] w-full px-8 py-8">
      <h1 className="text-xl font-medium mb-6">Train New Model</h1>

      {success && (
        <div className="mb-6 p-4 bg-green-100 text-green-800 rounded-lg">
          Model training started successfully! You can view its status in the Models tab.
        </div>
      )}

      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-800 rounded-lg">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label htmlFor="model_name" className="block text-sm font-medium">
              Model Name *
            </label>
            <input
              type="text"
              id="model_name"
              name="model_name"
              value={formData.model_name}
              onChange={handleInputChange}
              required
              className="w-full p-2 border rounded-md"
              placeholder="My Collision Prediction Model"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="version" className="block text-sm font-medium">
              Version
            </label>
            <input
              type="text"
              id="version"
              name="version"
              value={formData.version}
              onChange={handleInputChange}
              className="w-full p-2 border rounded-md"
              placeholder="1.0.0"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="model_type" className="block text-sm font-medium">
              Model Type *
            </label>
            <select
              id="model_type"
              name="model_type"
              value={formData.model_type}
              onChange={handleInputChange}
              required
              className="w-full p-2 border rounded-md"
            >
              <option value="collision_probability">Collision Probability</option>
              <option value="conjunction_risk">Risk Classification</option>
              <option value="miss_distance">Miss Distance</option>
            </select>
          </div>

          <div className="space-y-2">
            <label htmlFor="algorithm" className="block text-sm font-medium">
              Algorithm *
            </label>
            <select
              id="algorithm"
              name="algorithm"
              value={formData.algorithm}
              onChange={handleInputChange}
              required
              className="w-full p-2 border rounded-md"
            >
              <option value="random_forest">Random Forest</option>
              <option value="gradient_boosting">Gradient Boosting</option>
              <option value="xgboost">XGBoost</option>
            </select>
          </div>

          <div className="space-y-2 md:col-span-2">
            <label htmlFor="description" className="block text-sm font-medium">
              Description
            </label>
            <textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              className="w-full p-2 border rounded-md"
              rows={3}
              placeholder="A brief description of this model..."
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="test_size" className="block text-sm font-medium">
              Test Size (0-1)
            </label>
            <input
              type="number"
              id="test_size"
              name="test_size"
              value={formData.test_size}
              onChange={handleInputChange}
              className="w-full p-2 border rounded-md"
              step="0.1"
              min="0.1"
              max="0.5"
            />
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="tune"
              name="tune"
              checked={formData.tune}
              onChange={handleInputChange}
              className="h-4 w-4 text-blue-600 rounded"
            />
            <label htmlFor="tune" className="text-sm font-medium">
              Perform Hyperparameter Tuning
            </label>
          </div>
        </div>

        <div className="pt-4">
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300 flex items-center justify-center min-w-[120px]"
          >
            {loading ? (
              <>
                <span className="animate-spin mr-2 h-4 w-4 border-t-2 border-b-2 border-white rounded-full"></span>
                Training...
              </>
            ) : (
              "Train Model"
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
