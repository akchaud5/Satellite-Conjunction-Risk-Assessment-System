"use client";

import React, { useEffect, useState } from "react";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { useParams } from "next/navigation";
import Link from "next/link";
import Loading from "@/app/loading";

interface HeatmapDataPoint {
  T_hours: number;
  dv: number;
  miss_distance: number;
  pc: number;
}

interface BackendResponse {
  original: {
    sat1_initial_position: number[];
    sat1_initial_velocity: number[];
    sat2_initial_position: number[];
    sat2_initial_velocity: number[];
    miss_distance: number;
    pc_value: number;
  };
  best_maneuver: {
    T_hours_before_TCA: number;
    delta_v_m_s: number;
    pc_value: number;
    miss_distance: number;
    sat1_final_position: number[];
    sat1_final_velocity: number[];
  };
  heatmap_data: HeatmapDataPoint[];
}

export default function ManeuveringHeatmapPage() {
  const params = useParams();
  const id = params?.id as string | undefined;
  const [heatmapData, setHeatmapData] = useState<HeatmapDataPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    import("highcharts/modules/heatmap").then(({ default: HeatmapModule }) => {
      try {
        // Force the module call and ignore any errors it might throw.
        (HeatmapModule as any)(Highcharts);
      } catch (e) {
        // Ignore the error
        console.warn("Ignoring heatmap module error:", e);
      }
    });
  }, []);

  useEffect(() => {
    async function fetchHeatmap() {
      try {
        const accessToken = localStorage.getItem("token");
        if (!accessToken) {
          throw new Error("Please login to view this page");
        }

        const headers = {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        };

        const response = await fetch("http://localhost:8000/api/tradespace/", {
          method: "POST",
          headers: headers,
          body: JSON.stringify({ cdm_id: Number(id) }),
        });
        if (!response.ok) {
          throw new Error("Failed to fetch tradespace data");
        }
        const data: BackendResponse = await response.json();
        setHeatmapData(data.heatmap_data);
      } catch (err: any) {
        setError(err.message || "An error occurred");
      } finally {
        setLoading(false);
      }
    }
    if (id) {
      fetchHeatmap();
    }
  }, [id]);

  // Prepare the data for both heatmaps.
  // 1. Get unique sorted time (T_hours) and Δv (dv) values.
  const uniqueT = Array.from(new Set(heatmapData.map((pt) => pt.T_hours))).sort(
    (a, b) => a - b
  );
  const uniqueDv = Array.from(new Set(heatmapData.map((pt) => pt.dv))).sort(
    (a, b) => a - b
  );

  // 2. Build chartData for collision probability (pc).
  const pcChartData = heatmapData.map((pt) => {
    const x = uniqueT.indexOf(pt.T_hours);
    const y = uniqueDv.indexOf(pt.dv);
    return [x, y, pt.pc];
  });

  // 3. Build chartData for miss distance.
  const mdChartData = heatmapData.map((pt) => {
    const x = uniqueT.indexOf(pt.T_hours);
    const y = uniqueDv.indexOf(pt.dv);
    return [x, y, pt.miss_distance];
  });

  // Collision Probability Heatmap Options
  const pcOptions: Highcharts.Options = {
    chart: { type: "heatmap" },
    title: { text: "Maneuvering Tradespace Heatmap (Collision Probability)" },
    xAxis: {
      categories: uniqueT.map((t) => `${t.toFixed(2)} hr`),
      title: { text: "Time Before TCA" },
    },
    yAxis: {
      categories: uniqueDv.map((dv) => `${dv.toFixed(2)} m/s`),
      title: { text: "Δv" },
      reversed: true,
    },
    colorAxis: {
      min: 0,
      max: Math.max(...pcChartData.map((d) => d[2])),
      stops: [
        [0, "#ffffff"],
        [0.5, "#fdae61"],
        [1, "#d73027"],
      ],
    },
    tooltip: {
      formatter: function () {
        // @ts-ignore
        const timeLabel = this.series.xAxis.categories[this.point.x];
        // @ts-ignore
        const dvLabel = this.series.yAxis.categories[this.point.y];
        return `<b>Time:</b> ${timeLabel}<br/><b>Δv:</b> ${dvLabel}<br/><b>PC:</b> ${this.point.value.toExponential(3)}`;
      },
    },
    series: [
      {
        type: "heatmap",
        name: "Collision Probability",
        borderWidth: 1,
        data: pcChartData,
        dataLabels: { enabled: false },
      },
    ],
  };

  // Miss Distance Heatmap Options
  const mdOptions: Highcharts.Options = {
    chart: { type: "heatmap" },
    title: { text: "Maneuvering Tradespace Heatmap (Miss Distance)" },
    xAxis: {
      categories: uniqueT.map((t) => `${t.toFixed(2)} hr`),
      title: { text: "T_hours" },
    },
    yAxis: {
      categories: uniqueDv.map((dv) => `${dv.toFixed(2)} m/s`),
      title: { text: "Δv" },
      reversed: true,
    },
    colorAxis: {
      min: 0,
      max: Math.max(...mdChartData.map((d) => d[2])),
      stops: [
        [0, "#ffffff"],
        [0.5, "#fdae61"],
        [1, "#d73027"],
      ],
    },
    tooltip: {
      formatter: function () {
        // @ts-ignore
        const timeLabel = this.series.xAxis.categories[this.point.x];
        // @ts-ignore
        const dvLabel = this.series.yAxis.categories[this.point.y];
        return `<b>Time:</b> ${timeLabel}<br/><b>Δv:</b> ${dvLabel}<br/><b>Miss Distance:</b> ${this.point.value.toFixed(3)}`;
      },
    },
    series: [
      {
        type: "heatmap",
        name: "Miss Distance",
        borderWidth: 1,
        data: mdChartData,
        dataLabels: { enabled: false },
      },
    ],
  };

  if (loading) return <div className="h-screen"><Loading /></div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="p-4 ml-[250px] w-full">
      <div className="mb-4">
        <Link href="/dashboard">
          <button className="bg-gray-200 hover:bg-gray-300 text-black py-2 px-4 rounded flex items-center gap-2">
            <svg
              className="w-4 h-4"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Dashboard
          </button>
        </Link>
      </div>
      <h1 className="text-2xl font-bold mb-4">Maneuvering Heatmap</h1>
      <HighchartsReact highcharts={Highcharts} options={pcOptions} />
      <div className="mt-8">
        <HighchartsReact highcharts={Highcharts} options={mdOptions} />
      </div>
      {id && (
        <div className="mt-4 flex justify-center">
          <Link href={`/maneuvering/linear/${id}`}>
            <button className="bg-black hover:bg-gray-800 text-white py-2 px-4 rounded">
              View linear plot with best ΔV every time
            </button>
          </Link>
        </div>
      )}
    </div>
  );
}