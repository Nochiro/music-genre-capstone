import React from 'react';
import { ComparisonResponse } from '../types/api';
import { CheckCircle } from 'lucide-react';

interface ModelComparisonProps {
  comparison: ComparisonResponse;
}

export default function ModelComparison({ comparison }: ModelComparisonProps) {
  const modelColors: Record<string, string> = {
    cnn: 'from-blue-500 to-blue-600',
    lstm: 'from-purple-500 to-purple-600',
    resnet: 'from-pink-500 to-pink-600',
  };

  const maxConfidence = Math.max(...comparison.predictions.map(p => p.confidence));

  return (
    <div className="space-y-8 fade-in">
      {/* Model Predictions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {comparison.predictions.map((pred) => (
          <div key={pred.model} className="glass-panel p-6 rounded-xl border-slate-700 hover:border-slate-600 transition-all">
            <p className="text-xs font-semibold text-slate-400 uppercase mb-4">{pred.model}</p>

            <div className="mb-4">
              <p className="text-2xl font-bold text-slate-100 mb-2">{pred.genre}</p>
              <div className="flex items-baseline gap-2">
                <span className="text-lg font-bold gradient-text">
                  {(pred.confidence * 100).toFixed(2)}%
                </span>
              </div>
            </div>

            {/* Confidence bar */}
            <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full bg-gradient-to-r ${modelColors[pred.model]}`}
                style={{ width: `${(pred.confidence / maxConfidence) * 100}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Consensus */}
      <div className="glass-panel p-8 rounded-2xl border-emerald-500/30 bg-emerald-500/5">
        <div className="flex items-center gap-3 mb-4">
          <CheckCircle className="w-6 h-6 text-emerald-400" />
          <h3 className="text-xl font-bold text-slate-100">Model Consensus</h3>
        </div>

        <p className="text-4xl font-bold gradient-text">{comparison.consensus_genre}</p>
        <p className="text-sm text-slate-400 mt-2">
          {comparison.predictions.filter(p => p.genre === comparison.consensus_genre).length} of 3 models agree
        </p>
      </div>

      {/* Confidence Comparison Chart */}
      <div className="glass-panel p-6 rounded-xl">
        <h3 className="text-lg font-semibold text-slate-100 mb-6">Confidence Comparison</h3>

        <div className="space-y-4">
          {comparison.predictions.map((pred) => (
            <div key={pred.model} className="group">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-semibold text-slate-300 capitalize">{pred.model}</span>
                <span className="text-sm font-bold text-violet-300">{(pred.confidence * 100).toFixed(1)}%</span>
              </div>

              <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full bg-gradient-to-r ${modelColors[pred.model]} transition-all duration-500 group-hover:shadow-lg`}
                  style={{ width: `${pred.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
