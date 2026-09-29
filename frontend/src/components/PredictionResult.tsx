import React from 'react';
import { PredictionResponse } from '../types/api';

interface PredictionResultProps {
  prediction: PredictionResponse;
}

export default function PredictionResult({ prediction }: PredictionResultProps) {
  const sortedGenres = Object.entries(prediction.probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 16);

  const maxProb = Math.max(...Object.values(prediction.probabilities));

  return (
    <div className="space-y-8 fade-in">
      {/* Main Prediction */}
      <div className="glass-panel p-8 rounded-2xl border-violet-500/20">
        <p className="text-sm text-slate-400 mb-3">Predicted Genre</p>
        <h2 className="text-5xl font-bold gradient-text mb-4">{prediction.genre}</h2>
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-bold text-slate-100">
            {(prediction.confidence * 100).toFixed(2)}%
          </span>
          <span className="text-slate-400">Confidence</span>
        </div>
      </div>

      {/* Probability Distribution */}
      <div className="glass-panel p-6 rounded-2xl">
        <h3 className="text-lg font-semibold text-slate-100 mb-6">Genre Probabilities</h3>

        <div className="space-y-3">
          {sortedGenres.map(([genre, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            const barWidth = (prob / maxProb) * 100;

            return (
              <div key={genre} className="group">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-slate-300 group-hover:text-violet-300 transition-colors">
                    {genre}
                  </span>
                  <span className="text-xs font-semibold text-slate-400">{percentage}%</span>
                </div>

                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-violet-500 to-cyan-500 rounded-full transition-all duration-500 group-hover:shadow-lg group-hover:shadow-violet-500/30"
                    style={{ width: `${barWidth}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Model Info */}
      <div className="glass-panel p-4 rounded-lg text-center">
        <p className="text-xs text-slate-400">
          Analyzed with <span className="font-semibold text-violet-300">{prediction.model.toUpperCase()}</span>
        </p>
      </div>
    </div>
  );
}
