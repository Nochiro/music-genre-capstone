import React from 'react';
import { Brain } from 'lucide-react';

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
  disabled?: boolean;
}

const MODELS = [
  {
    id: 'cnn',
    name: 'CNN',
    fullName: 'Convolutional Neural Network',
    description: 'Spectrogram pattern recognition',
    params: '102,416',
  },
  {
    id: 'lstm',
    name: 'LSTM',
    fullName: 'Long Short-Term Memory',
    description: 'Temporal sequence modeling',
    params: '273,488',
  },
  {
    id: 'resnet',
    name: 'ResNet18',
    fullName: 'Residual Network',
    description: 'Deep spectrogram feature extraction',
    params: '11,178,448',
  },
  {
    id: 'compare',
    name: 'Compare All',
    fullName: 'Multi-Model Ensemble',
    description: 'Run all three models simultaneously',
    params: 'Combined',
  },
];

export default function ModelSelector({ selectedModel, onModelChange, disabled }: ModelSelectorProps) {
  return (
    <div className="w-full fade-in">
      <p className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
        <Brain className="w-4 h-4" />
        Select Analysis Model
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {MODELS.map((model) => (
          <button
            key={model.id}
            onClick={() => onModelChange(model.id)}
            disabled={disabled}
            className={`
              relative p-4 rounded-lg text-left transition-all duration-300
              ${
                selectedModel === model.id
                  ? 'glass-panel ring-2 ring-violet-400 bg-violet-500/10'
                  : 'glass-panel hover:bg-white/10 border-slate-700'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <div className="flex items-start justify-between mb-2">
              <span className="font-semibold text-sm text-slate-100">{model.name}</span>
              {selectedModel === model.id && (
                <div className="w-2 h-2 rounded-full bg-violet-400"></div>
              )}
            </div>

            <p className="text-xs text-slate-400 mb-2">{model.description}</p>
            <p className="text-xs text-slate-500">
              {model.params}{model.params !== 'Combined' && ' params'}
            </p>

            {selectedModel === model.id && (
              <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-violet-500/0 to-cyan-500/0 pointer-events-none"></div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
