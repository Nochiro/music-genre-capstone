import React from 'react';
import { Brain, Database, Zap } from 'lucide-react';

export default function ModelInfo() {
  const models = [
    {
      name: 'CNN',
      description: 'Analyzes spatial patterns in mel-spectrograms through stacked convolutional layers.',
      params: '102,416',
      icon: Brain,
    },
    {
      name: 'LSTM',
      description: 'Models temporal sequence information from spectrogram features across time steps.',
      params: '273,488',
      icon: Zap,
    },
    {
      name: 'ResNet18',
      description: 'Deeper residual CNN architecture enabling very deep networks without degradation.',
      params: '11,178,448',
      icon: Database,
    },
  ];

  return (
    <section className="py-12">
      <h2 className="text-2xl font-bold text-slate-100 mb-8">The Models</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {models.map((model) => {
          const Icon = model.icon;
          return (
            <div key={model.name} className="glass-panel p-6 rounded-xl border-slate-700 hover:border-violet-500/30 transition-all">
              <div className="flex items-start gap-3 mb-4">
                <div className="p-2 bg-gradient-to-br from-violet-500/20 to-cyan-500/20 rounded-lg">
                  <Icon className="w-5 h-5 text-violet-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-slate-100">{model.name}</h3>
                  <p className="text-xs text-slate-500">{model.params} parameters</p>
                </div>
              </div>

              <p className="text-sm text-slate-400 leading-relaxed">{model.description}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
