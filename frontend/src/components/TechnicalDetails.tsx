import React from 'react';
import { Music, Headphones, Radio, Zap } from 'lucide-react';

export default function TechnicalDetails() {
  const specs = [
    { label: 'Dataset', value: 'FMA Medium', icon: Music },
    { label: 'Classes', value: '16 genres', icon: Radio },
    { label: 'Input', value: '30-second mono audio', icon: Headphones },
    { label: 'Sample Rate', value: '22.05 kHz', icon: Zap },
    { label: 'Representation', value: '128-bin mel spectrogram', icon: Music },
    { label: 'Backend', value: 'FastAPI + PyTorch', icon: Zap },
  ];

  return (
    <section className="py-12 border-t border-slate-800">
      <h2 className="text-2xl font-bold text-slate-100 mb-8">Under the Hood</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {specs.map((spec) => {
          const Icon = spec.icon;
          return (
            <div key={spec.label} className="glass-panel p-4 rounded-lg border-slate-700 hover:border-slate-600 transition-all">
              <div className="flex items-center gap-3 mb-2">
                <Icon className="w-4 h-4 text-violet-400" />
                <p className="text-xs text-slate-500 uppercase">{spec.label}</p>
              </div>
              <p className="font-semibold text-slate-100">{spec.value}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
