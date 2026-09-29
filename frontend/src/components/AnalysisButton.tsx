import React from 'react';
import { Zap } from 'lucide-react';

interface AnalysisButtonProps {
  onClick: () => void;
  loading: boolean;
  disabled: boolean;
}

export default function AnalysisButton({ onClick, loading, disabled }: AnalysisButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={`
        w-full py-4 px-6 rounded-lg font-semibold
        transition-all duration-300 flex items-center justify-center gap-2
        ${
          disabled || loading
            ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
            : 'bg-gradient-to-r from-violet-500 to-cyan-500 text-white button-hover hover:shadow-xl'
        }
      `}
    >
      {loading ? (
        <>
          <div className="w-5 h-5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
          <span>Analyzing...</span>
        </>
      ) : (
        <>
          <Zap className="w-5 h-5" />
          <span>Analyze Track</span>
        </>
      )}
    </button>
  );
}
