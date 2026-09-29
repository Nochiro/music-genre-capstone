import { Music } from 'lucide-react';

export default function Header({ apiStatus }: { apiStatus: boolean }) {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-violet-500 to-cyan-500 rounded-lg">
              <Music className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">GenreLab</h1>
              <p className="text-xs text-slate-400">Deep Learning Music Analysis</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 text-sm">
              <span className={`w-2 h-2 rounded-full ${apiStatus ? 'bg-emerald-400' : 'bg-red-400'}`}></span>
              <span className="text-slate-300">
                {apiStatus ? 'API Online' : 'API Offline'}
              </span>
            </div>
            <div className="px-3 py-1 bg-white/5 border border-slate-700 rounded-full text-xs text-slate-300">
              3 Models Online
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
